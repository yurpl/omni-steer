from __future__ import annotations
import inspect
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Callable, Iterable

import torch
from transformers import (
    Qwen3OmniMoeForConditionalGeneration,
    Qwen3OmniMoeThinkerForConditionalGeneration,
    Qwen3OmniMoeTalkerForConditionalGeneration,
    Qwen3OmniMoeProcessor,
)
from transformers.generation.utils import ModelOutput

@dataclass
class OmniHandles:
    handles: List[Any]
    cache: Dict[str, torch.Tensor]

class OmniRunner:
    """
    Wrapper around Qwen3-Omni to:
      - build processor inputs from MELD items
      - run Thinker forward with hidden_states (for probing)
      - register forward hooks (for patching/steering)
      - run joint generation (text_ids, audio)
    """
    def __init__(self,
                 model_name: str = "Qwen/Qwen3-Omni-30B-A3B-Instruct",
                 thinker_only: bool = False,
                 dtype: str = "auto",
                 device_map: str = "auto",
                 enable_audio_output: bool = True):
        self.processor = Qwen3OmniMoeProcessor.from_pretrained(model_name)
        if thinker_only:
            self.model = Qwen3OmniMoeThinkerForConditionalGeneration.from_pretrained(
                model_name, dtype=dtype, device_map=device_map
            ).eval()
            self.thinker = self.model
            self.talker = None
        else:
            self.model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
                model_name, dtype=dtype, device_map=device_map, enable_audio_output=enable_audio_output
            ).eval()
            # The combined model exposes thinker/talker submodules:
            self.thinker = getattr(self.model, "thinker", None)
            self.talker = getattr(self.model, "talker", None)

        self.device = next(self.model.parameters()).device
        self.dtype = next(self.model.parameters()).dtype

    # ----- Conversation builders -----

    def convo_from_text(self, text: str) -> List[Dict[str, Any]]:
        return [{"role": "user", "content": [{"type": "text", "text": text}]}]

    def convo_from_video(self, video_path: str, prompt: Optional[str] = None) -> List[Dict[str, Any]]:
        content = [{"type": "video", "video": video_path}]
        if prompt:
            content.append({"type": "text", "text": prompt})
        return [{"role": "user", "content": content}]

    # ----- Input preparation -----

    def make_inputs(self, conversation: List[Dict[str, Any]],
                    use_audio_in_video: bool = True,
                    add_generation_prompt: bool = True,
                    video_fps: int = 1,
                    tokenize: bool = True,
                    return_tensors: str = "pt",
                    **proc_kwargs) -> Dict[str, torch.Tensor]:
        inputs = self.processor.apply_chat_template(
            conversation,
            load_audio_from_video=use_audio_in_video,
            use_audio_in_video=use_audio_in_video,
            add_generation_prompt=add_generation_prompt,
            tokenize=tokenize,
            return_dict=True,
            return_tensors=return_tensors,
            video_fps=video_fps,
            **proc_kwargs
        )
        # Move to device and convert floating point tensors to model dtype
        result = {}
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                v = v.to(self.device)
                if v.dtype in (torch.float32, torch.float64):
                    v = v.to(self.dtype)
                result[k] = v
            else:
                result[k] = v
        return result

    # ----- Running: hidden states for Thinker -----

    @torch.no_grad()
    def thinker_hidden_states(self, inputs: Dict[str, torch.Tensor]) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Returns (hidden_states, logits) for the Thinker given multimodal inputs.
        hidden_states is a list: one per layer (plus embeddings if config enables).
        """
        # Forward through the thinker only
        # For convenience, call the combined model with output_hidden_states to get thinker states.
        out = self.thinker(
            **{k: v for k, v in inputs.items() if isinstance(v, torch.Tensor)},
            output_hidden_states=True,
            return_dict=True
        )
        hs = list(out.hidden_states) if isinstance(out, ModelOutput) else out[2]
        logits = out.logits if isinstance(out, ModelOutput) else out[0]
        return hs, logits

    # ----- Hooks -----

    def _iter_thinker_layers(self) -> List[torch.nn.Module]:
        # try common locations
        candidates = []
        if hasattr(self.thinker, "model") and hasattr(self.thinker.model, "layers"):
            candidates = list(self.thinker.model.layers)
        elif hasattr(self.thinker, "layers"):
            candidates = list(self.thinker.layers)
        else:
            # fallback: search for modules with submodules named 'self_attn' or 'mlp'
            for m in self.thinker.modules():
                if hasattr(m, "self_attn") or hasattr(m, "mlp"):
                    candidates.append(m)
        return candidates

    def register_thinker_hooks(self,
                               names: Iterable[str] = ("resid_post", "attn_out", "mlp_out")
                               ) -> OmniHandles:
        cache: Dict[str, torch.Tensor] = {}
        handles: List[Any] = []

        layers = self._iter_thinker_layers()
        def save(name):
            def _hook(_m, _inp, out):
                # some submodules return tuples
                val = out[0] if isinstance(out, (tuple, list)) else out
                cache[name] = val.detach()
                return out
            return _hook

        for i, layer in enumerate(layers):
            # Residual-post: hook on layer output
            if "resid_post" in names:
                handles.append(layer.register_forward_hook(save(f"blocks.{i}.resid_post")))
            # Attn out
            if hasattr(layer, "self_attn") and "attn_out" in names:
                handles.append(layer.self_attn.register_forward_hook(save(f"blocks.{i}.attn_out")))
            # MLP out
            if hasattr(layer, "mlp") and "mlp_out" in names:
                handles.append(layer.mlp.register_forward_hook(save(f"blocks.{i}.mlp_out")))

        # Talker: capture code predictor logits if available
        if self.talker is not None:
            for name, mod in self.talker.named_modules():
                if "code_predictor" in name.lower():
                    handles.append(mod.register_forward_hook(save("talker.code_predictor.out")))
                    break

        return OmniHandles(handles=handles, cache=cache)

    @staticmethod
    def remove_handles(h: OmniHandles):
        for hd in h.handles:
            try: hd.remove()
            except Exception: pass

    # ----- Generation (text + audio) -----

    @torch.no_grad()
    def generate_any2any(self,
                         inputs: Dict[str, torch.Tensor],
                         thinker_do_sample: bool = False,
                         talker_do_sample: bool = True,
                         **gen_kwargs) -> Tuple[List[int], torch.Tensor]:
        """
        Joint generation: returns (text_ids, audio_waveform)
        """
        text_ids, audio = self.model.generate(
            **inputs,
            use_audio_in_video=True,
            thinker_do_sample=thinker_do_sample,
            talker_do_sample=talker_do_sample,
            **gen_kwargs
        )
        return text_ids, audio
