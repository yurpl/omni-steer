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
        
        # Choose a safe default: fp16 on CUDA, fp32 otherwise
        _torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        if thinker_only:
            self.model = Qwen3OmniMoeThinkerForConditionalGeneration.from_pretrained(
                model_name, torch_dtype=_torch_dtype, device_map=device_map, attn_implementation="flash_attention_2"
            ).eval()
            self.thinker = self.model
            self.talker = None
        else:
            self.model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
                model_name, torch_dtype=_torch_dtype, device_map=device_map, enable_audio_output=enable_audio_output, attn_implementation="flash_attention_2"
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
                    video_fps: int = 1,  # kept for BC
                    tokenize: bool = True,
                    return_tensors: str = "pt",
                    **proc_kwargs) -> Dict[str, torch.Tensor]:
        # Use `fps` (not `video_fps`) and ensure padding so attention_mask is set
        fps = proc_kwargs.pop("fps", video_fps)
        inputs = self.processor.apply_chat_template(
            conversation,
            load_audio_from_video=use_audio_in_video,
            use_audio_in_video=use_audio_in_video,
            add_generation_prompt=add_generation_prompt,
            tokenize=tokenize,
            return_dict=True,
            return_tensors=return_tensors,
            fps=fps,                 # <- replaces video_fps
            padding=True,            # <- ensures attention_mask
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

    def decode_text(self, text_ids, inputs=None):
        """
        Robustly decode text from model.generate(...) for Qwen3-Omni.
        Accepts:
          - torch.LongTensor [B, T]
          - GenerateDecoderOnlyOutput with .sequences
          - list[list[int]]
          - list[str] (already-decoded)
        Optionally slices off the prompt using inputs["input_ids"] length.
        """
        # 1) Normalize to sequences (token IDs) when possible
        seq = text_ids
        if hasattr(seq, "sequences"):   # e.g., GenerateDecoderOnlyOutput
            seq = seq.sequences

        # 2) If it's already a list of strings, return as-is
        if isinstance(seq, list) and len(seq) > 0 and isinstance(seq[0], str):
            return seq

        # 3) If it's a tensor/ids, optionally slice off the prompt tokens
        if isinstance(seq, torch.Tensor):
            if inputs is not None and "input_ids" in inputs and seq.dim() == 2:
                start = inputs["input_ids"].shape[1]
                seq = seq[:, start:]
            return self.processor.batch_decode(seq, skip_special_tokens=True,
                                               clean_up_tokenization_spaces=False)

        # 4) If it's a Python list of lists of ints
        if isinstance(seq, list) and len(seq) > 0 and isinstance(seq[0], (list, tuple)):
            return self.processor.batch_decode(seq, skip_special_tokens=True,
                                               clean_up_tokenization_spaces=False)

        # 5) Last resort
        return [str(seq)]

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
                         **gen_kwargs) -> Tuple[List[str], torch.Tensor]:
        """
        Joint generation: returns (decoded_text, audio_waveform)
        """
        # Filter inputs to only include tensors for generation
        gen_inputs = {k: v for k, v in inputs.items() if isinstance(v, torch.Tensor)}
        outputs = self.model.generate(
            **gen_inputs,
            use_audio_in_video=True,
            thinker_do_sample=thinker_do_sample,
            talker_do_sample=talker_do_sample,
            return_dict_in_generate=True,
            output_scores=False,
            **gen_kwargs
        )

        # Text sequences - use robust decoding
        text_ids = self.decode_text(outputs, inputs=inputs)

        # Audio waveform
        audio = None
        for attr in ("audio", "audios", "audio_values", "waveforms"):
            if hasattr(outputs, attr):
                audio = getattr(outputs, attr)
                break
        if audio is None and isinstance(outputs, (tuple, list)) and len(outputs) > 1:
            audio = outputs[1]
        if audio is None:
            raise RuntimeError("Failed to retrieve generated audio waveform.")

        if isinstance(audio, torch.Tensor):
            audio_tensor = audio
        elif isinstance(audio, list) and audio and isinstance(audio[0], torch.Tensor):
            audio_tensor = audio[0]
        else:
            raise RuntimeError("Audio waveform is not a tensor or list of tensors.")

        return text_ids, audio_tensor
