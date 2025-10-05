import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sae_lens import SAE, SAETrainer, SAETrainingConfig
import torch

def train_emotion_probe(data, layer, run_cache_fn, pos=-1, test_size=0.2):
    X, y = [], []
    for ex in data:
        conv = [{"role": "user", "content": [{"type": "text", "text": ex["utterance"]}]}]  # Simplify for probing
        cache = run_cache_fn(conv)
        H = cache[f"blocks.{layer}.resid_post"][0, pos, :].cpu().numpy()
        X.append(H)
        y.append(ex["emotion"])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    clf = LogisticRegression(penalty="l2", C=1.0, max_iter=200, multi_class="ovr").fit(X_train, y_train)
    preds = clf.predict(X_test)
    print(f"Macro F1: {f1_score(y_test, preds, average='macro')}")
    return clf

def train_sae(activations, d_model, expansion_factor=8, l1_coeff=0.01):
    config = SAETrainingConfig(
        model_dim=d_model,
        dict_size=d_model * expansion_factor,
        l1_coefficient=l1_coeff
    )
    trainer = SAETrainer(config)
    activations = torch.tensor(np.array(activations)).view(-1, d_model)  # Flatten
    sae = trainer.train(activations)
    return sae