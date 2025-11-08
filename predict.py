#!/usr/bin/env python3
import argparse
import json
import os
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from tabulate import tabulate


# ---------------------------
# Model
# ---------------------------
class LSTMModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.lstm(x)      # (B, L, H)
        out = out[:, -1, :]        # (B, H)
        out = self.fc(out)         # (B, 1)
        return self.sigmoid(out)   # (B, 1)


# ---------------------------
# Utilities
# ---------------------------
def select_device(use_gpu: bool) -> torch.device:
    if use_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    if use_gpu and not torch.cuda.is_available():
        print("Warning: --use_gpu was set but no CUDA device is available. Falling back to CPU.")
    return torch.device("cpu")


def load_meta(meta_path: str) -> dict:
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"{meta_path} not found. Please run training first.")
    with open(meta_path, "r") as f:
        return json.load(f)


def infer_model_dims_from_state_dict(sd: dict) -> Tuple[int, int, int]:
    w_ih = sd["lstm.weight_ih_l0"]
    hidden_size = w_ih.shape[0] // 4
    input_size = w_ih.shape[1]
    num_layers = 0
    while f"lstm.weight_ih_l{num_layers}" in sd:
        num_layers += 1
    return input_size, hidden_size, num_layers


def build_model_from_ckpt(path: str, device: torch.device) -> Tuple[LSTMModel, int, int, int]:
    sd = torch.load(path, map_location="cpu")
    input_size, hidden_size, num_layers = infer_model_dims_from_state_dict(sd)
    model = LSTMModel(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers).to(device)
    model.load_state_dict(sd, strict=True)
    model.eval()
    return model, input_size, hidden_size, num_layers


def scale_sequence(seq_2d: np.ndarray) -> np.ndarray:
    scaler = MinMaxScaler()
    return scaler.fit_transform(seq_2d)


# ---------------------------
# Predictor
# ---------------------------
class MultiModelPredictor:
    def __init__(self, model_paths: List[str], seq_lengths: List[int], device: torch.device):
        # Filter to models that actually exist
        if len(model_paths) != len(seq_lengths):
            existing = [(p, L) for p, L in zip(model_paths, seq_lengths) if os.path.exists(p)]
            if not existing:
                raise FileNotFoundError("No matching model files found. Please train first.")
            model_paths, seq_lengths = zip(*existing)
            model_paths, seq_lengths = list(model_paths), list(seq_lengths)

        self.device = device
        self.seq_lengths = seq_lengths
        self.models: List[LSTMModel] = []

        first_input_size = None
        for p in model_paths:
            model, in_size, _, _ = build_model_from_ckpt(p, device)
            if first_input_size is None:
                first_input_size = in_size
            elif in_size != first_input_size:
                raise RuntimeError(
                    f"Inconsistent input_size across checkpoints. Got {in_size} vs {first_input_size}."
                )
            self.models.append(model)
        self.input_size = first_input_size

        s = sum(self.seq_lengths)
        self.default_weights = [L / s for L in self.seq_lengths]

    def _as_vec(self, x):
        """
        Ensure one time-step is a vector of length input_size.
        - If input_size==1 and x is scalar -> [x]
        - If input_size==1 and x is list/array -> take first element
        - If input_size>1 -> ensure list/array of that length
        """
        if self.input_size == 1:
            if isinstance(x, (list, tuple, np.ndarray)):
                return [float(x[0])]
            else:
                return [float(x)]
        else:
            arr = np.asarray(x, dtype=np.float32).ravel()
            if arr.size != self.input_size:
                raise ValueError(f"Expected vector of length {self.input_size}, got {arr.size}.")
            return arr.tolist()

    def _prep_tensor(self, history: list, L: int) -> torch.Tensor:
        """
        Build the input window for inference:
        - Uses the last L steps of history (NO target leak).
        - Scales each window independently.
        Returns tensor of shape (1, L, input_size).
        """
        hist_vecs = [self._as_vec(h) for h in history]
        if len(hist_vecs) < L:
            raise ValueError(f"Not enough history for L={L}. Have {len(hist_vecs)}.")
        seq = np.asarray(hist_vecs[-L:], dtype=np.float32)  # (L, input_size)
        seq = scale_sequence(seq)                           # per-window scaling
        return torch.tensor(seq[None, ...], dtype=torch.float32).to(self.device)

    def predict(self, history: list, weights: List[float] = None) -> Tuple[float, List[float]]:
        """
        Returns:
          combined_yes_prob: float
          per_length_yes_probs: List[float or None]
        """
        # Prepare weights
        weights = weights or self.default_weights
        if len(weights) != len(self.seq_lengths):
            print("Warning: --weights length does not match number of models. Ignoring custom weights.")
            weights = self.default_weights
        wsum = sum(weights)
        if wsum <= 0:
            raise ValueError("Weights must sum to > 0.")
        weights = [w / wsum for w in weights]

        probs = []
        eff_weights = []
        for model, L, w in zip(self.models, self.seq_lengths, weights):
            if len(history) < L:
                probs.append(None)
                continue
            X = self._prep_tensor(history, L)
            with torch.no_grad():
                p_yes = model(X).item()
            probs.append(p_yes)
            eff_weights.append(w)

        # Re-normalize across models that produced a prediction
        usable_probs = [p for p in probs if p is not None]
        if usable_probs:
            wsum_eff = sum(eff_weights)
            eff_weights = [w / wsum_eff for w in eff_weights]
            combined = sum(p * w for p, w in zip(usable_probs, eff_weights))
        else:
            raise ValueError(
                "History is shorter than the smallest trained sequence length. "
                f"Min required: {min(self.seq_lengths)}, provided: {len(history)}."
            )
        return combined, probs


# ---------------------------
# CLI
# ---------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Predict yes/no probability using multi-sequence LSTM ensembles")
    p.add_argument("--history", type=str, help='Comma-separated history, e.g. "no,no,yes,no" (simple ensemble)')
    p.add_argument("--next", type=str, choices=["yes", "no"], required=True, help='The next label ("yes" or "no")')
    p.add_argument("--history_csv", type=str, help="CSV path (datetime,returned,number) for time-feature ensemble")
    p.add_argument("--weights", nargs="*", type=float, help="Custom weights (same length as number of models)")
    p.add_argument("--use_gpu", action="store_true", help="Use GPU if available (CPU by default)")
    p.add_argument("--artifacts_dir", type=str, default="artifacts", help="Directory to read models and metadata from")
    return p.parse_args()


def main():
    args = parse_args()
    device = select_device(args.use_gpu)
    artifacts_dir = args.artifacts_dir
    meta_path = os.path.join(artifacts_dir, "models_meta.json")

    print(f"Using device: {device}")
    print(f"Artifacts dir: {artifacts_dir}")

    meta = load_meta(meta_path)
    seq_lengths = meta["seq_lengths"]
    ensembles = meta["ensembles"]

    # Choose ensemble by inputs and build history array
    if args.history_csv:
        # time-feature ensemble
        prefix = ensembles["time"]["prefix"]                 # "model_seq"
        expected_input_size = ensembles["time"]["input_size"]  # 6

        hist_path = args.history_csv
        if not os.path.exists(hist_path):
            # try artifacts/<file> if relative path doesn't exist in CWD
            maybe = os.path.join(artifacts_dir, hist_path)
            if os.path.exists(maybe):
                hist_path = maybe

        df = pd.read_csv(hist_path)
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
        df = df.dropna(subset=["datetime"]).sort_values("datetime")
        df["year"] = df["datetime"].dt.year
        df["month"] = df["datetime"].dt.month
        df["day"] = df["datetime"].dt.day
        df["hour"] = df["datetime"].dt.hour
        df["weekday"] = df["datetime"].dt.weekday
        # returned is not part of the input features but can be kept for reference
        df["returned"] = df["returned"].map({"yes": 1, "no": 0})

        features = ["year", "month", "day", "hour", "weekday", "number"]
        history = df[features].astype(np.float32).values.tolist()

    elif args.history:
        # simple ensemble
        prefix = ensembles["simple"]["prefix"]               # "model_simple_seq"
        expected_input_size = ensembles["simple"]["input_size"]  # 1
        history = [1.0 if s.strip().lower() == "yes" else 0.0 for s in args.history.split(",")]
    else:
        raise ValueError("Provide either --history_csv (time-feature mode) or --history (simple mode).")

    # Build model paths from artifacts and create predictor
    model_paths = [os.path.join(artifacts_dir, f"{prefix}{L}.pth") for L in seq_lengths]
    predictor = MultiModelPredictor(model_paths, seq_lengths, device)

    # Validate matched input sizes
    if predictor.input_size != expected_input_size:
        raise RuntimeError(
            f"Loaded models expect input_size={predictor.input_size}, but this mode expects {expected_input_size}."
        )

    # Predict YES probability from history windows only
    p_yes_combined, p_yes_perlen = predictor.predict(history, weights=args.weights)

    # Map to requested label
    want_yes = (args.next.lower() == "yes")
    combined = p_yes_combined if want_yes else (1.0 - p_yes_combined)
    perlen = [None if p is None else (p if want_yes else (1.0 - p)) for p in p_yes_perlen]

    # Output
    print("\n=== Prediction Results ===")
    print(f"Combined Probability for '{args.next.upper()}': {combined:.4f}")
    table = []
    for L, p in zip(seq_lengths, perlen):
        table.append([L, "N/A" if p is None else f"{p:.4f}"])
    print(tabulate(table, headers=["Seq Length", "Probability"], tablefmt="grid"))

    print("\nProbability Breakdown:")
    for L, p in zip(seq_lengths, perlen):
        if p is not None:
            bars = "#" * int(float(p) * 50)
            print(f"Seq {L:>4}: {float(p):.4f} {bars}")


if __name__ == "__main__":
    main()