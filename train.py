import argparse
import gc
import json
import os
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score
from tqdm import tqdm


# ---------------------------
# Dataset: lazy slice generator (no precomputed X_seq/y_seq)
# ---------------------------
class LazySequenceDataset(Dataset):
    def __init__(self, X_base: np.ndarray, y_base: np.ndarray, seq_len: int):
        """
        X_base: (N, D) float32
        y_base: (N,) float32
        seq_len: sequence length L (target is y[i+L])
        """
        self.X = X_base
        self.y = y_base
        self.L = seq_len
        # usable indices are [0 .. N-L-1]; target at i+L
        self.n = max(0, len(self.X) - self.L)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        xs = self.X[i : i + self.L]           # (L, D)
        yt = self.y[i + self.L]               # scalar
        xs_t = torch.from_numpy(xs).to(torch.float32).contiguous()
        yt_t = torch.tensor(yt, dtype=torch.float32)
        return xs_t, yt_t


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
        out, _ = self.lstm(x)           # (B, L, H)
        out = out[:, -1, :]             # (B, H)
        out = self.fc(out)              # (B, 1)
        return self.sigmoid(out)        # (B, 1)


# ---------------------------
# Helpers
# ---------------------------
def select_device(use_gpu: bool) -> torch.device:
    if use_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    if use_gpu and not torch.cuda.is_available():
        print("Warning: --use_gpu was set but no CUDA device is available. Falling back to CPU.")
    return torch.device("cpu")


def maybe_autocast(precision: str, device: torch.device):
    if precision == "fp16" and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    if precision == "bf16":
        dt_dev = "cuda" if device.type == "cuda" else "cpu"
        return torch.autocast(device_type=dt_dev, dtype=torch.bfloat16)
    return torch.autocast(device_type="cpu", enabled=False)


def parse_seq_lengths(arg: str) -> List[int]:
    if not arg:
        return []
    raw = arg.replace(",", " ").split()
    vals = []
    for t in raw:
        try:
            v = int(t)
            if v >= 1:
                vals.append(v)
        except ValueError:
            pass
    vals = sorted(set(vals))
    return vals


# ---------------------------
# Argparse
# ---------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Memory-efficient multi-sequence LSTM training (time-features + simple ensembles)")
    # default data in artifacts
    p.add_argument("--data_path", type=str, default=os.path.join("artifacts", "data.csv"),
                   help="CSV with datetime,returned,number (default: artifacts/data.csv)")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden_size", type=int, default=64)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--val_split", type=float, default=0.2)

    # fixed sequence lengths (default 2,3,5,8,12,24)
    p.add_argument("--seq_lengths", type=str, default="2,3,5,8,12,24",
                   help='Sequence lengths to train, comma or space separated. Example: "2,3,5,8,12,24"')

    # artifacts location
    p.add_argument("--artifacts_dir", type=str, default="artifacts",
                   help="Directory to read/write models and metadata")

    # performance / memory knobs
    p.add_argument("--num_workers", type=int, default=0, help="DataLoader workers (0 avoids process copies)")
    p.add_argument("--pin_memory", action="store_true", help="Pin host memory for faster HtoD copies")
    p.add_argument("--prefetch_factor", type=int, default=2, help="Batches prefetched per worker (only if workers>0)")
    p.add_argument("--use_gpu", action="store_true", help="Opt-in to CUDA if available")
    p.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="fp32",
                   help="Compute precision (fp16 only on CUDA; bf16 on CUDA/modern CPU)")

    p.add_argument("--grad_accum_steps", type=int, default=1, help="Accumulate gradients across this many steps")
    return p.parse_args()


# ---------------------------
# Training one ensemble (helper)
# ---------------------------
def train_ensemble(X_np: np.ndarray,
                   y_np: np.ndarray,
                   seq_lengths: list[int],
                   device: torch.device,
                   args,
                   model_prefix: str,
                   input_size: int,
                   artifacts_dir: str):
    """
    Trains and saves one ensemble (either time-feature or simple).
    Saves checkpoints as {artifacts_dir}/{model_prefix}{L}.pth
    """
    for L in tqdm(seq_lengths, desc=f"Training models ({model_prefix.strip('_')})", leave=True):
        dataset = LazySequenceDataset(X_np, y_np, seq_len=L)
        if len(dataset) == 0:
            print(f"Skipping L={L}: not enough data.")
            continue

        val_size = int(len(dataset) * args.val_split)
        train_size = len(dataset) - val_size
        train_ds, val_ds = random_split(dataset, [train_size, val_size])

        loader_kwargs = dict(
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            drop_last=False,
        )
        if args.num_workers > 0:
            loader_kwargs["prefetch_factor"] = args.prefetch_factor

        train_loader = DataLoader(train_ds, **loader_kwargs)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_workers, pin_memory=args.pin_memory)

        model = LSTMModel(input_size=input_size, hidden_size=args.hidden_size, num_layers=args.num_layers).to(device)
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        for epoch in tqdm(range(1, args.epochs + 1), desc=f"L={L} epochs", leave=False):
            model.train()
            running = []
            step = 0
            optimizer.zero_grad(set_to_none=True)
            with maybe_autocast(args.precision, device):
                for Xb, yb in tqdm(train_loader, desc=f"L={L} train", leave=False):
                    Xb = Xb.to(device, non_blocking=args.pin_memory)
                    yb = yb.to(device, non_blocking=args.pin_memory)

                    out = model(Xb).squeeze(1)
                    loss = criterion(out, yb)
                    loss.backward()

                    step += 1
                    if step % args.grad_accum_steps == 0:
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                    running.append(loss.item())

                if step % args.grad_accum_steps != 0:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

            # Validation
            model.eval()
            val_losses, preds_all, t_all = [], [], []
            with torch.no_grad():
                for Xb, yb in tqdm(val_loader, desc=f"L={L} val", leave=False):
                    Xb = Xb.to(device, non_blocking=args.pin_memory)
                    yb = yb.to(device, non_blocking=args.pin_memory)
                    out = model(Xb).squeeze(1)
                    loss = criterion(out, yb)
                    val_losses.append(loss.item())
                    preds_all.extend((out.detach().cpu().numpy() > 0.5).astype(np.int32))
                    t_all.extend(yb.detach().cpu().numpy().astype(np.int32))

            val_acc = (np.array(preds_all) == np.array(t_all)).mean() if len(t_all) else float("nan")
            print(f"[{model_prefix}{L}] Epoch {epoch}/{args.epochs}  "
                  f"TrainLoss={np.mean(running):.4f}  ValLoss={np.mean(val_losses):.4f}  ValAcc={val_acc:.4f}")

        out_path = os.path.join(artifacts_dir, f"{model_prefix}{L}.pth")
        torch.save(model.state_dict(), out_path)
        print(f"Saved: {out_path}")

        # free memory
        del model, train_loader, val_loader, train_ds, val_ds, dataset
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


# ---------------------------
# Main
# ---------------------------
def main():
    args = parse_args()
    device = select_device(args.use_gpu)
    artifacts_dir = args.artifacts_dir
    os.makedirs(artifacts_dir, exist_ok=True)

    print(f"Using device: {device} | precision: {args.precision}")
    print(f"Artifacts dir: {artifacts_dir}")
    print(f"Data path:     {args.data_path}")

    # Load & preprocess once
    df = pd.read_csv(args.data_path)
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime"]).sort_values("datetime")

    # Features from datetime
    df["year"] = df["datetime"].dt.year
    df["month"] = df["datetime"].dt.month
    df["day"] = df["datetime"].dt.day
    df["hour"] = df["datetime"].dt.hour
    df["weekday"] = df["datetime"].dt.weekday

    # Target
    le = LabelEncoder()
    df["returned"] = le.fit_transform(df["returned"])  # yes=1, no=0

    # Build base arrays
    features = ["year", "month", "day", "hour", "weekday", "number"]
    X_time = df[features].to_numpy(dtype=np.float32, copy=True)  # (N, 6)
    y_np = df["returned"].to_numpy(dtype=np.float32, copy=True)  # (N,)
    # Scale time features
    scaler = MinMaxScaler()
    X_time[:] = scaler.fit_transform(X_time)

    # Simple (returned-only) features: just the binary stream as (N,1)
    X_simple = y_np.reshape(-1, 1).astype(np.float32)            # (N, 1)

    N = len(X_time)
    if N < 2:
        raise ValueError("Not enough rows to create sequences (need at least 2).")

    # Parse and validate fixed sequence lengths
    requested = parse_seq_lengths(args.seq_lengths)
    if not requested:
        raise ValueError("No valid --seq_lengths provided.")
    usable = [L for L in requested if (N - L) >= 1]
    skipped = sorted(set(requested) - set(usable))
    print(f"Requested lengths: {requested}")
    print(f"Usable lengths:    {usable}")
    if skipped:
        print(f"Skipping (too long for dataset): {skipped}")
    if not usable:
        raise ValueError("None of the requested sequence lengths are usable for this dataset.")

    # Save metadata for predict.py
    meta = {
        "seq_lengths": usable,
        "ensembles": {
            "time":   {"prefix": "model_seq",         "input_size": X_time.shape[1]},
            "simple": {"prefix": "model_simple_seq",  "input_size": X_simple.shape[1]},
        },
        "precision": args.precision,
        "scaler": "MinMax",
        "mode": "fixed",
        "requested_seq_lengths": requested
    }
    meta_path = os.path.join(artifacts_dir, "models_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote metadata: {meta_path}")

    # Train ensembles (write into artifacts/)
    train_ensemble(X_time,   y_np, usable, device, args, model_prefix="model_seq",        input_size=X_time.shape[1],   artifacts_dir=artifacts_dir)
    train_ensemble(X_simple, y_np, usable, device, args, model_prefix="model_simple_seq", input_size=X_simple.shape[1], artifacts_dir=artifacts_dir)

    print("Done.")


if __name__ == "__main__":
    main()