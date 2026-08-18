"""Point d'arret ML-2 : le centralise doit atteindre ~99 % sur MNIST.

    python scripts/check_baselines.py --mode centralized --rounds 8
    python scripts/check_baselines.py --mode local --alpha 0.1 --rounds 3
"""

import argparse

from contracts.schemas import Algo, RunConfig
from fl_core.baselines import run_centralized, run_local_only

ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["centralized", "local"], default="centralized")
ap.add_argument("--dataset", default="mnist")
ap.add_argument("--rounds", type=int, default=8)
ap.add_argument("--alpha", type=float, default=0.1)
ap.add_argument("--clients", type=int, default=10)
ap.add_argument("--lr", type=float, default=0.01)
args = ap.parse_args()

cfg = RunConfig(
    algo=Algo.centralized if args.mode == "centralized" else Algo.local,
    dataset=args.dataset, rounds=args.rounds, alpha=args.alpha,
    n_clients=args.clients, lr=args.lr, batch_size=64,
)

print(f"{args.mode} — {args.dataset}, {args.rounds} rounds, lr={args.lr}"
      + (f", alpha={args.alpha}, {args.clients} clients" if args.mode == "local" else ""))
print(f"{'round':>6}{'accuracy':>11}{'loss':>9}{'ecart-type':>12}{'temps':>9}")

def show(m):
    print(f"{m.round:>6}{m.global_acc:>11.4f}{m.global_loss:>9.4f}"
          f"{m.std_client_acc:>12.4f}{m.wall_time_s:>8.0f}s", flush=True)

(run_centralized if args.mode == "centralized" else run_local_only)("check", cfg, show)
