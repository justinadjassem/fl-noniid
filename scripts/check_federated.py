"""Point d'arret ML-3 : FedAvg doit atteindre 98-99 % en quasi-IID.

    python scripts/check_federated.py --algo fedavg --alpha 100 --clients 2 --rounds 20

Commencer en quasi-IID n'est pas un detail : c'est ce qui separe un bug d'un
phenomene. A alpha=0.1, une accuracy mediocre peut venir de l'heterogeneite
(normal) ou d'une agregation cassee (bug), sans moyen de trancher. A alpha=100
avec 2 clients on est a un cheveu du centralise : sous 90 %, il n'y a plus
qu'une explication possible.

Puis allumer l'heterogeneite :

    for a in 100 1.0 0.5 0.1 0.05; do
      python scripts/check_federated.py --algo fedavg --alpha $a --clients 10 --rounds 60
    done
"""

import argparse

from contracts.schemas import Algo, RunConfig
from fl_core.server import run_federated

ap = argparse.ArgumentParser()
ap.add_argument("--algo", choices=["fedavg", "fedprox"], default="fedavg")
ap.add_argument("--mu", type=float, default=0.0)
ap.add_argument("--dataset", default="mnist")
ap.add_argument("--alpha", type=float, default=100.0)
ap.add_argument("--clients", type=int, default=10)
ap.add_argument("--rounds", type=int, default=20)
ap.add_argument("--epochs", type=int, default=2)
ap.add_argument("--lr", type=float, default=0.01)
ap.add_argument("--batch", type=int, default=64)
ap.add_argument("--participation", type=float, default=1.0)
ap.add_argument("--systems", action="store_true",
                help="heterogeneite systemes : epoques tirees dans [1, E]")
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

if args.algo == "fedprox" and args.mu <= 0:
    ap.error("fedprox exige --mu > 0, sinon c'est exactement fedavg.")

cfg = RunConfig(
    algo=Algo(args.algo), mu=args.mu, dataset=args.dataset,
    alpha=args.alpha, n_clients=args.clients, rounds=args.rounds,
    local_epochs=args.epochs, lr=args.lr, batch_size=args.batch,
    participation=args.participation, systems_heterogeneity=args.systems,
    seed=args.seed,
)

print(f"{cfg.label()} — {args.dataset}, {args.clients} clients, "
      f"{args.rounds} rounds, {args.epochs} epoques locales, lr={args.lr}, "
      f"batch={args.batch}" + (", heterogeneite systemes" if args.systems else ""))
print(f"{'round':>6}{'acc globale':>13}{'loss':>9}{'acc clients':>13}"
      f"{'ecart-type':>12}{'Mo cumules':>12}{'temps':>9}")


def show(m):
    print(f"{m.round:>6}{m.global_acc:>13.4f}{m.global_loss:>9.4f}"
          f"{m.mean_client_acc:>13.4f}{m.std_client_acc:>12.4f}"
          f"{m.comm_mb:>12.1f}{m.wall_time_s:>8.0f}s", flush=True)


run_federated("check", cfg, show)
