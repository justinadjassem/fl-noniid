"""Execute le plan experimental, et sait reprendre ou il s'est arrete.

    python experiments/run_grid.py --config experiments/grid.yaml

Reprise : chaque run porte une cle deterministe. Au demarrage, le script
interroge MLflow et saute ceux qui sont deja termines. Une deconnexion Colab a
la 40e heure ne fait donc rien recommencer.

Calibration — avant la grille, mesurer le cout reel d'un round sur GPU :

    python experiments/run_grid.py --config experiments/grid.yaml \\
        --limit 1 --rounds 10

Ce script ne contient AUCUN code scientifique : toute la logique est dans
fl_core/, donc testee. Ici il n'y a qu'une boucle et de l'affichage.
"""

import argparse
import time
from datetime import datetime, timezone

from contracts.schemas import Run, RunStatus
from fl_core.grid import (charger_grille, cles_terminees, enumerer_runs,
                          moteur_pour, verifier_tracking)
from fl_core.metrics import final_accuracy, rounds_to_target
from fl_core.tracking import track

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="experiments/grid.yaml")
ap.add_argument("--rounds", type=int, default=None,
                help="surcharge le nombre de rounds (calibration)")
ap.add_argument("--limit", type=int, default=None,
                help="n'execute que les N premiers runs (calibration)")
ap.add_argument("--dry-run", action="store_true",
                help="liste ce qui reste a faire, sans rien executer")
args = ap.parse_args()

if not args.dry_run:
    try:
        print(f"journalisation vers {verifier_tracking()}")
    except RuntimeError as e:
        raise SystemExit(f"\nARRET : {e}\n")

spec = charger_grille(args.config)
runs = enumerer_runs(spec, rounds=args.rounds, limite=args.limit)
faits = cles_terminees()
a_faire = [(cle, cfg) for cle, cfg in runs if cle not in faits]

print(f"{len(runs)} runs au plan · {len(runs) - len(a_faire)} deja termines "
      f"· {len(a_faire)} a faire")

if args.dry_run:
    for cle, _ in a_faire:
        print("   ", cle)
    raise SystemExit(0)

debut = time.time()
for i, (cle, cfg) in enumerate(a_faire, start=1):
    t0 = time.time()
    accs: list[float] = []

    with track(cle, cfg) as tracker:
        def sink(m, _t=tracker, _a=accs):
            _a.append(m.global_acc)
            _t.on_round(m)

        moteur_pour(cfg)(cle, cfg, sink)

        run = Run(
            id=cle, config=cfg, status=RunStatus.done,
            created_at=datetime.now(timezone.utc),
            final_acc=final_accuracy(accs),
            rounds_to_target=rounds_to_target(accs, cfg.target_acc),
        )
        tracker.summarize(run)

    cible = run.rounds_to_target if run.rounds_to_target is not None else "jamais"
    print(f"[{i}/{len(a_faire)}] {cle} · final_acc={run.final_acc:.4f} "
          f"· cible={cible} · {time.time() - t0:.0f}s "
          f"· total {(time.time() - debut) / 60:.0f} min", flush=True)

print(f"grille terminee en {(time.time() - debut) / 3600:.2f} h")
