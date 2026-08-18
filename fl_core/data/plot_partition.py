"""Heatmap clients x classes : la figure qui prouve le régime non-IID.

Produit `docs/partition_heatmap.png` et un manifeste JSON par valeur d'alpha.
C'est la figure du rapport, et la source de données de l'onglet Partition du
dashboard.

    python -m fl_core.data.plot_partition
    python -m fl_core.data.plot_partition --dataset cifar10 --clients 10
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from fl_core.data.loaders import load
from fl_core.data.partition import (
    dirichlet_partition,
    heterogeneity_score,
    partition_matrix,
    save_manifest,
)

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="mnist", choices=["mnist", "cifar10"])
    ap.add_argument("--clients", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--alphas", type=float, nargs="+",
                    default=[0.05, 0.1, 0.5, 100.0])
    args = ap.parse_args()

    split = load(args.dataset)
    fig, axes = plt.subplots(1, len(args.alphas),
                             figsize=(3.6 * len(args.alphas), 3.4))
    axes = [axes] if len(args.alphas) == 1 else list(axes)

    print(f"{'alpha':>8} {'score':>7} {'tirages':>8} {'classes/client':>16} {'taille min':>11}")
    for ax, alpha in zip(axes, args.alphas):
        parts, draws = dirichlet_partition(
            split.labels, args.clients, alpha, seed=args.seed
        )
        matrix = partition_matrix(split.labels, parts)
        score = heterogeneity_score(matrix)

        payload = save_manifest(
            ROOT / "docs" / "partitions" / f"{args.dataset}_a{alpha:g}_s{args.seed}.json",
            split.labels, parts, alpha, args.seed, draws,
        )
        cpc = payload["classes_per_client"]
        print(f"{alpha:>8g} {score:>7.3f} {draws:>8} "
              f"{f'{min(cpc)}-{max(cpc)}':>16} {min(payload['sizes']):>11}")

        # Proportions par client, et NON effectifs bruts : une échelle par
        # panneau rendrait les quatre visuellement comparables alors qu'ils ne
        # le sont pas (6000 vs 700 en valeur max). Ici 0-1 partout : une case
        # sombre veut dire « ce client concentre cette classe », quel que soit
        # le panneau.
        props = matrix / matrix.sum(axis=1, keepdims=True)
        im = ax.imshow(props, aspect="auto", cmap="Blues", vmin=0.0, vmax=1.0)
        ax.set_title(f"α = {alpha:g}\nscore {score:.2f}", fontsize=11)
        ax.set_xlabel("classe")
        ax.set_ylabel("client" if ax is axes[0] else "")
        ax.set_xticks(range(matrix.shape[1]))
        ax.set_yticks(range(args.clients))

    fig.suptitle(
        f"Partition de Dirichlet — {args.dataset}, {args.clients} clients "
        f"(seed {args.seed})\npart de chaque classe dans les données du client",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 0.92, 1))
    cax = fig.add_axes((0.94, 0.18, 0.012, 0.62))
    fig.colorbar(im, cax=cax, label="part de la classe chez le client")

    out = ROOT / "docs" / f"partition_heatmap_{args.dataset}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    print(f"\nfigure : {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
