"""Le plan expérimental : énumération, nommage, reprise.

La grille ne se lance qu'une fois et coûte des heures. Deux propriétés
priment donc : énumérer exactement ce qu'annonce le plan, et être reprenable.

La logique vit ici plutôt que dans `experiments/run_grid.py` parce que
`experiments/` n'est pas un paquet installé : rien de ce qui s'y trouve ne
serait testable. Le script n'est qu'une ligne de commande.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from contracts.schemas import Algo, RunConfig

# Le centralisé ignore la partition ; alpha doit néanmoins être valide.
_ALPHA_INUTILISE = 0.5


@dataclass(frozen=True)
class GridSpec:
    """Le plan expérimental, tel que lu depuis le YAML."""

    name: str
    base: dict
    seeds: list[int]
    alphas: list[float]
    arms: list[dict]
    baselines: list[dict]


def charger_grille(chemin: str | Path) -> GridSpec:
    chemin = Path(chemin)
    d = yaml.safe_load(chemin.read_text(encoding="utf-8"))
    reserves = {"name", "seeds", "alphas", "arms", "baselines"}
    return GridSpec(
        name=d.get("name", chemin.stem),
        base={k: v for k, v in d.items() if k not in reserves},
        seeds=d["seeds"],
        alphas=d["alphas"],
        arms=d["arms"],
        baselines=d.get("baselines", []),
    )


def cle_du_run(cfg: RunConfig, plan: str) -> str:
    """Identifiant déterministe et lisible d'un run, unique DANS SON PLAN.

    C'est ce qui rend la grille reprenable : la même config redonne toujours la
    même clé, donc interroger la base suffit à savoir ce qui reste à faire.

    Le nom du plan et le nombre de rounds en font partie, et ce n'est pas
    cosmétique. Sans eux :

    - le run de calibration (`--rounds 10`) marquerait `fedavg_..._s0` comme
      terminé, et la grille finale à 100 rounds le SAUTERAIT ;
    - le balayage de mu à 40 rounds ferait sauter deux bras entiers de la
      grille.

    Dans les deux cas, un trou silencieux dans le tableau final — qu'aucune
    moyenne ne signalerait.

    Le centralisé n'a pas d'alpha : il ne dépend pas de la partition, un run
    par seed suffit.

    `:g` plutôt que `str()` : 0.1 s'écrit « 0.1 » et 100.0 « 100 », sinon deux
    écritures du même nombre produiraient deux clés différentes.
    """
    if cfg.algo is Algo.centralized:
        corps = f"centralized_s{cfg.seed}"
    else:
        corps = f"{cfg.algo.value}_mu{cfg.mu:g}_a{cfg.alpha:g}_s{cfg.seed}"
    return f"{plan}/{corps}_r{cfg.rounds}"


def enumerer_runs(
    spec: GridSpec, rounds: int | None = None, limite: int | None = None
) -> list[tuple[str, RunConfig]]:
    """Tous les runs du plan, dans un ordre stable.

    `rounds` et `limite` servent au run de calibration : mesurer le coût réel
    sur GPU sans éditer le YAML, qui décrit le protocole et ne doit pas changer
    entre deux exécutions.
    """
    base = dict(spec.base)
    if rounds is not None:
        base["rounds"] = rounds

    runs: list[tuple[str, RunConfig]] = []

    for arm in spec.arms:
        for alpha in spec.alphas:
            for seed in spec.seeds:
                cfg = RunConfig(algo=Algo(arm["algo"]), mu=arm.get("mu", 0.0),
                                alpha=alpha, seed=seed, **base)
                runs.append((cle_du_run(cfg, spec.name), cfg))

    for baseline in spec.baselines:
        for seed in spec.seeds:
            cfg = RunConfig(algo=Algo(baseline["algo"]), mu=baseline.get("mu", 0.0),
                            alpha=_ALPHA_INUTILISE, seed=seed, **base)
            runs.append((cle_du_run(cfg, spec.name), cfg))

    return runs[:limite] if limite is not None else runs


def moteur_pour(cfg: RunConfig):
    """Le moteur qui sait exécuter cette configuration.

    FedAvg et FedProx partagent `run_federated` : c'est l'argument de propreté
    de la comparaison, une seule variable change entre les deux bras.
    """
    from fl_core.baselines import run_centralized, run_local_only
    from fl_core.server import run_federated

    if cfg.algo is Algo.centralized:
        return run_centralized
    if cfg.algo is Algo.local:
        return run_local_only
    return run_federated


def cles_terminees() -> set[str]:
    """Les clés des runs déjà menés à leur terme, lues dans MLflow.

    C'est ce qui rend la grille reprenable. Un run interrompu n'est PAS dans
    cet ensemble : son statut est FAILED ou RUNNING, et ses métriques
    s'arrêtent au milieu de la courbe — il doit être refait.

    Sans base interrogeable, l'ensemble est vide : on ne saute rien. Renvoyer
    autre chose ferait silencieusement disparaître des runs de la grille.
    """
    import os

    import mlflow

    from fl_core.tracking import EXPERIMENT, TAG_RUN_ID

    uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not uri:
        return set()

    mlflow.set_tracking_uri(uri)
    client = mlflow.MlflowClient()
    exp = client.get_experiment_by_name(EXPERIMENT)
    if exp is None:
        return set()

    return {
        r.data.tags[TAG_RUN_ID]
        for r in client.search_runs([exp.experiment_id],
                                    filter_string="attributes.status = 'FINISHED'")
        if TAG_RUN_ID in r.data.tags
    }


def verifier_tracking() -> str:
    """Refuse de lancer la grille si rien ne sera journalisé.

    `track()` est volontairement inerte sans `MLFLOW_TRACKING_URI` — c'est ce
    qui permet à pytest et à un uvicorn local de tourner sans serveur. Mais
    pour la grille, cette même propriété est un piège : quarante heures de
    calcul pour un fichier vide, sans le moindre avertissement.
    """
    import os

    uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not uri:
        raise RuntimeError(
            "MLFLOW_TRACKING_URI n'est pas défini : la grille calculerait sans "
            "rien journaliser. Sur Colab :\n"
            '    os.environ["MLFLOW_TRACKING_URI"] = '
            '"sqlite:////content/drive/MyDrive/fl-results/mlflow.db"'
        )
    return uri
