"""Puits MLflow : où atterrissent les métriques d'un run.

Le moteur d'entraînement ne connaît pas ce module — il appelle `on_round`,
sans savoir où ça va (cf. le protocole `Runner`). C'est ici qu'on décide que
« là » veut dire MLflow.

**Le puits est inerte quand `MLFLOW_TRACKING_URI` est absent.** C'est
délibéré : sans ça, `pytest`, un `uvicorn --reload` en local et l'exécution
sur Colab exigeraient tous un serveur MLflow joignable. Le suivi
d'expériences est une commodité, pas une dépendance du cœur.

La variable accepte indifféremment `http://mlflow:5000` (le service Docker)
et `sqlite:///…` (Colab, tests) : c'est le client MLflow qui gère la
différence, ce module ne fait que lire l'environnement. En revanche MLflow 3.x
REFUSE le magasin de fichiers (`file://`, l'ancien `./mlruns`) : il faut un
backend base de données.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterator, Protocol

import mlflow

from contracts.schemas import RoundMetric, Run, RunConfig

# Une seule expérience pour tout le projet : les runs se distinguent par
# leurs paramètres (algo, alpha, mu, seed), pas par leur rangement.
EXPERIMENT = "fl-noniid"

# Clé de jointure entre l'identifiant que l'API génère et le run MLflow.
# C'est par ce tag que BACK-2 retrouvera un run avec search_runs().
TAG_RUN_ID = "fl_run_id"


class Tracker(Protocol):
    """Ce qu'un puits sait faire."""

    def on_round(self, m: RoundMetric) -> None: ...
    def summarize(self, run: Run) -> None: ...


class _Inerte:
    """Puits sans effet, quand aucun serveur n'est configuré."""

    def on_round(self, m: RoundMetric) -> None:
        pass

    def summarize(self, run: Run) -> None:
        pass


class _Mlflow:
    """Puits réel. Suppose un run MLflow déjà ouvert par `track`."""

    def on_round(self, m: RoundMetric) -> None:
        # step=m.round : MLflow est conçu pour des métriques indexées par pas,
        # et le round de communication en est un. C'est ce qui permet de tracer
        # les courbes et de comparer des runs de longueurs différentes.
        metriques = {
            "global_acc": m.global_acc,
            "global_loss": m.global_loss,
            "mean_client_acc": m.mean_client_acc,
            "std_client_acc": m.std_client_acc,
            "comm_mb": m.comm_mb,
            "wall_time_s": m.wall_time_s,
        }

        # Le préfixe `client_<id>/` est ce qui permet à MLflow de regrouper et
        # de superposer les courbes. Sans lui, les dix clients écraseraient la
        # même clé et il ne resterait que celle du dernier.
        #
        # La liste est vide pour les bornes et le moteur factice : c'est la
        # contrepartie du défaut vide qui rend l'ajout au contrat non cassant.
        for c in m.clients:
            metriques.update({
                f"client_{c.client_id}/drift": c.drift,
                f"client_{c.client_id}/local_acc": c.local_acc,
                f"client_{c.client_id}/local_loss": c.local_loss,
                f"client_{c.client_id}/epochs_run": c.epochs_run,
                f"client_{c.client_id}/n_samples": c.n_samples,
                f"client_{c.client_id}/wall_time_s": c.wall_time_s,
            })

        # Un seul appel par round : dix clients feraient sinon soixante
        # allers-retours HTTP là où un seul suffit.
        mlflow.log_metrics(metriques, step=m.round)

    def summarize(self, run: Run) -> None:
        """Verse ce que MLflow ne sait pas calculer pour nous."""
        if run.final_acc is not None:
            mlflow.log_metric("final_acc", run.final_acc)
        # `None` = cible jamais atteinte. Le journaliser à 0 ferait croire à une
        # convergence immédiate et fausserait toute moyenne calculée dessus.
        if run.rounds_to_target is not None:
            mlflow.log_metric("rounds_to_target", run.rounds_to_target)


@contextmanager
def track(run_id: str, cfg: RunConfig) -> Iterator[Tracker]:
    """Ouvre un run MLflow le temps d'une expérience.

    Une exception qui traverse le bloc marque le run `FAILED` : un
    entraînement interrompu ne doit pas ressembler à un résultat valide,
    simplement plus court.
    """
    uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not uri:
        yield _Inerte()
        return

    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(EXPERIMENT)

    with mlflow.start_run(run_name=run_id):
        mlflow.set_tag(TAG_RUN_ID, run_id)
        # mode="json" : sans lui les énumérations arrivent en `Algo.fedprox`
        # au lieu de `fedprox`, et le filtrage MLflow devient inutilisable.
        mlflow.log_params(cfg.model_dump(mode="json"))
        yield _Mlflow()
