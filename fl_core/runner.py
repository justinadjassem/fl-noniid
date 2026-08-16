"""La couture entre le cœur scientifique et le reste du système.

Tant que le vrai entrainement n'existe pas encore, Ce `FakeRunner` produit des courbes
plausibles pour que le Backend et le Frontend travaillent en parallèle. Le jour où
`TorchRunner` arrive, on change une ligne dans `get_runner` :
ni l'API ni le dashboard ne bougent.
"""

from __future__ import annotations

import math
import random
import time
from typing import Callable, Protocol

from contracts.schemas import Algo, RoundMetric, RunConfig

OnRound = Callable[[RoundMetric], None]

class Runner(Protocol):
    """Contrat d'un moteur d'entrainement."""

    def run(self, run_id:str, cfg: RunConfig, on_round: OnRound) -> None:
        """Exécute l'expérience et appelle `on_round` après chaque round.`"""
        ...


class FakeRunner:
    """Simulateur analytique: aucune donée, aucun réseau de neurones.
    Les courbes suivent acc(t) = plafond * (1 - exp(-t / tau)) + bruit, où le plafond et le tau
    dépendent de l'hétérogénéité. Les tendances reproduites sont celles attendues, elles servent à valider le pipeline
    pas à produire un résultat scientifique
    """
    def __init__(self, seconds_per_round: float = 0.12) -> None:
        self.seconds_per_round = seconds_per_round

    @staticmethod
    def _profile(cfg: RunConfig) -> tuple[float, float, float]:
        """ Retourne (plafond, tau, amplitude du bruit)"""
        # het = 0 (IID) -> 1 (extrêmement non-IID)
        het = 1.0 / (1.0 + cfg.alpha)

        if cfg.algo is Algo.centralized:
            return 0.990, 6.0, 0.002
        if cfg.algo is Algo.local:
            # Un client ne voit qu'une poignée de classes : plafond très bas
            # quand alpha est petit, proche du centralisé quand les données sont IID.
            return 0.985 - 0.55 * het, 5.0, 0.004 + 0.02 * het

        ceiling = 0.985 - 0.30 * het
        tau = 8.0 + 40.0 * het
        noise = 0.003 + 0.030 * het

        if cfg.algo is Algo.fedprox:
            # Le terme proximal aide surtout en forte hétérogénéité, et il existe
            # un mu optimal : trop grand, le modèle local est figé sur le global.
            benefit = math.exp(-((math.log10(max(cfg.mu, 1e-4)) + 1.7) ** 2) / 1.2)
            ceiling += 0.30 * het * 0.45 * benefit
            tau -= 30.0 * het * 0.45 * benefit
            noise *= 1.0 - 0.5 * benefit

        if cfg.systems_heterogeneity:
            # Les stragglers pénalisent FedAvg bien plus que FedProx.
            penalty = 0.06 if cfg.algo is Algo.fedavg else 0.015
            ceiling -= penalty * het
            noise *= 1.3

        return max(ceiling, 0.10), max(tau, 2.0), noise

    def run(self, run_id: str, cfg: RunConfig, on_round: OnRound) -> None:
        rng = random.Random(cfg.seed)
        ceiling, tau, noise = self._profile(cfg)

        # Taille approximative d'un modèle CNN échangé à chaque round.
        model_mb = 0.42 if cfg.dataset.value == "mnist" else 1.8
        n_selected = max(1, round(cfg.n_clients * cfg.participation))

        for r in range(1, cfg.rounds + 1):
            time.sleep(self.seconds_per_round)

            base = ceiling * (1.0 - math.exp(-r / tau))
            acc = min(0.999, max(0.05, base + rng.gauss(0, noise)))
            loss = max(0.01, 2.4 * math.exp(-r / tau) + rng.gauss(0, noise))

            # Plus l'hétérogénéité est forte, plus les clients divergent entre eux.
            spread = (0.5 / (1.0 + cfg.alpha)) * (1.0 - base) + 0.01

            on_round(
                RoundMetric(
                    run_id=run_id,
                    round=r,
                    global_acc=round(acc, 5),
                    global_loss=round(loss, 5),
                    mean_client_acc=round(max(0.0, acc - 0.4 * spread), 5),
                    std_client_acc=round(spread, 5),
                    comm_mb=round(2 * n_selected * model_mb * r, 2),
                    wall_time_s=round(r * self.seconds_per_round, 3),
                )
            )


def get_runner() -> Runner:
    """Point de bascule unique vers le vrai moteur."""
    return FakeRunner()


