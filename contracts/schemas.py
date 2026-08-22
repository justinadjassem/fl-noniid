"""Contrat partagé entre les trois rôles (Data Scientist, Backend Engineer, Frontend Developer).

À ne pas modifier unilatéralement : l'API, le dashboard et le
cœur d'entrainement dépendent tous de ces structures. Toute évolution se
discute à trois (et se versionne).

    Data + ML  -> produit des RoundMetric conformes à une RunConfig
    Backend    -> persiste et expose Run / RoundMetric / GridCell
    Frontend   -> construit une RunConfig, consomme RoundMetric et GridCell
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class Algo(str, Enum):
    """Les quatre méthodes comparées par l'étude."""

    fedavg = "fedavg"
    fedprox = "fedprox"
    centralized = "centralized"  # borne haute
    local = "local"  # borne basse (aucune agrégation)


class Dataset(str, Enum):
    mnist = "mnist"
    cifar10 = "cifar10"


class RunStatus(str, Enum):
    pending = "pending"
    running = "running"
    done = "done"
    failed = "failed"


class RunConfig(BaseModel):
    """Tout ce qui définit une experience reproductible."""

    algo: Algo
    dataset: Dataset = Dataset.mnist

    # --- héterogénéité des données ---
    alpha: float = Field(0.5, gt=0, description="Paramètre de la partition de Dirichlet")
    n_clients: int = Field(10, ge=2, le=100)

    # --- protocole fédéré ---
    rounds: int = Field(50, ge=1, le=1000)
    local_epochs: int = Field(2, ge=1, le=20)
    participation: float = Field(1.0, gt=0, le=1, description="Fraction de clients par round")

    # --- optimisation ---
    lr: float = Field(0.01, gt=0)
    batch_size: int = Field(32, ge=1)
    mu: float = Field(0.0, ge=0, description="Terme proximal FedProx ; ignore si algo != fedprox")

    # --- héterogénéité des systèmes (cf. papier FedProx) ---
    systems_heterogeneity: bool = Field(
        False, description="Si vrai, chaque client tire son nombre d'époques locales dans [1, local_epochs]"
    )

    seed: int = 0
    target_acc: float = Field(0.90, gt=0, lt=1, description="Cible utilisée pour rounds_to_target")

    def label(self) -> str:
        """Nom court et lisible pour les legendes de courbes."""
        if self.algo is Algo.fedprox:
            return f"FedProx mu={self.mu:g} (alpha={self.alpha:g})"
        if self.algo is Algo.fedavg:
            return f"FedAvg (alpha={self.alpha:g})"
        if self.algo is Algo.centralized:
            return "Centralisé (borne haute)"
        return f"Local pur (alpha={self.alpha:g}, borne basse)"


class ClientMetric(BaseModel):
    """Une ligne par client et par round. C'est le grain fin du monitoring.

    `drift` est la métrique centrale : c'est EXACTEMENT la quantité que le
    terme proximal de FedProx pénalise, ||w_k - w_global||_2. La journaliser
    par client donne la preuve directe que FedProx contient la dérive, et pas
    seulement qu'il améliore le score — un test plus fin que l'accuracy.
    """

    client_id: int = Field(ge=0)
    n_samples: int = Field(ge=0, description="Taille du shard : le déséquilibre de quantité")
    epochs_run: int = Field(ge=0, description="< local_epochs si hétérogénéité systèmes")
    local_acc: float = Field(description="Sur le jeu de test GLOBAL, donc comparable")
    local_loss: float
    drift: float = Field(ge=0, description="||w_k - w_global||_2 en fin d'entraînement local")
    wall_time_s: float = Field(0.0, ge=0, description="Qui ralentit le round")


class RoundMetric(BaseModel):
    """Une ligne par round de communication. C'est l'unité de streaming."""

    run_id: str
    round: int = Field(ge=0)

    global_acc: float
    global_loss: float

    # Dispersion des accuracies locales : proxy mesurable du client drift.
    mean_client_acc: float
    std_client_acc: float

    comm_mb: float = Field(0.0, description="Volume cumulé d'échange sur ce round")
    wall_time_s: float = 0.0

    # Défaut vide : l'ajout est NON CASSANT. Le moteur factice, les bornes et
    # le dashboard continuent de fonctionner sans une ligne modifiée. C'est la
    # forme à privilégier pour toute évolution du contrat.
    clients: list[ClientMetric] = []


class Run(BaseModel):
    """Etat d'une expérience, du lancement au résultat."""

    id: str
    config: RunConfig
    status: RunStatus
    created_at: datetime
    finished_at: datetime | None = None

    current_round: int = 0
    final_acc: float | None = None
    rounds_to_target: int | None = None  # None = cible jamais atteinte
    error: str | None = None


class GridCell(BaseModel):
    """Une cellule du tableau croise algorithme x hétérogénéité, agrégée sur les seeds."""

    algo: Algo
    alpha: float
    mu: float
    n_seeds: int

    final_acc_mean: float
    final_acc_std: float
    rounds_to_target_mean: float | None
    n_converged: int  # nombre de seeds ayant atteint la cible


class CreateRunRequest(BaseModel):
    config: RunConfig
