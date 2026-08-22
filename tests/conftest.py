"""Outillage partagé par les tests de la boucle fédérée.

Le jeu injecté remplace MNIST : ce que les tests vérifient, c'est la mécanique
fédérée — sélection, broadcast, ancrage proximal, agrégation — pas la qualité
de l'apprentissage. Sans cette substitution, chaque exécution téléchargerait
puis parcourrait 60 000 images, et la suite passerait de quelques secondes à
plusieurs minutes.

La validation scientifique, elle, ne passe pas par pytest : ce sont les points
d'arrêt de `scripts/check_federated.py`.
"""

import mlflow
import numpy as np
import pytest
import torch
from torch.utils.data import TensorDataset

from fl_core.data.loaders import Split
from fl_core.tracking import EXPERIMENT


def jeu_minuscule(n_train: int = 120, n_test: int = 40) -> Split:
    """120 images 1x28x28 étiquetées sur les 10 classes.

    Dimensionné au minimum utile : 12 exemples par classe suffisent à faire
    tourner la partition de Dirichlet sans clients vides, et chaque seconde
    gagnée ici est payée à chaque exécution de la suite.
    """
    g = torch.Generator().manual_seed(0)
    y_train = torch.arange(n_train) % 10
    y_test = torch.arange(n_test) % 10
    return Split(
        train=TensorDataset(torch.randn(n_train, 1, 28, 28, generator=g), y_train),
        test=TensorDataset(torch.randn(n_test, 1, 28, 28, generator=g), y_test),
        labels=y_train.numpy().astype(np.int64),
    )


@pytest.fixture
def sans_mnist(monkeypatch):
    """Substitue le jeu minuscule à MNIST dans la boucle fédérée."""
    import fl_core.server as server_mod

    monkeypatch.setattr(server_mod, "load", lambda *a, **k: jeu_minuscule())


@pytest.fixture
def magasin_jetable(tmp_path, monkeypatch):
    """Isole complètement MLflow dans tmp_path, et restaure l'état global.

    Backend sqlite et non magasin de fichiers : MLflow 3.x refuse `file://`
    (« maintenance mode »). C'est aussi ce que sert le conteneur, donc le test
    s'exécute sur le même moteur de stockage que la production.

    L'expérience est créée ici avec un `artifact_location` explicite : sinon
    MLflow la poserait dans un `./mlruns` relatif au répertoire courant, et la
    suite de tests salirait le dépôt.
    """
    avant = mlflow.get_tracking_uri()
    uri = f"sqlite:///{tmp_path}/mlflow.db"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    mlflow.set_tracking_uri(uri)
    mlflow.create_experiment(EXPERIMENT, artifact_location=str(tmp_path / "artifacts"))
    yield tmp_path
    if mlflow.active_run():
        mlflow.end_run()
    mlflow.set_tracking_uri(avant)
