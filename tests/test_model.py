"""Tests du modèle.

Ils gardent les trois décisions de conception documentées dans cnn.py — ce
sont exactement les points où une régression serait invisible à l'oeil mais
fausserait toute l'étude.
"""

from __future__ import annotations

import pytest
import torch

from fl_core.models.cnn import SmallCNN, build_model


@pytest.mark.parametrize(
    "in_channels,size,dataset",
    [(1, 28, "mnist"), (3, 32, "cifar10")],
)
def test_les_deux_datasets_passent(in_channels: int, size: int, dataset: str):
    """La même architecture doit accepter MNIST 28x28 et CIFAR-10 32x32."""
    model = SmallCNN(in_channels=in_channels)
    out = model(torch.randn(4, in_channels, size, size))
    assert out.shape == (4, 10)
    assert build_model(dataset)(torch.randn(2, in_channels, size, size)).shape == (2, 10)


def test_aucun_buffer_dans_le_state_dict():
    """La vérification concrète du choix GroupNorm.

    Avec BatchNorm, cet ensemble contiendrait running_mean, running_var et
    num_batches_tracked par couche — trois tenseurs qu'une moyenne pondérée
    corromprait, dont un compteur entier.
    """
    model = SmallCNN()
    parametres = {nom for nom, _ in model.named_parameters()}
    assert set(model.state_dict()) - parametres == set()


def test_pooling_en_4x4():
    """Garde-fou contre un retour à AdaptiveAvgPool((1,1)).

    Le pooling global ferait tomber la tête de 2048 à 128 entrées : mesuré à
    0,39 d'accuracy contre 0,95, sans qu'aucune forme ne change en sortie.
    """
    feat = SmallCNN().features(torch.randn(2, 1, 28, 28))
    assert feat.shape[1:] == (128, 4, 4)


def test_taille_du_modele():
    """Le volume échangé par round est une métrique de l'étude."""
    n_mnist = sum(p.numel() for p in build_model("mnist").parameters())
    n_cifar = sum(p.numel() for p in build_model("cifar10").parameters())

    assert n_mnist == 356_682
    assert n_cifar == 357_258
    assert n_mnist * 4 / 1e6 == pytest.approx(1.43, abs=0.01)   # Mo par round


def test_dataset_inconnu_rejete():
    with pytest.raises(ValueError, match="Dataset inconnu"):
        build_model("fashion-mnist")
