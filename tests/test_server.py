"""Tests de la boucle fédérée.

Ils injectent un jeu de données minuscule à la place de MNIST : ce qui est
vérifié ici, c'est la MÉCANIQUE fédérée — sélection, broadcast, ancrage
proximal, agrégation — pas la qualité de l'apprentissage. La validation
scientifique, elle, est le point d'arrêt à 98-99 % en quasi-IID, qui se lance
par `scripts/check_federated.py`.

Sans cette injection, chaque exécution téléchargerait 404 Mo et la suite
passerait de quelques secondes à plusieurs minutes.
"""

import numpy as np
import pytest
import torch
from torch.utils.data import TensorDataset

import fl_core.server as server_mod
from contracts.schemas import Algo, RunConfig
from fl_core.data.loaders import Split
from fl_core.server import run_federated

CFG = dict(algo=Algo.fedavg, alpha=100.0, n_clients=4, rounds=2,
           local_epochs=1, batch_size=32, seed=0)


def _jeu_minuscule(n_train: int = 120, n_test: int = 40) -> Split:
    """120 images 1x28x28 et leurs étiquettes, réparties sur les 10 classes.

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


@pytest.fixture(autouse=True)
def sans_mnist(monkeypatch):
    monkeypatch.setattr(server_mod, "load", lambda *a, **k: _jeu_minuscule())


def _courbe(**surcharges) -> list:
    metriques = []
    cfg = RunConfig(**{**CFG, **surcharges})
    run_federated("t", cfg, metriques.append)
    return metriques


def test_les_rounds_sont_emis_de_1_a_R():
    m = _courbe(rounds=3)

    assert [x.round for x in m] == [1, 2, 3]
    assert all(0.0 <= x.global_acc <= 1.0 for x in m)
    assert all(x.comm_mb > 0 for x in m), "un round fédéré échange forcément des poids"


def test_deux_executions_de_meme_seed_donnent_la_meme_courbe():
    """Sans déterminisme, les trois seeds appariés du protocole ne veulent
    rien dire et l'écart FedAvg/FedProx devient indéfendable."""
    a = [x.global_acc for x in _courbe()]
    b = [x.global_acc for x in _courbe()]

    assert a == b


def test_la_participation_limite_le_nombre_de_contributeurs(monkeypatch):
    """participation=0.5 sur 4 clients : 2 modèles agrégés, pas 4."""
    tailles = []
    vrai = server_mod.average_weights

    def espion(state_dicts, n_samples):
        tailles.append(len(state_dicts))
        return vrai(state_dicts, n_samples)

    monkeypatch.setattr(server_mod, "average_weights", espion)
    _courbe(participation=0.5, rounds=2)

    assert tailles == [2, 2]


def test_tous_les_clients_d_un_round_partagent_la_meme_ancre_proximale(monkeypatch):
    """LE test qui protège FedProx.

    `global_params` est l'ancre w^t du terme proximal. Tous les clients d'un
    même round doivent recevoir EXACTEMENT la même, capturée une seule fois
    avant la boucle. Si l'on passait par erreur les paramètres du modèle local
    de chaque client, chacun recevrait une ancre différente, le terme
    proximal se réduirait à zéro et FedProx redeviendrait FedAvg — sans
    erreur, sans crash, avec des courbes plausibles.
    """
    ancres, mus = [], []
    vrai = server_mod.train_local

    def espion(model, loader, epochs, lr, device="cpu", mu=0.0, global_params=None):
        mus.append(mu)
        ancres.append([g.clone() for g in global_params] if global_params else None)
        return vrai(model, loader, epochs, lr, device, mu, global_params)

    monkeypatch.setattr(server_mod, "train_local", espion)
    _courbe(algo=Algo.fedprox, mu=0.01, rounds=1)

    assert mus == [0.01] * 4, "mu doit être transmis à chaque client"
    assert all(a is not None for a in ancres), "l'ancre ne doit jamais être None"

    premiere = ancres[0]
    for autre in ancres[1:]:
        assert all(torch.equal(x, y) for x, y in zip(premiere, autre))


def test_sans_heterogeneite_systemes_tous_font_le_meme_nombre_d_epoques(monkeypatch):
    epoques = []
    vrai = server_mod.train_local

    def espion(model, loader, epochs, lr, device="cpu", mu=0.0, global_params=None):
        epoques.append(epochs)
        return vrai(model, loader, epochs, lr, device, mu, global_params)

    monkeypatch.setattr(server_mod, "train_local", espion)
    _courbe(local_epochs=3, rounds=1)

    assert epoques == [3, 3, 3, 3]


def test_avec_heterogeneite_systemes_les_epoques_varient(monkeypatch):
    """Le régime pour lequel le terme proximal de FedProx a été conçu :
    chaque client tire son nombre d'époques dans [1, E], ce qui simule les
    stragglers du papier de Li et al."""
    epoques = []
    vrai = server_mod.train_local

    def espion(model, loader, epochs, lr, device="cpu", mu=0.0, global_params=None):
        epoques.append(epochs)
        return vrai(model, loader, epochs, lr, device, mu, global_params)

    monkeypatch.setattr(server_mod, "train_local", espion)
    _courbe(local_epochs=4, n_clients=6, systems_heterogeneity=True, rounds=1)

    assert len(set(epoques)) > 1, f"toutes les époques identiques : {epoques}"
    assert all(1 <= e <= 4 for e in epoques)


def test_le_device_est_resolu_automatiquement(monkeypatch):
    """Sans GPU on reste sur CPU ; avec GPU on le prend, sans rien configurer.

    C'est ce qui rend le passage sur Colab utile : sans résolution
    automatique, le code y tournerait sur le CPU de la machine virtuelle,
    c'est-à-dire plus lentement que sur un portable — et rien ne le
    signalerait.
    """
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert server_mod.resolve_device() == "cuda"

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert server_mod.resolve_device() == "cpu"

    # Un choix explicite l'emporte toujours : utile pour forcer le CPU sur une
    # machine à GPU, par exemple pour comparer deux exécutions.
    assert server_mod.resolve_device("cpu") == "cpu"


def test_le_device_est_propage_a_l_entrainement_et_a_l_evaluation(monkeypatch):
    """Le device doit atteindre les DEUX. Le passer au seul entraînement
    laisserait l'évaluation sur CPU et transférerait les tenseurs à chaque
    round, ce qui annulerait une bonne part du gain."""
    devices_train, devices_eval = [], []
    vrai_train, vrai_eval = server_mod.train_local, server_mod.evaluate

    def espion_train(model, loader, epochs, lr, device="cpu", mu=0.0, global_params=None):
        devices_train.append(device)
        return vrai_train(model, loader, epochs, lr, device, mu, global_params)

    def espion_eval(model, loader, device="cpu"):
        devices_eval.append(device)
        return vrai_eval(model, loader, device)

    monkeypatch.setattr(server_mod, "train_local", espion_train)
    monkeypatch.setattr(server_mod, "evaluate", espion_eval)

    metriques = []
    run_federated("t", RunConfig(**{**CFG, "rounds": 1}), metriques.append, device="cpu")

    assert devices_train == ["cpu"] * 4
    # 4 modèles locaux + le modèle global agrégé
    assert devices_eval == ["cpu"] * 5
