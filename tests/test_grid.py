"""La grille d'ablation : énumération, nommage, reprise.

La grille ne se lance qu'une fois et coûte des heures. Deux propriétés
comptent donc plus que tout : elle doit énumérer EXACTEMENT ce qu'annonce le
plan expérimental, et elle doit être REPRENABLE — une déconnexion Colab à la
40e heure ne doit pas tout faire recommencer.
"""

import pytest

from contracts.schemas import Algo
from fl_core.grid import GridSpec, cle_du_run, charger_grille, enumerer_runs

SPEC = GridSpec(
    name="grid",
    base=dict(dataset="mnist", n_clients=10, rounds=100, local_epochs=2,
              lr=0.01, batch_size=64, target_acc=0.90),
    seeds=[0, 1, 2],
    alphas=[0.05, 0.1, 0.5, 1.0, 100.0],
    arms=[{"algo": "fedavg", "mu": 0.0},
          {"algo": "fedprox", "mu": 0.01},
          {"algo": "local", "mu": 0.0}],
    baselines=[{"algo": "centralized", "mu": 0.0}],
)


def test_la_grille_enumere_45_runs_plus_3_bornes():
    """3 bras x 5 alphas x 3 seeds = 45, plus un centralisé par seed."""
    runs = enumerer_runs(SPEC)

    assert len(runs) == 48
    assert sum(1 for k, _ in runs if "centralized" in k) == 3


def test_le_centralise_ne_depend_pas_d_alpha():
    """Toutes les données sont réunies : la partition n'intervient jamais. Un
    centralisé par alpha serait 5 fois le même calcul, et le tableau du rapport
    afficherait 5 valeurs identiques comme si elles voulaient dire quelque
    chose."""
    cles = [k for k, _ in enumerer_runs(SPEC) if "centralized" in k]

    assert cles == ["grid/centralized_s0_r100", "grid/centralized_s1_r100",
                    "grid/centralized_s2_r100"]


def test_les_cles_sont_lisibles_et_stables():
    """La clé est ce qui permet la reprise : elle doit être déterministe et
    identifier le run sans ambiguïté."""
    cles = dict(enumerer_runs(SPEC))

    assert "grid/fedprox_mu0.01_a0.1_s2_r100" in cles
    assert "grid/fedavg_mu0_a100_s0_r100" in cles
    assert "grid/local_mu0_a0.05_s1_r100" in cles


def test_chaque_cle_decrit_bien_sa_config():
    runs = dict(enumerer_runs(SPEC))
    cfg = runs["grid/fedprox_mu0.01_a0.1_s2_r100"]

    assert cfg.algo is Algo.fedprox
    assert cfg.mu == 0.01
    assert cfg.alpha == 0.1
    assert cfg.seed == 2
    assert cfg.rounds == 100
    assert cfg.n_clients == 10


def test_deux_runs_distincts_ne_partagent_jamais_une_cle():
    """Une collision ferait sauter un run par erreur à la reprise, et le
    tableau final aurait un trou qu'aucune moyenne ne signalerait."""
    cles = [k for k, _ in enumerer_runs(SPEC)]

    assert len(cles) == len(set(cles))


def test_la_cle_se_recalcule_a_partir_de_la_config():
    """Même fonction pour énumérer et pour interroger la base : sinon les deux
    divergeraient et la reprise relancerait tout."""
    runs = dict(enumerer_runs(SPEC))
    cle = "grid/fedprox_mu0.01_a0.5_s1_r100"

    assert cle_du_run(runs[cle], "grid") == cle


def test_le_yaml_du_depot_est_lisible():
    """Le fichier livré doit charger sans surprise : c'est lui qui définit le
    plan expérimental cité au rapport."""
    spec = charger_grille("experiments/grid.yaml")

    assert spec.seeds == [0, 1, 2]
    assert 0.05 in spec.alphas
    assert len(enumerer_runs(spec)) == 48


def test_les_surcharges_permettent_un_run_de_calibration():
    """Avant la grille, un run court mesure le coût réel sur GPU. Sans cette
    surcharge, il faudrait éditer le YAML — donc versionner un fichier qui ne
    décrit plus le plan expérimental."""
    runs = enumerer_runs(SPEC, rounds=10, limite=2)

    assert len(runs) == 2
    assert all(cfg.rounds == 10 for _, cfg in runs)


def test_le_moteur_depend_de_l_algorithme():
    """Quatre méthodes, trois moteurs : FedAvg et FedProx partagent le même,
    ce qui est précisément l'argument de propreté de la comparaison."""
    from fl_core.baselines import run_centralized, run_local_only
    from fl_core.grid import moteur_pour
    from fl_core.server import run_federated

    runs = dict(enumerer_runs(SPEC))

    assert moteur_pour(runs["grid/fedavg_mu0_a0.5_s0_r100"]) is run_federated
    assert moteur_pour(runs["grid/fedprox_mu0.01_a0.5_s0_r100"]) is run_federated
    assert moteur_pour(runs["grid/local_mu0_a0.5_s0_r100"]) is run_local_only
    assert moteur_pour(runs["grid/centralized_s0_r100"]) is run_centralized


def test_sans_mlflow_aucune_cle_n_est_connue(monkeypatch):
    """Sans base interrogeable, on ne peut rien sauter : tout reste à faire.
    Renvoyer autre chose ferait silencieusement disparaître des runs."""
    from fl_core.grid import cles_terminees

    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)

    assert cles_terminees() == set()


def test_les_runs_deja_termines_sont_reconnus(magasin_jetable):
    """LE point qui rend la grille reprenable : une déconnexion Colab à la 40e
    heure ne doit pas tout faire recommencer.

    Un run interrompu — donc non FINISHED — doit en revanche être refait :
    ses métriques s'arrêtent au milieu de la courbe.
    """
    from contracts.schemas import RunConfig
    from fl_core.grid import cles_terminees
    from fl_core.tracking import track

    with track("fedavg_mu0_a0.1_s0", RunConfig(algo=Algo.fedavg, alpha=0.1)):
        pass

    try:
        with track("fedprox_mu0.01_a0.1_s0", RunConfig(algo=Algo.fedprox, mu=0.01, alpha=0.1)):
            raise RuntimeError("déconnexion Colab")
    except RuntimeError:
        pass

    terminees = cles_terminees()

    assert "fedavg_mu0_a0.1_s0" in terminees
    assert "fedprox_mu0.01_a0.1_s0" not in terminees, "un run interrompu est à refaire"


def test_la_grille_refuse_de_demarrer_sans_journalisation(monkeypatch):
    """Le garde-fou le plus rentable du projet.

    `track()` est silencieusement inerte sans MLFLOW_TRACKING_URI. Lancer la
    grille sans cette variable calculerait quarante heures pour ne rien
    journaliser — et rien, absolument rien, ne le signalerait avant la fin.
    """
    from fl_core.grid import verifier_tracking

    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    with pytest.raises(RuntimeError, match="MLFLOW_TRACKING_URI"):
        verifier_tracking()

    monkeypatch.setenv("MLFLOW_TRACKING_URI", "sqlite:///x.db")
    assert verifier_tracking() == "sqlite:///x.db"


def test_la_calibration_et_le_balayage_ne_collisionnent_pas_avec_la_grille():
    """LE bug que la reprise aurait introduit.

    Sans le plan ni le nombre de rounds dans la clé, un run de calibration à
    10 rounds marquerait la configuration comme terminée, et la grille à 100
    rounds la SAUTERAIT. Un trou dans le tableau final, qu'aucune moyenne ne
    signalerait.
    """
    grille = {k for k, _ in enumerer_runs(SPEC)}
    calibration = {k for k, _ in enumerer_runs(SPEC, rounds=10, limite=3)}

    balayage = GridSpec(name="sweep_mu", base={**SPEC.base, "rounds": 40},
                        seeds=SPEC.seeds, alphas=[0.1],
                        arms=[{"algo": "fedprox", "mu": 0.01}], baselines=[])
    sweep = {k for k, _ in enumerer_runs(balayage)}

    assert not (grille & calibration), "la calibration ferait sauter des runs"
    assert not (grille & sweep), "le balayage ferait sauter des runs"
