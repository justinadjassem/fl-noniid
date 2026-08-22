"""Les métriques qui répondent au livrable central.

« À partir de quel niveau d'hétérogénéité FedProx surpasse-t-il
*significativement* FedAvg ? » — le mot est dans l'énoncé, et y répondre exige
trois choses : lisser le bruit de fin de courbe, mesurer la vitesse et pas
seulement le plateau, et comparer sur des seeds APPARIÉS.
"""

import pytest

from fl_core.metrics import compare_arms, final_accuracy, rounds_to_target


# --- final_accuracy ---------------------------------------------------------

def test_final_accuracy_moyenne_la_fenetre_de_fin():
    """La fin de courbe oscille, surtout aux petits alpha. S'arrêter au dernier
    round, c'est mesurer du bruit : selon qu'on coupe au round 99 ou 100, le
    résultat peut varier de plusieurs points."""
    accs = [0.1, 0.2, 0.3, 0.90, 0.92, 0.88, 0.94, 0.86]

    assert final_accuracy(accs, window=5) == pytest.approx(0.90)


def test_final_accuracy_accepte_une_courbe_plus_courte_que_la_fenetre():
    assert final_accuracy([0.8, 0.9], window=5) == pytest.approx(0.85)


def test_final_accuracy_refuse_une_courbe_vide():
    """Silencieusement renvoyer 0.0 ferait passer un run raté pour un mauvais
    résultat, et il finirait dans une moyenne de la grille."""
    with pytest.raises(ValueError):
        final_accuracy([])


# --- rounds_to_target -------------------------------------------------------

def test_rounds_to_target_rend_un_numero_de_round_pas_un_indice():
    """Numérotation à partir de 1, comme `RoundMetric.round`. Un décalage
    d'un round se propagerait silencieusement jusqu'au tableau du rapport."""
    assert rounds_to_target([0.5, 0.7, 0.91, 0.93], target=0.90) == 3


def test_rounds_to_target_rend_none_si_jamais_atteinte():
    """None est une INFORMATION — la cible n'a pas été atteinte — pas une
    valeur manquante. La journaliser à 0 ferait croire à une convergence
    immédiate et fausserait toute moyenne calculée dessus."""
    assert rounds_to_target([0.5, 0.7, 0.8], target=0.90) is None


def test_rounds_to_target_exige_optionnellement_une_stabilite():
    """Le guide demande « le premier round atteignant la cible ». À alpha=0,05
    les courbes oscillent de plusieurs points : un pic chanceux donnerait une
    convergence apparente au round 2 alors que le modèle repasse dessous juste
    après. `consecutive` permet d'exiger que ça tienne."""
    oscillante = [0.5, 0.91, 0.80, 0.85, 0.91, 0.92, 0.93]

    assert rounds_to_target(oscillante, 0.90) == 2                    # premier contact
    assert rounds_to_target(oscillante, 0.90, consecutive=3) == 7     # premier palier tenu


# --- compare_arms -----------------------------------------------------------

def test_compare_arms_compare_par_paires_et_non_par_moyennes():
    """L'appariement élimine la variance due à la partition, qui est la plus
    grosse source de bruit du protocole. Ici les deux bras ont des moyennes
    très dispersées, mais l'écart PAR SEED est constant : +0,02 à chaque fois.
    Une comparaison non appariée n'y verrait que du bruit."""
    a = [0.70, 0.85, 0.60]
    b = [0.72, 0.87, 0.62]

    c = compare_arms(a, b)

    assert c.mean_diff == pytest.approx(0.02)
    assert c.std_diff == pytest.approx(0.0, abs=1e-9)
    assert c.n_seeds == 3
    assert c.significant is True


def test_compare_arms_ne_conclut_pas_quand_l_ecart_est_noye_dans_le_bruit():
    """Le cas qu'il faut savoir rapporter honnêtement : en moyenne FedProx
    gagne, mais l'écart change de signe selon le seed."""
    a = [0.70, 0.85, 0.60]
    b = [0.75, 0.83, 0.63]

    c = compare_arms(a, b)

    assert c.mean_diff > 0
    assert c.significant is False


def test_compare_arms_refuse_des_bras_de_tailles_differentes():
    """Des bras désappariés produiraient un écart calculé entre des seeds
    différents — la comparaison perdrait exactement ce qui fait sa valeur."""
    with pytest.raises(ValueError):
        compare_arms([0.7, 0.8], [0.7])


def test_compare_arms_refuse_un_seul_seed():
    """Avec un seul seed il n'y a pas d'écart-type, donc rien à dire sur la
    significativité. Renvoyer 0.0 laisserait croire à une certitude parfaite."""
    with pytest.raises(ValueError):
        compare_arms([0.7], [0.8])
