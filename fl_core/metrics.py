"""Les métriques d'analyse : ce que MLflow ne calcule pas pour nous.

MLflow stocke des séries indexées par round. Transformer ces séries en réponse
au livrable central — « à partir de quel niveau d'hétérogénéité FedProx
surpasse-t-il *significativement* FedAvg ? » — relève de l'algorithme, pas du
stockage. C'est pourquoi ces fonctions vivent ici et non dans l'API.

Elles sont pures : une liste d'accuracies entre, un nombre sort. Aucune
dépendance à torch, à MLflow ni au réseau, donc testables en millisecondes.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass


def final_accuracy(accs: list[float], window: int = 5) -> float:
    """Moyenne des `window` derniers rounds.

    La fin de courbe oscille, d'autant plus que alpha est petit. Prendre le
    dernier round seul, c'est mesurer du bruit : selon qu'on s'arrête au round
    99 ou 100, le résultat peut varier de plusieurs points.

    La fenêtre est un CHOIX MÉTHODOLOGIQUE à annoncer dans le rapport, pas un
    détail d'implémentation.
    """
    if not accs:
        raise ValueError("Courbe vide : impossible de calculer une accuracy finale.")
    if window < 1:
        raise ValueError("La fenêtre doit valoir au moins 1.")
    return statistics.fmean(accs[-window:])


def rounds_to_target(
    accs: list[float], target: float, consecutive: int = 1
) -> int | None:
    """Premier round atteignant la cible, numéroté à partir de 1.

    Retourne `None` si elle n'est jamais atteinte. C'est une INFORMATION — la
    cible n'a pas été atteinte — et non une valeur manquante : la remplacer par
    0 ferait croire à une convergence immédiate et fausserait toute moyenne.

    Cette métrique mesure la VITESSE là où `final_accuracy` mesure le plateau.
    Deux méthodes peuvent finir au même niveau, l'une en 30 rounds et l'autre
    en 90 — et chaque round coûte `2 x K x taille_du_modele` en communication.

    `consecutive` exige que la cible tienne sur k rounds d'affilée. Aux petits
    alpha les courbes oscillent de plusieurs points : avec la valeur par défaut
    de 1, un pic chanceux suffit à déclarer la convergence.
    """
    if consecutive < 1:
        raise ValueError("`consecutive` doit valoir au moins 1.")

    tenus = 0
    for i, a in enumerate(accs, start=1):
        tenus = tenus + 1 if a >= target else 0
        if tenus == consecutive:
            return i
    return None


@dataclass(frozen=True)
class Comparison:
    """Résultat d'une comparaison appariée entre deux bras."""

    mean_diff: float      # moyenne de (b - a) sur les seeds
    std_diff: float       # écart-type de cet écart
    n_seeds: int
    t_stat: float         # mean_diff / (std_diff / sqrt(n)) — indicatif à n=3
    significant: bool     # |mean_diff| > std_diff


def compare_arms(a: list[float], b: list[float]) -> Comparison:
    """Comparaison APPARIÉE de deux bras, seed par seed.

    `a` et `b` contiennent une valeur par seed, DANS LE MÊME ORDRE : même seed
    = même partition = même initialisation. On compare donc les écarts par
    paire, et non deux moyennes indépendantes.

    C'est ce qui donne sa puissance au protocole : l'appariement élimine la
    variance due à la partition, qui est la plus grosse source de bruit en
    non-IID. À budget de calcul égal, on détecte des écarts bien plus fins.

    `significant` applique le critère du livrable — l'écart moyen dépasse-t-il
    la variabilité inter-seeds ? C'est une heuristique lisible, pas un test
    formel. `t_stat` est fourni pour le rapport ; à trois seeds il reste
    indicatif (2 degrés de liberté, seuil à 4,30 pour p < 0,05).
    """
    if len(a) != len(b):
        raise ValueError(
            f"Bras désappariés : {len(a)} valeurs contre {len(b)}. "
            "La comparaison perdrait exactement ce qui fait sa valeur."
        )
    if len(a) < 2:
        raise ValueError(
            "Au moins deux seeds sont nécessaires : sans écart-type, il n'y a "
            "rien à dire sur la significativité."
        )

    diffs = [y - x for x, y in zip(a, b)]
    mean_diff = statistics.fmean(diffs)
    std_diff = statistics.stdev(diffs)

    if std_diff == 0.0:
        t_stat = 0.0 if mean_diff == 0.0 else float("inf")
    else:
        t_stat = mean_diff / (std_diff / len(diffs) ** 0.5)

    return Comparison(
        mean_diff=mean_diff,
        std_diff=std_diff,
        n_seeds=len(diffs),
        t_stat=t_stat,
        significant=abs(mean_diff) > std_diff,
    )
