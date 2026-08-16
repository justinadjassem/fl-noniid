"""Dashboard de suivi — le livrable « application » du projet.

Quatre zones :
  · barre latérale  : lancer une expérience
  · Convergence     : accuracy globale par round de communication
  · Tableau croisé  : algorithme × hétérogénéité, agrégé sur les seeds
  · Client drift    : dispersion des accuracies locales

Ce module ne connaît ni PyTorch, ni Flower, ni MLflow : il ne parle qu'à
l'API en HTTP. C'est ce qui lui permet de fonctionner à l'identique avec le
moteur factice d'aujourd'hui et le vrai entraînement.

Palette : les couleurs de séries encodent une donnée et ne doivent jamais
être alignées sur le thème de l'interface. Le violet du thème reste sur les
widgets, il n'entre jamais dans un graphique.
"""

from __future__ import annotations

import os

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

from contracts.schemas import Algo, Dataset, RunConfig

API = os.environ.get("API_URL", "http://localhost:8000")
TIMEOUT = 10

# --- couleurs porteuses de sens (validées pour la lisibilité daltonienne) ---
SERIES = {
    Algo.fedavg.value: "#2a78d6",    # bleu
    Algo.fedprox.value: "#eb6834",   # orange
}
REFERENCE = "#898781"                # gris pointillé : les deux bornes

# --- chrome, aligné sur .streamlit/config.toml ------------------------------
SURFACE = "#FFFFFF"                  # = backgroundColor
GRID = "#E6E9EF"
AXIS = "#C9CDD6"
INK = "#31333F"                      # = textColor
INK_MUTED = "#808495"

ALGO_LABEL = {
    "fedavg": "FedAvg",
    "fedprox": "FedProx",
    "centralized": "Centralisé (borne haute)",
    "local": "Local pur (borne basse)",
}
# Deux registres pour les statuts : la syntaxe :material/…: n'est interprétée
# que dans les contextes markdown (titres, captions, labels). Les options d'un
# multiselect sont du texte brut — il y faut des mots.
STATUS_TEXT = {
    "pending": "en attente",
    "running": "en cours",
    "done": "terminé",
    "failed": "échec",
}
STATUS_ICON = {
    "pending": ":material/schedule:",
    "running": ":material/sync:",
    "done": ":material/check_circle:",
    "failed": ":material/error:",
}

st.set_page_config(
    page_title="FL non-IID — FedAvg vs FedProx",
    page_icon=":material/hub:",     # un nœud central et ses satellites
    layout="wide",
)


# ---------------------------------------------------------------- accès API
def api_get(path: str, **params):
    r = requests.get(f"{API}{path}", params=params, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def api_post(path: str, payload: dict):
    r = requests.post(f"{API}{path}", json=payload, timeout=TIMEOUT)
    if r.status_code >= 400:
        try:
            raise RuntimeError(r.json()["detail"])
        except (ValueError, KeyError):
            raise RuntimeError(r.text)
    return r.json()


# -------------------------------------------------------------- présentation
def run_label(run: dict) -> str:
    cfg = run["config"]
    algo = cfg["algo"]
    if algo == "fedprox":
        return f"FedProx μ={cfg['mu']:g} · α={cfg['alpha']:g}"
    if algo == "fedavg":
        return f"FedAvg · α={cfg['alpha']:g}"
    return ALGO_LABEL[algo]


def is_reference(algo: str) -> bool:
    """Centralisé et local pur ne sont pas des séries concurrentes : ce sont
    les bornes qui encadrent la comparaison. Elles se lisent comme telles."""
    return algo in ("centralized", "local")


def style_axes(fig: go.Figure, x_title: str, y_title: str) -> go.Figure:
    """Le graphique doit disparaître dans la page : fond, texte et trame
    dérivent du thème Streamlit, pas d'une palette parallèle."""
    fig.update_layout(
        plot_bgcolor=SURFACE,
        paper_bgcolor=SURFACE,
        font=dict(family="system-ui, -apple-system, 'Segoe UI', sans-serif", color=INK),
        hovermode="x unified",
        margin=dict(l=8, r=8, t=8, b=8),
        legend=dict(orientation="h", y=-0.18, x=0, font=dict(size=12)),
        height=440,
    )
    axis = dict(
        gridcolor=GRID,
        linecolor=AXIS,
        zeroline=False,
        tickfont=dict(color=INK_MUTED, size=11),
        title_font=dict(color=INK_MUTED, size=12),
    )
    fig.update_xaxes(title_text=x_title, **axis)
    fig.update_yaxes(title_text=y_title, **axis)
    return fig


def add_curve(fig: go.Figure, run: dict, df: pd.DataFrame, column: str) -> None:
    algo = run["config"]["algo"]
    ref = is_reference(algo)
    label = run_label(run)
    fig.add_trace(
        go.Scatter(
            x=df["round"],
            y=df[column],
            name=label,
            mode="lines",
            line=dict(
                color=REFERENCE if ref else SERIES.get(algo, "#1baf7a"),
                width=2,
                dash="dash" if ref else "solid",
            ),
            hovertemplate="%{y:.3f}<extra>" + label + "</extra>",
        )
    )


# ------------------------------------------------------- barre latérale
with st.sidebar:
    st.header("Lancer une expérience")

    try:
        health = api_get("/health")
        st.caption(f"API en ligne · moteur `{health['runner']}`")
    except Exception as exc:  # noqa: BLE001
        st.error(f"API injoignable sur {API}\n\n{exc}")
        st.info("Démarrez-la : `uvicorn api.main:app --reload`")
        st.stop()

    algo = st.selectbox(
        "Algorithme", [a.value for a in Algo], format_func=lambda a: ALGO_LABEL[a]
    )
    dataset = st.selectbox("Dataset", [d.value for d in Dataset])

    alpha = st.select_slider(
        "α — hétérogénéité",
        options=[0.05, 0.1, 0.3, 0.5, 1.0, 10.0, 100.0],
        value=0.1,
        help="Petit α = très non-IID. α ≥ 100 ≈ IID.",
    )

    mu = 0.0
    if algo == "fedprox":
        mu = st.select_slider(
            "μ — terme proximal",
            options=[0.001, 0.01, 0.1, 1.0],
            value=0.01,
            help="Trop petit : indiscernable de FedAvg. Trop grand : le modèle "
                 "local reste figé sur le global.",
        )

    rounds = st.slider("Rounds de communication", 10, 200, 60, step=10)
    n_clients = st.slider("Clients", 5, 20, 10)
    local_epochs = st.slider("Époques locales E", 1, 10, 2)
    seed = st.number_input("Seed", min_value=0, max_value=999, value=0, step=1)

    sys_het = st.checkbox(
        "Hétérogénéité systèmes",
        help="Chaque client tire son nombre d'époques dans [1, E]. C'est le "
             "régime pour lequel FedProx a été conçu.",
    )

    if st.button("Lancer", type="primary", use_container_width=True):
        cfg = RunConfig(
            algo=algo,
            dataset=dataset,
            alpha=alpha,
            mu=mu,
            rounds=rounds,
            n_clients=n_clients,
            local_epochs=local_epochs,
            seed=int(seed),
            systems_heterogeneity=sys_het,
        )
        try:
            api_post("/runs", {"config": cfg.model_dump(mode="json")})
            st.success("Expérience lancée.")
            st.rerun()
        except Exception as exc:  # noqa: BLE001
            st.error(str(exc))


# ------------------------------------------------------------------- en-tête
st.title("Apprentissage fédéré sur données non-IID")
st.caption("FedAvg vs FedProx à hétérogénéité contrôlée, encadrés par deux bornes")

runs = api_get("/runs")
if not runs:
    st.info("Aucune expérience pour l'instant. Lancez-en une depuis la barre latérale.")
    st.stop()

running = [r for r in runs if r["status"] == "running"]

by_id = {r["id"]: r for r in runs}


def option_label(run_id: str) -> str:
    """Libellé du sélecteur. Texte brut : les options de multiselect ne sont
    pas interprétées comme du markdown."""
    run = by_id[run_id]
    label = f"{run_label(run)} · seed {run['config']['seed']}"
    if run["status"] != "done":
        label += f" · {STATUS_TEXT[run['status']]}"
    return label


# Les options sont les identifiants, uniques par construction : deux runs de
# même configuration et même seed ne peuvent plus s'écraser mutuellement.
selected_ids = st.multiselect(
    "Expériences à comparer",
    options=list(by_id),
    default=list(by_id)[: min(4, len(by_id))],
    format_func=option_label,
)

tab_conv, tab_grid, tab_drift = st.tabs(
    ["Convergence", "Tableau croisé", "Client drift"]
)


# --------------------------------------------------------------- convergence
@st.fragment(run_every="2s" if running else None)
def convergence_panel() -> None:
    if running:
        st.caption(
            f"{STATUS_ICON['running']} {len(running)} expérience(s) en cours — "
            "rafraîchissement automatique"
        )

    fig = go.Figure()
    rows = []

    for run_id in selected_ids:
        run = api_get(f"/runs/{run_id}")
        metrics = api_get(f"/runs/{run_id}/metrics")
        if not metrics:
            continue

        add_curve(fig, run, pd.DataFrame(metrics), "global_acc")
        rows.append(
            {
                "Expérience": run_label(run),
                "Seed": run["config"]["seed"],
                "Statut": STATUS_TEXT[run["status"]],
                "Rounds": run["current_round"],
                "Accuracy finale": run["final_acc"],
                "Rounds → cible": run["rounds_to_target"] or "jamais",
            }
        )

    st.plotly_chart(
        style_axes(fig, "Round de communication", "Accuracy globale (test)"),
        use_container_width=True,
    )

    # Vue tabulaire : indispensable dès qu'une couleur porte du sens, et
    # directement reprenable dans le rapport.
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


with tab_conv:
    convergence_panel()


# ------------------------------------------------------------ tableau croisé
with tab_grid:
    st.subheader("Accuracy finale par algorithme et niveau d'hétérogénéité")
    st.caption(
        "C'est le livrable central du sujet. Il se remplit avec les expériences "
        "terminées ; comptez trois seeds par cellule pour que l'écart soit défendable."
    )

    done = [r for r in runs if r["status"] == "done" and r["final_acc"] is not None]
    if not done:
        st.info("Aucune expérience terminée pour l'instant.")
    else:
        df = pd.DataFrame(
            [
                {
                    "Algorithme": (
                        f"FedProx (μ={r['config']['mu']:g})"
                        if r["config"]["algo"] == "fedprox"
                        else ALGO_LABEL[r["config"]["algo"]]
                    ),
                    "alpha": r["config"]["alpha"],
                    "seed": r["config"]["seed"],
                    "final_acc": r["final_acc"],
                    "rounds_to_target": r["rounds_to_target"],
                }
                for r in done
            ]
        )

        pivot = df.pivot_table(
            index="Algorithme", columns="alpha", values="final_acc", aggfunc="mean"
        )
        st.dataframe(
            pivot.style.format("{:.3f}").background_gradient(cmap="Blues", axis=None),
            use_container_width=True,
        )
        st.caption("α croissant de gauche à droite : l'hétérogénéité décroît.")

        st.subheader("Détail par cellule")
        detail = (
            df.groupby(["Algorithme", "alpha"])
            .agg(
                seeds=("seed", "count"),
                acc_moyenne=("final_acc", "mean"),
                acc_ecart_type=("final_acc", "std"),
                rounds_cible=("rounds_to_target", "mean"),
            )
            .reset_index()
        )
        st.dataframe(
            detail.style.format(
                {
                    "acc_moyenne": "{:.3f}",
                    "acc_ecart_type": "{:.3f}",
                    "rounds_cible": "{:.0f}",
                },
                na_rep="—",
            ),
            use_container_width=True,
            hide_index=True,
        )

        if detail["seeds"].max() < 3:
            st.warning(
                "Moins de trois seeds par configuration : aucun écart ne peut être "
                "qualifié de significatif à ce stade."
            )


# ---------------------------------------------------------------- drift
with tab_drift:
    st.subheader("Dispersion des accuracies locales")
    st.caption(
        "Écart-type des performances entre clients. Plus il est élevé, plus les "
        "modèles locaux divergent du modèle global : c'est la mesure directe du "
        "client drift, le phénomène que FedProx cherche à contenir."
    )

    fig = go.Figure()
    for run_id in selected_ids:
        run = api_get(f"/runs/{run_id}")
        metrics = api_get(f"/runs/{run_id}/metrics")
        if metrics:
            add_curve(fig, run, pd.DataFrame(metrics), "std_client_acc")

    st.plotly_chart(
        style_axes(fig, "Round de communication", "Écart-type inter-clients"),
        use_container_width=True,
    )
