"""Streamlit dashboard for Brazilian greenhouse gas emissions."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import pydeck as pdk
import streamlit as st

# ---------------------------------------------------------------------------
# Global configuration and constants
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Emissões GEE Brasil",
    page_icon="🌎",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_PATH = Path("emissoes_estado_filtrado.csv")
RAW_DATA_PATH = Path("dados_gases.xlsx")
RAW_SHEET_NAME = "GEE Estados"
COLOR_SCALE = ["#003f5c", "#2f4b7c", "#665191", "#a05195", "#d45087", "#f95d6a", "#ff7c43", "#ffa600"]
PLOTLY_CONFIG = {"locale": "pt-BR"}

STATE_COORDS = {
    "AC": (-8.77, -70.55),
    "AL": (-9.62, -36.82),
    "AP": (1.41, -51.77),
    "AM": (-3.47, -65.10),
    "BA": (-12.97, -38.51),
    "CE": (-3.71, -38.54),
    "DF": (-15.83, -47.86),
    "ES": (-19.19, -40.34),
    "GO": (-16.64, -49.31),
    "MA": (-2.55, -44.30),
    "MT": (-12.64, -55.42),
    "MS": (-20.51, -54.54),
    "MG": (-18.10, -44.38),
    "PA": (-1.35, -48.20),
    "PB": (-7.12, -34.86),
    "PR": (-24.89, -51.55),
    "PE": (-8.28, -35.07),
    "PI": (-8.28, -43.68),
    "RJ": (-22.91, -43.17),
    "RN": (-5.81, -35.21),
    "RS": (-30.01, -51.22),
    "RO": (-11.22, -62.80),
    "RR": (2.82, -60.67),
    "SC": (-27.60, -48.55),
    "SP": (-23.55, -46.64),
    "SE": (-10.57, -37.45),
    "TO": (-10.25, -48.25),
}

REGION_MAP = {
    "AC": "Norte",
    "AL": "Nordeste",
    "AP": "Norte",
    "AM": "Norte",
    "BA": "Nordeste",
    "CE": "Nordeste",
    "DF": "Centro-Oeste",
    "ES": "Sudeste",
    "GO": "Centro-Oeste",
    "MA": "Nordeste",
    "MT": "Centro-Oeste",
    "MS": "Centro-Oeste",
    "MG": "Sudeste",
    "PA": "Norte",
    "PB": "Nordeste",
    "PR": "Sul",
    "PE": "Nordeste",
    "PI": "Nordeste",
    "RJ": "Sudeste",
    "RN": "Nordeste",
    "RS": "Sul",
    "RO": "Norte",
    "RR": "Norte",
    "SC": "Sul",
    "SP": "Sudeste",
    "SE": "Nordeste",
    "TO": "Norte",
}


# ---------------------------------------------------------------------------
# Data utilities
# ---------------------------------------------------------------------------
def winsorize_iqr(series: pd.Series) -> pd.Series:
    """Replicates the notebook's IQR-based winsorization."""

    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        return series
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return series.clip(lower, upper)


def build_dataset_from_notebook_source() -> pd.DataFrame:
    """Reproduz o pipeline do notebook para gerar o dataset final."""

    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            "Arquivo 'dados_gases.xlsx' não encontrado. Gere-o com o notebook ou mantenha-o na raiz do projeto."
        )

    emissoes_raw = pd.read_excel(RAW_DATA_PATH, sheet_name=RAW_SHEET_NAME)

    cols_to_drop = ["Produto"]
    fill_defaults = {
        "Estado": "BRASIL",
        "Atividade Econômica": "Não informado",
        "Nível 3": "Não informado",
        "Nível 4": "Não informado",
        "Nível 5": "Não informado",
        "Nível 6": "Não informado",
    }

    emissoes = (
        emissoes_raw.loc[emissoes_raw["Emissão / Remoção / Bunker"] == "Emissão"]
        .drop(columns=cols_to_drop)
        .copy()
    )

    demographic_cols = list(fill_defaults.keys())
    emissoes[demographic_cols] = emissoes[demographic_cols].fillna(fill_defaults)
    emissoes = emissoes.drop(columns=["Emissão / Remoção / Bunker"])

    info_cols = [col for col in emissoes.columns if not isinstance(col, int)]
    year_cols = sorted([col for col in emissoes.columns if isinstance(col, int)])
    emissoes_long = emissoes.melt(
        id_vars=info_cols,
        value_vars=year_cols,
        var_name="Ano",
        value_name="Emissao",
    )

    emissoes_long["Ano"] = emissoes_long["Ano"].astype(int)
    emissoes_estado = (
        emissoes_long.groupby(["Estado", "Ano"], as_index=False)["Emissao"].sum()
        .query("Estado != 'BRASIL'")
    )
    emissoes_estado["Emissao_sem_outlier"] = (
        emissoes_estado.groupby("Ano")["Emissao"].transform(winsorize_iqr)
    )
    return emissoes_estado


@st.cache_data(show_spinner=False)
def load_default_dataset() -> pd.DataFrame:
    """Load cached CSV or rebuild it automatically from the Excel source."""

    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH)

    try:
        df = build_dataset_from_notebook_source()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    df.to_csv(DATA_PATH, index=False)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names, derive helpers and ensure numeric types."""

    # Normalize column names
    df = df.rename(
        columns=lambda col: col.strip().replace(" ", "_").lower()
    )

    # Expected columns from notebook pipeline
    required_cols = {"estado", "ano", "emissao", "emissao_sem_outlier"}
    missing = required_cols.intersection(df.columns) ^ required_cols
    if missing:
        st.warning(
            "Colunas esperadas não encontradas. Verifique se o CSV veio do notebook atualizado."
        )

    df["ano"] = df["ano"].astype(int)
    metric_col = "emissao_sem_outlier" if "emissao_sem_outlier" in df.columns else "emissao"
    df[metric_col] = pd.to_numeric(df[metric_col], errors="coerce")
    df = df.dropna(subset=[metric_col])

    df["estado"] = df["estado"].str.upper()
    df["regiao"] = df["estado"].map(REGION_MAP)
    df["emissao_base"] = df[metric_col]
    return df


def filter_data(
    df: pd.DataFrame,
    year_range: Tuple[int, int],
    states: list[str],
    min_emission: float,
) -> pd.DataFrame:
    """Apply dynamic filters coming from sidebar controls."""

    filtered = df.query("@year_range[0] <= ano <= @year_range[1]")
    if states:
        filtered = filtered[filtered["estado"].isin(states)]
    filtered = filtered[filtered["emissao_base"] >= min_emission]
    return filtered


def compute_metrics(df: pd.DataFrame) -> dict[str, float]:
    """Summarize key KPIs to highlight in metric cards."""

    total = df["emissao_base"].sum()
    growth = (
        df.groupby("ano")["emissao_base"].sum().pct_change().iloc[-1] * 100
        if df["ano"].nunique() > 1
        else np.nan
    )
    avg_state = df.groupby("estado")["emissao_base"].mean().mean()
    top_state = (
        df.groupby("estado")["emissao_base"].sum().idxmax()
        if not df.empty
        else "—"
    )
    return {
        "total": total,
        "growth": growth,
        "avg_state": avg_state,
        "top_state": top_state,
    }


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------
def format_br_number(value: float, decimals: int = 2) -> str:
    """Format numbers using Brazilian separators (thousand='.' decimal=',')."""

    if value is None:
        return "—"
    if isinstance(value, (int, np.integer)):
        value = float(value)
    if isinstance(value, (float, np.floating)) and np.isnan(value):
        return "—"
    try:
        formatted = f"{value:,.{decimals}f}"
    except (TypeError, ValueError):
        return str(value)
    return formatted.replace(",", "X").replace(".", ",").replace("X", ".")


# ---------------------------------------------------------------------------
# Visualization functions
# ---------------------------------------------------------------------------
def line_trend_chart(df: pd.DataFrame):
    agg = df.groupby("ano", as_index=False)["emissao_base"].sum()
    fig = px.line(
        agg,
        x="ano",
        y="emissao_base",
        markers=True,
        title="Evolução anual das emissões",
        color_discrete_sequence=[COLOR_SCALE[0]],
    )
    fig.update_layout(
        yaxis_title="Emissões (t CO₂e)",
        xaxis_title="Ano",
        separators=",.",
        yaxis_tickformat=",.0f",
    )
    fig.update_traces(
        hovertemplate="Ano: %{x}<br>Emissões: %{y:,.2f} t CO₂e<extra></extra>"
    )
    return fig


def top_states_bar(df: pd.DataFrame):
    agg = (
        df.groupby("estado", as_index=False)["emissao_base"].sum()
        .sort_values("emissao_base", ascending=False)
        .head(10)
    )
    fig = px.bar(
        agg,
        x="estado",
        y="emissao_base",
        title="Top 10 estados emissores",
        color="estado",
        color_discrete_sequence=COLOR_SCALE,
    )
    fig.update_layout(
        xaxis_title="Estado",
        yaxis_title="Emissões (t CO₂e)",
        showlegend=False,
        separators=",.",
        yaxis_tickformat=",.0f",
    )
    fig.update_traces(
        hovertemplate="%{x}: %{y:,.2f} t CO₂e<extra></extra>",
        selector=dict(type="bar"),
    )
    return fig


def regional_donut(df: pd.DataFrame):
    agg = df.groupby("regiao", dropna=False)["emissao_base"].sum().reset_index()
    agg["regiao"] = agg["regiao"].fillna("Não mapeado")
    total_emission = agg["emissao_base"].sum()
    if total_emission == 0:
        agg["participacao"] = 0.0
    else:
        agg["participacao"] = agg["emissao_base"] / total_emission * 100
    agg["participacao_fmt"] = agg["participacao"].apply(lambda val: format_br_number(val, decimals=1))
    fig = px.pie(
        agg,
        names="regiao",
        values="emissao_base",
        hole=0.5,
        color_discrete_sequence=COLOR_SCALE,
        title="Participação por região",
        custom_data=["participacao_fmt"],
    )
    fig.update_layout(separators=",.")
    fig.update_traces(
        textposition="inside",
        texttemplate="%{label}<br>%{customdata[0]}%",
        hovertemplate=(
            "%{label}<br>Participação: %{customdata[0]}%"
            "<br>Emissões: %{value:,.2f} t CO₂e<extra></extra>"
        ),
    )
    return fig


def geo_heatmap(df: pd.DataFrame):
    geo_df = (
        df.groupby("estado", as_index=False)["emissao_base"].mean()
    )
    geo_df["lat"] = geo_df["estado"].map(lambda s: STATE_COORDS.get(s, (np.nan, np.nan))[0])
    geo_df["lon"] = geo_df["estado"].map(lambda s: STATE_COORDS.get(s, (np.nan, np.nan))[1])
    geo_df = geo_df.dropna(subset=["lat", "lon"])
    geo_df["emissao_fmt"] = geo_df["emissao_base"].apply(lambda val: format_br_number(val, decimals=2))

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=geo_df,
        get_position="[lon, lat]",
        get_fill_color="[255, 140, 0, 180]",
        get_radius="emissao_base / 2000",
        pickable=True,
    )
    view_state = pdk.ViewState(latitude=-14.235, longitude=-51.9253, zoom=3.5)
    return pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        tooltip={"text": "{estado}: {emissao_fmt} t CO₂e"},
    )


def emission_correlation(df: pd.DataFrame):
    pivot = (
        df.pivot_table(index="estado", columns="ano", values="emissao_base", aggfunc="mean")
    )
    corr = pivot.corr().round(2)
    fig = px.imshow(
        corr,
        color_continuous_scale="RdBu_r",
        title="Correlação entre anos",
        aspect="auto",
    )
    fig.update_layout(separators=",.", coloraxis_colorbar_tickformat=",.2f")
    fig.update_traces(
        hovertemplate=(
            "Ano %{x}<br>Ano %{y}<br>Correlação: %{z:,.2f}<extra></extra>"
        )
    )
    return fig


def stats_table(df: pd.DataFrame) -> pd.DataFrame:
    summary = df.groupby("estado")["emissao_base"].agg(
        Total="sum",
        Média="mean",
        Mediana="median",
    )
    summary = summary.sort_values("Total", ascending=False)
    formatted = summary.applymap(lambda val: format_br_number(val, decimals=2))
    return formatted


# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
def sidebar_filters(base_df: pd.DataFrame) -> tuple:
    st.sidebar.header("⚙️ Controles")
    uploaded_file = st.sidebar.file_uploader("Upload do CSV (opcional)", type=["csv"])
    if uploaded_file is not None:
        st.sidebar.success("Arquivo carregado!", icon="📂")
        df = clean_data(pd.read_csv(uploaded_file))
    else:
        df = base_df

    min_year, max_year = int(df["ano"].min()), int(df["ano"].max())
    year_range = st.sidebar.slider("Intervalo de anos", min_year, max_year, (min_year, max_year))

    states = sorted(df["estado"].unique())
    selected_states = st.sidebar.multiselect(
        "Estados",
        options=states,
        default=[],
        help="Deixe vazio para considerar todos",
    )

    min_value = float(df["emissao_base"].min())
    max_value = float(df["emissao_base"].max())
    min_emission = st.sidebar.number_input(
        "Emissão mínima",
        min_value=min_value,
        max_value=max_value,
        value=min_value,
        step=(max_value - min_value) / 100,
    )

    return df, year_range, selected_states, min_emission


# ---------------------------------------------------------------------------
# Main dashboard layout
# ---------------------------------------------------------------------------
def main():
    st.title("Dashboard de Emissões de GEE do Brasil")
    st.markdown(
        """
        Painel desenvolvido para evidenciar tendências estaduais e apoiar decisões rápidas. Use os filtros na lateral
        para ajustar a narrativa e exportar o dataset filtrado para outras ferramentas.
        """
    )

    base_df = clean_data(load_default_dataset())
    data, year_range, selected_states, min_emission = sidebar_filters(base_df)
    filtered_df = filter_data(data, year_range, selected_states, min_emission)

    metrics = compute_metrics(filtered_df)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(
        "Emissões totais",
        f"{format_br_number(metrics['total']/1e6, decimals=2)} mi t"
    )
    col2.metric(
        "Variação anual",
        f"{format_br_number(metrics['growth'], decimals=1)}%" if not np.isnan(metrics["growth"]) else "—",
        help="Comparação com o ano anterior"
    )
    col3.metric(
        "Média por estado",
        f"{format_br_number(metrics['avg_state']/1e6, decimals=2)} mi t"
    )
    col4.metric("Estado destaque", metrics["top_state"])

    st.subheader("Visão geral")
    col_a, col_b = st.columns((2, 1))
    col_a.plotly_chart(line_trend_chart(filtered_df), use_container_width=True, config=PLOTLY_CONFIG)
    col_b.plotly_chart(regional_donut(filtered_df), use_container_width=True, config=PLOTLY_CONFIG)

    st.subheader("Comparações por estado")
    col_c, col_d = st.columns((3, 2))
    col_c.plotly_chart(top_states_bar(filtered_df), use_container_width=True, config=PLOTLY_CONFIG)
    col_d.pydeck_chart(geo_heatmap(filtered_df))

    st.subheader("Relações e correlações")
    st.plotly_chart(emission_correlation(filtered_df), use_container_width=True, config=PLOTLY_CONFIG)

    st.subheader("Tabela detalhada")
    summary = stats_table(filtered_df)
    st.dataframe(summary, use_container_width=True)

    st.markdown("### Exportar recorte")
    csv = filtered_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Baixar CSV filtrado",
        data=csv,
        file_name="emissoes_filtrado_dashboard.csv",
        mime="text/csv",
    )

    st.markdown("---")
    st.caption("Fonte: SEEG (dados tratados no notebook de portfólio)")


if __name__ == "__main__":
    main()
