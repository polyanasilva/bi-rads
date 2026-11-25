"""Figuras Plotly do dashboard."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dashboard import dados as d

# Paleta (rampa azul ordinal para o BI-RADS, a mesma do notebook)
CORES_BIRADS = dict(zip(d.BIRADS, ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]))
CORES_GRUPO = dict(zip(d.GRUPOS, ["#86b6ef", "#2a78d6", "#104281"]))
CORES_TIPO = {"lesão": "#2a78d6", "contexto": "#eb6834", "clínica": "#1baf7a"}
VAZIO = "#c3c2b7"

SUPERFICIE = "#fcfcfb"
TINTA = "#0b0b0b"
TINTA_2 = "#52514e"
MUDO = "#898781"
GRADE = "#e1e0d9"
EIXO = "#c3c2b7"
FONTE = 'system-ui, -apple-system, "Segoe UI", sans-serif'


def _layout(fig, titulo=None, altura=360, legenda=True, **kw):
    fig.update_layout(
        title=dict(text=titulo, x=0, xanchor="left", font=dict(size=15, color=TINTA)) if titulo else None,
        height=altura,
        font=dict(family=FONTE, size=12, color=TINTA_2),
        paper_bgcolor=SUPERFICIE,
        plot_bgcolor=SUPERFICIE,
        margin=dict(l=10, r=20, t=50 if titulo else 20, b=10),
        showlegend=legenda,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1, traceorder="normal",
                    font=dict(color=TINTA_2), title_font=dict(color=TINTA_2)),
        separators=",.",
        hoverlabel=dict(bgcolor="white", bordercolor=GRADE, font=dict(family=FONTE, color=TINTA)),
        barcornerradius=4,
        bargap=0.3,
        **kw,
    )
    eixo = dict(gridcolor=GRADE, linecolor=EIXO, zerolinecolor=EIXO, tickfont=dict(color=MUDO),
                title_font=dict(color=TINTA_2))
    fig.update_xaxes(**eixo)
    fig.update_yaxes(**eixo)
    return fig


def _pct(eixo):
    eixo.update(tickformat=".0%", range=[0, 1])


# ---------------------------------------------------------------- alvo

def birads_barras(modo="categorias"):
    df = d.carregar()
    n = len(df)
    if modo == "grupos":
        cont = df["grupo"].value_counts().reindex(d.GRUPOS)
        rotulos, cores = d.GRUPOS, [CORES_GRUPO[g] for g in d.GRUPOS]
    else:
        cont = df["birads_ultrasound"].dropna().astype(int).value_counts().reindex(d.BIRADS)
        rotulos, cores = [str(b) for b in d.BIRADS], [CORES_BIRADS[b] for b in d.BIRADS]
    fig = go.Figure(go.Bar(
        x=rotulos, y=cont.values, marker_color=cores,
        text=[f"{v} ({v / n:.1%})" for v in cont.values], textposition="outside",
        textfont=dict(color=TINTA_2),
        hovertemplate="BI-RADS %{x}<br>%{y} mamas<extra></extra>",
    ))
    _layout(fig, "BI-RADS por mama", legenda=False)
    fig.update_xaxes(title="BI-RADS", type="category", showgrid=False)
    fig.update_yaxes(title="mamas", range=[0, cont.max() * 1.18])
    return fig


def lados_heatmap():
    t, _ = d.lados()
    fig = go.Figure(go.Heatmap(
        z=t.values, x=[str(c) for c in t.columns], y=[str(i) for i in t.index],
        colorscale=[[0, "#f0efec"], [0.02, "#cde2fb"], [0.3, "#6da7ec"], [1, "#104281"]],
        text=t.values, texttemplate="%{text}", textfont=dict(size=12),
        xgap=2, ygap=2, showscale=False,
        hovertemplate="direita %{y} · esquerda %{x}<br>%{z} exames<extra></extra>",
    ))
    _layout(fig, "BI-RADS da mama direita × esquerda (exames)", altura=360, legenda=False)
    fig.update_xaxes(title="esquerda", showgrid=False)
    fig.update_yaxes(title="direita", showgrid=False, autorange="reversed", scaleanchor="x")
    return fig


# ---------------------------------------------------------------- lesões

def mapa_vazios():
    m, linhas = d.ordem_mapa_vazios()
    birads = linhas["birads_ultrasound"].map(lambda b: "vazio" if pd.isna(b) else str(int(b))).values
    custom = np.tile(np.stack([linhas["sample_id"].values, birads], axis=-1), (m.shape[0], 1, 1))
    fig = go.Figure(go.Heatmap(
        z=m.values, x=np.arange(m.shape[1]), y=[d.nome(c) for c in m.index],
        colorscale=[[0, "#ebeae5"], [0.5, "#ebeae5"], [0.5, "#2a78d6"], [1, "#2a78d6"]],
        zmin=0, zmax=1, showscale=False, customdata=custom,
        hovertemplate="%{customdata[0]} (BI-RADS %{customdata[1]})<br>%{y}<extra></extra>",
    ))
    # marca onde começam as mamas com lesão
    n_sem = int((linhas["n_descritores"] == 0).sum())
    fig.add_vline(x=n_sem - 0.5, line=dict(color=TINTA, width=1, dash="dot"))
    fig.add_annotation(x=n_sem / 2, y=1.02, yref="paper", text=f"{n_sem} mamas sem descritor",
                       showarrow=False, font=dict(color=TINTA_2))
    fig.add_annotation(x=n_sem + (m.shape[1] - n_sem) / 2, y=1.02, yref="paper",
                       text=f"{m.shape[1] - n_sem} com lesão", showarrow=False, font=dict(color=TINTA_2))
    _layout(fig, None, altura=620, legenda=False)
    fig.update_layout(margin=dict(t=30))
    fig.update_xaxes(title="mamas (ordenadas pelo nº de descritores de lesão preenchidos)",
                     showticklabels=False, showgrid=False)
    fig.update_yaxes(showgrid=False, tickfont=dict(size=11, color=TINTA_2), autorange="reversed")
    return fig


def lesao_por_birads():
    df = d.carregar().dropna(subset=["birads_ultrasound"])
    t = pd.crosstab(df["birads_ultrasound"].astype(int), df["tem_lesao"]).reindex(d.BIRADS, fill_value=0)
    fig = go.Figure()
    for col, cor in [("sem lesão", VAZIO), ("com lesão", "#2a78d6")]:
        fig.add_bar(x=[str(b) for b in t.index], y=t[col], name=col, marker_color=cor,
                    marker_line=dict(color=SUPERFICIE, width=2),
                    hovertemplate=f"BI-RADS %{{x}}<br>{col}: %{{y}} mamas<extra></extra>")
    _layout(fig, "Lesão descrita × BI-RADS", barmode="stack")
    fig.update_xaxes(title="BI-RADS", type="category", showgrid=False)
    fig.update_yaxes(title="mamas")
    return fig


def empilhado_grupo(coluna, dados, titulo=None):
    """Proporção de cada grupo de BI-RADS dentro de cada categoria da variável."""
    x = dados[coluna].fillna("vazio")
    t = pd.crosstab(x, dados["grupo"], normalize="index").reindex(columns=d.GRUPOS, fill_value=0)
    n = x.value_counts().reindex(t.index)
    ordem = n.sort_values().index
    t, n = t.loc[ordem], n.loc[ordem]
    rotulos = [f"{k}  (n={v})" for k, v in n.items()]
    cont = pd.crosstab(x, dados["grupo"]).reindex(index=ordem, columns=d.GRUPOS, fill_value=0)
    fig = go.Figure()
    for g in d.GRUPOS:
        fig.add_bar(y=rotulos, x=t[g], name=g, orientation="h", marker_color=CORES_GRUPO[g],
                    marker_line=dict(color=SUPERFICIE, width=2), customdata=cont[g],
                    hovertemplate=f"%{{y}}<br>BI-RADS {g}: %{{x:.0%}} (%{{customdata}} mamas)<extra></extra>")
    _layout(fig, titulo or d.nome(coluna), altura=max(220, 60 + 42 * len(t)), barmode="stack",
            legend_title_text="BI-RADS")
    fig.update_layout(barcornerradius=0)
    _pct(fig.layout.xaxis)
    fig.update_yaxes(showgrid=False)
    return fig


def descritor_contagem(coluna):
    df = d.carregar()
    com = df[(df["tem_lesao"] == "com lesão") & df["birads_ultrasound"].notna()]
    t = pd.crosstab(com[coluna].fillna("vazio"), com["birads_ultrasound"].astype(int))
    t = t.reindex(columns=[b for b in d.BIRADS if b in t.columns], fill_value=0)
    return t


# ---------------------------------------------------------------- 1 vs 2

def um_vs_dois(coluna):
    df = d.carregar()
    sem = df[(df["tem_lesao"] == "sem lesão") & df["birads_ultrasound"].isin([1, 2])]
    t = pd.crosstab(sem[coluna].fillna("vazio"), sem["birads_ultrasound"].astype(int)).reindex(columns=[1, 2])
    fig = go.Figure()
    for b in [1, 2]:
        tot = t[b].sum()
        fig.add_bar(x=t.index, y=t[b] / tot, name=f"BI-RADS {b}", marker_color={1: CORES_BIRADS[1], 2: CORES_BIRADS[3]}[b],
                    customdata=np.stack([t[b], np.full(len(t), tot)], axis=-1),
                    text=[f"{v}" for v in t[b]], textposition="outside", textfont=dict(color=TINTA_2),
                    hovertemplate=f"BI-RADS {b}<br>%{{x}}: %{{customdata[0]}} de %{{customdata[1]}} mamas "
                                  f"(%{{y:.0%}})<extra></extra>")
    _layout(fig, d.nome(coluna), altura=340, barmode="group", bargroupgap=0.08)
    fig.update_xaxes(showgrid=False, type="category")
    fig.update_yaxes(title="% das mamas do BI-RADS", tickformat=".0%", range=[0, 1.1])
    return fig


# ---------------------------------------------------------------- clínicas

def numerica_box(coluna):
    df = d.carregar().dropna(subset=["birads_ultrasound"])
    fig = go.Figure()
    for b in d.BIRADS:
        v = df.loc[df["birads_ultrasound"] == b, coluna].dropna()
        fig.add_box(y=v, name=str(b), marker_color=CORES_BIRADS[b], line=dict(width=1.5),
                    boxpoints="outliers", marker_size=4,
                    hovertemplate=f"BI-RADS {b}<br>%{{y}}<extra></extra>")
    _layout(fig, f"{d.nome(coluna).capitalize()} por BI-RADS (mamas)", legenda=False)
    fig.update_xaxes(title="BI-RADS", showgrid=False)
    fig.update_yaxes(title=d.nome(coluna))
    return fig


def numerica_hist(coluna):
    v = d.exames()[coluna].dropna()
    tamanho = 1 if coluna != "age_years" else 3
    fig = go.Figure(go.Histogram(x=v, xbins=dict(size=tamanho), marker_color="#2a78d6",
                                 marker_line=dict(color=SUPERFICIE, width=1),
                                 hovertemplate="%{x}<br>%{y} exames<extra></extra>"))
    if coluna == "age_at_menopause_years":
        fig.add_vline(x=40, line=dict(color=TINTA_2, dash="dash", width=1))
        fig.add_annotation(x=40, y=1, yref="paper", text=" 40 anos", showarrow=False, xanchor="left",
                           font=dict(color=TINTA_2))
    _layout(fig, f"Distribuição de {d.nome(coluna)} (exames, n={len(v)})", legenda=False)
    fig.update_layout(barcornerradius=2, bargap=0.05)
    fig.update_xaxes(title=d.nome(coluna), showgrid=False)
    fig.update_yaxes(title="exames")
    return fig


def categorica_freq(coluna):
    f = d.exames()[coluna].fillna("vazio").value_counts().sort_values()
    fig = go.Figure(go.Bar(y=f.index, x=f.values, orientation="h",
                           marker_color=[VAZIO if k == "vazio" else "#2a78d6" for k in f.index],
                           text=f.values, textposition="outside", textfont=dict(color=TINTA_2),
                           hovertemplate="%{y}: %{x} exames<extra></extra>"))
    _layout(fig, f"{d.nome(coluna).capitalize()} (exames)", altura=max(220, 60 + 42 * len(f)), legenda=False)
    fig.update_xaxes(range=[0, f.max() * 1.15])
    fig.update_yaxes(showgrid=False)
    return fig


# ---------------------------------------------------------------- associação

def associacao_barras():
    a = d.associacao().sort_values("grupos")
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.04,
                        subplot_titles=["grupos 1–2 / 3 / 4–5 (todas as mamas)", "BI-RADS 1 vs 2"])
    for tipo, cor in CORES_TIPO.items():
        s = a[a["tipo"] == tipo]
        for col, (coluna, mostrar) in enumerate([("grupos", True), ("1 vs 2", False)], start=1):
            fig.add_bar(y=s["nome"], x=s[coluna].fillna(0), orientation="h", name=tipo, marker_color=cor,
                        legendgroup=tipo, showlegend=mostrar,
                        hovertemplate="%{y}<br>V de Cramér = %{x:.2f}<extra>" + tipo + "</extra>",
                        row=1, col=col)
    _layout(fig, None, altura=740, legend_title_text="tipo")
    fig.update_layout(margin=dict(t=90), bargap=0.25, legend=dict(y=1.08, x=0, xanchor="left"))
    fig.update_yaxes(categoryorder="array", categoryarray=a["nome"].tolist(), showgrid=False)
    fig.update_xaxes(range=[0, 1], title="V de Cramér")
    for anot in fig.layout.annotations:
        anot.font = dict(size=13, color=TINTA)
    return fig


# ---------------------------------------------------------------- qualidade

def faltantes_barras():
    f = d.faltantes()
    cores = ["#2a78d6" if c in d.LESAO else ("#eb6834" if c in ("age_at_menopause_years",
                                                                "dense_content_cyst_ultrasound")
                                             else VAZIO) for c in f.index]
    fig = go.Figure(go.Bar(y=[d.nome(c) for c in f.index], x=f.values, orientation="h", marker_color=cores,
                           text=[f"{v:.1%}" for v in f.values], textposition="outside",
                           textfont=dict(color=TINTA_2, size=11),
                           hovertemplate="%{y}<br>%{x:.1%} das mamas vazias<extra></extra>"))
    _layout(fig, "Proporção de mamas com o campo vazio", altura=560, legenda=False)
    fig.update_xaxes(tickformat=".0%", range=[0, 1.12])
    fig.update_yaxes(showgrid=False)
    return fig


def rotulo_barras():
    t = d.rotulo_cancer()
    prop = t.div(t.sum(axis=1), axis=0)
    fig = go.Figure()
    for col, cor in [("1 (câncer)", "#104281"), ("vazio", VAZIO)]:
        fig.add_bar(x=[str(b) for b in t.index], y=prop[col], name=col, marker_color=cor,
                    marker_line=dict(color=SUPERFICIE, width=2), customdata=t[col],
                    text=[str(v) if v else "" for v in t[col]], textposition="inside",
                    insidetextanchor="middle",
                    textfont=dict(color="white" if col != "vazio" else TINTA),
                    hovertemplate=f"pior BI-RADS %{{x}}<br>rótulo {col}: %{{customdata}} exames "
                                  f"(%{{y:.0%}})<extra></extra>")
    _layout(fig, "Rótulo de câncer pelo pior BI-RADS do exame", barmode="stack",
            legend_title_text="rótulo")
    fig.update_layout(barcornerradius=0)
    fig.update_xaxes(title="pior BI-RADS do exame", type="category", showgrid=False)
    fig.update_yaxes(title="% dos exames", tickformat=".0%", range=[0, 1])
    return fig
