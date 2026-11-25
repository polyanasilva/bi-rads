"""Dashboard da análise exploratória do BI-RADS.

Rodar a partir da raiz do projeto:
    python -m dashboard.app
e abrir http://127.0.0.1:8050
"""

import pandas as pd
from dash import Dash, Input, Output, dcc, html

from dashboard import dados as d
from dashboard import graficos as g

app = Dash(__name__, title="BI-RADS · análise exploratória")
server = app.server

CONFIG = {"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]}


# ---------------------------------------------------------------- componentes

def grafico(fig=None, id=None):
    kw = {"id": id} if id else {}
    return dcc.Graph(figure=fig, config=CONFIG, **kw) if fig is not None else dcc.Graph(config=CONFIG, **kw)


def card(*filhos, classe=""):
    return html.Div(list(filhos), className=f"card {classe}")


def nota(*itens):
    """Lista curta de leituras do gráfico."""
    return html.Ul([html.Li(i) for i in itens], className="nota")


def tabela(df, numericas=()):
    cab = html.Thead(html.Tr([html.Th(c) for c in df.columns]))
    corpo = html.Tbody([
        html.Tr([html.Td(v, className="num" if c in numericas else None) for c, v in zip(df.columns, linha)])
        for linha in df.itertuples(index=False)
    ])
    return html.Div(html.Table([cab, corpo]), className="tabela")


def kpi(valor, rotulo, detalhe=None):
    return html.Div([
        html.Div(rotulo, className="kpi-rotulo"),
        html.Div(f"{valor:,}".replace(",", "."), className="kpi-valor"),
        html.Div(detalhe, className="kpi-detalhe") if detalhe else None,
    ], className="kpi")


def opcoes(colunas):
    return [{"label": d.nome(c), "value": c} for c in colunas]


# ---------------------------------------------------------------- abas

R = d.resumo()
_, IGUAIS = d.lados()


def aba_alvo():
    grupos = d.tabela_grupos()
    return html.Div([
        html.Div([
            card(
                html.H3("O que é uma linha"),
                html.P("Cada exame gera duas linhas, uma por mama. São 500 exames de 495 pacientes, "
                       "então 5 pacientes aparecem em dois exames."),
                html.Pre("paciente (495)\n└── exame (500)\n    ├── mama direita (R)\n    └── mama esquerda (L)",
                         className="arvore"),
                html.P("As duas mamas de um exame compartilham idade, histórico e hábitos. As linhas não são "
                       "independentes, e a divisão treino/teste precisa ser feita por paciente."),
                html.P(["Nível do exame: dados clínicos, densidade, assimetria, linfonodos e rótulo de câncer. "
                        "Nível da mama: BI-RADS, descritores da lesão, ectasia ductal e cisto de conteúdo denso."],
                       className="secundario"),
            ),
            card(
                html.H3("Pacientes com mais de um exame"),
                tabela(d.repetidas()),
                html.P("Um par é idêntico (provável duplicata). Nos outros, mudam campos que não deveriam mudar "
                       "na mesma paciente, como etnia e idade da menarca.", className="secundario"),
            ),
        ], className="grade-2"),
        html.Div([
            card(
                dcc.RadioItems(id="modo-birads", value="categorias", inline=True, className="filtro",
                               options=[{"label": "5 categorias", "value": "categorias"},
                                        {"label": "3 grupos (conduta clínica)", "value": "grupos"}]),
                grafico(id="fig-birads"),
                nota("Só aparecem as categorias 1 a 5. Há 3 mamas sem BI-RADS.",
                     "BI-RADS 4 e 5 somam 16 mamas (1,6%), contra 858 nas categorias 1 e 2."),
            ),
            card(
                html.H3("Agrupamento pela conduta"),
                html.P("1–2: rotina · 3: controle em 6 meses · 4–5: biópsia", className="secundario"),
                tabela(grupos, numericas=grupos.columns[1:]),
                html.P("Mesmo agrupado, o grupo 4–5 tem 16 mamas de 16 pacientes, cerca de 1,6 por fold. "
                       "Acertar ou errar uma mama muda a sensibilidade do fold em 50 a 100 pontos percentuais."),
            ),
        ], className="grade-2"),
        html.Div([
            card(
                grafico(g.lados_heatmap()),
                nota(f"{IGUAIS:.0%} dos exames têm o mesmo BI-RADS nas duas mamas "
                     "(correlação de Spearman 0,39).",
                     "A distribuição é parecida entre os lados."),
            ),
            card(
                grafico(g.rotulo_barras()),
                nota("O rótulo de câncer existe só para 8 exames, e não há nenhum negativo confirmado: "
                     "o vazio quer dizer \"não confirmado\".",
                     "O rótulo é da paciente, não da mama. Todos os positivos têm alguma mama BI-RADS 4 ou 5.",
                     "Por isso ele não entra nem como alvo nem como preditora."),
            ),
        ], className="grade-2"),
    ])


def aba_lesao():
    return html.Div([
        card(
            html.H3("Mapa de vazios: azul = preenchido, cinza = vazio"),
            grafico(g.mapa_vazios()),
            nota("Os 9 descritores da lesão faltam juntos: 871 mamas não têm nenhum, e as outras 129 têm "
                 "quase todos.",
                 "O vazio nesses campos quer dizer \"não há lesão descrita nesta mama\". Imputar com a moda "
                 "inventaria uma lesão oval e circunscrita em 871 mamas."),
        ),
        html.Div([
            card(
                grafico(g.lesao_por_birads()),
                nota("BI-RADS 1 e 2: sem lesão descrita em 856 de 858 mamas.",
                     "BI-RADS 3, 4 e 5: com lesão em 127 de 139 mamas.",
                     "Ter ou não ter lesão separa quase perfeitamente {1, 2} de {3, 4, 5}."),
            ),
            card(
                html.Div([html.Label("Descritor"),
                          dcc.Dropdown(id="descritor", options=opcoes(d.LESAO), value="lesion_margin_ultrasound",
                                       clearable=False)], className="filtro"),
                grafico(id="fig-descritor"),
                html.Div(id="tab-descritor"),
                html.P("Só as 129 mamas com lesão. Os descritores reproduzem o léxico do ACR: forma irregular, "
                       "margem não circunscrita, sombra e distorção aparecem quase só no 4–5.",
                       className="secundario"),
            ),
        ], className="grade-2"),
        card(
            html.H3("Achados de contexto"),
            html.Div([html.Label("Variável"),
                      dcc.Dropdown(id="contexto", options=opcoes(d.CONTEXTO), value="lymph_nodes_ultrasound",
                                   clearable=False)], className="filtro"),
            grafico(id="fig-contexto"),
            nota("Linfonodos: Absent e Negative parecem dizer a mesma coisa, mas nenhuma mama 4–5 tem Negative.",
                 "Densidade: 88% das mamas são densas, e nenhuma mama não densa é BI-RADS 1."),
        ),
    ])


def aba_um_dois():
    return html.Div([
        card(
            html.H3("O que separa o BI-RADS 1 do 2?"),
            html.P("BI-RADS 2 quer dizer \"achado benigno\", mas 536 das 538 mamas BI-RADS 2 não têm nenhum "
                   "descritor de lesão. Entre as mamas sem lesão, só dois campos separam bem as duas categorias."),
        ),
        html.Div([
            card(
                grafico(g.um_vs_dois("dense_content_cyst_ultrasound")),
                nota("O campo é vazio em 440 de 536 mamas BI-RADS 2, contra 47 de 320 BI-RADS 1.",
                     "Hipótese: o vazio quer dizer \"há cisto, mas não é denso\" (cisto simples), o achado "
                     "benigno clássico do BI-RADS 2. Se for isso, imputar o campo destrói o sinal."),
            ),
            card(
                grafico(g.um_vs_dois("breast_density_ultrasound")),
                nota("Todas as 320 mamas BI-RADS 1 são densas. \"Not dense\" aparece em 111 mamas BI-RADS 2 "
                     "e em nenhuma BI-RADS 1.",
                     "Não há motivo clínico claro para isso. Pode ser efeito de como a base foi preenchida."),
            ),
        ], className="grade-2"),
    ])


def aba_clinicas():
    return html.Div([
        card(
            html.H3("Variáveis numéricas"),
            html.Div([html.Label("Variável"),
                      dcc.Dropdown(id="numerica", options=opcoes(d.NUMERICAS), value="age_years",
                                   clearable=False)], className="filtro"),
            html.Div([grafico(id="fig-num-hist"), grafico(id="fig-num-box")], className="grade-2 sem-borda"),
            html.Div(id="nota-numerica"),
        ),
        card(
            html.H3("Variáveis categóricas"),
            html.Div([html.Label("Variável"),
                      dcc.Dropdown(id="categorica", options=opcoes(d.CLINICAS), value="breastfeeding_history",
                                   clearable=False)], className="filtro"),
            html.Div([grafico(id="fig-cat-freq"), grafico(id="fig-cat-grupo")], className="grade-2 sem-borda"),
            nota("Na maioria das variáveis as proporções de BI-RADS mudam pouco.",
                 "Amamentação \"No\": cerca de um quarto é BI-RADS 3, contra ~11% entre quem amamentou. "
                 "Provável efeito da idade (mediana de 34 contra 45 anos).",
                 "Categorias com menos de ~10 exames (Indigenous, Asian, álcool \"Yes\") não permitem estimar "
                 "efeito e podem ser agrupadas."),
        ),
    ])


NOTAS_NUM = {
    "age_years": "Média de 43,5 anos (14 a 72). População jovem para rastreamento. O BI-RADS 3 aparece em "
                 "mulheres mais jovens (mediana 41) e o 5 em mais velhas (mediana 49).",
    "age_at_menarche_years": "Entre 8 e 18 anos, centrada em 13. Não varia entre as categorias de BI-RADS.",
    "age_at_menopause_years": "Preenchida em 146 de 500 exames. Cerca de um quarto das menopausas registradas "
                              "aconteceu antes dos 40 anos, o que é muito acima do esperado (~1%).",
    "parity": "De 0 a 13, mediana 3. O BI-RADS 3 aparece em mulheres com menos filhos.",
}


def aba_associacao():
    return html.Div([
        card(
            html.H3("Força da associação com o BI-RADS (V de Cramér)"),
            html.P("0 = nenhuma associação, 1 = uma variável determina a outra. As numéricas foram cortadas em "
                   "quartis e os vazios entram como categoria própria.", className="secundario"),
            grafico(g.associacao_barras()),
            nota("Nos grupos, os descritores da lesão dominam (V entre 0,64 e 0,86). Quase todo esse sinal vem "
                 "de ter ou não ter lesão. Nenhuma variável clínica passa de 0,17.",
                 "Entre 1 e 2, os descritores não ajudam. Quem separa é o cisto de conteúdo denso (V ≈ 0,66, por "
                 "causa do vazio) e, bem atrás, a densidade (≈ 0,30).",
                 "Boa parte do sinal vem de como os campos foram preenchidos, não de uma medida clínica."),
        ),
    ])


def aba_qualidade():
    inc = d.inconsistencias()
    return html.Div([
        html.Div([
            card(
                grafico(g.faltantes_barras()),
                html.Div([
                    html.Span([html.I(className="ponto", style={"background": "#2a78d6"}), "descritores da lesão"]),
                    html.Span([html.I(className="ponto", style={"background": "#eb6834"}),
                               "vazio \"não aplicável\""]),
                    html.Span([html.I(className="ponto", style={"background": g.VAZIO}), "vazios espalhados"]),
                ], className="legenda"),
                nota("Descritores da lesão: 87% a 90% vazios (vascularização 99%).",
                     "Menopausa (71%) e cisto denso (56%): vazio com significado.",
                     "Os demais campos têm até 2% de vazios."),
            ),
            card(
                grafico(g.numerica_hist("age_at_menopause_years")),
                html.H3("Checagens de plausibilidade (por exame)"),
                tabela(inc, numericas=["exames"]),
            ),
        ], className="grade-2"),
    ])


def aba_modelagem():
    return html.Div([
        card(
            html.H3("O que a análise muda na modelagem"),
            html.Ol([
                html.Li([html.B("Divisão por paciente. "),
                         "Treino/teste e folds agrupados por patient_id (StratifiedGroupKFold), para que as duas "
                         "mamas e os exames repetidos fiquem sempre do mesmo lado."]),
                html.Li([html.B("Vazio como categoria. "),
                         "Nos descritores da lesão e no cisto denso, o vazio vira categoria ou uma indicadora "
                         "tem_lesao. Nada de imputar com a moda."]),
                html.Li([html.B("Limpeza. "),
                         "Remover a duplicata (P0104), as 3 mamas sem BI-RADS, os identificadores, o rótulo de "
                         "câncer e, provavelmente, a vascularização."]),
                html.Li([html.B("Alvo agrupado "), "(1–2 / 3 / 4–5) com métricas que não escondem a classe rara: "
                         "acurácia balanceada, recall por classe e matriz de confusão."]),
                html.Li([html.B("Categorias raras "), "agrupadas, e uma variável \"menopausa: sim / não / "
                         "desconhecido\" no lugar da idade da menopausa sozinha."]),
            ]),
        ),
        html.Div([
            kpi(R["preditoras"], "preditoras", "4 numéricas e 23 categóricas"),
            kpi(R["colunas_one_hot"], "colunas depois do one-hot", "com o vazio como categoria"),
            kpi(R["categorias_raras"], "categorias com menos de 10 mamas"),
        ], className="kpis"),
    ])


# ---------------------------------------------------------------- layout

app.layout = html.Div([
    html.Header([
        html.H1("BI-RADS na ultrassonografia"),
        html.P("Análise exploratória do dataset de ultrassonografia mamária"),
    ]),
    html.Div([
        kpi(R["mamas"], "mamas (linhas)"),
        kpi(R["exames"], "exames", "2 mamas por exame"),
        kpi(R["pacientes"], "pacientes", "5 com dois exames"),
        kpi(R["birads_45"], "mamas BI-RADS 4–5", "1,6% da base"),
        kpi(R["sem_lesao"], "mamas sem lesão descrita", "87% da base"),
    ], className="kpis"),
    dcc.Tabs(className="abas", value="alvo", children=[
        dcc.Tab(label="Estrutura e alvo", value="alvo", children=aba_alvo()),
        dcc.Tab(label="Lesões", value="lesao", children=aba_lesao()),
        dcc.Tab(label="BI-RADS 1 vs 2", value="um_dois", children=aba_um_dois()),
        dcc.Tab(label="Variáveis clínicas", value="clinicas", children=aba_clinicas()),
        dcc.Tab(label="Associação", value="associacao", children=aba_associacao()),
        dcc.Tab(label="Qualidade dos dados", value="qualidade", children=aba_qualidade()),
        dcc.Tab(label="Implicações", value="modelagem", children=aba_modelagem()),
    ]),
], className="pagina")


# ---------------------------------------------------------------- callbacks

@app.callback(Output("fig-birads", "figure"), Input("modo-birads", "value"))
def _birads(modo):
    return g.birads_barras(modo)


@app.callback(Output("fig-descritor", "figure"), Output("tab-descritor", "children"), Input("descritor", "value"))
def _descritor(coluna):
    com = d.carregar()
    com = com[(com["tem_lesao"] == "com lesão") & com["grupo"].notna()]
    t = g.descritor_contagem(coluna).reset_index()
    t.columns = [d.nome(coluna)] + [f"BI-RADS {c}" for c in t.columns[1:]]
    return g.empilhado_grupo(coluna, com), tabela(t, numericas=t.columns[1:])


@app.callback(Output("fig-contexto", "figure"), Input("contexto", "value"))
def _contexto(coluna):
    return g.empilhado_grupo(coluna, d.carregar().dropna(subset=["grupo"]))


@app.callback(Output("fig-num-hist", "figure"), Output("fig-num-box", "figure"), Output("nota-numerica", "children"),
              Input("numerica", "value"))
def _numerica(coluna):
    return g.numerica_hist(coluna), g.numerica_box(coluna), nota(NOTAS_NUM[coluna])


@app.callback(Output("fig-cat-freq", "figure"), Output("fig-cat-grupo", "figure"), Input("categorica", "value"))
def _categorica(coluna):
    return g.categorica_freq(coluna), g.empilhado_grupo(coluna, d.carregar().dropna(subset=["grupo"]),
                                                       "Grupo de BI-RADS em cada categoria (mamas)")


if __name__ == "__main__":
    import os
    app.run(debug=os.environ.get("DASH_DEBUG") == "1")
