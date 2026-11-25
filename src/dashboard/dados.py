"""Carrega o dataset de ultrassonografia e calcula as tabelas usadas no dashboard.

As definições seguem o notebook notebooks/01_exploracao_bi_rads.ipynb.
"""

from functools import lru_cache

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

from config import RAW_DATA

PASTA = RAW_DATA

BIRADS = [1, 2, 3, 4, 5]
GRUPOS = ["1–2", "3", "4–5"]

NUMERICAS = ["age_years", "age_at_menarche_years", "age_at_menopause_years", "parity"]
CLINICAS = ["ethnicity", "alcohol_use", "smoking_status", "obesity", "oral_contraceptive_use",
            "hormone_replacement_therapy", "breastfeeding_history", "family_history_breast_cancer",
            "other_cancer_family_history"]
LESAO = ["lesion_shape_ultrasound", "lesion_orientation_ultrasound", "lesion_margin_ultrasound",
         "lesion_echogenicity_ultrasound", "acoustic_enhancement_ultrasound",
         "posterior_shadowing_ultrasound", "tissue_distortion_ultrasound",
         "lesion_calcification_ultrasound", "vascularization_ultrasound"]
CONTEXTO = ["breast_density_ultrasound", "asymmetry_ultrasound", "lymph_nodes_ultrasound",
            "duct_ectasia_ultrasound", "dense_content_cyst_ultrasound"]
IDS = ["sample_id", "patient_id", "side", "exame"]

NOMES = {
    "age_years": "idade",
    "age_at_menarche_years": "idade da menarca",
    "age_at_menopause_years": "idade da menopausa",
    "parity": "paridade",
    "ethnicity": "etnia",
    "alcohol_use": "álcool",
    "smoking_status": "tabagismo",
    "obesity": "obesidade",
    "oral_contraceptive_use": "anticoncepcional oral",
    "hormone_replacement_therapy": "reposição hormonal",
    "breastfeeding_history": "amamentação",
    "family_history_breast_cancer": "hist. familiar de câncer de mama",
    "other_cancer_family_history": "hist. familiar de outros cânceres",
    "lesion_shape_ultrasound": "forma da lesão",
    "lesion_orientation_ultrasound": "orientação da lesão",
    "lesion_margin_ultrasound": "margem da lesão",
    "lesion_echogenicity_ultrasound": "ecogenicidade",
    "acoustic_enhancement_ultrasound": "reforço acústico",
    "posterior_shadowing_ultrasound": "sombra posterior",
    "tissue_distortion_ultrasound": "distorção tecidual",
    "lesion_calcification_ultrasound": "calcificação",
    "vascularization_ultrasound": "vascularização",
    "breast_density_ultrasound": "densidade mamária",
    "asymmetry_ultrasound": "assimetria",
    "lymph_nodes_ultrasound": "linfonodos",
    "duct_ectasia_ultrasound": "ectasia ductal",
    "dense_content_cyst_ultrasound": "cisto de conteúdo denso",
    "birads_ultrasound": "BI-RADS",
    "patient_cancer_label": "rótulo de câncer",
}

TIPO = {**{c: "clínica" for c in CLINICAS + NUMERICAS},
        **{c: "lesão" for c in LESAO},
        **{c: "contexto" for c in CONTEXTO}}


def nome(coluna):
    return NOMES.get(coluna, coluna)


def agrupa(b):
    if pd.isna(b):
        return np.nan
    return "1–2" if b <= 2 else ("3" if b == 3 else "4–5")


@lru_cache(maxsize=1)
def carregar():
    """Uma linha por mama, com as colunas derivadas do notebook."""
    df = pd.read_csv(PASTA / "Dataset_ultrassonografia.csv")
    df["exame"] = df["sample_id"].str.split("_").str[0]
    df["grupo"] = df["birads_ultrasound"].map(agrupa)
    df["n_descritores"] = df[LESAO].notna().sum(axis=1)
    df["tem_lesao"] = np.where(df["n_descritores"] > 0, "com lesão", "sem lesão")
    return df


@lru_cache(maxsize=1)
def exames():
    """Uma linha por exame (as variáveis clínicas são iguais nas duas mamas)."""
    return carregar().drop_duplicates("exame").set_index("exame")


def resumo():
    df = carregar()
    categoricas = CLINICAS + LESAO + CONTEXTO
    return {
        "mamas": len(df),
        "exames": df["exame"].nunique(),
        "pacientes": df["patient_id"].nunique(),
        "preditoras": len(CLINICAS + NUMERICAS + LESAO + CONTEXTO),
        "colunas_one_hot": len(NUMERICAS) + sum(df[c].nunique() + int(df[c].isna().any()) for c in categoricas),
        "categorias_raras": int(sum((df[c].fillna("vazio").value_counts() < 10).sum() for c in categoricas)),
        "birads_45": int((df["birads_ultrasound"] >= 4).sum()),
        "sem_lesao": int((df["tem_lesao"] == "sem lesão").sum()),
    }


def tabela_grupos():
    df = carregar()
    por_mama = df["grupo"].value_counts().reindex(GRUPOS)
    pior = df.groupby("exame")["birads_ultrasound"].max().map(agrupa)
    por_exame = pior.value_counts().reindex(GRUPOS)
    pacientes = df.dropna(subset=["grupo"]).groupby("grupo")["patient_id"].nunique().reindex(GRUPOS)
    return pd.DataFrame({
        "grupo": GRUPOS,
        "mamas": por_mama.values,
        "exames (pior mama)": por_exame.values,
        "pacientes": pacientes.values,
        "mamas por fold (10 folds)": (por_mama / 10).round(1).values,
    })


def lados():
    """Crosstab do BI-RADS da mama direita (linhas) contra a esquerda (colunas)."""
    df = carregar()
    p = df.pivot_table(index="exame", columns="side", values="birads_ultrasound", aggfunc="first")
    t = pd.crosstab(p["R"], p["L"]).reindex(index=BIRADS, columns=BIRADS, fill_value=0)
    validos = p.dropna()
    iguais = (validos["R"] == validos["L"]).mean()
    return t, iguais


def repetidas():
    df = carregar()
    rep = df["patient_id"].value_counts().loc[lambda s: s > 2].index
    conteudo = [c for c in df.columns if c not in IDS]
    linhas = []
    for p in rep:
        sub = df[df["patient_id"] == p]
        grupos = [g[conteudo].reset_index(drop=True) for _, g in sub.groupby("exame")]
        a, b = grupos
        diferentes = [nome(c) for c in conteudo
                      if c not in ("grupo", "n_descritores", "tem_lesao")
                      and not a[c].equals(b[c])]
        linhas.append({
            "exames": " × ".join(sorted(sub["exame"].unique())),
            "situação": "idênticos" if not diferentes else "diferentes",
            "campos que mudam": ", ".join(diferentes) if diferentes else "nenhum",
        })
    return pd.DataFrame(linhas).sort_values("exames")


def rotulo_cancer():
    df = carregar()
    r = df.groupby("exame").agg(pior=("birads_ultrasound", "max"), cancer=("patient_cancer_label", "max"))
    r = r.dropna(subset=["pior"])
    t = pd.crosstab(r["pior"].astype(int), r["cancer"].fillna(0).astype(int).map({0: "vazio", 1: "1 (câncer)"}))
    return t.reindex(index=BIRADS, fill_value=0).reindex(columns=["1 (câncer)", "vazio"], fill_value=0)


def faltantes():
    df = carregar()
    colunas = [c for c in df.columns if c not in IDS + ["grupo", "n_descritores", "tem_lesao"]]
    f = df[colunas].isna().mean()
    return f[f > 0].sort_values()


def ordem_mapa_vazios():
    """Matriz de vazios (colunas × mamas), mamas ordenadas por nº de descritores e BI-RADS."""
    df = carregar()
    colunas = [c for c in df.columns if c not in IDS + ["grupo", "n_descritores", "tem_lesao"]]
    ordem = df.sort_values(["n_descritores", "birads_ultrasound"]).index
    return df.loc[ordem, colunas].notna().astype(int).T, df.loc[ordem]


def cramer_v(x, y):
    t = pd.crosstab(x, y)
    if t.shape[0] < 2 or t.shape[1] < 2:
        return np.nan
    chi2 = chi2_contingency(t, correction=False)[0]
    return float(np.sqrt(chi2 / (t.values.sum() * (min(t.shape) - 1))))


@lru_cache(maxsize=1)
def associacao():
    """V de Cramér de cada preditora com o grupo (todas as mamas) e com 1 vs 2."""
    tabela = carregar().dropna(subset=["grupo"]).copy()
    for c in NUMERICAS:
        tabela[c] = pd.qcut(tabela[c], 4, duplicates="drop").astype(str)
    tabela = tabela.fillna({c: "vazio" for c in TIPO})
    um_dois = tabela[tabela["birads_ultrasound"] <= 2]
    linhas = [{"variável": c, "nome": nome(c), "tipo": TIPO[c],
               "grupos": cramer_v(tabela[c], tabela["grupo"]),
               "1 vs 2": cramer_v(um_dois[c], um_dois["birads_ultrasound"])}
              for c in TIPO]
    return pd.DataFrame(linhas)


def inconsistencias():
    e = exames()
    checagens = {
        "idade < 18 anos": e["age_years"] < 18,
        "menopausa antes dos 40 anos": e["age_at_menopause_years"] < 40,
        "menopausa antes dos 30 anos": e["age_at_menopause_years"] < 30,
        "55 anos ou mais sem idade da menopausa": (e["age_years"] >= 55) & e["age_at_menopause_years"].isna(),
        "nulípara que amamentou": (e["parity"] == 0) & (e["breastfeeding_history"] == "Yes"),
        "paridade ≥ 8": e["parity"] >= 8,
        "menarca ≤ 9 anos": e["age_at_menarche_years"] <= 9,
        "reposição hormonal sem menopausa registrada": (e["hormone_replacement_therapy"] == "Yes")
                                                      & e["age_at_menopause_years"].isna(),
    }
    return pd.DataFrame([{"checagem": k, "exames": int(v.sum()), "códigos": ", ".join(v[v].index)}
                         for k, v in checagens.items()])
