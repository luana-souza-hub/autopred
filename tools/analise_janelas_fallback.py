"""Verificacao post hoc (fora da analise registrada): mediana de MAPE e MAE dos 10 modelos
usando todas as janelas e apenas as janelas 2 e 3, nas quais o SARIMA heuristico nao
recorre ao fallback (na janela 1, com 24 observacoes, ele recorre em todas as series).

Uso (a partir da raiz do repositorio):
    python tools/analise_janelas_fallback.py
Le results/v2/{m4,tm}/limpo/*_by_window.csv e grava
results/v2/analise_janelas_fallback.csv.
Agregacao igual a do artigo: mediana das janelas dentro da serie, depois mediana entre series.
"""
import glob
import pandas as pd

linhas = []
for base, pre in (("m4", "m4"), ("tm", "tm")):
    arquivos = sorted(glob.glob(f"results/v2/{base}/limpo/{pre}_limpo_*_by_window.csv"))
    D = pd.concat([pd.read_csv(f) for f in arquivos])
    for nome, sub in (("todas_as_janelas", D), ("janelas_2_e_3", D[D.janela >= 2])):
        for metrica in ("mape", "mae"):
            med = sub.groupby(["model", "unique_id"])[metrica].median().groupby("model").median()
            rank = med.rank(method="min").astype(int)
            for modelo in med.index:
                linhas.append(dict(base=base, janelas=nome, metrica=metrica, modelo=modelo,
                                   mediana=round(float(med[modelo]), 4), rank=int(rank[modelo])))
pd.DataFrame(linhas).to_csv("results/v2/analise_janelas_fallback.csv", index=False)
print("ok", len(linhas), "linhas")
