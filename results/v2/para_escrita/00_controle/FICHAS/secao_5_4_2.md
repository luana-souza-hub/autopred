# Ficha: 5.4.2 Questão 1: consistência entre séries (M4)

Fonte única dos números: `00_controle/FATOS_CONGELADOS.csv`. Caminhos relativos a `results/v2/` (cópias em `para_escrita/arquivos/`).

| id | fato | valor | IC 95% | origem |
|---|---|---|---|---|
| M4X-08 | M4: Q1, CV entre séries (reanálise, SHAP) | normalizado 1,211 / 1,015 / 0,700; escala original 1,702 / 1,493 / 1,167 (RF/GB/SVR) | normalizado RF [0,955; 1,298]; GB [0,849; 1,069]; SVR [0,621; 0,748] | results/v2/fase5/reanalise_cv_q1_m4.csv |
| DEF-14 | Normalização das importâncias | soma 1 em cada série antes da média entre séries |  | src/pipeline_v2.py (normalizar_por_serie) |

**Arquivos**

- `fase5/reanalise_cv_q1_m4.csv`
- `fase5/figuras/fig18_v2_m4_variabilidade_shap.png`
- `fase5/figuras/fig18_v2_m4_variabilidade_shap_dados.csv`
