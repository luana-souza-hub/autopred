# Ficha: 5.4.1 Importância global por SHAP (M4)

Fonte única dos números: `00_controle/FATOS_CONGELADOS.csv`. Caminhos relativos a `results/v2/` (cópias em `para_escrita/arquivos/`).

| id | fato | valor | IC 95% | origem |
|---|---|---|---|---|
| M4X-06 | M4: lag_12 (limpo) | posição global SHAP 3º / 3º / 5º e LIME 3º / 3º / 4º (RF/GB/SVR); no top-3 de 16,7% / 26,7% / 20,0% (SHAP) e 16,7% / 26,7% / 30,0% (LIME) das séries |  | results/v2/m4/xai/xai_m4_posicao_lag12.csv |
| PRT-10 | Características dos modelos de ML | 15: lag_1 a lag_12, roll_mean_3, roll_std_3, roll_mean_6; sem lag_season (= lag_12) e sem lag_24 no oficial |  | src/pipeline_v2.py (feature_names) |
| PRT-15 | SHAP do SVR | KernelExplainer; fundo de 50 linhas; nsamples = 2p + 2048 (p = 15); semente fixa; escala original |  | src/pipeline_v2.py (explicar_pontos) |
| DEF-14 | Normalização das importâncias | soma 1 em cada série antes da média entre séries |  | src/pipeline_v2.py (normalizar_por_serie) |

**Arquivos**

- `m4/xai/xai_m4_top10_global.csv`
- `m4/xai/xai_m4_posicao_lag12.csv`
- `fase5/figuras/fig17_v2_m4_top10_shap.png`
- `fase5/figuras/fig17_v2_m4_top10_shap_dados.csv`
