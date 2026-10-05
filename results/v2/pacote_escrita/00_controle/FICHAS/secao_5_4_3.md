# Ficha: 5.4.3 Questão 2: lag sazonal (M4)

Fonte única dos números: `00_controle/FATOS_CONGELADOS.csv`. Caminhos relativos a `results/v2/` (cópias em `para_escrita/arquivos/`).

| id | fato | valor | IC 95% | origem |
|---|---|---|---|---|
| M4X-06 | M4: lag_12 (limpo) | posição global SHAP 3º / 3º / 5º e LIME 3º / 3º / 4º (RF/GB/SVR); no top-3 de 16,7% / 26,7% / 20,0% (SHAP) e 16,7% / 26,7% / 30,0% (LIME) das séries |  | results/v2/m4/xai/xai_m4_posicao_lag12.csv |
| M4X-10 | M4: lag_24 (suplementar) | lag_24 em 5º a 7º; lag_12 + lag_24 = 11% a 15% da importância |  | results/v2/m4/xai/xai_m4_suplementar_lag24.csv |
| LIM-05 | H3 no M4 é circular | limiar de 30,0% = máximo observado no M4 (16,7% a 30,0%): 0 de 6 por construção |  | results/v2/m4/xai/xai_m4_posicao_lag12.csv |

**Arquivos**

- `m4/xai/xai_m4_posicao_lag12.csv`
- `m4/xai/xai_m4_suplementar_lag24.csv`
- `m4/xai/xai_m4_comparacao_dissertacao.csv`
- `fase5/figuras/fig19_v2_m4_top10_lime.png`
- `fase5/figuras/fig19_v2_m4_top10_lime_dados.csv`
