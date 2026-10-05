# Ficha: 5.4.4 Concordância SHAP × LIME (M4)

Fonte única dos números: `00_controle/FATOS_CONGELADOS.csv`. Caminhos relativos a `results/v2/` (cópias em `para_escrita/arquivos/`).

| id | fato | valor | IC 95% | origem |
|---|---|---|---|---|
| M4X-01 | M4: Pearson SHAP × LIME (limpo) | RF 0,997; GB 0,996; SVR 0,993 |  | results/v2/m4/xai/xai_m4_concordancia_shap_lime.csv |
| M4X-02 | M4: Spearman por série SHAP × LIME (limpo) | RF 0,653; GB 0,702; SVR 0,824 | RF [0,553; 0,746]; GB [0,621; 0,780]; SVR [0,748; 0,881] | results/v2/m4/xai/xai_m4_concordancia_shap_lime.csv |
| M4X-03 | M4: Spearman global SHAP × LIME (limpo) | RF 0,954; GB 0,907; SVR 0,946 |  | results/v2/m4/xai/xai_m4_concordancia_shap_lime.csv |
| M4X-09 | M4: sensibilidade do LIME (200, 1.000, 5.000) | Pearson SHAP × LIME de 0,986 a 0,999 |  | results/v2/m4/xai/xai_m4_sensibilidade_lime.csv |
| DEF-12 | Concordância SHAP × LIME | Pearson entre importâncias globais (cada vetor ÷ máximo); Spearman médio por série; condição limpa |  | results/v2/PREVISOES_tourism.md |

**Arquivos**

- `m4/xai/xai_m4_concordancia_shap_lime.csv`
- `m4/xai/xai_m4_sensibilidade_lime.csv`
- `fase5/figuras/fig20_v2_m4_shap_x_lime.png`
- `fase5/figuras/fig20_v2_m4_shap_x_lime_dados.csv`
