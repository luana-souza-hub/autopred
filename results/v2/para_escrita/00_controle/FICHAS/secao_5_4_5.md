# Ficha: 5.4.5 Questão 3: estabilidade sob ruído (M4)

Fonte única dos números: `00_controle/FATOS_CONGELADOS.csv`. Caminhos relativos a `results/v2/` (cópias em `para_escrita/arquivos/`).

| id | fato | valor | IC 95% | origem |
|---|---|---|---|---|
| M4X-04 | M4: estabilidade SHAP (média; mediana) | RF 0,561; GB 0,436; SVR 0,616 (medianas 0,584 / 0,405 / 0,630) | RF [0,480; 0,632]; GB [0,372; 0,500]; SVR [0,528; 0,695] | results/v2/m4/xai/xai_m4_estabilidade.csv |
| M4X-05 | M4: estabilidade LIME (média) | RF 0,485; GB 0,442; SVR 0,572 | RF [0,399; 0,566]; GB [0,379; 0,510]; SVR [0,483; 0,663] | results/v2/m4/xai/xai_m4_estabilidade.csv |
| M4X-07 | M4: redistribuição sob ruído (SHAP, Δ normalizado) | RF lag_1 −0,188 e roll_mean_6 +0,102; GB lag_1 −0,160 e roll_mean_6 +0,047; SVR lag_1 −0,054 e roll_mean_6 −0,001 | RF [−0,266; −0,110] e [+0,057; +0,152]; GB [−0,225; −0,094] e [+0,018; +0,076]; SVR [−0,077; −0,031] e [−0,006; +0,004] | results/v2/m4/xai/xai_m4_redistribuicao_ruido.csv |
| PRT-16 | Ruído da análise de estabilidade (XAI) | σ = 30% do desvio padrão do treino, só no treino |  | src/pipeline_v2.py (treino_xai) |
| DEF-13 | Estabilidade das explicações | Spearman por série entre limpo e ruído 30%; Jaccard top-3 e top-5; média entre séries |  | src/pipeline_v2.py (estabilidade) |
| LIM-04 | Leitura pós-hoc da redistribuição | "a característica dominante perde importância e as médias móveis ganham": formulada depois do Tourism, não registrada, não altera a H2 |  | results/v2/CHANGELOG_conclusoes.md (4c) |

**Arquivos**

- `m4/xai/xai_m4_estabilidade.csv`
- `m4/xai/xai_m4_redistribuicao_ruido.csv`
- `m4/xai/xai_m4_comparacao_fig22.csv`
- `fase5/figuras/fig21_v2_m4_estabilidade.png`
- `fase5/figuras/fig21_v2_m4_estabilidade_dados.csv`
- `fase5/figuras/fig22_v2_m4_shap_limpo_x_ruido.png`
- `fase5/figuras/fig22_v2_m4_shap_limpo_x_ruido_dados.csv`
