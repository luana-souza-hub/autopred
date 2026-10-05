# Ficha: Resumo e Abstract

Fonte única dos números: `00_controle/FATOS_CONGELADOS.csv`. Caminhos relativos a `results/v2/` (cópias em `para_escrita/arquivos/`).

| id | fato | valor | IC 95% | origem |
|---|---|---|---|---|
| BAS-01 | M4 mensal: séries no arquivo e amostra | 48.000; amostra de 200 (semente 42, ordem do arquivo) |  | results/v2/pacote_escrita/02_caracterizacao_bases/caracterizacao_bases.csv |
| BAS-03 | Car Parts: séries, comprimento e período | 2.674 séries; 51 obs.; jan/1998 a mar/2002 |  | results/v2/pacote_escrita/02_caracterizacao_bases/caracterizacao_bases.csv; data/raw (.tsf) |
| BAS-04 | Car Parts: zeros | mediana 80,4%; 100,0% das séries com algum zero |  | results/v2/pacote_escrita/02_caracterizacao_bases/caracterizacao_bases.csv |
| BAS-05 | Tourism: séries, comprimento e período | 366 séries (200 oficiais); 91 / 332 / 333 obs. (200) e 91 / 330 / 333 (366); início de jan/1979 a jan/2000; fim mais recente set/2007 |  | results/v2/pacote_escrita/02_caracterizacao_bases/caracterizacao_bases.csv; data/raw (.tsf) |
| BAS-10 | Papel de cada base | M4: desenvolvimento (origem dos achados, hipóteses e limiares); Car Parts: exploratória; Tourism 200: replicação com previsões pré-registradas; Tourism 366: sensibilidade |  | Tabela 8 (4.7) do .docx; results/v2/HANDOFF.md |
| PRT-17 | Cenários adversos | 4 tipos, 9 cenários: ruído 10% e 30%; ausências 10%, 20% e 30%; outliers 2% e 5% (média ± 3σ do treino); truncamento 50% e 25% (mínimo 24 obs.) |  | src/pipeline_v2.py (CENARIOS, OUTLIER_K) |
| PRT-13 | Pontos explicados por série (XAI) | até 29 linhas de treino mais recentes + ponto de previsão (30); Car Parts 22 |  | src/pipeline_v2.py (pontos_explicados) |
| HIP-04 | Tourism (oficial, 200 séries) | H1 confirmou parcialmente (5 de 6); H2 confirmou parcialmente (6 de 10); H3 confirmou (6 de 6); H4 confirmou parcialmente |  | results/v2/tm/replicacao_previsoes_tm.csv |
| CRZ-01 | Spearman dos rankings, MASE | M4 × Car Parts −0,200; Tourism × M4 +0,188; Tourism × Car Parts −0,770 | M4 × CP [−0,685; +0,248]; TM × M4 [−0,176; +0,636]; TM × CP [−0,867; −0,661] | notebooks/11a_desempenho_tourism_v2.ipynb (saída); results/v2/fase5/replicacao_h4_pares.csv |
| CRZ-06 | Queda do Holt-Winters M4 → Car Parts (mesma métrica) | MAE 3º → 10º; MASE 2º → 10º; ICs separados com e sem fallback | MAE [1; 7] → [10; 10]; MASE [1; 6] → [10; 10] | results/v2/cp/limpo/reversao_hierarquia_m4_cp_mesma_metrica.csv |
| M4L-01 | M4 limpo: Holt-Winters | MAPE 8,36%; posto MAPE 1º; MAE 325,29; R² −1,22; MASE 0,918; posto MASE 2º [1; 6] | MAPE [6,19; 10,25]; posto [1; 3] | results/v2/fase5/tabela3_antigo_v2_m4.csv |
| M4A-05 | M4: escore de robustez, ajustados | SES 3,0 < RF 3,6 < SVR 4,1 < GB 8,5 < HW 10,5 < SARIMA heur. 12,9 | sem IC | results/v2/m4/adversos/m4_adversos_hierarquia_robustez.csv |
| TMA-02 | Tourism: escore de robustez, ajustados | SES 1,5 < RF 3,4 < SVR 6,1 < SARIMA heur. 6,2 < GB 6,3 < HW 13,7 | sem IC | results/v2/tm/adversos/tm_adversos_hierarquia_robustez.csv |
| TMA-09 | Spearman do escore de robustez M4 × Tourism | +0,829 (6 ajustados) | sem IC | notebooks/11b_cenarios_adversos_tourism_v2.ipynb (saída) |
| M4X-01 | M4: Pearson SHAP × LIME (limpo) | RF 0,997; GB 0,996; SVR 0,993 |  | results/v2/m4/xai/xai_m4_concordancia_shap_lime.csv |
| M4X-02 | M4: Spearman por série SHAP × LIME (limpo) | RF 0,653; GB 0,702; SVR 0,824 | RF [0,553; 0,746]; GB [0,621; 0,780]; SVR [0,748; 0,881] | results/v2/m4/xai/xai_m4_concordancia_shap_lime.csv |
| M4X-04 | M4: estabilidade SHAP (média; mediana) | RF 0,561; GB 0,436; SVR 0,616 (medianas 0,584 / 0,405 / 0,630) | RF [0,480; 0,632]; GB [0,372; 0,500]; SVR [0,528; 0,695] | results/v2/m4/xai/xai_m4_estabilidade.csv |
| TMX-05 | Tourism: lag_12 | 1ª característica global nos 6 casos; no top-3 de 100% das séries, exceto SVR LIME (93,3%, 28 de 30) |  | results/v2/tm/xai/xai_tm_posicao_lag12.csv |
| BAS-07 | Séries com F_S acima do p95 do piso de ruído | M4 56,5%; Car Parts 18,0%; Tourism 98,5%; Tourism 366 99,2% |  | results/v2/pacote_escrita/02_caracterizacao_bases/caracterizacao_bases.csv |
| DEF-15 | Terminologia de hierarquia | "inversão de hierarquia" só para a queda do Holt-Winters e para o par Tourism × Car Parts; no restante, "dependência do regime" |  | results/v2/HANDOFF.md (Redação) |

**Arquivos**

- `fase5/replicacao_H1_H4_resumo.csv`
