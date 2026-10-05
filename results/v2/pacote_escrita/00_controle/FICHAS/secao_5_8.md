# Ficha: 5.8 Hipóteses H1 a H4 nas três bases

Fonte única dos números: `00_controle/FATOS_CONGELADOS.csv`. Caminhos relativos a `results/v2/` (cópias em `para_escrita/arquivos/`).

| id | fato | valor | IC 95% | origem |
|---|---|---|---|---|
| HIP-01 | H1: condições e rótulos | Pearson agregado > 0,90 e Spearman por série em [0,65; 0,85], por modelo: 6 condições; 6 confirmou; 3 a 5 parcial; 0 a 2 não confirmou |  | results/v2/PREVISOES_tourism.md |
| HIP-02 | H2: condições e rótulos | (a) estabilidade < 0,80 por modelo e método; (b) em RF e GB, lag_1 perde e roll_mean_3 ou roll_mean_6 ganha, ICs excluindo zero: 10 condições (6 de (a), 4 de (b)); 10 confirmou; 5 a 9 parcial; 0 a 4 não confirmou |  | results/v2/PREVISOES_tourism.md |
| HIP-03 | H3: condições e rótulos | 6 condições, fração > 30,0%; 6 confirmou; 1 a 5 parcial; 0 não confirmou |  | results/v2/PREVISOES_tourism.md |
| HIP-04 | Tourism (oficial, 200 séries) | H1 confirmou parcialmente (5 de 6); H2 confirmou parcialmente (6 de 10); H3 confirmou (6 de 6); H4 confirmou parcialmente |  | results/v2/tm/replicacao_previsoes_tm.csv |
| HIP-05 | Tourism 366 (sensibilidade) | H1 5 de 6; H2 8 de 10; H3 6 de 6; H4 ordem idêntica; mesmos rótulos |  | results/v2/fase5/replicacao_sensibilidade_tm366.csv |
| HIP-06 | M4 (critérios a posteriori) | H1 6 de 6; H2 10 de 10; H3 0 de 6 (circular); H4 par M4 × Car Parts: ordem difere, nenhuma família com IC separado no MASE (no MAE, Baseline separado) |  | results/v2/fase5/replicacao_H1_H4_resumo.csv |
| HIP-07 | Car Parts (exploratória, critérios a posteriori) | H1 4 de 6; H2 6 de 10; H3 0 de 6 |  | results/v2/fase5/replicacao_H1_H4_resumo.csv |
| LIM-03 | H1 a H4 no M4 e no Car Parts | critérios aplicados depois de conhecidos os resultados: descritivo |  | results/v2/fase5/replicacao_H1_H4.csv |
| LIM-05 | H3 no M4 é circular | limiar de 30,0% = máximo observado no M4 (16,7% a 30,0%): 0 de 6 por construção |  | results/v2/m4/xai/xai_m4_posicao_lag12.csv |
| REG-01 | Registro das previsões do Tourism | 2 de outubro de 2026, commit 1a08fa1 |  | results/v2/PREVISOES_tourism.md |
| REG-02 | Interpretação da faixa da H1 | 3 de outubro de 2026, commit 1e3d9f0 (faixa fechada [0,65; 0,85]) |  | results/v2/INTERPRETACAO_criterios_4c.md |
| M4X-01 | M4: Pearson SHAP × LIME (limpo) | RF 0,997; GB 0,996; SVR 0,993 |  | results/v2/m4/xai/xai_m4_concordancia_shap_lime.csv |
| M4X-02 | M4: Spearman por série SHAP × LIME (limpo) | RF 0,653; GB 0,702; SVR 0,824 | RF [0,553; 0,746]; GB [0,621; 0,780]; SVR [0,748; 0,881] | results/v2/m4/xai/xai_m4_concordancia_shap_lime.csv |
| M4X-04 | M4: estabilidade SHAP (média; mediana) | RF 0,561; GB 0,436; SVR 0,616 (medianas 0,584 / 0,405 / 0,630) | RF [0,480; 0,632]; GB [0,372; 0,500]; SVR [0,528; 0,695] | results/v2/m4/xai/xai_m4_estabilidade.csv |
| M4X-06 | M4: lag_12 (limpo) | posição global SHAP 3º / 3º / 5º e LIME 3º / 3º / 4º (RF/GB/SVR); no top-3 de 16,7% / 26,7% / 20,0% (SHAP) e 16,7% / 26,7% / 30,0% (LIME) das séries |  | results/v2/m4/xai/xai_m4_posicao_lag12.csv |
| M4X-07 | M4: redistribuição sob ruído (SHAP, Δ normalizado) | RF lag_1 −0,188 e roll_mean_6 +0,102; GB lag_1 −0,160 e roll_mean_6 +0,047; SVR lag_1 −0,054 e roll_mean_6 −0,001 | RF [−0,266; −0,110] e [+0,057; +0,152]; GB [−0,225; −0,094] e [+0,018; +0,076]; SVR [−0,077; −0,031] e [−0,006; +0,004] | results/v2/m4/xai/xai_m4_redistribuicao_ruido.csv |
| CPX-01 | Car Parts: concordância SHAP × LIME (exploratória) | Pearson RF 0,911, GB 0,708, SVR 0,791; Spearman por série 0,842 / 0,792 / 0,658 | Spearman RF [0,783; 0,890]; GB [0,737; 0,843]; SVR [0,530; 0,764] | results/v2/cp/xai/xai_cp_concordancia_shap_lime.csv |
| CPX-02 | Car Parts: estabilidade SHAP (exploratória) | RF 0,173; GB 0,204; SVR 0,469 (29 séries no SVR); Jaccard top-3 de 0,220 a 0,376 | RF [0,061; 0,271]; GB [0,104; 0,296]; SVR [0,377; 0,558] | results/v2/cp/xai/xai_cp_estabilidade.csv |
| CPX-03 | Car Parts: lag_12 | posição global 7º a 9º; no top-3 de 13,3% a 23,3% das séries |  | results/v2/cp/xai/xai_cp_posicao_lag12.csv |
| CPX-04 | Car Parts: roll_mean_6 sob ruído | perde: RF SHAP −0,100; GB SHAP −0,060 | RF [−0,170; −0,041]; GB [−0,102; −0,019] | results/v2/cp/xai/xai_cp_redistribuicao_ruido.csv |
| TMX-01 | Tourism: concordância SHAP × LIME (limpo) | Pearson RF 1,000, GB 1,000, SVR 0,997; Spearman por série 0,553 / 0,686 / 0,839 | RF [0,451; 0,648]; GB [0,593; 0,763]; SVR [0,806; 0,869] | results/v2/tm/xai/xai_tm_concordancia_shap_lime.csv |
| TMX-03 | Tourism: estabilidade (média) | SHAP RF 0,679, GB 0,625, SVR 0,753; LIME 0,612 / 0,592 / 0,727 | SHAP RF [0,615; 0,739]; GB [0,551; 0,690]; SVR [0,691; 0,810]; LIME RF [0,548; 0,670]; GB [0,519; 0,662]; SVR [0,672; 0,778] | results/v2/tm/xai/xai_tm_estabilidade.csv |
| TMX-04 | Tourism: H2(b), Δlag_1 e Δroll_mean_6 | lag_1: RF SHAP −0,016, RF LIME −0,004, GB SHAP −0,007, GB LIME +0,003 (ICs incluem zero); roll_mean_6: +0,066 / +0,046 / +0,028 / +0,024 (ICs excluem zero) | lag_1 RF SHAP [−0,059; +0,019]; roll_mean_6 RF SHAP [+0,023; +0,117] | results/v2/tm/xai/xai_tm_avaliacao_previsoes_condicoes.csv |
| TMX-05 | Tourism: lag_12 | 1ª característica global nos 6 casos; no top-3 de 100% das séries, exceto SVR LIME (93,3%, 28 de 30) |  | results/v2/tm/xai/xai_tm_posicao_lag12.csv |
| T366-04 | Condições que mudam com 366 | H1 RF Spearman por série 0,571 (continua fora); H2 GB lag_1 passa a perder: SHAP −0,026, LIME −0,013 | H1 [0,494; 0,649]; SHAP [−0,055; −0,004]; LIME [−0,027; −0,000] | results/v2/tm_366/sensibilidade/condicoes_h1_h3_200_366.csv |
| CRZ-01 | Spearman dos rankings, MASE | M4 × Car Parts −0,200; Tourism × M4 +0,188; Tourism × Car Parts −0,770 | M4 × CP [−0,685; +0,248]; TM × M4 [−0,176; +0,636]; TM × CP [−0,867; −0,661] | notebooks/11a_desempenho_tourism_v2.ipynb (saída); results/v2/fase5/replicacao_h4_pares.csv |
| CRZ-02 | Spearman dos rankings, MAE | M4 × Car Parts +0,067; Tourism × M4 −0,091; Tourism × Car Parts −0,602 | M4 × CP [−0,588; +0,273]; TM × M4 [−0,297; +0,624]; TM × CP [−0,673; −0,248] | notebooks/11a_desempenho_tourism_v2.ipynb (saída); results/v2/fase5/replicacao_h4_pares.csv |
| CRZ-07 | Ordem das famílias por MASE nas três bases | M4: ML > Suav. > Baseline > ARIMA; Car Parts: Baseline > ML > Suav. > ARIMA; Tourism: Suav. > ARIMA > Baseline > ML. IC separado: Tourism × Car Parts em ARIMA e ML; Tourism × M4 nenhuma; M4 × Car Parts nenhuma (no MAE, Baseline) |  | results/v2/fase5/replicacao_h4_pares.csv |
| BAS-07 | Séries com F_S acima do p95 do piso de ruído | M4 56,5%; Car Parts 18,0%; Tourism 98,5%; Tourism 366 99,2% |  | results/v2/pacote_escrita/02_caracterizacao_bases/caracterizacao_bases.csv |
| DEF-15 | Terminologia de hierarquia | "inversão de hierarquia" só para a queda do Holt-Winters e para o par Tourism × Car Parts; no restante, "dependência do regime" |  | results/v2/HANDOFF.md (Redação) |

**Arquivos**

- `fase5/replicacao_H1_H4_resumo.csv`
- `fase5/replicacao_H1_H4.csv`
- `fase5/replicacao_condicoes_h1_h3.csv`
- `fase5/replicacao_h4_pares.csv`
- `fase5/replicacao_sensibilidade_tm366.csv`
- `tm/replicacao_previsoes_tm.csv`
- `PREVISOES_tourism.md`
- `INTERPRETACAO_criterios_4c.md`
