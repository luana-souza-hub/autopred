# Ficha: 5.7 Tourism Monthly (replicação com previsões pré-registradas)

Fonte única dos números: `00_controle/FATOS_CONGELADOS.csv`. Caminhos relativos a `results/v2/` (cópias em `para_escrita/arquivos/`).

| id | fato | valor | IC 95% | origem |
|---|---|---|---|---|
| BAS-05 | Tourism: séries, comprimento e período | 366 séries (200 oficiais); 91 / 332 / 333 obs. (200) e 91 / 330 / 333 (366); início de jan/1979 a jan/2000; fim mais recente set/2007 |  | results/v2/pacote_escrita/02_caracterizacao_bases/caracterizacao_bases.csv; data/raw (.tsf) |
| BAS-06 | Força sazonal F_S (mediana) | M4 0,506; Car Parts 0,585; Tourism 0,838; Tourism 366 0,859 |  | results/v2/pacote_escrita/02_caracterizacao_bases/caracterizacao_bases.csv |
| BAS-07 | Séries com F_S acima do p95 do piso de ruído | M4 56,5%; Car Parts 18,0%; Tourism 98,5%; Tourism 366 99,2% |  | results/v2/pacote_escrita/02_caracterizacao_bases/caracterizacao_bases.csv |
| PRT-05 | Horizonte H por base | M4 18; Car Parts 18; Tourism 24 |  | scripts/run_v2.py |
| PRT-08 | Janelas de avaliação, Tourism 200 | 600 |  | results/v2/tm/limpo/*_by_window.csv |
| PRT-09 | Janelas de avaliação, Tourism 366 (sensibilidade) | 1.098 |  | results/v2/tm_366/limpo/*_by_window.csv |
| REG-01 | Registro das previsões do Tourism | 2 de outubro de 2026, commit 1a08fa1 |  | results/v2/PREVISOES_tourism.md |
| TML-01 | Tourism limpo: Holt-Winters | MAPE 21,61% (1º); MAE 365,00 (4º); MASE 1,349 (1º) | ver ranking_ic_tm.csv | results/v2/tm/limpo/tabela_desempenho_v2_tm.csv |
| TML-02 | Tourism limpo: Sazonal ingênuo | MAPE 23,36% (2º); MAE 354,54 (3º); MASE 1,395 (3º) | ver ranking_ic_tm.csv | results/v2/tm/limpo/tabela_desempenho_v2_tm.csv |
| TML-03 | Tourism limpo: SARIMA automático | MAPE 23,41% (3º); MAE 329,13 (2º); MASE 1,462 (4º) | ver ranking_ic_tm.csv | results/v2/tm/limpo/tabela_desempenho_v2_tm.csv |
| TML-04 | Tourism limpo: SARIMA heurístico | MAPE 23,76% (4º); MAE 315,10 (1º); MASE 1,360 (2º) | ver ranking_ic_tm.csv | results/v2/tm/limpo/tabela_desempenho_v2_tm.csv |
| TML-05 | Tourism limpo: SVR | MAPE 25,55% (5º); MAE 392,29 (6º); MASE 1,618 (7º) | ver ranking_ic_tm.csv | results/v2/tm/limpo/tabela_desempenho_v2_tm.csv |
| TML-06 | Tourism limpo: Random Forest | MAPE 26,00% (6º); MAE 384,51 (5º); MASE 1,509 (5º) | ver ranking_ic_tm.csv | results/v2/tm/limpo/tabela_desempenho_v2_tm.csv |
| TML-07 | Tourism limpo: Gradient Boosting | MAPE 26,25% (7º); MAE 422,83 (7º); MASE 1,549 (6º) | ver ranking_ic_tm.csv | results/v2/tm/limpo/tabela_desempenho_v2_tm.csv |
| TML-08 | Tourism limpo: SES | MAPE 33,34% (8º); MAE 563,44 (8º); MASE 2,116 (8º) | ver ranking_ic_tm.csv | results/v2/tm/limpo/tabela_desempenho_v2_tm.csv |
| TML-09 | Tourism limpo: Persistência | MAPE 37,47% (9º); MAE 653,48 (10º); MASE 2,495 (9º) | ver ranking_ic_tm.csv | results/v2/tm/limpo/tabela_desempenho_v2_tm.csv |
| TML-10 | Tourism limpo: Média histórica | MAPE 44,21% (10º); MAE 634,61 (9º); MASE 2,952 (10º) | ver ranking_ic_tm.csv | results/v2/tm/limpo/tabela_desempenho_v2_tm.csv |
| TML-11 | Tourism: famílias por MASE | Suavização exponencial 1º; ARIMA 2º; Baseline 3º; ML 4º | Suav. [1; 3]; ARIMA [1; 3]; Baseline [1; 3]; ML [4; 4] | results/v2/tm/limpo/familias_tres_bases_mase_tm.csv |
| TML-12 | Tourism: SARIMA heurístico sem fallback | MAPE 22,58%; MAE 463,11; fallback 33,3% |  | results/v2/tm/limpo/tabela_desempenho_v2_tm.csv |
| TML-13 | Tourism: SARIMA automático sem explosivas | MAPE 23,16% (8 janelas explosivas, 1,3%) |  | results/v2/tm/limpo/tabela_desempenho_v2_tm.csv |
| TMA-01 | Tourism: ICs do Δ% que incluem zero | 53 de 81 |  | results/v2/tm/adversos/tm_adversos_delta_ic95.csv |
| TMA-02 | Tourism: escore de robustez, ajustados | SES 1,5 < RF 3,4 < SVR 6,1 < SARIMA heur. 6,2 < GB 6,3 < HW 13,7 | sem IC | results/v2/tm/adversos/tm_adversos_hierarquia_robustez.csv |
| TMA-03 | Tourism: escore de robustez, triviais | Persistência 1,0 < Média histórica 5,1 < Sazonal ingênuo 8,6 | sem IC | results/v2/tm/adversos/tm_adversos_hierarquia_robustez.csv |
| TMA-04 | Tourism: ausências (10/20/30%) | HW +4,7 / +15,0 / +17,8%; GB +1,9 / +7,2 / +13,6%; baselines ≈ 0 |  | results/v2/tm/adversos/tm_adversos_delta_pct.csv |
| TMA-05 | Tourism: outliers 5% | HW +28,2%; Sazonal ingênuo +25,2%; SARIMA heur. +19,4%; GB +14,5%; SVR +10,5%; RF +6,3%; SES −0,3% |  | results/v2/tm/adversos/tm_adversos_delta_pct.csv |
| TMA-06 | Tourism: outliers, média dos dois níveis (ML) | RF 4,3% < SVR 8,9% < GB 11,5% |  | results/v2/tm/adversos/tm_adversos_hierarquia_robustez.csv |
| TMA-07 | Tourism: truncamento (50/25%) | HW +6,3 / +9,1% (ICs incluem zero); média histórica −19,1 / −22,1% |  | results/v2/tm/adversos/tm_adversos_delta_pct.csv |
| TMA-08 | Tourism: SARIMA heurístico com e sem fallback (ausências) | com fallback 0,6 / 4,8 / 7,4%; sem fallback 6,3 / 12,6 / 10,8% |  | results/v2/tm/adversos/tm_adversos_sarima_variantes.csv |
| TMA-09 | Spearman do escore de robustez M4 × Tourism | +0,829 (6 ajustados) | sem IC | notebooks/11b_cenarios_adversos_tourism_v2.ipynb (saída) |
| TMA-10 | Sinal do Δ% M4 × Tourism | mesmo sinal em 54 de 81; das 29 combinações com IC excluindo zero em alguma base, 28 com mesmo sinal |  | results/v2/tm/adversos/comparacao_m4_tm_delta.csv |
| TMX-01 | Tourism: concordância SHAP × LIME (limpo) | Pearson RF 1,000, GB 1,000, SVR 0,997; Spearman por série 0,553 / 0,686 / 0,839 | RF [0,451; 0,648]; GB [0,593; 0,763]; SVR [0,806; 0,869] | results/v2/tm/xai/xai_tm_concordancia_shap_lime.csv |
| TMX-02 | Tourism: Spearman global SHAP × LIME (limpo) | RF 0,764; GB 0,743; SVR 0,964 |  | results/v2/tm/xai/xai_tm_concordancia_shap_lime.csv |
| TMX-03 | Tourism: estabilidade (média) | SHAP RF 0,679, GB 0,625, SVR 0,753; LIME 0,612 / 0,592 / 0,727 | SHAP RF [0,615; 0,739]; GB [0,551; 0,690]; SVR [0,691; 0,810]; LIME RF [0,548; 0,670]; GB [0,519; 0,662]; SVR [0,672; 0,778] | results/v2/tm/xai/xai_tm_estabilidade.csv |
| TMX-04 | Tourism: H2(b), Δlag_1 e Δroll_mean_6 | lag_1: RF SHAP −0,016, RF LIME −0,004, GB SHAP −0,007, GB LIME +0,003 (ICs incluem zero); roll_mean_6: +0,066 / +0,046 / +0,028 / +0,024 (ICs excluem zero) | lag_1 RF SHAP [−0,059; +0,019]; roll_mean_6 RF SHAP [+0,023; +0,117] | results/v2/tm/xai/xai_tm_avaliacao_previsoes_condicoes.csv |
| TMX-05 | Tourism: lag_12 | 1ª característica global nos 6 casos; no top-3 de 100% das séries, exceto SVR LIME (93,3%, 28 de 30) |  | results/v2/tm/xai/xai_tm_posicao_lag12.csv |
| TMX-06 | Tourism: lag_12 perde sob ruído (pós-hoc) | RF SHAP −0,132; GB SHAP −0,109 | RF [−0,188; −0,088]; GB [−0,149; −0,074] | results/v2/tm/xai/xai_tm_redistribuicao_ruido.csv |
| TMX-07 | Tourism: suplementares | LIME 1.000 e 5.000: Pearson 0,997 a 1,000; lag_24 em 2º em todos os casos; lag_12 + lag_24 = 25% a 59% |  | results/v2/tm/xai/xai_tm_sensibilidade_lime.csv; xai_tm_suplementar_lag24.csv |
| T366-01 | Spearman do ranking 200 × 366 | MASE 0,988; MAPE 0,939; MAE 0,939 | MASE [0,927; 1,000]; MAPE [0,842; 1,000]; MAE [0,818; 0,988] (aproximados: amostras não independentes) | results/v2/tm_366/sensibilidade/spearman_ranking_200_366.csv |
| T366-02 | Escore de robustez com 366 | SES 1,3 < RF 6,3 < SVR 6,4 < SARIMA heur. 7,3 < GB 7,7 < HW 16,4; Spearman 200 × 366 +1,000 (6 ajustados) e +0,967 (9); mesmo sinal em 67 de 81; Pearson dos Δ% 0,946 | sem IC | results/v2/tm_366/sensibilidade/robustez_200_366.csv |
| T366-03 | ICs do Δ% que incluem zero (366) | 42 de 81 |  | results/v2/tm_366/sensibilidade/delta_ic95_366.csv |
| T366-04 | Condições que mudam com 366 | H1 RF Spearman por série 0,571 (continua fora); H2 GB lag_1 passa a perder: SHAP −0,026, LIME −0,013 | H1 [0,494; 0,649]; SHAP [−0,055; −0,004]; LIME [−0,027; −0,000] | results/v2/tm_366/sensibilidade/condicoes_h1_h3_200_366.csv |
| T366-05 | SARIMA automático com 366 | 17 janelas explosivas (1,5%); MAPE 23,55% com e 23,25% sem explosivas |  | results/v2/tm_366/sensibilidade/status_200_366.csv |
| HIP-04 | Tourism (oficial, 200 séries) | H1 confirmou parcialmente (5 de 6); H2 confirmou parcialmente (6 de 10); H3 confirmou (6 de 6); H4 confirmou parcialmente |  | results/v2/tm/replicacao_previsoes_tm.csv |
| HIP-05 | Tourism 366 (sensibilidade) | H1 5 de 6; H2 8 de 10; H3 6 de 6; H4 ordem idêntica; mesmos rótulos |  | results/v2/fase5/replicacao_sensibilidade_tm366.csv |
| FBK-02 | Fallback do SARIMA heurístico, linha de base | M4 33,3% (200 de 600); Car Parts 67,1% (5.380 de 8.022); Tourism 33,3% (200 de 600); Tourism 366 33,3% (366 de 1.098) |  | results/v2/sarima_limites_fallback_v2.csv; tabela3_v2_m4.csv; tabela7_v2_cp.csv |
| FBK-06 | Janelas explosivas do SARIMA automático, linha de base | M4 14; Car Parts 246; Tourism 8; Tourism 366 17 |  | results/v2/sarima_limites_fallback_v2.csv |
| FBK-07 | Janelas explosivas do SARIMA heurístico | linha de base: M4 1; Car Parts 60; Tourism 0; Tourism 366 0. Cenários oficiais do M4: 4 (ausências 10% 1, outliers 2% 1, outliers 5% 2); Tourism e Tourism 366: 1 cada (truncamento 25%) |  | results/v2/sarima_limites_fallback_v2.csv; m4_adversos_status.csv |
| CRZ-01 | Spearman dos rankings, MASE | M4 × Car Parts −0,200; Tourism × M4 +0,188; Tourism × Car Parts −0,770 | M4 × CP [−0,685; +0,248]; TM × M4 [−0,176; +0,636]; TM × CP [−0,867; −0,661] | notebooks/11a_desempenho_tourism_v2.ipynb (saída); results/v2/fase5/replicacao_h4_pares.csv |
| CRZ-02 | Spearman dos rankings, MAE | M4 × Car Parts +0,067; Tourism × M4 −0,091; Tourism × Car Parts −0,602 | M4 × CP [−0,588; +0,273]; TM × M4 [−0,297; +0,624]; TM × CP [−0,673; −0,248] | notebooks/11a_desempenho_tourism_v2.ipynb (saída); results/v2/fase5/replicacao_h4_pares.csv |
| CRZ-03 | Spearman dos rankings, MAPE, Tourism × M4 | +0,382 | [+0,152; +0,745] | notebooks/11a_desempenho_tourism_v2.ipynb (saída); results/v2/fase5/replicacao_h4_pares.csv |
| CRZ-07 | Ordem das famílias por MASE nas três bases | M4: ML > Suav. > Baseline > ARIMA; Car Parts: Baseline > ML > Suav. > ARIMA; Tourism: Suav. > ARIMA > Baseline > ML. IC separado: Tourism × Car Parts em ARIMA e ML; Tourism × M4 nenhuma; M4 × Car Parts nenhuma (no MAE, Baseline) |  | results/v2/fase5/replicacao_h4_pares.csv |
| LIM-04 | Leitura pós-hoc da redistribuição | "a característica dominante perde importância e as médias móveis ganham": formulada depois do Tourism, não registrada, não altera a H2 |  | results/v2/CHANGELOG_conclusoes.md (4c) |
| CUS-02 | Tempo de parede da linha de base (10 modelos) | M4 44,2 min (42,1 do SARIMA automático); Car Parts 166,1 min (158,0); Tourism 34,2 min (33,2) |  | results/v2/fase5/tabela6_antigo_v2.csv |

**Arquivos**

- `tm/limpo/tabela_desempenho_v2_tm.csv`
- `tm/limpo/ranking_ic_tm.csv`
- `tm/limpo/ranks_tres_bases_tm.csv`
- `tm/limpo/familias_tres_bases_mase_tm.csv`
- `tm/limpo/desempenho_por_janela_tm.csv`
- `tm/limpo/tm_limpo_taxas_status.csv`
- `tm/adversos/tm_adversos_delta_ic95.csv`
- `tm/adversos/tm_adversos_delta_pct.csv`
- `tm/adversos/tm_adversos_hierarquia_robustez.csv`
- `tm/adversos/tm_adversos_categorias.csv`
- `tm/adversos/tm_adversos_sarima_variantes.csv`
- `tm/adversos/comparacao_m4_tm_delta.csv`
- `tm/xai/xai_tm_concordancia_shap_lime.csv`
- `tm/xai/xai_tm_estabilidade.csv`
- `tm/xai/xai_tm_posicao_lag12.csv`
- `tm/xai/xai_tm_redistribuicao_ruido.csv`
- `tm/xai/xai_tm_sensibilidade_lime.csv`
- `tm/xai/xai_tm_suplementar_lag24.csv`
- `tm/xai/xai_tm_avaliacao_previsoes_condicoes.csv`
- `tm/replicacao_previsoes_tm.csv`
- `tm_366/sensibilidade/spearman_ranking_200_366.csv`
- `tm_366/sensibilidade/posicoes_200_366.csv`
- `tm_366/sensibilidade/robustez_200_366.csv`
- `tm_366/sensibilidade/delta_ic95_366.csv`
- `tm_366/sensibilidade/condicoes_h1_h3_200_366.csv`
- `tm_366/sensibilidade/status_200_366.csv`

**Dados ainda sem tabela:** L06, L02 (ver `00_controle/LACUNAS.md`)
