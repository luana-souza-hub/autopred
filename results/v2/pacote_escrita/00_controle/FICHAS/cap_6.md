# Ficha: Capítulo 6 Conclusões

Fonte única dos números: `00_controle/FATOS_CONGELADOS.csv`. Caminhos relativos a `results/v2/` (cópias em `para_escrita/arquivos/`).

| id | fato | valor | IC 95% | origem |
|---|---|---|---|---|
| BAS-10 | Papel de cada base | M4: desenvolvimento (origem dos achados, hipóteses e limiares); Car Parts: exploratória; Tourism 200: replicação com previsões pré-registradas; Tourism 366: sensibilidade |  | Tabela 8 (4.7) do .docx; results/v2/HANDOFF.md |
| PRT-05 | Horizonte H por base | M4 18; Car Parts 18; Tourism 24 |  | scripts/run_v2.py |
| HIP-04 | Tourism (oficial, 200 séries) | H1 confirmou parcialmente (5 de 6); H2 confirmou parcialmente (6 de 10); H3 confirmou (6 de 6); H4 confirmou parcialmente |  | results/v2/tm/replicacao_previsoes_tm.csv |
| HIP-05 | Tourism 366 (sensibilidade) | H1 5 de 6; H2 8 de 10; H3 6 de 6; H4 ordem idêntica; mesmos rótulos |  | results/v2/fase5/replicacao_sensibilidade_tm366.csv |
| HIP-06 | M4 (critérios a posteriori) | H1 6 de 6; H2 10 de 10; H3 0 de 6 (circular); H4 par M4 × Car Parts: ordem difere, nenhuma família com IC separado no MASE (no MAE, Baseline separado) |  | results/v2/fase5/replicacao_H1_H4_resumo.csv |
| HIP-07 | Car Parts (exploratória, critérios a posteriori) | H1 4 de 6; H2 6 de 10; H3 0 de 6 |  | results/v2/fase5/replicacao_H1_H4_resumo.csv |
| CRZ-01 | Spearman dos rankings, MASE | M4 × Car Parts −0,200; Tourism × M4 +0,188; Tourism × Car Parts −0,770 | M4 × CP [−0,685; +0,248]; TM × M4 [−0,176; +0,636]; TM × CP [−0,867; −0,661] | notebooks/11a_desempenho_tourism_v2.ipynb (saída); results/v2/fase5/replicacao_h4_pares.csv |
| CRZ-06 | Queda do Holt-Winters M4 → Car Parts (mesma métrica) | MAE 3º → 10º; MASE 2º → 10º; ICs separados com e sem fallback | MAE [1; 7] → [10; 10]; MASE [1; 6] → [10; 10] | results/v2/cp/limpo/reversao_hierarquia_m4_cp_mesma_metrica.csv |
| CRZ-07 | Ordem das famílias por MASE nas três bases | M4: ML > Suav. > Baseline > ARIMA; Car Parts: Baseline > ML > Suav. > ARIMA; Tourism: Suav. > ARIMA > Baseline > ML. IC separado: Tourism × Car Parts em ARIMA e ML; Tourism × M4 nenhuma; M4 × Car Parts nenhuma (no MAE, Baseline) |  | results/v2/fase5/replicacao_h4_pares.csv |
| M4L-01 | M4 limpo: Holt-Winters | MAPE 8,36%; posto MAPE 1º; MAE 325,29; R² −1,22; MASE 0,918; posto MASE 2º [1; 6] | MAPE [6,19; 10,25]; posto [1; 3] | results/v2/fase5/tabela3_antigo_v2_m4.csv |
| M4L-03 | M4 limpo: Gradient Boosting | MAPE 9,06%; posto MAPE 3º; MAE 290,35; R² −1,09; MASE 0,900; posto MASE 1º [1; 5] | MAPE [7,72; 10,91]; posto [1; 5] | results/v2/fase5/tabela3_antigo_v2_m4.csv |
| M4L-04 | M4 limpo: SES | MAPE 9,68%; posto MAPE 4º; MAE 328,89; R² −1,06; MASE 0,938; posto MASE 3º [1; 7] | MAPE [8,29; 11,53]; posto [3; 7] | results/v2/fase5/tabela3_antigo_v2_m4.csv |
| M4L-06 | M4 limpo: Persistência | MAPE 9,75%; posto MAPE 6º; MAE 332,19; R² −1,29; MASE 0,953; posto MASE 4º [1; 8] | MAPE [8,30; 12,37]; posto [3; 8] | results/v2/fase5/tabela3_antigo_v2_m4.csv |
| M4A-02 | M4: Holt-Winters, ruído 30% | +17,6% (+1,47 p.p.) | [+2,5; +37,7] | results/v2/m4/adversos/m4_adversos_delta_ic95.csv |
| M4A-03 | M4: Holt-Winters, outliers 2% | +13,1% (+1,09 p.p.) | [+2,5; +38,6] | results/v2/m4/adversos/m4_adversos_delta_ic95.csv |
| M4A-05 | M4: escore de robustez, ajustados | SES 3,0 < RF 3,6 < SVR 4,1 < GB 8,5 < HW 10,5 < SARIMA heur. 12,9 | sem IC | results/v2/m4/adversos/m4_adversos_hierarquia_robustez.csv |
| M4A-12 | SARIMA automático fora dos cenários | 42 min na linha de base do M4; 5 a 6 h estimadas para os 9 cenários |  | results/v2/logs/m4_limpo_sarima_auto.log; HANDOFF.md |
| TMA-02 | Tourism: escore de robustez, ajustados | SES 1,5 < RF 3,4 < SVR 6,1 < SARIMA heur. 6,2 < GB 6,3 < HW 13,7 | sem IC | results/v2/tm/adversos/tm_adversos_hierarquia_robustez.csv |
| TMA-09 | Spearman do escore de robustez M4 × Tourism | +0,829 (6 ajustados) | sem IC | notebooks/11b_cenarios_adversos_tourism_v2.ipynb (saída) |
| M4X-01 | M4: Pearson SHAP × LIME (limpo) | RF 0,997; GB 0,996; SVR 0,993 |  | results/v2/m4/xai/xai_m4_concordancia_shap_lime.csv |
| M4X-02 | M4: Spearman por série SHAP × LIME (limpo) | RF 0,653; GB 0,702; SVR 0,824 | RF [0,553; 0,746]; GB [0,621; 0,780]; SVR [0,748; 0,881] | results/v2/m4/xai/xai_m4_concordancia_shap_lime.csv |
| M4X-04 | M4: estabilidade SHAP (média; mediana) | RF 0,561; GB 0,436; SVR 0,616 (medianas 0,584 / 0,405 / 0,630) | RF [0,480; 0,632]; GB [0,372; 0,500]; SVR [0,528; 0,695] | results/v2/m4/xai/xai_m4_estabilidade.csv |
| M4X-08 | M4: Q1, CV entre séries (reanálise, SHAP) | normalizado 1,211 / 1,015 / 0,700; escala original 1,702 / 1,493 / 1,167 (RF/GB/SVR) | normalizado RF [0,955; 1,298]; GB [0,849; 1,069]; SVR [0,621; 0,748] | results/v2/fase5/reanalise_cv_q1_m4.csv |
| TMX-03 | Tourism: estabilidade (média) | SHAP RF 0,679, GB 0,625, SVR 0,753; LIME 0,612 / 0,592 / 0,727 | SHAP RF [0,615; 0,739]; GB [0,551; 0,690]; SVR [0,691; 0,810]; LIME RF [0,548; 0,670]; GB [0,519; 0,662]; SVR [0,672; 0,778] | results/v2/tm/xai/xai_tm_estabilidade.csv |
| TMX-05 | Tourism: lag_12 | 1ª característica global nos 6 casos; no top-3 de 100% das séries, exceto SVR LIME (93,3%, 28 de 30) |  | results/v2/tm/xai/xai_tm_posicao_lag12.csv |
| BAS-07 | Séries com F_S acima do p95 do piso de ruído | M4 56,5%; Car Parts 18,0%; Tourism 98,5%; Tourism 366 99,2% |  | results/v2/pacote_escrita/02_caracterizacao_bases/caracterizacao_bases.csv |
| BAS-11 | Car Parts na explicabilidade | 30 séries entre 2.505 elegíveis (169 descartadas por alvo constante); treino da última janela com 33 obs.; 21 linhas de treino; 22 pontos |  | results/v2/cp/xai/*.meta.json |
| CPL-01 | Car Parts: Persistência | MAE 0,333; posto 1º; sem fallback 0,333; MASE 0,568 (1º) | MAE [0,33; 0,39]; posto [1; 1] | results/v2/fase5/tabela7_antigo_v2_cp.csv |
| CPL-10 | Car Parts: Holt-Winters | MAE 0,741; posto 10º; sem fallback 0,741; MASE 1,351 (10º) | MAE [0,72; 0,77]; posto [10; 10] | results/v2/fase5/tabela7_antigo_v2_cp.csv |
| TML-01 | Tourism limpo: Holt-Winters | MAPE 21,61% (1º); MAE 365,00 (4º); MASE 1,349 (1º) | ver ranking_ic_tm.csv | results/v2/tm/limpo/tabela_desempenho_v2_tm.csv |
| MOD-08 | Busca de hiperparâmetros | nenhuma: configurações fixas a priori, iguais em séries, janelas e bases |  | varredura do backup_pre_correcoes (sessão de 2026-10-04) |

**Arquivos**

- `fase5/replicacao_H1_H4_resumo.csv`
- `CHANGELOG_conclusoes.md`

**Dados ainda sem tabela:** L10 (ver `00_controle/LACUNAS.md`)
