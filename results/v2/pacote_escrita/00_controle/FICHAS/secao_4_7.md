# Ficha: 4.7 Hipóteses, pré-registro e status dos critérios

Fonte única dos números: `00_controle/FATOS_CONGELADOS.csv`. Caminhos relativos a `results/v2/` (cópias em `para_escrita/arquivos/`).

| id | fato | valor | IC 95% | origem |
|---|---|---|---|---|
| REG-01 | Registro das previsões do Tourism | 2 de outubro de 2026, commit 1a08fa1 |  | results/v2/PREVISOES_tourism.md |
| REG-02 | Interpretação da faixa da H1 | 3 de outubro de 2026, commit 1e3d9f0 (faixa fechada [0,65; 0,85]) |  | results/v2/INTERPRETACAO_criterios_4c.md |
| REG-03 | Código de avaliação das hipóteses | commit 6b49e3d, antes dos resultados |  | results/v2/CHANGELOG_conclusoes.md (4c) |
| HIP-01 | H1: condições e rótulos | Pearson agregado > 0,90 e Spearman por série em [0,65; 0,85], por modelo: 6 condições; 6 confirmou; 3 a 5 parcial; 0 a 2 não confirmou |  | results/v2/PREVISOES_tourism.md |
| HIP-02 | H2: condições e rótulos | (a) estabilidade < 0,80 por modelo e método; (b) em RF e GB, lag_1 perde e roll_mean_3 ou roll_mean_6 ganha, ICs excluindo zero: 10 condições (6 de (a), 4 de (b)); 10 confirmou; 5 a 9 parcial; 0 a 4 não confirmou |  | results/v2/PREVISOES_tourism.md |
| HIP-03 | H3: condições e rótulos | 6 condições, fração > 30,0%; 6 confirmou; 1 a 5 parcial; 0 não confirmou |  | results/v2/PREVISOES_tourism.md |
| LIM-01 | Categorias de robustez (mapa síntese) | sobre a média do Δ% de cada tipo: Alta −2 < Δ% < 2; Moderada 2 ≤ Δ% < 10; Baixa Δ% ≥ 10; Melhora* Δ% ≤ −2 |  | tools/nb_builders/nb08v2.py:226 a 246 |
| LIM-02 | Critérios das recomendações | definidos depois do M4; os das recomendações 5 (restrição operacional) e 6 (explicabilidade) depois do Tourism |  | results/v2/fase5/tabela5_antigo_v2.csv |
| LIM-03 | H1 a H4 no M4 e no Car Parts | critérios aplicados depois de conhecidos os resultados: descritivo |  | results/v2/fase5/replicacao_H1_H4.csv |
| LIM-04 | Leitura pós-hoc da redistribuição | "a característica dominante perde importância e as médias móveis ganham": formulada depois do Tourism, não registrada, não altera a H2 |  | results/v2/CHANGELOG_conclusoes.md (4c) |
| LIM-05 | H3 no M4 é circular | limiar de 30,0% = máximo observado no M4 (16,7% a 30,0%): 0 de 6 por construção |  | results/v2/m4/xai/xai_m4_posicao_lag12.csv |
| BAS-10 | Papel de cada base | M4: desenvolvimento (origem dos achados, hipóteses e limiares); Car Parts: exploratória; Tourism 200: replicação com previsões pré-registradas; Tourism 366: sensibilidade |  | Tabela 8 (4.7) do .docx; results/v2/HANDOFF.md |
| BAS-11 | Car Parts na explicabilidade | 30 séries entre 2.505 elegíveis (169 descartadas por alvo constante); treino da última janela com 33 obs.; 21 linhas de treino; 22 pontos |  | results/v2/cp/xai/*.meta.json |
| DEF-02 | MASE | erro absoluto ÷ média de \|y_t − y_(t−12)\| no treino da janela; indefinido se a escala é zero |  | src/pipeline_v2.py (escala_mase) |
| DEF-10 | Posto de uma família | posição do seu melhor modelo |  | tools/nb_builders/nb10v2.py:187 a 190 |
| M4X-06 | M4: lag_12 (limpo) | posição global SHAP 3º / 3º / 5º e LIME 3º / 3º / 4º (RF/GB/SVR); no top-3 de 16,7% / 26,7% / 20,0% (SHAP) e 16,7% / 26,7% / 30,0% (LIME) das séries |  | results/v2/m4/xai/xai_m4_posicao_lag12.csv |
| M4L-12 | M4: famílias por MASE | ML 1º; Suavização exponencial 2º; Baseline 3º; ARIMA 4º | ML [1; 4]; Suav. [1; 3]; Baseline [1; 4]; ARIMA [2; 4] | results/v2/tm/limpo/familias_tres_bases_mase_tm.csv |
| CRZ-07 | Ordem das famílias por MASE nas três bases | M4: ML > Suav. > Baseline > ARIMA; Car Parts: Baseline > ML > Suav. > ARIMA; Tourism: Suav. > ARIMA > Baseline > ML. IC separado: Tourism × Car Parts em ARIMA e ML; Tourism × M4 nenhuma; M4 × Car Parts nenhuma (no MAE, Baseline) |  | results/v2/fase5/replicacao_h4_pares.csv |

**Arquivos**

- `PREVISOES_tourism.md`
- `INTERPRETACAO_criterios_4c.md`
- `fase5/tabela4_antigo_v2.csv`
- `fase5/tabela5_antigo_v2.csv`
