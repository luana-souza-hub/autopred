# Pacote para a escrita da dissertação e do artigo

Cópias (os originais continuam nas pastas de `results/`) dos arquivos necessários para reescrever a dissertação e o artigo com os resultados do protocolo v2. Montado em 2026-10-04, sobre o commit `1ea7977` (Fase 5). Todos os números vêm do pipeline `src/pipeline_v2.py` (sha256 `bbe78767…`); a origem de cada cópia está indicada na coluna "Original".

**Controle da reescrita (2026-10-05):** `00_controle/` traz os fatos congelados (`FATOS_CONGELADOS.csv`), a matriz de mudanças do `.docx` (`MATRIZ_MUDANCAS.csv`), uma ficha por seção (`FICHAS/`), as lacunas (`LACUNAS.md`) e a lista do que é obsoleto (`OBSOLETOS.md`); ver `00_controle/LEIAME_controle.md`. O verificador de texto é `tools/checar_texto.py`. A pasta `results/v2/para_escrita/` reúne `00_controle` e os arquivos citados nas fichas.

**Por onde começar:** `01_documentos/CHANGELOG_conclusoes.md` (achado da dissertação, valor antigo, valor v2, situação, por fase), depois `09_replicacao_H1_H4/replicacao_H1_H4_resumo.csv` e as tabelas da Fase 5 (`11_fase5_tabelas_dissertacao/` e as cópias de `tabela3`, `tabela6` e `tabela7`).

**Convenções:** M4 = M4 Monthly (200 séries, H=18); CP = Car Parts (2.674 séries, H=18); TM = Tourism Monthly (200 séries, H=24); TM366 = Tourism com as 366 séries (sensibilidade). "Δ%" é relativo e "p.p." é absoluto. IC = 95%, bootstrap pareado por série (2000). Rótulos de previsão do Tourism: só os de `PREVISOES_tourism.md` valem como oficiais (200 séries); M4 e CP recebem os mesmos critérios **a posteriori**; TM366 é sensibilidade.

---

## 01_documentos

| Arquivo | Base | Pergunta que responde | Original |
|---|---|---|---|
| `AUDITORIA.md` | M4, CP | Quais defeitos havia nos notebooks 01 a 10 (C1 a C20, S1 a S3, D1 a D12), com que evidência, e onde o texto da dissertação diverge dos CSVs? Onde cada item foi corrigido? | criado para o pacote (relatório da auditoria de 2026-10-02) |
| `CHANGELOG_conclusoes.md` | todas | Cada achado da dissertação se mantém, muda ou não se sustenta no v2? Com que valores e em que arquivo? | `results/v2/` |
| `HANDOFF.md` | todas | Em que estado está cada fase, que decisões de protocolo foram tomadas e como reproduzir tudo? | `results/v2/` |
| `PREVISOES_tourism.md` | TM | O que foi previsto para o Tourism antes de qualquer execução (H1 a H4) e com que critérios? (commit 1a08fa1; não alterar) | `results/v2/` |
| `hiperparametros_v2.csv`, `hiperparametros_v2.md` | todas | Quais os hiperparâmetros efetivos de cada um dos 10 modelos (inclusive os padrões de biblioteca), as regras de seleção (aditivo ou multiplicativo no HW; grade de 140 candidatos e AIC no SARIMA automático), as sementes e as regras de fallback? Lido do código por `tools/gerar_hiperparametros_v2.py` | `results/v2/` |
| `sarima_limites_fallback_v2.csv` | M4, CP, TM, TM366 | Quantas janelas do SARIMA automático tiveram a busca cortada pelo limite de 300 s e quantas do heurístico passaram de 60 s? Quantas caíram em fallback por motivo (treino constante, nenhum candidato válido, exceção, verossimilhança degenerada, tempo, previsão inválida)? Por base, condição (limpo e os 9 cenários; `_alvo_ruidoso` é suplementar), modelo e janela de origem, com linha "todas". Respostas: nenhuma busca cortada nem estouro de tempo; todo fallback é verossimilhança degenerada ou treino constante. O SARIMA automático só tem cache em dados limpos. Contado dos caches por `tools/contar_limites_sarima_v2.py` | `results/v2/` |
| `INTERPRETACAO_criterios_4c.md` | TM | Como foi interpretada a faixa da H1 e por que os rótulos não serão reinterpretados? (commit 1e3d9f0) | `results/v2/` |

## 02_caracterizacao_bases

| Arquivo | Base | Pergunta que responde | Original |
|---|---|---|---|
| `caracterizacao_bases.csv` | M4, CP, TM, TM366 | Em que regime de dados está cada base (comprimento, zeros, CV, força de tendência F_T e de sazonalidade F_S, fração com sazonalidade acima do piso de ruído)? | `results/tourism/` (NB11, Etapa 1; independe do protocolo de modelagem) |
| `piso_ruido_fs.csv` | M4, CP, TM | Quanto F_S o STL atribui a ruído branco no comprimento típico de cada base? (mostra que o F_S bruto do Car Parts é ruído) | `results/tourism/` |
| `tabela6_antigo_v2.csv` | M4, CP, TM | Tabela 6 refeita: características metodológicas das três bases, antigo x v2, com tempo de execução incluindo o SARIMA automático | `results/v2/fase5/` |

## 03_m4_limpo (Fase 2a)

| Arquivo | Base | Pergunta que responde | Original |
|---|---|---|---|
| `tabela3_v2_m4.csv` | M4 | Qual o desempenho de cada modelo em dados limpos (MAPE, MAE, R², rank), **com e sem fallback** e **sem explosivas**, com a taxa de fallback e de explosivas? | `results/v2/m4/limpo/` |
| `tabela3_antigo_v2_m4.csv` | M4 | Tabela 3 final (Fase 5): valor antigo x v2 com IC, variantes com e sem fallback e explosivas, MASE | `results/v2/fase5/` |
| `comparacao_tabela3_antiga_v2_m4.csv` | M4 | Quanto cada modelo mudou de MAPE e de posição em relação à Tabela 3 antiga, e de qual amostra antiga vinha? | `results/v2/m4/limpo/` |
| `ranking_familias_v2_m4.csv` | M4 | Qual a ordem das famílias (melhor modelo de cada uma) em dados limpos? | `results/v2/m4/limpo/` |
| `m4_limpo_taxas_status.csv` | M4 | Que fração das janelas ficou OK, em fallback, em erro ou sem janela, por modelo? | `results/v2/m4/limpo/` |
| `m4_limpo_fallback_por_janela.csv` | M4 | Em que janela (1ª, 2ª, 3ª) ocorrem os fallbacks? (SARIMA heurístico: toda a 1ª janela) | `results/v2/m4/limpo/` |
| `sanidade_amostra_antiga_m4.csv` | M4 | Onde o protocolo não mudou, o v2 reproduz os caches antigos janela a janela? (sim, diferença ≤ 3×10⁻¹⁴) | `results/v2/m4/limpo/` |
| `m4_limpo_por_serie.csv` | M4 | Tabela por série (unique_id x modelo): MAPE, MASE, MAE e R² da mediana das 3 janelas (com fallback, versão oficial), com nº de janelas, de fallbacks e de explosivas de cada série, para filtrar | gerado aqui por `tools/detalhar_m4_limpo_v2.py` (dos caches, sem rodar modelos) |
| `m4_limpo_distribuicao_por_serie.csv` | M4 | Como a mediana por série se distribui entre as 200 séries? Mínimo, Q1, mediana, Q3 e máximo de MAPE, MASE, MAE e R², com o unique_id do mínimo e do máximo do MAPE e do R². Versão oficial (com fallback) para os 10 modelos; SARIMA heurístico e SVR também sem fallback; SARIMA automático e heurístico também sem explosivas | idem |
| `m4_limpo_mape_acima_100.csv` | M4 | Quantas séries têm MAPE (mediana por série) acima de 100% em cada modelo, qual o MAPE máximo e em que série? (SARIMA automático: 184,31% em M20937, 5 séries acima de 100%; sem explosivas, 4) | idem |
| `desempenho_por_janela_m4.csv` | M4 | MAPE, MAE e MASE (mediana entre séries) e posto de cada um dos 10 modelos em cada janela de origem (1ª, 2ª, 3ª), no formato de `desempenho_por_janela_tm.csv` mais as colunas `posto MAPE` e `posto MASE`. Na 1ª janela o SARIMA heurístico empata com o sazonal ingênuo (100% em fallback) | idem |
| `m4_limpo_explosivas_denominador.csv` | M4 | Qual o denominador da proporção de janelas explosivas da Tabela 3 v2? (todas as janelas do modelo: SARIMA heurístico 1 de 600 = 0,17%, exibido como 0,2%; sobre as 400 janelas OK seria 0,25%; SARIMA automático 14 de 600 = 2,33%, igual nos dois denominadores) | idem |
| `fig09_v2_m4_10_modelos.png` | M4 | Fig. 9 v2 refeita: MAPE (truncado em 100%) e MASE (truncado em 6) por série dos 10 modelos, sem o espaço vazio à esquerda do painel do MAPE (a versão em `results/v2/fase5/figuras/` não foi alterada) | idem |

## 04_m4_adversos (Fase 2b)

| Arquivo | Base | Pergunta que responde | Original |
|---|---|---|---|
| `m4_adversos_delta_pct.csv` | M4 | Quanto cada modelo degrada em cada um dos 9 cenários, em ΔMAPE relativo (%)? | `results/v2/m4/adversos/` |
| `m4_adversos_delta_pp.csv` | M4 | A mesma degradação em pontos percentuais absolutos | idem |
| `m4_adversos_delta_pareado.csv` | M4 | Mediana, entre séries, da diferença pareada de MAPE (p.p.) | idem |
| `m4_adversos_delta_longo.csv` | M4 | Formato longo com MAPE base, MAPE no cenário, Δ em % e p.p. e nº de séries | idem |
| `m4_adversos_delta_ic95.csv` | M4 | Qual o IC 95% de cada Δ% e quais incluem zero (66 de 81)? | idem |
| `m4_adversos_pares_ic95.csv` | M4 | Quais pares de modelos são distinguíveis em cada cenário? | idem |
| `m4_adversos_hierarquia_robustez.csv` | M4 | Qual a hierarquia de robustez, com **baselines triviais separados** dos modelos ajustados e Δ% com sinal ao lado do escore em módulo? | idem |
| `m4_adversos_status.csv` | M4 | Taxas de fallback, erro e nº de explosivas por cenário e modelo | idem |
| `m4_adversos_suplementar_alvo_ruidoso.csv` | M4 | (Suplementar) quanto da degradação antiga vinha de avaliar contra um alvo perturbado? | idem |
| `comparacao_delta_nb08_v2.csv` | M4 | Como o Δ% de cada modelo e cenário mudou do NB08 para o v2 (e onde mudou de sinal)? | idem |
| `comparacao_tabela4_robustez.csv` | M4 | Categorias de robustez da Tabela 4: antigo x v2 (limiares definidos depois de ver os resultados) | idem |
| `comparacao_tabela5_recomendacoes.csv` | M4 | As recomendações da Tabela 5 se sustentam no v2? (critérios definidos depois de ver os resultados) | idem |

## 05_car_parts_limpo (Fase 3)

| Arquivo | Base | Pergunta que responde | Original |
|---|---|---|---|
| `tabela7_v2_cp.csv` | CP | Desempenho em dados limpos com MAE como métrica principal; MAE **com e sem fallback** e **sem explosivas**; cobertura do MAPE por janela e por série; R² sem teste constante; MAE por faixa de zeros | `results/v2/cp/limpo/` |
| `tabela7_antigo_v2_cp.csv` | CP | Tabela 7 final (Fase 5): antigo x v2 com IC, com e sem fallback, MASE | `results/v2/fase5/` |
| `comparacao_tabela7_antiga_v2_cp.csv` | CP | Quanto cada modelo mudou de MAE e de posição em relação à Tabela 7 antiga, com MAE v2 sem fallback | `results/v2/cp/limpo/` |
| `cp_limpo_taxas_status.csv` | CP | Taxas de status por modelo (SARIMA heurístico 67% e SVR 13% em fallback) | idem |
| `cp_limpo_fallback_por_janela.csv` | CP | Em que janela ocorrem os fallbacks? | idem |
| `reversao_hierarquia_m4_cp_mesma_metrica.csv` | M4 x CP | Com a mesma métrica nas duas bases (MAE e MASE), quanto muda o rank de cada modelo, com IC, **com fallback e com SVR e SARIMA sem fallback**? | idem |
| `reversao_familias_m4_cp_mesma_metrica.csv` | M4 x CP | A ordem das famílias com a mesma métrica, nas duas versões (com e sem fallback de SVR e SARIMA) | idem |
| `reversao_hierarquia_m4_cp.csv` | M4 x CP | (Referência) a comparação antiga com métricas diferentes (MAPE no M4, MAE no CP) | idem |
| `reversao_familias_m4_cp.csv` | M4 x CP | (Referência) a ordem das famílias com métricas diferentes | idem |

## 06_tourism_limpo (Fase 4a)

| Arquivo | Base | Pergunta que responde | Original |
|---|---|---|---|
| `tabela_desempenho_v2_tm.csv` | TM | Desempenho em dados limpos por MAPE, MAE e MASE, com ranks; MAPE e MAE **sem fallback**; MAPE **sem explosivas**; taxas de fallback e explosivas | `results/v2/tm/limpo/` |
| `ranking_ic_tm.csv` | TM | Rank de cada modelo com IC por bootstrap (MASE, MAE, MAPE) | idem |
| `ranks_tres_bases_tm.csv` | M4, CP, TM | Rank de cada modelo nas três bases com a mesma métrica (MAE e MASE; MAPE entre M4 e TM), com IC | idem |
| `familias_tres_bases_mase_tm.csv` | M4, CP, TM | Ordem das famílias nas três bases por MASE, com IC (critério principal da H4) | idem |
| `familias_tres_bases_mae_tm.csv` | M4, CP, TM | O mesmo por MAE (secundária) | idem |
| `desempenho_por_janela_tm.csv` | TM | O desempenho do ML depende da 1ª janela (12 linhas de treino)? (não: é nela que o ML fica mais bem colocado) | idem |
| `tm_limpo_taxas_status.csv` | TM | Taxas de status por modelo | idem |
| `tm_limpo_fallback_por_janela.csv` | TM | Em que janela ocorrem os fallbacks? | idem |

## 07_tourism_adversos (Fase 4b)

| Arquivo | Base | Pergunta que responde | Original |
|---|---|---|---|
| `tm_adversos_delta_pct.csv`, `tm_adversos_delta_pp.csv`, `tm_adversos_delta_pareado.csv`, `tm_adversos_delta_longo.csv` | TM | Degradação por modelo e cenário em %, p.p. e mediana pareada | `results/v2/tm/adversos/` |
| `tm_adversos_delta_ic95.csv` | TM | IC de cada Δ% (53 de 81 incluem zero) | idem |
| `tm_adversos_pares_ic95.csv` | TM | Pares de modelos distinguíveis por cenário (14 de 135 entre os ajustados) | idem |
| `tm_adversos_hierarquia_robustez.csv` | TM | Hierarquia de robustez com **baselines triviais separados** e Δ% com sinal | idem |
| `tm_adversos_categorias.csv` | TM | Categorias de robustez com os mesmos limiares do M4 (fixados antes desta execução) | idem |
| `tm_adversos_sarima_variantes.csv` | TM | Δ% do SARIMA heurístico **com fallback, sem fallback e sem explosivas** | idem |
| `tm_adversos_status.csv` | TM | Taxas de fallback, erro e explosivas por cenário | idem |
| `comparacao_m4_tm_delta.csv` | M4 x TM | O Δ% de cada modelo e cenário tem o mesmo sinal nas duas bases? (a ordem de robustez se repete: Spearman do escore +0,83) | idem |

## 08_xai (Fases 2c, 3, 4c e 4d)

Em cada subpasta (`m4`, `cp`, `tm`, `tm_366`), as mesmas tabelas, calculadas com SHAP e LIME nos mesmos 30 pontos do modelo da última janela (no CP, 22 pontos), importâncias normalizadas por série, limpo e com ruído σ=30% no treino. O CP é **exploratório** (21 linhas de treino; sorteio entre 2.505 séries elegíveis); o TM366 é **sensibilidade**.

| Arquivo (prefixo `xai_<base>_`) | Bases | Pergunta que responde |
|---|---|---|
| `top10_global.csv` | M4, CP, TM, TM366 | Quais as 10 features mais importantes por modelo, método e condição, e em que posição está o lag_12? |
| `estabilidade.csv` | M4, CP, TM, TM366 | As explicações se mantêm sob ruído? (Spearman e Jaccard top-3 e top-5 por série, com IC) |
| `concordancia_shap_lime.csv` | M4, CP, TM, TM366 | SHAP e LIME concordam? (Pearson e Spearman globais e Spearman por série com IC) |
| `posicao_lag12.csv` | M4, CP, TM, TM366 | Onde fica o lag sazonal (global, mediana por série e % de séries com lag_12 no top-3)? |
| `redistribuicao_ruido.csv` | M4, CP, TM, TM366 | Sob ruído, quais features perdem e quais ganham importância (Δ com IC)? (equivalente à Fig. 22) |
| `sensibilidade_lime.csv` | M4, CP, TM, TM366 | Os resultados dependem do nº de amostras do LIME (200, 1000, 5000)? (CP e TM366: só 200, sem as rodadas extras) |
| `suplementar_lag24.csv` | M4, TM | (Suplementar) o que muda com lag_24 incluído? |
| `avaliacao_previsoes_condicoes.csv` | TM, TM366 | Cada condição de H1 a H3, com o valor e se vale (TM: avaliação oficial; TM366: sensibilidade) |
| `comparacao_dissertacao.csv` | M4 | Valores antigos da dissertação x v2 (Pearson da Fig. 20, Spearman de estabilidade) |
| `comparacao_fig22.csv` | M4 | Fig. 22 antiga (bruta) x antiga normalizada x v2 normalizada |

## 09_replicacao_H1_H4 (Fase 5)

| Arquivo | Base | Pergunta que responde | Original |
|---|---|---|---|
| `replicacao_H1_H4_resumo.csv` | M4, CP, TM | Quadro final: cada hipótese replicou, replicou parcialmente ou não replicou em cada base? | `results/v2/fase5/` |
| `replicacao_H1_H4.csv` | M4, CP, TM | O mesmo em formato longo, com a evidência numérica e o tipo de avaliação (oficial ou a posteriori) | idem |
| `replicacao_condicoes_h1_h3.csv` | M4, CP, TM | Cada condição de H1 a H3 nas três bases, com valor e decisão | idem |
| `replicacao_h4_pares.csv` | M4, CP, TM | H4 nos três pares de bases (MASE principal, MAE de apoio), com IC | idem |
| `replicacao_previsoes_tm.csv` | TM | Avaliação oficial das previsões registradas (H1 parcial, H2 parcial, H3 confirmou, H4 parcial) | `results/v2/tm/` |

## 10_sensibilidade_366 (Fase 4d)

| Arquivo | Base | Pergunta que responde | Original |
|---|---|---|---|
| `spearman_ranking_200_366.csv` | TM x TM366 | O ranking dos modelos muda com as 366 séries? (Spearman MASE 0,988, MAPE 0,939) | `results/v2/tm_366/sensibilidade/` |
| `posicoes_200_366.csv` | TM x TM366 | Quais modelos mudam de posição e quanto? | idem |
| `desempenho_200_366.csv` | TM x TM366 | MAPE, MAE e MASE nas duas amostras, **com e sem fallback** e **sem explosivas** | idem |
| `status_200_366.csv` | TM x TM366 | Fallback e explosivas nas duas amostras | idem |
| `condicoes_h1_h3_200_366.csv` | TM x TM366 | Cada condição de H1 a H3 nas duas amostras: a decisão é a mesma? | idem |
| `h1_h3_200_366.csv` | TM x TM366 | Rótulos de H1 a H3: oficial (200) x sensibilidade (366) | idem |
| `sensibilidade_previsoes_tm366.csv` | TM366 | Rótulos de sensibilidade de H1 a H3 (não substituem os oficiais) | `results/v2/tm_366/` |
| `replicacao_sensibilidade_tm366.csv` | TM366 | Sensibilidade no formato da tabela de replicação da Fase 5 | `results/v2/fase5/` |
| `robustez_200_366.csv` | TM x TM366 | A hierarquia de robustez se mantém? (**baselines triviais separados**; Spearman do escore +1,000 entre os ajustados) | `results/v2/tm_366/sensibilidade/` |
| `delta_200_366.csv` | TM x TM366 | Δ% de cada modelo e cenário nas duas amostras | idem |
| `delta_ic95_366.csv` | TM366 | IC dos Δ% com 366 séries (42 de 81 incluem zero) | idem |

## 11_fase5_tabelas_dissertacao (Fase 5)

| Arquivo | Base | Pergunta que responde | Original |
|---|---|---|---|
| `tabela4_antigo_v2.csv` | M4, TM | Tabela 4 (mapa síntese) refeita: desempenho, robustez por tipo de adversidade e custo medido, antigo x v2 | `results/v2/fase5/` |
| `tabela5_antigo_v2.csv` | M4, CP, TM | Tabela 5 (recomendações) com critérios verificáveis nas três bases | idem |
| `custo_computacional_v2.csv` | M4, CP, TM | Quanto custa cada modelo (tempos medidos no v2)? | idem |
| `reanalise_cv_q1_m4.csv` | M4 | Reanálise do CV de consistência da Questão 1 (item C11), base da Fig. 18 | idem |
| `figuras_indice.csv` | M4, CP | Índice das Figs. 8 a 26: o que mudou em cada uma (antigo x v2) e onde está a figura nova | idem (as figuras estão em `results/v2/fase5/figuras/`, não copiadas) |
| `indice_fase5.csv` | todas | Índice dos arquivos da Fase 5 | idem |
| `tabela3_dissertacao.csv` a `tabela7_dissertacao.csv` | M4, CP | Valores exatamente como estão hoje no `.docx` da dissertação (lidos sem alterar o arquivo), para a comparação antigo x v2 | `results/v2/fase5/antigo_dissertacao/` |

---

## Lacunas: o que não está em tabela

Verificadas ao montar o pacote, pela varredura das colunas de cada arquivo:

1. **SARIMA heurístico nos cenários adversos do M4, sem fallback e sem explosivas: não existe tabela.** No Tourism existe (`tm_adversos_sarima_variantes.csv`), no M4 não. É relevante: no M4 o SARIMA heurístico fica em fallback em 33% a 66% das janelas dos cenários (66% no truncamento de 25%) e tem 4 janelas explosivas. Os Δ% do M4 (`m4_adversos_delta_*.csv`) incluem o fallback. Pode ser gerada a partir dos caches existentes, sem rodar modelos.
2. **SVR sem fallback nos cenários adversos: não existe tabela em nenhuma base.** No M4 o SVR fica em fallback em 0% a 0,8% das janelas dos cenários (impacto pequeno); no Tourism, 0% (sem efeito). Também pode ser gerada dos caches.
3. **Rankings com IC no Tourism (`ranking_ic_tm.csv`, `ranks_tres_bases_tm.csv`, `familias_tres_bases_*_tm.csv`) só na versão com fallback.** As medianas sem fallback estão em `tabela_desempenho_v2_tm.csv`, mas sem IC nem rank. No Car Parts a versão sem fallback de SVR e SARIMA existe (`*_mesma_metrica.csv`).
4. **Baselines triviais separados só nas tabelas de robustez** (`m4_adversos_hierarquia_robustez.csv`, `tm_adversos_hierarquia_robustez.csv`, `robustez_200_366.csv`). Nas tabelas de Δ% e de categorias (`*_delta_*.csv`, `tm_adversos_categorias.csv`, `comparacao_tabela4_robustez.csv`, `tabela4_antigo_v2.csv`) eles aparecem misturados aos modelos ajustados, identificáveis pela família "Baseline". Nas tabelas de desempenho em dados limpos a separação não se aplica (os baselines são referência de desempenho).
5. **Explicabilidade:** não há fallback (o SVR só é explicado quando o alvo do treino não é constante); no Car Parts, 169 séries foram excluídas do sorteio por esse motivo (registrado nos `.meta.json` de `results/v2/cp/xai/`).
6. **Tabela 8 da dissertação** (atendimento dos objetivos): listada como pendência na Fase 5, não existe tabela v2.
