# CHANGELOG das conclusões: dissertação x protocolo v2

Registro, por fase, de como cada achado da dissertação (`Dissertacao_PPGI_AutoPred_2.docx`) se comporta quando refeito com o protocolo único v2 (`src/pipeline_v2.py`). Situação: **mantém** (direção e magnitude compatíveis), **muda** (direção mantida, mas magnitude, posição ou rótulo diferentes) ou **não se sustenta** (o achado deixa de valer).

Convenções: MAPE e ΔMAPE em medianas entre séries; "Δ%" é relativo e "p.p." é absoluto; IC = intervalo de 95% por bootstrap pareado por série (2000 reamostragens). Atualizado em 2026-10-02.

---

## Fase 2a: M4 Monthly, desempenho em dados limpos (Tabela 3)

Mudanças de protocolo que explicam as diferenças: amostra única de 200 séries (antes, estatísticos e ML vinham de amostras diferentes, item C6), features sem a duplicata `lag_season` (C5), SARIMA automático sem vazamento (C4) e ajustes SARIMA degenerados rejeitados. A sanidade mostrou que, nos modelos cujo protocolo não mudou, o v2 reproduz os caches antigos janela a janela (diferença ≤ 3×10⁻¹⁴); para eles, a diferença vem só da amostra.

| Achado da dissertação | Valor antigo | Valor v2 | Situação | Arquivo-fonte |
|---|---|---|---|---|
| Holt-Winters é o melhor em dados limpos | 8,26% (1º) | 8,36% (1º) | mantém | `m4/limpo/tabela3_v2_m4.csv` |
| Persistência em 2º lugar ("desempenho surpreendente", seção 5.2.3) | 8,70% (2º) | 9,75% (6º) | não se sustenta (era efeito da amostra) | `m4/limpo/comparacao_tabela3_antiga_v2_m4.csv` |
| SES em 3º | 8,78% (3º) | 9,68% (4º) | muda | `m4/limpo/tabela3_v2_m4.csv` |
| SARIMA automático em 4º | 9,12% (4º) | 8,91% (2º); 8,70% sem as 14 janelas explosivas (2,3%) | muda (sobe com a seleção sem vazamento) | `m4/limpo/tabela3_v2_m4.csv` |
| Gradient Boosting é o melhor modelo de ML | 9,90% (5º) | 9,06% (3º) | mantém (sobe de posição) | `m4/limpo/tabela3_v2_m4.csv` |
| Modelos de ML nas posições 5ª a 9ª | 5º, 7º, 9º | 3º (GB), 7º (RF), 8º (SVR) | muda | `m4/limpo/tabela3_v2_m4.csv` |
| SARIMA heurístico em 6º | 10,10% | 9,72% (5º) com fallback; 10,69% (7º) sem; 1ª janela 100% em fallback | muda | `m4/limpo/tabela3_v2_m4.csv`, `m4/limpo/m4_limpo_fallback_por_janela.csv` |
| Média histórica em último | 18,60% (10º) | 21,82% (10º) | mantém | `m4/limpo/tabela3_v2_m4.csv` |
| Diferença GB x RF de "≈1,2 p.p." | texto 1,2; tabela antiga 0,69 | 1,07 p.p. | muda (o texto não batia com a própria tabela) | `m4/limpo/tabela3_v2_m4.csv` |
| Hierarquia entre famílias (melhor modelo de cada uma) | não reportada | Suavização (8,36) > ARIMA (8,91) > ML (9,06) > Baseline (9,75) | novo | `m4/limpo/ranking_familias_v2_m4.csv` |
| Séries do M4 com 42 a 126 observações (5.2.2, Tabela 6) | 42 a 126 | 66 a 918 (mediana 198) | não se sustenta (erro de texto) | `../tourism/caracterizacao_series_m4.csv` |
| Fig. 11: ganho do HW sobre a Persistência de "1,1 p.p." | 1,1 "p.p." | 1,1% relativo = 0,06 p.p. (amostra antiga) | muda (rótulo de unidade) | `notebooks/02_baselines_e_holt_winters.ipynb` |

## Fase 2b: M4 Monthly, cenários adversos (Seção 5.3, Tabelas 4 e 5)

Mudanças de protocolo: previsão recursiva de ML sem deslocamento (C1), perturbação só no treino com o teste limpo (C2), ΔMAPE em % e em p.p. (C3), os 9 cenários (C7) e truncamento comparável à linha de base (S1). Com 200 séries, 66 das 81 combinações modelo × cenário têm IC do Δ% incluindo zero, e quase todos os pares de modelos ajustados são indistinguíveis (exceção: outliers de 5%).

| Achado da dissertação | Valor antigo | Valor v2 | Situação | Arquivo-fonte |
|---|---|---|---|---|
| HW tem a maior degradação com ruído σ=30% | "+49,2 p.p." (era Δ% relativo) | +17,6% [+2,5; +37,7], ou +1,47 p.p. | muda (magnitude cerca de 1/3; HW entre os mais degradados, mas não distinguível de SES, RF, GB e SVR) | `m4/adversos/m4_adversos_delta_ic95.csv` |
| HW tem a maior degradação com outliers de 2% | "+45,7 p.p." | +13,1% [+2,5; +38,6], ou +1,09 p.p.; o sazonal ingênuo (+24,0%) e o SARIMA heurístico (+19,0%) degradam mais | muda | `m4/adversos/m4_adversos_delta_ic95.csv` |
| Outliers são o cenário mais devastador para todos | +16% a +46% (2%) | +0,5% a +24% (2%); +1% a +43% (5%) | muda (forte só para HW, GB, SARIMA e sazonal ingênuo) | `m4/adversos/m4_adversos_delta_pct.csv` |
| Paradoxo: ausências reduzem o MAPE de GB, RF e Persistência | GB −4,2/−7,9/−0,7; Persistência −2,5/−6,2/−10,8 | GB +0,6/+1,8/+5,2; Persistência 0,0/+1,2/+2,5; ICs incluem zero | não se sustenta (16 das 36 combinações mudam de sinal; vinha do teste interpolado) | `m4/adversos/comparacao_delta_nb08_v2.csv` |
| SES é o mais robusto a ruído | +0,1 / +27,9 | −3,8 / +1,1 (1º entre os ajustados) | mantém | `m4/adversos/m4_adversos_hierarquia_robustez.csv` |
| SVR é o melhor ML sob outliers (Tabela 5) | +18,1 | SVR +8,2% contra RF +7,4% (média dos dois níveis), com ICs sobrepostos | não se sustenta (indistinguíveis) | `m4/adversos/comparacao_tabela5_recomendacoes.csv` |
| Truncamento sem efeito ("ΔMAPE ≈ 0", Fig. 15 e 16) | figura vazia (o cenário nunca rodou) | ≈ 0 para baselines, SES, RF e SVR; HW +4,2/+9,1%; SARIMA +8,1/+15,9% | muda (agora há medição) | `m4/adversos/m4_adversos_delta_pct.csv` |
| Taxa de sucesso 0,93 / 0,85 por "falha de convergência" (5.3.1) | 0,93 e 0,85 | eram séries curtas; no v2, fallback 0% exceto SARIMA heurístico (33%, 1ª janela) e SVR (< 1%) | não se sustenta | `m4/adversos/m4_adversos_status.csv` |
| HW tem o perfil de robustez mais desfavorável | pior em ruído e outliers | entre os ajustados: SES 3,0 < RF 3,6 < SVR 4,1 < GB 8,5 < HW 10,5 < SARIMA heurístico 12,9 (escore \|Δ%\|) | muda (HW em 5º de 6; o pior é o SARIMA heurístico, que antes não foi avaliado; a ordem tem pouco suporte estatístico) | `m4/adversos/m4_adversos_hierarquia_robustez.csv` |
| Tabela 5: recomendações por tipo de adversidade | 5 recomendações | sustentam-se: HW em dados limpos, SES com ruído, SES como padrão; não se sustentam: SVR com outliers, GB ou Persistência com ausências | muda (critérios definidos depois de ver os resultados) | `m4/adversos/comparacao_tabela5_recomendacoes.csv` |
| SARIMA automático fora dos cenários adversos por custo (7 h) | > 7 h para 1 cenário | 42 min na linha de base v2; cerca de 6 h estimadas para 9 cenários | mantém (restrição operacional) | `logs/m4_limpo_sarima_auto.log` |
| Degradação reportada antes medida contra o alvo perturbado | n/a | a análise suplementar "alvo ruidoso" infla o Δ% em +15 (ruído 30%), +16 (outliers 2%) e +34 (outliers 5%) p.p. de Δ% | explica as diferenças | `m4/adversos/m4_adversos_suplementar_alvo_ruidoso.csv` |

## Fase 2c: M4 Monthly, explicabilidade (Seção 5.4)

Mudanças de protocolo: LIME por índice, features sem duplicata (lag_12 é o lag sazonal), SHAP e LIME nos mesmos 30 pontos (29 linhas de treino mais recentes + ponto de previsão), SHAP do SVR semeado e na escala original, importâncias normalizadas por série.

| Achado da dissertação | Valor antigo | Valor v2 | Situação | Arquivo-fonte |
|---|---|---|---|---|
| H1: concordância SHAP x LIME alta (Pearson > 0,90, Fig. 20) | 0,959 / 0,952 / 0,908 | Pearson 0,997 / 0,996 / 0,993; Spearman global 0,954 / 0,907 / 0,946; Spearman por série 0,65 / 0,70 / 0,82 | mantém (com ressalva: por série a concordância é bem menor) | `m4/xai/xai_m4_concordancia_shap_lime.csv` |
| Divergência SHAP x LIME no lag sazonal (LIME top-3, SHAP 7º a 8º) | LIME 3º; SHAP 7º a 8º | lag_12 em 3º nos dois métodos (RF, GB), 5º/4º no SVR; no top-3 de só 17% a 30% das séries | não se sustenta (bug do LIME somado à importância dividida com a duplicata) | `m4/xai/xai_m4_posicao_lag12.csv` |
| Estabilidade SHAP sob ruído σ=30% (Spearman mediano) | 0,572 / 0,344 / 0,520 | 0,584 / 0,405 / 0,630 (médias 0,561 [0,480; 0,632], 0,436 [0,372; 0,500], 0,616 [0,528; 0,695]) | mantém (moderada; GB a menor; nenhum ≥ 0,80) | `m4/xai/xai_m4_estabilidade.csv` |
| Fig. 22: sob ruído, lag_1 perde e roll_mean_6 ganha (RF, GB) | RF 160→62 e 38→104; GB 182→98 e 51→105 (brutos) | normalizados: RF 0,36→0,17 e 0,09→0,19; GB 0,33→0,17 e 0,07→0,12; ICs excluem zero | mantém | `m4/xai/xai_m4_comparacao_fig22.csv` |
| Fig. 22 / texto: o mesmo vale para o SVR | lag_1 0,146→0,093; roll_mean_6 0,057→0,066 | lag_1 0,20→0,14 (cai); roll_mean_6 0,086→0,085, IC inclui zero | não se sustenta para o SVR | `m4/xai/xai_m4_redistribuicao_ruido.csv` |
| Q1: consistência entre séries (CV 1,75 / 1,61 / 0,51) | CV médio sobre 17 features, escala original | não reavaliado no v2 (o CV antigo mistura escalas, item C11) | pendente | n/a |
| Sensibilidade ao nº de amostras do LIME | n/a | 200, 1000 e 5000 dão Pearson de 0,986 a 0,999 e o mesmo top-5 | novo (robusto) | `m4/xai/xai_m4_sensibilidade_lime.csv` |
| Lag sazonal de 24 meses (lag_season2 em 5º a 6º no SHAP antigo) | 5º / 6º / 6º | fora do oficial (decisão 1A); suplementar: lag_24 em 5º a 7º; lag_12 + lag_24 = 11% a 15% da importância | muda (análise suplementar) | `m4/xai/xai_m4_suplementar_lag24.csv` |

---

## Fase 3: Car Parts (Seção 5.6, Tabelas 6 e 7)

Mudanças de protocolo: RF com 100 árvores (antes 30, C15), features sem duplicata, SARIMA automático incluído (seleção por janela), ajustes SARIMA degenerados rejeitados, R² indefinido com teste constante (C13), cobertura do MAPE por janela e por série (C20). Sem cenários adversos (decisão 3A). Sanidade: Persistência, média, sazonal ingênuo, SES, Holt-Winters e SARIMA heurístico (janelas OK) reproduzem o NB10 janela a janela (diferença ≤ 4×10⁻¹⁵).

| Achado da dissertação | Valor antigo | Valor v2 | Situação | Arquivo-fonte |
|---|---|---|---|---|
| Persistência é a melhor no Car Parts (MAE) | 0,333 (1º) | 0,333 (1º), rank IC [1; 1] | mantém | `cp/limpo/tabela7_v2_cp.csv` |
| Persistência domina na faixa Extremo de zeros | 0,111 | 0,111 | mantém | `cp/limpo/tabela7_v2_cp.csv` |
| RF em 2º e SVR em 3º | RF 0,485 (2º); SVR 0,488 (3º) | SVR 0,459 (2º); RF 0,500 (3º). Sem fallback: RF 0,500 e SVR 0,507 | muda (o 2º lugar do SVR depende dos 13% de janelas em que ele cai na média do treino) | `cp/limpo/tabela7_v2_cp.csv` |
| GB à frente do SES | 0,499 (4º) | 0,556 (5º, empatado com o sazonal ingênuo) | muda | `cp/limpo/comparacao_tabela7_antiga_v2_cp.csv` |
| Holt-Winters é o penúltimo (8º de 9) | 0,741 (8º) | 0,741 (10º de 10, último), rank IC [10; 10] | mantém (fica ainda pior em posição; o MAE é idêntico) | `cp/limpo/tabela7_v2_cp.csv` |
| SARIMA heurístico é o último | 0,885 (9º) | 0,611 (8º) com fallback; 0,826 sem; 67% das janelas em fallback (janelas de 24 e 28 obs) | muda (o MAE antigo vinha de ajustes degenerados) | `cp/limpo/cp_limpo_fallback_por_janela.csv` |
| SARIMA automático | não avaliado | 0,650 (9º); 0,692 sem fallback (6%); 0,630 sem as 246 janelas explosivas (3,1%) | novo | `cp/limpo/tabela7_v2_cp.csv` |
| MAPE indefinido em 15,8% das janelas; cobertura de 89% | "Cob. MAPE = proporção de janelas" (89%) | 84,2% das janelas e 89,0% das séries | muda (rótulo corrigido) | `cp/limpo/tabela7_v2_cp.csv` |
| R² inadequado no Car Parts | medianas de −0,09 a −1,67 | R² indefinido em 15,8% das janelas (teste constante); medianas de −0,13 a −1,18 | muda | `cp/limpo/tabela7_v2_cp.csv` |
| H4: reversão de hierarquia M4 → Car Parts (HW de 1º a penúltimo) | HW 1º → 8º | HW 1º [1; 3] no M4 → 10º [10; 10] no Car Parts; Spearman entre rankings −0,48 [−0,66; −0,15] | mantém (agora com IC: a inversão é estatisticamente sustentada) | `cp/limpo/reversao_hierarquia_m4_cp.csv` |
| ML mais estável entre regimes que os estatísticos (5ª–9ª no M4 → 2ª–4ª no CP) | ML 5º–9º → 2º–4º | ML 3º/7º/8º → 2º/3º/5º; famílias: Suavização 1º → 3º, ARIMA 2º → 4º, ML 3º → 2º, Baseline 4º → 1º | muda (a família ML é a que menos muda de posição, mas dentro dela GB cai e SVR sobe) | `cp/limpo/reversao_familias_m4_cp.csv` |
| Explicabilidade não conduzida no Car Parts (limitação, 6.4) | não conduzida | EXPLORATÓRIA: 30 séries (de 2.505 elegíveis; 169 descartadas por alvo constante), 21 linhas de treino, 22 pontos explicados | novo | `cp/xai/` |
| (Car Parts) concordância SHAP x LIME | n/a | Pearson 0,91 / 0,71 / 0,79 (RF/GB/SVR); Spearman por série 0,84 / 0,79 / 0,66 | novo: H1 replica só parcialmente (GB e SVR abaixo de 0,90) | `cp/xai/xai_cp_concordancia_shap_lime.csv` |
| (Car Parts) estabilidade sob ruído σ=30% | n/a | Spearman SHAP 0,17 / 0,20 / 0,47 (médias); Jaccard top-3 de 0,24 a 0,38 | novo: muito menor que no M4 | `cp/xai/xai_cp_estabilidade.csv` |
| (Car Parts) lag sazonal | n/a | lag_12 em 7º a 9º (limpo); a top-1 é roll_mean_6 (RF, GB) ou lag_1 (SVR) | novo | `cp/xai/xai_cp_posicao_lag12.csv` |
| (Car Parts) redistribuição sob ruído | n/a | não reproduz o padrão do M4: roll_mean_6 **perde** importância em RF e GB (−0,10 e −0,06 no SHAP) e os modelos passam a usar lags isolados (lag_10, lag_6, lag_9) | novo: não replica (o ruído proporcional em série intermitente cria estrutura artificial) | `cp/xai/xai_cp_redistribuicao_ruido.csv` |
