# Auditoria dos notebooks 01 a 10, do src/ e dos caches (somente leitura)

Relatório da auditoria feita em 2026-10-02, **antes** das correções, sobre os notebooks 01 a 10, `src/` e os resultados em cache (`results/`). Nenhum arquivo foi alterado durante a auditoria (snapshot de nome, tamanho e data de 165 arquivos, idêntico antes e depois). Cada item foi reproduzido com teste mínimo ou com números do cache. A coluna "Correção" foi acrescentada depois, ao montar este pacote, e indica onde o item foi tratado no protocolo v2 (detalhes em `CHANGELOG_conclusoes.md`).

Classificação: **confirmado com evidência**, **suspeito sem reprodução** ou **descartado**. Gravidade: afeta resultado principal, afeta resultado secundário, cosmético.

## 1. Resumo

| Gravidade | Confirmados | Itens |
|---|---|---|
| Afeta resultado principal | 9 | C1 a C9 |
| Afeta resultado secundário | 6 | C10 a C15 |
| Cosmético ou só no texto | 5 | C16 a C20 |
| Suspeito, sem reprodução | 3 | S1 a S3 |
| Descartado | 12 | D1 a D12 |

Além desses, o bug do LIME no NB09 (associação de termos às features por substring, que somava lag_10 a lag_12 em lag_1 e lag_season2 em lag_season) foi encontrado antes da auditoria e corrigido no próprio NB09 (backup em `results/explicabilidade_original_bugado/`).

## 2. Itens confirmados

As células seguem a numeração do Jupyter (a partir de 0) nos notebooks originais.

### Afetam resultado principal

| id | Onde | Mecanismo | Evidência (reproduzida) | Afeta | Correção |
|---|---|---|---|---|---|
| C1 | NB08, célula [7], linhas 53, 68 e 85 (`clf.predict(Xt[[-1]])`) | Na previsão recursiva, a última linha de `build_features(hist)` tem alvo `y[n-1]`, já observado; cada passo "prevê" o ponto anterior, e a previsão de RF, GB e SVR fica deslocada em um passo | Série y = 100 + t (n = 80) com regressão linear perfeita: real 180, 181, 182; NB08 prevê 179, 180, 180,91 | Linha de base do NB08, todo ΔMAPE de ML, Figs. 13 a 16, Tabelas 4 e 5, H4 no M4 | v2: `ml_forecast` monta o próximo passo com a mesma função das linhas de treino; teste de regressão C1 (Fase 1) |
| C2 | NB08, célula [13], linhas 78 e 80; célula [11] | `perturb_fn(s)` aplicada à série inteira antes da avaliação: ruído, ausências interpoladas e outliers entram também no teste, e o erro é medido contra um alvo perturbado | Persistência, cache reproduzido: outliers 2% dão ΔMAPE +16,3% com teste perturbado e +0,9% com teste limpo; ruído 30%: +32,3% e +16,2%; ausências 30%: −10,8% e −7,6% | Seção 5.3, "outliers o cenário mais devastador", parte do paradoxo das ausências, conclusões, Tabelas 4 e 5 | v2: perturbação só no treino de cada janela; teste de regressão C2; "alvo ruidoso" só como análise suplementar (Fase 2b) |
| C3 | NB08, célula [15], linha 65 (`DELTA_pct`); texto da dissertação | O código calcula o ΔMAPE relativo (%), o texto o chama de "variação absoluta (p.p.)" | HW com ruído 30%: base 8,26, cenário 12,33, ou seja +4,07 p.p.; os "49,2" são % relativo | Resumo, 5.1, 5.3, Tabela 5, conclusões | v2: ΔMAPE em % e em p.p. com rótulos corretos (Fase 2b); o texto precisa ser corrigido |
| C4 | NB04b, célula [10], linha 35 (`select_sarima_order(values, ...)`) | A ordem do SARIMA automático é escolhida por AIC na série inteira, incluindo os testes das três janelas (vazamento na seleção de modelo) | 6 de 6 séries mudam de ordem quando a seleção usa só o treino da 1ª janela (ex.: M10314 de (2,1,2)(1,1,1) para (0,0,0)(0,1,1)) | Tabela 3 (SARIMA automático 9,12%), Fig. 9 | v2: seleção por janela, só com o treino (decisão 2A, Fase 2a) |
| C5 | NB09 célula [6], linha 15; NB08 [6]; NB05 a 07 e NB10 (`build_features`) | `lag_12` e `lag_season` são a mesma coluna (y[t−12]); a importância do lag sazonal se divide entre as duas | `np.array_equal` = True nos três protocolos; somadas, ocupariam 4ª (RF) e 3ª (GB) posição no SHAP, contra 7ª e 8ª separadas | H3, Figs. 17, 19 e 20, posição do lag sazonal | v2: 15 features sem duplicata; lag_season2 também fora do oficial (decisão 1A) |
| C6 | NB01 a 04b (ordem alfabética) x NB05 a 09 (ordem do arquivo) | Duas amostras piloto diferentes do M4 | Só 3 séries em comum em 200; os caches confirmam | Tabela 3 e Fig. 9 misturam as amostras (estatísticos e baselines de uma, GB, RF e SVR da outra); H4 | v2: amostra única (NB05 a 09, ordem do arquivo, SEED=42) |
| C7 | `results/adversos/` | Os cenários D (truncamento 50% e 25%) e C 5% nunca foram gerados (sem CSV e fora de `delta_mape_tabela.csv`) | A "escala ±0,05" citada no texto é o eixo padrão do matplotlib num painel vazio (ylim = −0,055 a 0,055) | 5.3.5 ("ΔMAPE próximo de zero"), Tabela 4 (truncamento "Neutro"), "6 subcenários" | v2: os 9 cenários executados (Fase 2b) |
| C8 | NB09, célula [10], linhas 24 e 28 | SHAP do SVR com KernelExplainer, `nsamples=50` para 17 features e sem semente: ruidoso e não reproduzível | Duas execuções na mesma série diferem em até 191% e o top-6 muda | SHAP do SVR, estabilidade do SVR (ρ = 0,520), H1 e H2 do SVR | v2: semente fixa, nsamples = 2p + 2048, escala original; teste de reprodutibilidade |
| C9 | NB05, célula [12], linha 30, e célula [10], linha 20 | O código atual do RF não roda na 1ª origem (n_train = 24): treina com 15 features e prevê com 16; o cache foi gerado às 11:02, o notebook salvo às 11:38 | Código atual: "X has 16 features..."; versão com `feature_cols`: MAPE difere do cache em até 2,47 p.p. (HW, GB e SVR reproduzem exatamente) | RF da Tabela 3 (10,59%), OE5 (reprodutibilidade) | v2: RF refeito no pipeline único (Fase 2a) |

### Afetam resultado secundário

| id | Onde | Mecanismo | Evidência | Afeta | Correção |
|---|---|---|---|---|---|
| C10 | NB09, células [14], [17] e [19] (`sub.mean()`) | Importância global = média simples de \|SHAP\| e \|LIME\| em escala original; séries de nível alto dominam | As 3 maiores séries (de 30) respondem por 39% (RF) e 36% (GB) da importância do lag_1; no SVR (escala normalizada), 17% | Figs. 17 a 20, Q1 | v2: importâncias normalizadas por série antes da média |
| C11 | NB09, célula [27], linha 6 | O CV de consistência mistura escalas (RF e GB em escala original, SVR normalizado) e é a média de 17 features, não o "CV do lag_1" do texto | Normalizando por série, CV 1,75 → 1,18 (RF) e 1,61 → 1,10 (GB); SVR 0,50 | Q1 ("SVR mais consistente") | Reanálise na Fase 5 (`reanalise_cv_q1_m4.csv`) |
| C12 | NB08, célula [6], linha 78 | Taxa de sucesso < 1 vem só de séries curtas (n < 24 + 3·18), não de falha de convergência | Persistência: 44 séries com taxa < 1, exatamente as 44 com n < 78; ML: 51 séries, todas com n < 94 | 5.3.1, Fig. 14 | v2: status separa SEM_JANELA, FALLBACK e ERRO |
| C13 | NB10, célula [10], linha 33 (`r2_score`) | Com teste constante (todo zero), o sklearn força R² = 0 ou 1 | 1.265 de 8.022 janelas (15,8%) | R² do Car Parts (Tabela 7) | v2: R² indefinido (NaN) com teste constante (Fase 3) |
| C14 | NB10, célula [12], linhas 78, 92 e 96 | SARIMA e HW caem silenciosamente no sazonal ingênuo e contam como OK; no M4 (NB04) a mesma falha vira ERRO | SARIMA do Car Parts em fallback em 4,4% das janelas da amostra (treino constante); HW 0% | SARIMA no Car Parts; consistência entre bases | v2: fallback registrado no status e reportado como taxa |
| C15 | NB05 e NB06 x texto (4.3); NB10, célula [2], linha 67 | O texto diz "valores padrão do scikit-learn" e "protocolo idêntico", mas RF tem min_samples_leaf=2, GB subsample=0,8, RF do Car Parts 30 árvores, SARIMA heurístico ordem fixa | Código | 4.3 e 5.6 | v2: mesmos hiperparâmetros nas três bases (RF com 100 árvores) e documentados |

### Cosméticos ou só no texto

| id | Onde | Mecanismo | Evidência | Correção |
|---|---|---|---|---|
| C16 | NB08, célula [11], linha 24 | Outliers: código usa média ± 3σ (substituição); texto diz ± 5σ | Código x seções 4.4 e 5.3.4 | v2 manteve 3σ (`OUTLIER_K`); o texto precisa ser corrigido |
| C17 | NB02, célula [24], linha 18 | Ganho da Fig. 11 é relativo (%), texto diz "p.p." | 1,1% = 0,06 p.p.; 50,1% = 8,24 p.p.; 19,6% = 1,00 p.p. | Fig. 11 refeita na Fase 5 |
| C18 | NB04b, célula [10], linha 24 | Registro SKIP grava `'unique_id': 'uid'` literal | Nenhum SKIP no cache (600 OK): sem efeito | n/a |
| C19 | NB03, célula do α | Comentário diz "última janela", ajuste usa a série inteira | Só figura descritiva | n/a |
| C20 | NB10, célula [21] | "Cob. MAPE" é por série (89,0%), texto diz por janela (84,2%) | Cache | v2: cobertura por janela e por série (Fase 3) |

## 3. Suspeitos sem reprodução

| id | Descrição | Desfecho no v2 |
|---|---|---|
| S1 | Truncamento do NB08 mantém o início da série: as janelas avaliadas diferem das da linha de base | Corrigido: truncamento mantém a fração mais recente do treino de cada janela, mesmo teste da linha de base |
| S2 | NB05 e NB10 incluem lag_season2 só quando n_train > 24: o conjunto de features muda entre janelas da mesma série | Resolvido pela decisão 1A (lag_season2 fora do oficial) |
| S3 | Origens por `linspace` geram testes sobrepostos em séries curtas (n = 66: origens 24, 36, 48); janelas correlacionadas, sem vazamento | Mantido (esquema de janelas do protocolo); registrar como limitação |

## 4. Descartados

| id | Verificação |
|---|---|
| D1 | Features só usam o passado (shift ≥ 1); scalers ajustados só no treino de cada janela (NB07, NB08, NB09, NB10, src) |
| D2 | A recursão do NB05 a 07 e do NB10 usa `history[n-i]` corretamente; o deslocamento é só do NB08 (C1) |
| D3 | `make_rolling_windows` e `rolling_origin_eval` sem sobreposição treino x teste; a última janela termina em n |
| D4 | Fórmulas de MAPE e sMAPE corretas; MASE e RMSE não eram usados (o MASE foi introduzido no v2, Fase 3) |
| D5 | Agregação (mediana por janela, depois mediana entre séries) bate com o texto |
| D6 | Mesma perturbação para todos os modelos (gerador por cenário) |
| D7 | Cache GBT do Car Parts vem do NB10 (100 árvores), não do script `src` (30) |
| D8 | HW do Car Parts nunca cai no fallback (0%) |
| D9 | Caches de HW (NB02), GB e SVR do M4 reproduzem exatamente a partir do código |
| D10 | lag_season = t−12 e lag_season2 = t−24 corretos |
| D11 | Nenhum outro casamento por substring ou posição (`FEATURE_NAMES` na ordem de `build_features`; `argsort` da Fig. 6 alinhado) |
| D12 | TreeExplainer usa o modelo da própria série |

## 5. Divergências entre a dissertação (`Dissertacao_PPGI_AutoPred_2.docx`) e os CSVs

Levantadas na auditoria sobre os CSVs da época (protocolo antigo). Os valores v2 correspondentes estão em `CHANGELOG_conclusoes.md` e nas tabelas da Fase 5.

| Local | Texto da dissertação | CSV / código | Causa |
|---|---|---|---|
| Resumo, 5.1, 5.3, conclusões | ΔMAPE em "p.p." (HW +49,2 e +45,7) | +4,07 e +3,78 p.p. absolutos (49,2 e 45,7 são %) | C3 (a ordenação se mantém, a magnitude não) |
| 5.3.1 | GB com taxa de sucesso 0,93; "falha de convergência" | GB 0,848; queda causada por séries curtas | C12 |
| 5.3.5, Figs. 15 e 16, Tabela 4 | Cenário D com ΔMAPE ≈ 0 | o cenário não existia | C7 |
| 4.4, 5.3.4 | Outliers de ± 5σ | média ± 3σ | C16 |
| Tabela 3 | SARIMA heurístico 10,10 | 10,06 | origem não identificada |
| Tabela 3 | MAE e R² vazios (célula com traço) para vários modelos | existiam no cache: SES 253,52 / −1,19; SARIMA auto 266,39 / −1,27; SARIMA heurístico 284,79 / −2,06; GB 321,68 / −1,18; RF 341,67 / −1,09; SVR 371,88 / −1,50 | tabela desatualizada |
| Tabela 3, Fig. 9 | "200 séries" iguais para todos | duas amostras diferentes | C6 |
| 5.2.4 | GB x RF: diferença de ≈ 1,2 p.p. | 0,69 p.p. | erro de texto |
| 5.2.2, Tabela 6 | séries do M4 com 42 a 126 observações | 64 a 702 (NB02); 66 a 918 (NB05 a 09) | erro de texto |
| 5.2.3 (Fig. 11) | ganhos de 1,1 / 50,1 / 19,6 "p.p." | relativos; 0,06 / 8,24 / 1,00 p.p. | C17 |
| 5.4.1 | SVR: lag_season em 6º e lag_season2 em 5º | 8º e 6º | erro de texto |
| 5.4.1 | lag_1 = 325 (RF) e 355 (GB) | 321,17 e 352,46 | arredondamento ou cache anterior |
| 5.4.2, 5.5.3 | "CV do lag_1" = 1,75 / 1,61 / 0,51 | média de 17 features; CV só do lag_1 = 1,27 / 1,13 / 0,40 | rótulo errado (C11) |
| 5.4.3, conclusões, contribuições | lag_season no top-3 do LIME; Pearson 0,959 / 0,952 / 0,908 | 8º a 11º; 0,992 / 0,994 / 0,980 (LIME corrigido) | bug do LIME |
| Tabela 5, 5.5.2 | RF "r = 0,959" | 0,992 | bug do LIME |
| Tabela 6, 5.6 | protocolo "idêntico" ao M4 | RF do Car Parts com 30 árvores | C15 |
| Tabela 7, nota | cobertura do MAPE "por janela" | calculada por série | C20 |

Conferiam com o cache na época: Tabela 7 inteira, Fig. 22 (160→62, 182→98, 0,146→0,093; 38→104 e 51→105), Spearman de estabilidade 0,572 / 0,344 / 0,520, Fig. 12 e as estatísticas do Car Parts (80,4%, 89,1%, 1.383 séries, 15,8%).
