# Previsões para o Tourism Monthly (registradas antes da execução)

Registradas em 2026-10-02, **antes** de qualquer execução de modelo, explicação ou cenário no Tourism Monthly com o protocolo v2. Até este registro, do Tourism só foram feitas a carga e a caracterização da base (NB11, Etapa 1: comprimentos, zeros, F_T e F_S). **Este arquivo não deve ser alterado depois do commit em que foi criado**; a avaliação das previsões será feita em outros arquivos (tabela de replicação e CHANGELOG).

## Configuração em que as previsões serão avaliadas

- Base: Tourism Monthly (Monash, Zenodo v3, DOI 10.5281/zenodo.4656096); amostra de 200 séries (sorteio SEED=42 na ordem do arquivo); H = 24; janelas por `linspace(24, n − 24, 3)`.
- Pipeline: `src/pipeline_v2.py` no estado do commit que registra este arquivo; mesmos modelos, hiperparâmetros e regras de fallback do M4 e do Car Parts.
- Explicabilidade (H1 a H3): 30 séries sorteadas com SEED+1 sobre as posições da amostra; RF, GB e SVR treinados na última janela (treino = série[:−24]); 15 features (lag_1 a lag_12, roll_mean_3, roll_std_3, roll_mean_6); SHAP e LIME nos mesmos 30 pontos (29 linhas de treino mais recentes + ponto de previsão); LIME com 200 amostras e associação por índice; importâncias normalizadas por série (soma 1) antes da média; ruído σ = 30% do desvio padrão do próprio treino. A análise com lag_24 e a sensibilidade do LIME **não** entram na avaliação das previsões.
- IC: 95%, bootstrap com 2000 reamostragens sobre as séries.
- A sensibilidade com as 366 séries (parte 4d) será reportada, mas a avaliação das previsões usa as 200 séries.

## Previsões e critérios de avaliação

### H1: concordância SHAP x LIME
**Previsão:** Pearson SHAP x LIME agregado > 0,90; Spearman por série entre 0,65 e 0,85.

- Pearson agregado: correlação entre as importâncias globais (média das importâncias normalizadas por série), cada vetor dividido pelo seu máximo, condição limpa, por modelo.
- Spearman por série: média, entre as 30 séries, do Spearman entre o vetor SHAP e o vetor LIME da mesma série, condição limpa, por modelo.
- São 6 condições (2 partes x 3 modelos). **Confirmou**: as 6 valem. **Confirmou parcialmente**: 3 a 5 valem. **Não confirmou**: 0 a 2 valem.

### H2: estabilidade das explicações sob ruído e redistribuição
**Previsão:** estabilidade limpo x ruído (Spearman) < 0,80 em todos os modelos; lag_1 perde e médias móveis ganham em RF e GB.

- (a) Spearman médio entre os vetores de importância limpo e com ruído da mesma série, por modelo e método: < 0,80. São 6 condições (3 modelos x 2 métodos).
- (b) Em RF e GB, em SHAP e em LIME: a variação da importância normalizada do lag_1 (ruído − limpo) é negativa com IC excluindo zero **e** a de roll_mean_3 ou a de roll_mean_6 é positiva com IC excluindo zero. São 4 condições (2 modelos x 2 métodos), cada uma exigindo as duas partes.
- Total de 10 condições. **Confirmou**: as 10 valem. **Confirmou parcialmente**: 5 a 9 valem. **Não confirmou**: 0 a 4 valem.

### H3: lag sazonal
**Previsão:** lag_12 no top-3 de uma fração maior das séries do que no M4 (17% a 30%).

- Fração das 30 séries em que o lag_12 está entre as 3 features mais importantes da própria série (importância normalizada; empates pelo menor rank), condição limpa, por modelo e método. Referência do M4 (protocolo v2): de 16,7% a 30,0% (`results/v2/m4/xai/xai_m4_posicao_lag12.csv`).
- São 6 condições (3 modelos x 2 métodos), cada uma exigindo fração > 30,0%. **Confirmou**: as 6 valem. **Confirmou parcialmente**: 1 a 5 valem. **Não confirmou**: nenhuma vale.

### H4: hierarquia entre famílias
**Previsão:** a hierarquia entre famílias difere da do M4 e da do Car Parts.

- Métrica principal: **MASE** (comum às três bases; escala pelo erro in-sample do sazonal ingênuo no treino da janela), com fallback incluído, mediana por série e depois entre séries. Rank da família = posição do seu melhor modelo. Secundária, só reportada: MAE.
- Referências (MASE, com fallback; `results/v2/cp/limpo/reversao_familias_m4_cp_mesma_metrica.csv`):
  - M4: ML 1º, Suavização exponencial 2º, Baseline 3º, ARIMA 4º.
  - Car Parts: Baseline 1º, ML 2º, Suavização exponencial 3º, ARIMA 4º.
- **Confirmou**: a ordem das famílias no Tourism é diferente da do M4 **e** da do Car Parts, e, em cada uma das duas comparações, pelo menos uma família tem IC do rank (bootstrap pareado por série dentro de cada base) sem sobreposição com o IC da mesma família na outra base. **Confirmou parcialmente**: a ordem difere das duas bases nas estimativas pontuais, mas a separação de IC ocorre em no máximo uma das comparações. **Não confirmou**: a ordem é idêntica à de pelo menos uma das bases.
