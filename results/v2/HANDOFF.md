# HANDOFF: correção do projeto autopred (protocolo v2)

Estado em 2026-10-02, ao fim da Fase 4b. Último commit de resultados: `a320755` (Fase 4b). Pipeline: `src/pipeline_v2.py`, sha256 `bbe78767…` (sem alteração desde `ba7a8d6`); 34 testes em `tests/test_pipeline_v2.py`, todos passando.

Documentos de referência em `results/v2/`: `CHANGELOG_conclusoes.md` (achado da dissertação, valor antigo, valor v2, situação, arquivo-fonte, por fase), `PREVISOES_tourism.md` (previsões registradas, **não alterar**), `tm/replicacao_previsoes_tm.csv` (previsões x resultados) e este arquivo.

---

## 1. Estado de cada fase

| Fase | Conteúdo | Estado | Notebook | Resultados |
|---|---|---|---|---|
| 0 | Verificação ao vivo de C1 e C2; protocolo v2; estimativas | concluída | n/a | n/a |
| 1 | `src/pipeline_v2.py` e testes unitários e de regressão | concluída | n/a | `tests/test_pipeline_v2.py` |
| 2a | M4: linha de base limpa (10 modelos, amostra única) | **concluída** | `02_desempenho_m4_v2.ipynb` | `m4/limpo/` |
| 2b | M4: 9 cenários adversos + suplementar "alvo ruidoso" | **concluída** | `08_cenarios_adversos_v2.ipynb` | `m4/<cenário>/`, `m4/adversos/` |
| 2c | M4: explicabilidade (SHAP e LIME nos mesmos pontos) | **concluída** | `09_explicabilidade_v2.ipynb` | `m4/xai/` |
| 3 | Car Parts: linha de base (2.674 séries) e XAI exploratória | **concluída** | `10_car_parts_pipeline_v2.ipynb`, `10b_explicabilidade_car_parts_v2.ipynb` | `cp/limpo/`, `cp/xai/` |
| 4a | Tourism: linha de base limpa (200 séries, H=24); H4 avaliada | **concluída** | `11a_desempenho_tourism_v2.ipynb` | `tm/limpo/` |
| 4b | Tourism: 9 cenários adversos | **concluída** | `11b_cenarios_adversos_tourism_v2.ipynb` | `tm/<cenário>/`, `tm/adversos/` |
| 4c | Tourism: explicabilidade; avaliar H1 a H3 | **pendente** | `11c_explicabilidade_tourism_v2.ipynb` (a gerar) | `tm/xai/` (a gerar) |
| 4d | Tourism: sensibilidade com as 366 séries | **pendente** | a definir | `tm_366/` (a gerar) |
| 5 | Tabelas comparativas (Tabelas 3 a 7 e figuras, antigo x novo) e tabela de replicação H1 a H4 | **pendente** | a definir | a definir |

Situação das previsões do Tourism (`tm/replicacao_previsoes_tm.csv`): H4 **confirmou parcialmente** (4a); H1, H2 e H3 pendentes (4c).

## 2. Decisões de protocolo já tomadas (não reabrir sem decisão explícita)

| Decisão | Detalhe | Onde está registrada |
|---|---|---|
| Amostra única do M4 | 200 séries dos NB05 a 09 (ordem do arquivo, SEED=42) para todos os modelos | `pipeline_v2.load_m4_piloto` |
| Janelas | `linspace(24, n − H, 3)` nas três bases; H = 18 (M4, Car Parts) e 24 (Tourism) | `make_rolling_windows` |
| Features | 15, sem duplicata: lag_1 a lag_12, roll_mean_3, roll_std_3, roll_mean_6; **lag_season retirado** (= lag_12) e **lag_season2 / lag_24 retirado** do oficial (decisão 1A); lag_24 só em análise suplementar de XAI | `feature_names` |
| Previsão recursiva | o vetor do próximo passo é montado pela mesma função das linhas de treino (corrige C1) | `ml_forecast`, teste C1 |
| SARIMA automático | ordem escolhida **em cada janela, só com o treino** (decisão 2A); ajustes com **verossimilhança degenerada rejeitados** (n efetivo ≤ nº de parâmetros ou llf nulo); se não sobra candidato, fallback registrado. Limitação: na 1ª janela a seleção usa 24 obs | `selecionar_ordem_sarima`, `verossimilhanca_degenerada` |
| SARIMA heurístico | mesma rejeição de ajustes degenerados; por isso a 1ª janela (24 obs) cai sempre no sazonal ingênuo | `model_sarima_heuristic` |
| SARIMA automático fora dos cenários adversos | restrição operacional (M4 42 min e Tourism 33 min só na linha de base; ~5 a 6 h para 9 cenários) | notebooks 08 v2 e 11b |
| Previsões explosivas | status continua OK, com sinalizador (|previsão| > 10x a amplitude do treino); sempre reportar com e sem | `flag_explosiva` |
| Perturbação | **só no treino** de cada janela, com estatísticas do próprio treino e semente por (série, origem, cenário); teste sempre limpo (corrige C2); "alvo ruidoso" só como análise suplementar rotulada | `perturbar`, `avaliar_serie` |
| Truncamento | mantém a fração mais recente do treino (mínimo 24 obs); teste igual ao da linha de base (corrige S1) | `perturbar` |
| Outliers | média ± 3σ do treino, como o NB08 executou (o texto da dissertação diz 5σ, item C16) | `OUTLIER_K` |
| Car Parts sem cenários adversos | decisão 3A: ruído e outliers proporcionais a σ não têm sentido físico em séries com ~80% de zeros | `10_car_parts_pipeline_v2.ipynb`, seção inicial |
| Explicabilidade | RF, GB e SVR treinados na última janela (série[:−H]); **SHAP e LIME nos mesmos pontos** (29 linhas de treino mais recentes + ponto de previsão); LIME por índice (`as_map()[1]`), 200 amostras; SHAP do SVR semeado, nsamples = 2p + 2048, escala original; importâncias **normalizadas por série** antes da média; ruído σ = 30% só no treino; 30 séries sorteadas com SEED+1 | `explicar_serie`, `scripts/run_xai_v2.py` |
| Car Parts na XAI | exploratória; sorteio só entre séries elegíveis (`--elegiveis`: 169 descartadas por alvo constante); 21 linhas de treino, 22 pontos | `cp/xai/*.meta.json` |
| Métrica comum entre bases | MAE e **MASE** (escala = erro in-sample do sazonal ingênuo no treino da janela; NaN se zero) | `escala_mase`, `adicionar_mase` |
| Previsões do Tourism | registradas no commit **`1a08fa1`**, antes de qualquer execução no Tourism; **não alterar** `PREVISOES_tourism.md`; avaliar só pelos critérios ali escritos (200 séries, configuração oficial) | `PREVISOES_tourism.md` |
| Redação | "inversão de hierarquia" só para a queda do HW e para o par Tourism x Car Parts; no restante, "dependência do regime" | `CHANGELOG_conclusoes.md` |

## 3. Próximos passos, em ordem

### 3.1 Fase 4c: explicabilidade do Tourism (avalia H1 a H3)
1. Rodar os caches (cerca de 10 min no total, 10 processos):
   ```
   .venv\Scripts\python.exe scripts\run_xai_v2.py --base tm --condicao limpo
   .venv\Scripts\python.exe scripts\run_xai_v2.py --base tm --condicao ruido30
   .venv\Scripts\python.exe scripts\run_xai_v2.py --base tm --condicao limpo --lag24
   .venv\Scripts\python.exe scripts\run_xai_v2.py --base tm --condicao ruido30 --lag24
   .venv\Scripts\python.exe scripts\run_xai_v2.py --base tm --condicao limpo --metodos lime --lime-samples 1000
   .venv\Scripts\python.exe scripts\run_xai_v2.py --base tm --condicao ruido30 --metodos lime --lime-samples 1000
   .venv\Scripts\python.exe scripts\run_xai_v2.py --base tm --condicao limpo --metodos lime --lime-samples 5000
   .venv\Scripts\python.exe scripts\run_xai_v2.py --base tm --condicao ruido30 --metodos lime --lime-samples 5000
   ```
   No Tourism não é preciso `--elegiveis` (séries sem zeros e longas), o que mantém a mesma regra de sorteio do M4.
2. Gerar e executar o notebook: `python tools\nb_builders\nb09v2.py tm` e depois `run_nb.py notebooks\11c_explicabilidade_tourism_v2.ipynb`.
3. **Acrescentar ao notebook a avaliação de H1, H2 e H3 exatamente pelos critérios de `PREVISOES_tourism.md`** (contagem de condições: H1 6 condições; H2 10; H3 6, com fração do lag_12 no top-3 > 30,0%). O gerador `nb09v2.py` ainda não tem essas células; seguir o modelo da avaliação da H4 em `nb11a_v2.py` e gravar o resultado em `tm/replicacao_previsoes_tm.csv`.
4. Atualizar o CHANGELOG (seção 4c), fazer commit e parar.

### 3.2 Fase 4d: sensibilidade com as 366 séries
- Linha de base e cenários: `scripts\run_v2.py --base tm --n-series 366 ...` grava em `results/v2/tm_366/` (já previsto em `pasta_saida`). SARIMA automático: cerca de 1 h com 10 processos.
- **Cuidado:** `scripts/run_xai_v2.py` grava sempre em `results/v2/<base>/xai/`; rodar a XAI com `--n-series 366` sobrescreveria os nomes de arquivo das 200 séries. Antes, acrescentar um sufixo ou pasta própria para 366.
- Com 366 séries o sorteio vira "todas na ordem do arquivo"; as posições das 30 séries de XAI mudam.
- Reportar como sensibilidade; as previsões continuam avaliadas nas 200 séries.

### 3.3 Fase 5: tabelas comparativas
- Um CSV por tabela da dissertação (Tabelas 3 a 7 e figuras correspondentes), com valor antigo e valor novo; parte já existe (`m4/limpo/comparacao_tabela3_antiga_v2_m4.csv`, `m4/adversos/comparacao_tabela4_robustez.csv`, `m4/adversos/comparacao_tabela5_recomendacoes.csv`, `cp/limpo/comparacao_tabela7_antiga_v2_cp.csv`). Falta a Tabela 6 e consolidar tudo em uma pasta, por exemplo `results/v2/fase5/`.
- Tabela de replicação final H1 a H4 nas três bases (M4, Car Parts, Tourism), com H3 reformulado (importância do lag sazonal em SHAP e LIME) e status "replicou / parcialmente / não replicou" com a evidência numérica.

## 4. Pendências e cuidados

- **SARIMA (heurístico e automático) e SVR: sempre com e sem as janelas de fallback**, e o SARIMA também **sem as janelas explosivas**, em toda tabela em que aparecerem.
- **Baselines triviais separados** (Persistência, Sazonal ingênuo, Média histórica) em todo ranking de robustez; Δ% **com sinal** ao lado do escore em módulo.
- **ICs por bootstrap pareado por série (2000)** em todo ranking e ΔMAPE; marcar comparações cujo IC inclui zero. Com 200 séries, a maioria das diferenças entre modelos ajustados no M4 não é distinguível.
- **Limiares da Tabela 4 e critérios da Tabela 5 foram definidos depois de ver os resultados do M4** (não são pré-registrados); no Tourism foram aplicados sem mudança.
- ΔMAPE sempre em **% relativo e em p.p.**, com rótulo correto (o texto da dissertação chama de p.p. o que é %; item C3).
- Q1 de consistência (CV entre séries) **não foi reavaliado** no v2 (CHANGELOG, Fase 2c).
- `PREVISOES_tourism.md` não pode ser editado. Qualquer critério novo vai em outro arquivo, rotulado como análise posterior.
- O `.docx` da dissertação **não** deve ser alterado.
- Tarefas em segundo plano têm limite de 2 h: para execuções longas, usar `scripts/run_v2.py`, que retoma do `.parcial.csv` (como foi feito com o SARIMA automático do Car Parts).
- Para comparar bases, usar a mesma métrica nas duas (MAE ou MASE); o MAPE é indefinido em 16% das janelas do Car Parts.

## 5. Como retomar

Ambiente (Windows, na raiz `autopred/`):
```
.venv\Scripts\python.exe -m unittest tests.test_pipeline_v2          # 34 testes, cerca de 90 s
git status                                                          # deve estar limpo
git log --oneline | head                                            # último commit de resultados: a320755
```

Execução (fora do Jupyter, com cache, checkpoint e `.meta.json` com hash do pipeline, versões, sementes e SHA-256 dos dados):
```
.venv\Scripts\python.exe scripts\run_v2.py --base {m4|cp|tm} --etapa {limpo|<cenário>} --modelos {rapidos|todos|lista} [--alvo-ruidoso] [--n-series N] [--n-jobs 10]
.venv\Scripts\python.exe scripts\run_xai_v2.py --base {m4|cp|tm} --condicao {limpo|ruido30} [--metodos shap,lime] [--lime-samples N] [--lag24] [--elegiveis]
```
- Um cache que já existe não é recalculado (use `--force` para refazer).
- Execução interrompida: rodar o mesmo comando de novo retoma do `<arquivo>.parcial.csv` (ignorado pelo git).
- Logs em `results/v2/logs/`.

Notebooks: os geradores e o executor estão em `tools/nb_builders/` (ver o `README.md` dessa pasta). Fluxo: `python tools\nb_builders\<gerador>.py` reescreve o notebook, depois `python tools\nb_builders\run_nb.py notebooks\<notebook>.ipynb 3600` o executa e grava as saídas.

Dados (fora do git, em `data/raw/`): `Monthly-train.csv` (M4), `car_parts/car_parts_dataset_without_missing_values.tsf`, `tourism_monthly_dataset.tsf`. O SHA-256 de cada um está nos `.meta.json`.

Backup anterior às correções: `backup_pre_correcoes/` (fora do git).
