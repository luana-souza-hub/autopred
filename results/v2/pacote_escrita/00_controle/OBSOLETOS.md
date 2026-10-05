# Arquivos obsoletos: não usar na escrita

**A fonte de números da dissertação e do artigo é só `results/v2/`** (e, dentro dele, `pacote_escrita/` e `para_escrita/`, que são cópias). Tudo o que está em `results/` fora de `v2/` pertence ao protocolo antigo (notebooks 01 a 11 originais, antes das correções C1 a C20) e não deve ser citado nem usado para conferir valores. Esses arquivos ficam no repositório só como registro do que a dissertação reportava.

| Caminho (em `results/`) | O que é | Por que não usar |
|---|---|---|
| Arquivos soltos na raiz (55 arquivos, de 11 a 23 de maio de 2026): `results_*_by_window.csv`, `results_*_by_series.csv`, `baseline_limpo_200series.csv`, `tabela_acumulada_nb0*.csv`, `tabela_comparativa_baselines_hw.csv`, `comparacao_*.csv`, `*_feature_importance_example.csv`, `sarima_auto_order_log.csv`, `run_log_*.json`, `fig_*.png` | Caches e figuras dos NB01 a NB07 | Duas amostras diferentes do M4 (C6), SARIMA automático com vazamento (C4), features com duplicata (C5), RF que não roda no código salvo (C9) |
| `adversos/` (14 arquivos, maio) | Cenários do NB08 | Previsão recursiva deslocada (C1), teste perturbado (C2), Δ% rotulado como p.p. (C3), cenários D e C 5% nunca gerados (C7), taxa de sucesso por séries curtas (C12) |
| `car_parts/` (19 arquivos, junho) | NB10 | RF com 30 árvores (C15), R² forçado com teste constante (C13), fallback silencioso (C14), cobertura do MAPE rotulada errada (C20), sem SARIMA automático |
| `explicabilidade/` (18 arquivos, maio e outubro) | NB09, inclusive a versão com o LIME corrigido e a sensibilidade do LIME | Mesmo com o LIME corrigido, usa 17 features com duplicata (C5), importâncias sem normalização (C10), SHAP do SVR sem semente (C8) e pontos diferentes para SHAP e LIME |
| `explicabilidade_original_bugado/` (14 arquivos) | Backup do NB09 com o bug de associação do LIME | Bug do LIME (associação por substring) |
| `tourism/` (15 arquivos, outubro) | NB11, Etapas 1 e 1b (protocolo antigo) | Etapa 1b reproduz a explicabilidade antiga do M4. **Exceção:** a caracterização das bases (`caracterizacao_bases.csv`, `piso_ruido_fs.csv`, `caracterizacao_series_*.csv`) não depende do protocolo de modelagem e é a fonte da Tabela 3 da seção 4.1; use as cópias de `caracterizacao_bases.csv` e `piso_ruido_fs.csv` em `pacote_escrita/02_caracterizacao_bases/` (os `caracterizacao_series_*.csv`, por série, só existem em `results/tourism/`) |

Fora de `results/`: `backup_pre_correcoes/` (fora do git) guarda os notebooks, o `src/` e os resultados anteriores às correções, também só como registro. O texto da dissertação deve ser conferido contra `00_controle/FATOS_CONGELADOS.csv` e, na dúvida, contra os CSVs de `results/v2/` citados na coluna `arquivo_origem`.
