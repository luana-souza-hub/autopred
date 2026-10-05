# Ficha: 5.3.5 Cenário D: truncamento e restrição operacional (M4)

Fonte única dos números: `00_controle/FATOS_CONGELADOS.csv`. Caminhos relativos a `results/v2/` (cópias em `para_escrita/arquivos/`).

| id | fato | valor | IC 95% | origem |
|---|---|---|---|---|
| M4A-09 | M4: truncamento (50/25%) | HW +4,2 / +9,1%; SARIMA heur. +8,1 / +15,9%; GB +7,5 / +1,8%; média histórica −19,8 / −31,5% (IC exclui zero); demais ≈ 0 | média [−36,4; −13,3] e [−46,3; −20,0] | results/v2/m4/adversos/m4_adversos_delta_ic95.csv |
| M4A-12 | SARIMA automático fora dos cenários | 42 min na linha de base do M4; 5 a 6 h estimadas para os 9 cenários |  | results/v2/logs/m4_limpo_sarima_auto.log; HANDOFF.md |
| FBK-03 | Fallback do SARIMA heurístico nos cenários do M4 | 33,3% em ruído, ausências e outliers; 52,2% no truncamento 50%; 65,8% no truncamento 25% |  | results/v2/m4/adversos/m4_adversos_status.csv |
| MOD-05 | SARIMA automático | grade p, q ∈ {0,1,2}, d ∈ {0,1}, P, D, Q ∈ {0,1}; 140 candidatos; menor AIC entre não degenerados; seleção por janela só com o treino; limite de 300 s por janela |  | results/v2/hiperparametros_v2.csv |
| MOD-09 | Limites de tempo nunca atingidos | maior seleção do SARIMA automático 92,87 s (limite 300 s); maior ajuste do heurístico 2,21 s (limite 60 s) |  | results/v2/sarima_limites_fallback_v2.csv |
| CUS-01 | Tempo de ajuste mediano por janela (M4) | SES 0,006 s; SVR 0,012 s; GB 0,069 s; SARIMA heur. 0,146 s; HW 0,306 s; RF 0,334 s; SARIMA automático 16,1 s |  | results/v2/fase5/custo_computacional_v2.csv |
| CUS-02 | Tempo de parede da linha de base (10 modelos) | M4 44,2 min (42,1 do SARIMA automático); Car Parts 166,1 min (158,0); Tourism 34,2 min (33,2) |  | results/v2/fase5/tabela6_antigo_v2.csv |
| CUS-03 | Ressalva dos tempos | dependem da máquina e do paralelismo; n_jobs do SARIMA automático no Car Parts e no Tourism não registrado |  | results/v2/HANDOFF.md |

**Arquivos**

- `m4/adversos/m4_adversos_delta_ic95.csv`
- `m4/adversos/m4_adversos_status.csv`
- `fase5/custo_computacional_v2.csv`
- `fase5/tabela6_antigo_v2.csv`
- `sarima_limites_fallback_v2.csv`
- `pacote_escrita/00_controle/tabelas/L01_m4_adversos_sarima_variantes.csv`
