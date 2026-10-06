# Ficha: 5.3.4 Cenário C: outliers pontuais (M4)

Fonte única dos números: `00_controle/FATOS_CONGELADOS.csv`. Caminhos relativos a `results/v2/` (cópias em `para_escrita/arquivos/`).

| id | fato | valor | IC 95% | origem |
|---|---|---|---|---|
| M4A-16 | M4: outliers 2%, todos os modelos (Δ%) | Sazonal ingênuo +24,0; SARIMA heur. +19,0; GB +15,1; HW +13,1; SVR +8,4; SES +6,6; RF +2,1; Persistência +0,5; Média +0,2 | GB [+1,7; +31,0]; SVR [−7,5; +24,8]; RF [−4,6; +13,3] | results/v2/m4/adversos/m4_adversos_delta_ic95.csv |
| M4A-04 | M4: outliers 5% por modelo | SARIMA heur. +43,1%; HW +32,5%; GB +31,4%; RF +12,7%; SES +9,5%; SVR +8,0% | SARIMA [+13,6; +56,0]; HW [+15,1; +59,7]; GB [+14,7; +43,7]; RF [+3,2; +29,5]; SES [−1,9; +22,2]; SVR [−2,9; +22,5] | results/v2/m4/adversos/m4_adversos_delta_ic95.csv |
| M4A-03 | M4: Holt-Winters, outliers 2% | +13,1% (+1,09 p.p.) | [+2,5; +38,6] | results/v2/m4/adversos/m4_adversos_delta_ic95.csv |
| M4A-10 | M4: outliers, média dos dois níveis | RF 7,4% e SVR 8,2% (ICs sobrepostos) |  | results/v2/m4/adversos/m4_adversos_hierarquia_robustez.csv; m4/adversos/comparacao_tabela5_recomendacoes.csv |
| PRT-17 | Cenários adversos | 4 tipos, 9 cenários: ruído 10% e 30%; ausências 10%, 20% e 30%; outliers 2% e 5% (média ± 3σ do treino); truncamento 50% e 25% (mínimo 24 obs.) |  | src/pipeline_v2.py (CENARIOS, OUTLIER_K) |

**Arquivos**

- `m4/adversos/m4_adversos_delta_ic95.csv`
- `m4/adversos/m4_adversos_pares_ic95.csv`
- `m4/adversos/m4_adversos_hierarquia_robustez.csv`
- `pacote_escrita/00_controle/tabelas/L01_m4_adversos_sarima_variantes.csv`
