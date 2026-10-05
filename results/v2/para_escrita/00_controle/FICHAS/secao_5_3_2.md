# Ficha: 5.3.2 Cenário A: ruído gaussiano (M4)

Fonte única dos números: `00_controle/FATOS_CONGELADOS.csv`. Caminhos relativos a `results/v2/` (cópias em `para_escrita/arquivos/`).

| id | fato | valor | IC 95% | origem |
|---|---|---|---|---|
| M4A-15 | M4: ruído 10% e 30%, todos os modelos (Δ%) | SES −3,8 / +1,1; RF +1,0 / +2,9; SVR −0,4 / +3,7; GB +1,6 / +11,9; HW −1,3 / +17,6; SARIMA heur. +2,6 / +17,8; Persistência +0,1 / +13,3; Sazonal ingênuo +4,5 / +15,8; Média −0,4 / −0,6 | ruído 30%: ver m4_adversos_delta_ic95.csv | results/v2/m4/adversos/m4_adversos_delta_pct.csv |
| M4A-02 | M4: Holt-Winters, ruído 30% | +17,6% (+1,47 p.p.) | [+2,5; +37,7] | results/v2/m4/adversos/m4_adversos_delta_ic95.csv |
| M4A-07 | M4: SES sob ruído | −3,8% (10%) e +1,1% (30%) |  | results/v2/m4/adversos/m4_adversos_delta_pct.csv |
| PRT-17 | Cenários adversos | 4 tipos, 9 cenários: ruído 10% e 30%; ausências 10%, 20% e 30%; outliers 2% e 5% (média ± 3σ do treino); truncamento 50% e 25% (mínimo 24 obs.) |  | src/pipeline_v2.py (CENARIOS, OUTLIER_K) |
| DEF-05 | Δ% e Δp.p. | Δ% = 100 × (MAPE mediano do cenário − MAPE mediano da base) ÷ MAPE mediano da base; Δp.p. = diferença; positivo = degradação; não é a mediana dos Δ% por série |  | src/pipeline_v2.py (tabela_delta, linhas 550 a 563) |

**Arquivos**

- `m4/adversos/m4_adversos_delta_ic95.csv`
- `m4/adversos/m4_adversos_delta_pct.csv`
- `m4/adversos/m4_adversos_delta_pp.csv`
- `fase5/figuras/fig15_v2_m4_mape_cenarios.png`
- `fase5/figuras/fig15_v2_m4_mape_cenarios_dados.csv`
- `fase5/figuras/fig16_v2_m4_perfil_robustez.png`
- `fase5/figuras/fig16_v2_m4_perfil_robustez_dados.csv`
- `pacote_escrita/00_controle/tabelas/L01_m4_adversos_sarima_variantes.csv`
