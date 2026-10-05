# Ficha: 5.3.1 Panorama e continuidade operacional (M4)

Fonte única dos números: `00_controle/FATOS_CONGELADOS.csv`. Caminhos relativos a `results/v2/` (cópias em `para_escrita/arquivos/`).

| id | fato | valor | IC 95% | origem |
|---|---|---|---|---|
| PRT-17 | Cenários adversos | 4 tipos, 9 cenários: ruído 10% e 30%; ausências 10%, 20% e 30%; outliers 2% e 5% (média ± 3σ do treino); truncamento 50% e 25% (mínimo 24 obs.) |  | src/pipeline_v2.py (CENARIOS, OUTLIER_K) |
| PRT-18 | Combinações modelo × cenário por base | 81 (9 modelos × 9 cenários); SARIMA automático só na linha de base |  | results/v2/m4/adversos/m4_adversos_delta_ic95.csv |
| PRT-20 | Análise suplementar "alvo ruidoso" | só M4, cenários A, B e C; fora dos rankings |  | results/v2/m4/adversos/m4_adversos_suplementar_alvo_ruidoso.csv |
| DEF-05 | Δ% e Δp.p. | Δ% = 100 × (MAPE mediano do cenário − MAPE mediano da base) ÷ MAPE mediano da base; Δp.p. = diferença; positivo = degradação; não é a mediana dos Δ% por série |  | src/pipeline_v2.py (tabela_delta, linhas 550 a 563) |
| M4A-01 | M4: ICs do Δ% que incluem zero | 66 de 81 (15 excluem zero) |  | results/v2/m4/adversos/m4_adversos_delta_ic95.csv |
| M4A-02 | M4: Holt-Winters, ruído 30% | +17,6% (+1,47 p.p.) | [+2,5; +37,7] | results/v2/m4/adversos/m4_adversos_delta_ic95.csv |
| M4A-03 | M4: Holt-Winters, outliers 2% | +13,1% (+1,09 p.p.) | [+2,5; +38,6] | results/v2/m4/adversos/m4_adversos_delta_ic95.csv |
| M4A-11 | M4: inflação pelo alvo ruidoso (mediana entre modelos) | +15,1 (ruído 30%), +16,2 (outliers 2%), +33,9 (outliers 5%) p.p. de Δ% |  | results/v2/m4/adversos/m4_adversos_suplementar_alvo_ruidoso.csv |
| M4A-12 | SARIMA automático fora dos cenários | 42 min na linha de base do M4; 5 a 6 h estimadas para os 9 cenários |  | results/v2/logs/m4_limpo_sarima_auto.log; HANDOFF.md |
| M4A-13 | Fig. 14: taxa sem fallback | 1,00 para todos, exceto SARIMA heurístico (0,34 a 0,67) e SVR (≥ 0,99) |  | results/v2/fase5/figuras/fig14_v2_m4_taxa_sem_fallback_dados.csv |
| FBK-03 | Fallback do SARIMA heurístico nos cenários do M4 | 33,3% em ruído, ausências e outliers; 52,2% no truncamento 50%; 65,8% no truncamento 25% |  | results/v2/m4/adversos/m4_adversos_status.csv |
| FBK-05 | Fallback do SVR | M4 limpo 0,5%; M4 cenários 0% a 0,8%; Car Parts 13,1%; Tourism 0% |  | tabela3_v2_m4.csv; m4_adversos_status.csv; tabela7_v2_cp.csv; tabela_desempenho_v2_tm.csv |
| FBK-07 | Janelas explosivas do SARIMA heurístico | linha de base: M4 1; Car Parts 60; Tourism 0; Tourism 366 0. Cenários oficiais do M4: 4 (ausências 10% 1, outliers 2% 1, outliers 5% 2); Tourism e Tourism 366: 1 cada (truncamento 25%) |  | results/v2/sarima_limites_fallback_v2.csv; m4_adversos_status.csv |

**Arquivos**

- `m4/adversos/m4_adversos_delta_ic95.csv`
- `m4/adversos/m4_adversos_delta_pct.csv`
- `m4/adversos/m4_adversos_delta_pp.csv`
- `m4/adversos/m4_adversos_status.csv`
- `m4/adversos/m4_adversos_suplementar_alvo_ruidoso.csv`
- `fase5/figuras/fig13_v2_m4_heatmap_delta.png`
- `fase5/figuras/fig13_v2_m4_heatmap_delta_dados.csv`
- `fase5/figuras/fig14_v2_m4_taxa_sem_fallback.png`
- `fase5/figuras/fig14_v2_m4_taxa_sem_fallback_dados.csv`

**Dados ainda sem tabela:** L01, L02 (ver `00_controle/LACUNAS.md`)
