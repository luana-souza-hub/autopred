# Ficha: 5.3.6 Hierarquia de robustez (M4)

Fonte única dos números: `00_controle/FATOS_CONGELADOS.csv`. Caminhos relativos a `results/v2/` (cópias em `para_escrita/arquivos/`).

| id | fato | valor | IC 95% | origem |
|---|---|---|---|---|
| DEF-06 | Escore de robustez | média simples de \|Δ%\| nos 9 cenários (cada nível conta: ausências pesam 3, os demais tipos 2); janelas OK e em fallback, sem excluir explosivas; ranking separado para 6 ajustados e 3 triviais; SARIMA automático fora; sem IC |  | tools/nb_builders/nb08v2.py:152; nb11b_v2.py:152 |
| M4A-05 | M4: escore de robustez, ajustados | SES 3,0 < RF 3,6 < SVR 4,1 < GB 8,5 < HW 10,5 < SARIMA heur. 12,9 | sem IC | results/v2/m4/adversos/m4_adversos_hierarquia_robustez.csv |
| M4A-06 | M4: escore de robustez, triviais | Persistência 2,1 < Média histórica 6,3 < Sazonal ingênuo 7,7 | sem IC | results/v2/m4/adversos/m4_adversos_hierarquia_robustez.csv |
| M4A-01 | M4: ICs do Δ% que incluem zero | 66 de 81 (15 excluem zero) |  | results/v2/m4/adversos/m4_adversos_delta_ic95.csv |
| M4A-10 | M4: outliers, média dos dois níveis | RF 7,4% e SVR 8,2% (ICs sobrepostos) |  | results/v2/m4/adversos/m4_adversos_hierarquia_robustez.csv; m4/adversos/comparacao_tabela5_recomendacoes.csv |
| LIM-01 | Categorias de robustez (mapa síntese) | sobre a média do Δ% de cada tipo: Alta −2 < Δ% < 2; Moderada 2 ≤ Δ% < 10; Baixa Δ% ≥ 10; Melhora* Δ% ≤ −2 |  | tools/nb_builders/nb08v2.py:226 a 246 |

**Arquivos**

- `m4/adversos/m4_adversos_hierarquia_robustez.csv`
- `m4/adversos/comparacao_tabela4_robustez.csv`
- `m4/adversos/m4_adversos_pares_ic95.csv`

**Dados ainda sem tabela:** L01, L11 (ver `00_controle/LACUNAS.md`)
