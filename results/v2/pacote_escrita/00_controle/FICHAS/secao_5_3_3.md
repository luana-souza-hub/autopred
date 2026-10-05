# Ficha: 5.3.3 Cenário B: ausências aleatórias (M4)

Fonte única dos números: `00_controle/FATOS_CONGELADOS.csv`. Caminhos relativos a `results/v2/` (cópias em `para_escrita/arquivos/`).

| id | fato | valor | IC 95% | origem |
|---|---|---|---|---|
| M4A-08 | M4: ausências (10/20/30%) | GB +0,6 / +1,8 / +5,2%; Persistência 0,0 / +1,2 / +2,5%; RF −6,4 / +2,2 / +3,9%; HW −0,3 / +4,7 / +11,8%; ICs incluem zero |  | results/v2/m4/adversos/m4_adversos_delta_pct.csv |
| PRT-19 | Ausências: regra de remoção e preenchimento | piso(fração × n), mínimo 1; na 1ª janela 2, 4 e 7 pontos; interpolação linear; extremidades recebem o valor válido mais próximo |  | src/pipeline_v2.py (perturbar, linhas 415 a 418) |
| M4A-01 | M4: ICs do Δ% que incluem zero | 66 de 81 (15 excluem zero) |  | results/v2/m4/adversos/m4_adversos_delta_ic95.csv |
| DEF-05 | Δ% e Δp.p. | Δ% = 100 × (MAPE mediano do cenário − MAPE mediano da base) ÷ MAPE mediano da base; Δp.p. = diferença; positivo = degradação; não é a mediana dos Δ% por série |  | src/pipeline_v2.py (tabela_delta, linhas 550 a 563) |

**Arquivos**

- `m4/adversos/m4_adversos_delta_ic95.csv`
- `m4/adversos/m4_adversos_delta_pct.csv`
- `m4/adversos/comparacao_delta_nb08_v2.csv`
- `fase5/figuras/fig16_v2_m4_perfil_robustez_dados.csv`

**Dados ainda sem tabela:** L01 (ver `00_controle/LACUNAS.md`)
