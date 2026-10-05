# Ficha: 5.2.3 Persistência (posição no M4)

Fonte única dos números: `00_controle/FATOS_CONGELADOS.csv`. Caminhos relativos a `results/v2/` (cópias em `para_escrita/arquivos/`).

| id | fato | valor | IC 95% | origem |
|---|---|---|---|---|
| M4L-06 | M4 limpo: Persistência | MAPE 9,75%; posto MAPE 6º; MAE 332,19; R² −1,29; MASE 0,953; posto MASE 4º [1; 8] | MAPE [8,30; 12,37]; posto [3; 8] | results/v2/fase5/tabela3_antigo_v2_m4.csv |
| M4L-01 | M4 limpo: Holt-Winters | MAPE 8,36%; posto MAPE 1º; MAE 325,29; R² −1,22; MASE 0,918; posto MASE 2º [1; 6] | MAPE [6,19; 10,25]; posto [1; 3] | results/v2/fase5/tabela3_antigo_v2_m4.csv |
| M4L-04 | M4 limpo: SES | MAPE 9,68%; posto MAPE 4º; MAE 328,89; R² −1,06; MASE 0,938; posto MASE 3º [1; 7] | MAPE [8,29; 11,53]; posto [3; 7] | results/v2/fase5/tabela3_antigo_v2_m4.csv |
| M4L-16 | Fig. 11: ganho do Holt-Winters sobre os baselines | Persistência: diferença das medianas 1,39 p.p. (14,3% relativo), mediana por série 8,2%, HW melhor em 54% das séries; Sazonal ingênuo: 3,41 p.p. (29,0%), 17,6%, 63%; Média histórica: 13,47 p.p. (61,7%), 55,4%, 79% |  | results/v2/fase5/figuras/fig11_v2_m4_ganho_hw_dados.csv |
| M4L-17 | Fig. 10: série de exemplo | M17734, origem 51 (M32652 não está na amostra v2) |  | results/v2/fase5/figuras_indice.csv |
| DEF-04 | Agregação | mediana das janelas de cada série, depois mediana entre séries; entram janelas OK e em fallback; saem ERRO e SEM_JANELA |  | src/pipeline_v2.py (agregar_por_serie) |

**Arquivos**

- `fase5/tabela3_antigo_v2_m4.csv`
- `m4/limpo/comparacao_tabela3_antiga_v2_m4.csv`
- `fase5/figuras/fig10_v2_m4_exemplo_M17734.png`
- `fase5/figuras/fig10_v2_m4_exemplo_M17734_dados.csv`
- `fase5/figuras/fig11_v2_m4_ganho_hw.png`
- `fase5/figuras/fig11_v2_m4_ganho_hw_dados.csv`

**Dados ainda sem tabela:** L05 (ver `00_controle/LACUNAS.md`)
