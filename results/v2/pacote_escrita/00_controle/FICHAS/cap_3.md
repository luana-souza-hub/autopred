# Ficha: Capítulo 3 Definições operacionais

Fonte única dos números: `00_controle/FATOS_CONGELADOS.csv`. Caminhos relativos a `results/v2/` (cópias em `para_escrita/arquivos/`).

| id | fato | valor | IC 95% | origem |
|---|---|---|---|---|
| PRT-17 | Cenários adversos | 4 tipos, 9 cenários: ruído 10% e 30%; ausências 10%, 20% e 30%; outliers 2% e 5% (média ± 3σ do treino); truncamento 50% e 25% (mínimo 24 obs.) |  | src/pipeline_v2.py (CENARIOS, OUTLIER_K) |
| PRT-19 | Ausências: regra de remoção e preenchimento | piso(fração × n), mínimo 1; na 1ª janela 2, 4 e 7 pontos; interpolação linear; extremidades recebem o valor válido mais próximo |  | src/pipeline_v2.py (perturbar, linhas 415 a 418) |
| FBK-01 | Motivos de fallback ocorridos | só treino constante e verossimilhança degenerada |  | results/v2/sarima_limites_fallback_v2.csv |
| DEF-13 | Estabilidade das explicações | Spearman por série entre limpo e ruído 30%; Jaccard top-3 e top-5; média entre séries |  | src/pipeline_v2.py (estabilidade) |
| M4A-12 | SARIMA automático fora dos cenários | 42 min na linha de base do M4; 5 a 6 h estimadas para os 9 cenários |  | results/v2/logs/m4_limpo_sarima_auto.log; HANDOFF.md |
| MOD-09 | Limites de tempo nunca atingidos | maior seleção do SARIMA automático 92,87 s (limite 300 s); maior ajuste do heurístico 2,21 s (limite 60 s) |  | results/v2/sarima_limites_fallback_v2.csv |

**Arquivos**

- `hiperparametros_v2.csv`
