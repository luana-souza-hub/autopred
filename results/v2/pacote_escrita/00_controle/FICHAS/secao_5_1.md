# Ficha: 5.1 Visão geral do protocolo e organização da análise

Fonte única dos números: `00_controle/FATOS_CONGELADOS.csv`. Caminhos relativos a `results/v2/` (cópias em `para_escrita/arquivos/`).

| id | fato | valor | IC 95% | origem |
|---|---|---|---|---|
| BAS-01 | M4 mensal: séries no arquivo e amostra | 48.000; amostra de 200 (semente 42, ordem do arquivo) |  | results/v2/pacote_escrita/02_caracterizacao_bases/caracterizacao_bases.csv |
| BAS-03 | Car Parts: séries, comprimento e período | 2.674 séries; 51 obs.; jan/1998 a mar/2002 |  | results/v2/pacote_escrita/02_caracterizacao_bases/caracterizacao_bases.csv; data/raw (.tsf) |
| BAS-05 | Tourism: séries, comprimento e período | 366 séries (200 oficiais); 91 / 332 / 333 obs. (200) e 91 / 330 / 333 (366); início de jan/1979 a jan/2000; fim mais recente set/2007 |  | results/v2/pacote_escrita/02_caracterizacao_bases/caracterizacao_bases.csv; data/raw (.tsf) |
| BAS-10 | Papel de cada base | M4: desenvolvimento (origem dos achados, hipóteses e limiares); Car Parts: exploratória; Tourism 200: replicação com previsões pré-registradas; Tourism 366: sensibilidade |  | Tabela 8 (4.7) do .docx; results/v2/HANDOFF.md |
| PRT-01 | Semente global (SEED) | 42 |  | src/pipeline_v2.py |
| PRT-04 | Número de origens (N_ORIGINS) e posição | 3 origens em linspace(24, n − H, 3); janela expansiva |  | src/pipeline_v2.py |
| PRT-05 | Horizonte H por base | M4 18; Car Parts 18; Tourism 24 |  | scripts/run_v2.py |
| PRT-12 | Bootstrap dos IC | 2.000 reamostragens; semente 42; percentis 2,5 e 97,5; pareado por série; só séries com valor em todos os modelos |  | src/pipeline_v2.py (bootstrap_delta, bootstrap_ranks) |
| DEF-04 | Agregação | mediana das janelas de cada série, depois mediana entre séries; entram janelas OK e em fallback; saem ERRO e SEM_JANELA |  | src/pipeline_v2.py (agregar_por_serie) |
| DEF-05 | Δ% e Δp.p. | Δ% = 100 × (MAPE mediano do cenário − MAPE mediano da base) ÷ MAPE mediano da base; Δp.p. = diferença; positivo = degradação; não é a mediana dos Δ% por série |  | src/pipeline_v2.py (tabela_delta, linhas 550 a 563) |
| DEF-11 | Métrica principal por base | M4: MAPE (MASE ao lado); Car Parts: MAE (MASE ao lado); Tourism: MASE na H4, MAPE e MAE em paralelo; entre bases: MAE ou MASE |  | results/v2/HANDOFF.md |
| FBK-09 | Critério de previsão explosiva | max\|previsão\| > 10 × max\|treino\|; só nos dois SARIMA e só em janelas OK; status continua OK |  | src/pipeline_v2.py (_diagnostico_previsao) |
| M4A-01 | M4: ICs do Δ% que incluem zero | 66 de 81 (15 excluem zero) |  | results/v2/m4/adversos/m4_adversos_delta_ic95.csv |
| LIM-01 | Categorias de robustez (mapa síntese) | sobre a média do Δ% de cada tipo: Alta −2 < Δ% < 2; Moderada 2 ≤ Δ% < 10; Baixa Δ% ≥ 10; Melhora* Δ% ≤ −2 |  | tools/nb_builders/nb08v2.py:226 a 246 |
| LIM-02 | Critérios das recomendações | definidos depois do M4; os das recomendações 5 (restrição operacional) e 6 (explicabilidade) depois do Tourism |  | results/v2/fase5/tabela5_antigo_v2.csv |
| LIM-04 | Leitura pós-hoc da redistribuição | "a característica dominante perde importância e as médias móveis ganham": formulada depois do Tourism, não registrada, não altera a H2 |  | results/v2/CHANGELOG_conclusoes.md (4c) |
