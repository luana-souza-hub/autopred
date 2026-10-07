# Figuras corrigidas (só apresentação): o que mudou

Gerado por `tools/figuras_apresentacao_v2.py` em 2026-10-07. Nenhum dado, cálculo, amostra ou semente mudou: os valores vêm dos CSVs congelados em `fase5/figuras/*_dados.csv` (e, para os boxplots por série da Fig. 15 e do Car Parts, dos caches by_window com a mesma agregação da Fase 5). Os valores anotados ou desenhados nas figuras (126, com as Figs. 18 e 25) foram conferidos contra os de antes e são idênticos (`conferencia_valores_anotados.csv`).

Regras aplicadas a todas: Arial, mesmos tamanhos de fonte das Figs. 8 e 9; vírgula decimal, ponto de milhar e sinal de menos tipográfico (−); sem título interno e sem "v2"; cores por família iguais às das Figs. 8 e 9 (suavização exponencial azul, ARIMA laranja, aprendizado de máquina verde, baseline cinza); 300 dpi, 15 cm de largura; mesmos nomes de arquivo de `fase5/figuras/` (os originais lá não foram alterados).

| Figura (numeração atual) | Arquivo | O que mudou |
|---|---|---|
| 15 | `fig15_v2_m4_mape_cenarios.png` | título interno removido; painéis 2 × 2 com espaçamento refeito; boxplots horizontais por família com legenda de família; medianas com vírgula |
| 16 | `fig16_v2_m4_perfil_robustez.png` | eixo y "Δ% do MAPE"; título interno removido; linhas na cor da família (marcador distingue modelos da mesma família; baselines tracejados); legenda abaixo dos painéis |
| 17 | `fig17_v2_m4_top10_shap.png` | margem esquerda maior (rótulos do eixo y não encostam nos painéis); título interno removido; barras em verde (aprendizado de máquina), lag_12 em verde escuro com legenda |
| 18 | `fig18_v2_m4_variabilidade_shap.png` | refeita (2026-10-07): boxplots da importância SHAP normalizada por série das 8 características mais importantes de cada modelo, um painel por modelo com eixo y próprio; CV médio normalizado (15 características) no título de cada painel (1,21 / 1,02 / 0,70; o do GB é 1,01505, que arredonda para 1,02); verde (aprendizado de máquina), lag_12 em verde escuro; sem título interno e sem "v2". Valores conferidos: CV igual a `fase5/reanalise_cv_q1_m4.csv` e médias iguais às da Fig. 17 |
| 19 | `fig19_v2_m4_top10_lime.png` | mesmas mudanças da Fig. 17 |
| 20 | `fig20_v2_m4_shap_x_lime.png` | título interno ("30 pontos") removido; anotação "15 características" em cada painel; r com vírgula; só lag_1 e roll_mean_3 rotulados (os demais rótulos se sobrepunham); lag_12 pela cor, com legenda |
| 21 | `fig21_v2_m4_estabilidade.png` | vírgula decimal; título com "v2" removido; boxplots horizontais em verde |
| 22 | `fig22_v2_m4_shap_limpo_x_ruido.png` | vírgula decimal; título com "v2" removido; "sem ruído" em cinza e "ruído σ = 30%" em verde |
| 25 (Tourism) | `fig25_v2_tm_redistribuicao_ruido.png` (e cópia com o nome original `fig_v2_tm_redistribuicao_ruido.png`; o arquivo em `tm/xai/` não foi alterado) | estilo da Fig. 22: cinza = sem ruído, verde = ruído σ = 30%, legenda no rodapé; título interno com "v2" removido; nomes "Random Forest" e "Gradient Boosting"; vírgula decimal; eixo y "SHAP normalizado" / "LIME normalizado". Mesmas 4 características, mesma ordem de painéis e eixo y comum, como antes. As 48 alturas das barras conferem com `tm/xai/xai_tm_redistribuicao_ruido.csv` e com as importâncias recalculadas dos caches de pontos |
| Car Parts, MAE (atual Fig. 23) | `fig24_v2_cp_mae.png` | cores por família corrigidas (aprendizado de máquina passa de roxo a verde), legenda de família; "p95 (1,94)"; boxplots horizontais |
| Car Parts, MAPE | `fig23_v2_cp_mape.png` | cores por família e legenda; eixo truncado no p95 (116,0%) e indicado no rótulo, porque o MAPE por série do SARIMA automático passa de 100.000% e achatava todas as caixas (truncamento só visual) |

## Pontos para conferir na dissertação (não alterados aqui)

- **Fig. 19:** a legenda atual diz "CV médio da importância SHAP entre as 30 séries", mas a imagem é o top-10 do LIME. Legenda e imagem não correspondem.
- **Fig. 20:** a legenda atual traz os r antigos (0,959; 0,952; 0,908); a figura mostra os do v2 (0,997; 0,996; 0,993).
- **Boxplot do MAPE no Car Parts:** não tem legenda no `.docx` atual.
- As Figs. 10, 11, 13, 14 e 18 não estavam na lista e continuam com ponto decimal e/ou "v2" no título interno.
