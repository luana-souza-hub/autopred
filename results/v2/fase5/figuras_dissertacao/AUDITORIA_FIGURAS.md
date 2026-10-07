# Auditoria das figuras 5 a 25 da dissertação

Gerado por `tools/figuras_auditoria_v2.py`. Padrão: vírgula decimal, sem título interno, sem "v2", cores por família das Figs. 15 a 25, 300 dpi, sem cortes nem sobreposição, rótulos em português. Regeneradas só as fora do padrão, com os mesmos dados (184 valores conferidos em `conferencia_auditoria.csv`).

| Figura | Fora do padrão | O que foi corrigido |
|---|---|---|
| 5 | sim | Não regerada: diagrama importado no Word (120 dpi), sem código nem dados no projeto; refazê-lo seria criar figura nova. |
| 6 | sim | Não regerada: diagrama importado (220 dpi), sem código no projeto. Rótulos "OE2" e "OE3" sobrepostos à esquerda da Etapa 3. |
| 7 | sim | Não regerada: diagrama importado (220 dpi), sem código no projeto. Texto "origem desloca" sobre a seta entre as janelas 1 e 2. |
| 8, 9 | não | Aprovadas; não alteradas. |
| 10 | sim | Título interno com "v2" removido; vírgula decimal no MAPE da legenda; cores por família (baselines em cinza, distintos por marcador; Holt-Winters em azul); rótulos dos eixos em português; legenda no rodapé; 300 dpi. |
| 11 | sim | Título interno com "v2" removido; vírgula decimal e sinal de menos tipográfico; boxplots horizontais em cinza (baselines); medianas anotadas; eixo limitado a [−200; 100] com a contagem das 16 séries fora da área visível anotada; 300 dpi. |
| 12 | — | Não consta no .docx atual (a numeração passa de 11 para 13). |
| 13 | sim | Título interno com "v2" removido; vírgula decimal e sinal de menos tipográfico nas 81 células; nota do asterisco (IC 95% exclui zero) no rodapé; rótulo da barra de cores "Δ% do MAPE"; 300 dpi. |
| 14 | sim | Título interno com "v2" e nota removidos; vírgula decimal nas 90 células e na barra de cores; texto branco nas células escuras; 300 dpi. |
| 15 a 23, 25 | não | Aprovadas; não alteradas. |
| 24 | sim | Ponto decimal no MAE e no MAPE trocado por vírgula; rótulos dos eixos ("Mês", "Demanda") acrescentados; legenda que cobria os dados movida para o rodapé (MAE e MAPE passam para o título de cada painel); Random Forest em verde; 300 dpi. |
