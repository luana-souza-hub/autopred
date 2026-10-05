# 00_controle: controle da reescrita

Montado em 2026-10-05, só com leitura do repositório, dos resultados e do texto da dissertação.

## O que foi lido

- Dissertação: `../dissertacao/Dissertacao_PPGI_AutoPred_2_atualizando.docx` (sha256 `b3474624…`, de 2026-10-04). **O arquivo `Dissertacao_PPGI_AutoPred_2.docx` pedido não existe mais na pasta**; o `.docx` lido na Fase 5 tinha sha256 `bbd41343…` (registrado em `fase5/fase5.meta.json`), portanto o `_atualizando` é uma versão editada depois. Nele, o capítulo 4 (4.1 a 4.7) já está reescrito e os capítulos 5 e 6 ainda estão na versão antiga.
- Seções reescritas fora do `.docx`: `C:\Users\Usuario\Downloads\secao_*.docx` (4.1 a 4.7, 5.1, 5.2.1 em duas versões, 5.2.2). Usadas só para marcar o status na matriz; não foram alteradas.
- `CHANGELOG_conclusoes.md`, `AUDITORIA.md`, `HANDOFF.md`, `PREVISOES_tourism.md`, `INTERPRETACAO_criterios_4c.md`, `fase5/indice_fase5.csv`, `pacote_escrita/LEIAME.md` e os CSVs de `results/v2/` citados em cada fato.

## Arquivos

| Arquivo | Para quê |
|---|---|
| `FATOS_CONGELADOS.csv` | 197 números e definições usados em mais de uma seção, com id, valor, IC, arquivo de origem, seções e status (`confirmado`, `DIVERGENTE`, `não confirmado`). É a única fonte de números do texto |
| `MATRIZ_MUDANCAS.csv` | 119 afirmações, números ou figuras do `.docx` que ficaram inválidos, seção por seção, com o motivo (id da auditoria ou fase do CHANGELOG), o valor novo, o id do fato e o status (`reescrita`, `pendente`, `conferir`, `seção nova`, `pendente (rascunho existe…)`) |
| `FICHAS/<seção>.md` | 23 fichas (4.6, 4.7, 5.1, 5.2.3, 5.2.4, 5.3.1 a 5.3.6, 5.4.1 a 5.4.5, 5.5, 5.6, 5.7, 5.8, cap. 3, cap. 6, resumo), cada uma só com os fatos e os arquivos que a seção usa. `FICHAS/_arquivos_citados.txt` lista os 106 arquivos citados |
| `LACUNAS.md` | Dados que as fichas precisam e ainda não estão em tabela, com a forma de gerá-los dos caches |
| `OBSOLETOS.md` | O que em `results/` (fora de `v2/`) é do protocolo antigo e não deve ser usado |
| `../../../../tools/checar_texto.py` | Confere um `.docx` ou `.md` contra os termos proibidos e os números do FATOS |

## Pontos que dependem de decisão

1. **Denominador das explosivas** (FBK-08): os CSVs usam o total de janelas; a seção 4.5 diz "sobre as janelas OK".
2. **Repositório aberto** (OE5): a seção 4.3 diz "disponíveis mediante solicitação"; 5.5.3, 6.2 e 6.6 falam em repositório aberto.
3. **Status de 4.6, 4.7 e 5.1**: têm rascunho em Downloads (e 4.6 e 4.7 já estão no `.docx`), mas não estavam na lista de seções reescritas. Ficaram como "pendente (rascunho existe; não listada como reescrita)".
4. **Referências cruzadas**: 4.5, 4.7 e o rascunho de 5.1 citam "Tabela 4" e "Tabela 5" para o mapa síntese e as recomendações; no `.docx` atual essas são a configuração dos modelos e os cenários.
5. **Referências ausentes**: Wang, Smith e Hyndman (2006), Athanasopoulos et al. (2011) e Makridakis, Spiliotis e Assimakopoulos (2018) são citados no texto novo e não estão em REFERÊNCIAS.

## Como conferir um texto

```
.venv\Scripts\python.exe tools\checar_texto.py caminho\secao.docx
.venv\Scripts\python.exe tools\checar_texto.py caminho\secao.docx --sem-numeros
```
Números de fora do experimento (por exemplo, 1.045 séries da M3 citadas de Makridakis et al., 2018, ou as bases candidatas da Tabela 2) aparecem como achado; basta ignorá-los ou acrescentá-los ao FATOS com a fonte.
