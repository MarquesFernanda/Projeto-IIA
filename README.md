# Projeto-IIA

Classificação de asteroides potencialmente perigosos com dados da NASA, comparando KNN, Regressão Logística e redes neurais MLP em PyTorch.

Projeto desenvolvido para a disciplina de **Introdução à Inteligência Artificial — UnB**.

## Dataset

Foi utilizado o [NASA — Nearest Earth Objects](https://www.kaggle.com/datasets/sameepvani/nasa-nearest-earth-objects), com **90.836 registros**.

A variável alvo é `hazardous`:
- **0:** não perigoso.
- **1:** potencialmente perigoso.

## Etapas

- Verificação de valores ausentes e remoção de colunas desnecessárias.
- Criação do diâmetro médio e análise de correlação.
- Balanceamento por subamostragem, resultando em **17.680 registros**.
- Transformação dos atributos com `PowerTransformer`.
- Busca de hiperparâmetros para KNN e Regressão Logística.
- Treinamento de cinco MLPs com early stopping.
- Avaliação com métricas, matrizes de confusão e curvas de perda.

## Tecnologias

Python, pandas, NumPy, Matplotlib, Seaborn, scikit-learn, PyTorch e kagglehub.

## Resultados

| Modelo | Acurácia no teste | F1 no teste |
|---|---:|---:|
| KNN | 0,8719 | 0,8838 |
| Regressão Logística | 0,8462 | 0,8556 |
| MLP 1 | 0,8801 | 0,8918 |
| MLP 2 | 0,8812 | 0,8928 |
| MLP 3 | 0,8801 | 0,8918 |
| MLP 4 | 0,8797 | 0,8913 |
| MLP 5 | 0,8759 | 0,8887 |

A **MLP 2**, com duas camadas ocultas de 64 e 32 neurônios e ativação ReLU, apresentou o maior F1. Aumentar o número de camadas não melhorou os resultados.

Os modelos clássicos e as redes usaram divisões diferentes de teste, o que limita a comparação direta. As métricas foram obtidas com classes balanceadas.

## Como executar

1. Abra `Trabalho_IAA_FernandaMarques.ipynb` no [Google Colab](https://colab.research.google.com/).
2. Execute as células na ordem.
3. O dataset será baixado automaticamente pelo `kagglehub`.

O notebook contém os códigos, gráficos, resultados e respostas dos exercícios.
