# Projeto-IIA

Classificação de asteroides potencialmente perigosos com dados da NASA. O projeto inclui preparação dos dados, análise exploratória e comparação entre KNN, Regressão Logística e cinco redes neurais MLP implementadas em PyTorch.

Trabalho desenvolvido para a disciplina de **Introdução à Inteligência Artificial**, da **Universidade de Brasília — UnB**.

## Autoria

**Fernanda Marques Pereira**

## Objetivo

Construir modelos capazes de prever a classificação `hazardous` de um asteroide a partir de características físicas e dos registros de aproximação com a Terra.

O projeto percorre as etapas de:

- Inspeção e limpeza dos dados.
- Criação e seleção de atributos.
- Análise de correlação e distribuição das variáveis.
- Balanceamento das classes.
- Transformação dos dados.
- Busca de hiperparâmetros dos modelos clássicos.
- Treinamento de redes neurais com early stopping.
- Avaliação e comparação dos resultados.

> A classificação “potencialmente perigoso” não significa que uma colisão esteja prevista. O projeto reproduz os rótulos do dataset e tem finalidade acadêmica.

## Conjunto de dados

Foi utilizado o [NASA — Nearest Earth Objects](https://www.kaggle.com/datasets/sameepvani/nasa-nearest-earth-objects), disponibilizado no Kaggle e baseado em registros do catálogo NASA/JPL.

O conjunto carregado possui:

| Característica | Quantidade |
|---|---:|
| Registros de aproximação | 90.836 |
| Asteroides distintos | 27.423 |
| Colunas originais | 10 |
| Registros não perigosos | 81.996 |
| Registros potencialmente perigosos | 8.840 |

Um mesmo asteroide pode aparecer em diferentes registros de aproximação. Portanto, a repetição de seu identificador não foi tratada automaticamente como uma duplicação indevida.

### Variável alvo

A coluna `hazardous` foi convertida para números inteiros:

| Valor | Classe |
|---|---|
| `0` | Não perigoso |
| `1` | Potencialmente perigoso |

### Atributos usados nos modelos

| Atributo | Descrição |
|---|---|
| `relative_velocity` | Velocidade relativa no encontro, em km/h |
| `miss_distance` | Distância de passagem, em km |
| `absolute_magnitude` | Magnitude absoluta do objeto |
| `diameter_mean` | Média das estimativas mínima e máxima do diâmetro, em km |

## Tecnologias utilizadas

- **Python**
- **Google Colab**
- **pandas** — leitura e manipulação dos dados.
- **NumPy** — operações numéricas.
- **Matplotlib** e **Seaborn** — gráficos e matrizes de confusão.
- **scikit-learn** — divisão dos dados, transformação, modelos clássicos e métricas.
- **PyTorch** — implementação e treinamento das redes neurais.
- **kagglehub** — download do dataset.

Também foram utilizados os módulos `os` e `copy` da biblioteca padrão do Python.

## Etapas do projeto

### 1. Inspeção e preparação dos dados

A inspeção inicial verificou os tipos das colunas, a presença de valores ausentes, a quantidade de valores distintos e a distribuição das classes.

**Não foram encontrados valores nulos.**

As seguintes colunas foram removidas:

| Coluna | Motivo |
|---|---|
| `id` | Identificador catalográfico |
| `name` | Nome ou identificação do asteroide |
| `orbiting_body` | Valor constante nesta base |
| `sentry_object` | Valor constante nesta base; sua retirada também segue a orientação do trabalho sobre prevenção de vazamento de informação |

As colunas `est_diameter_min` e `est_diameter_max` apresentaram correlação linear de **1,0**. Elas foram combinadas em um único atributo:

```python
diameter_mean = (est_diameter_min + est_diameter_max) / 2
```

Após essa combinação, as duas colunas originais foram removidas.

### 2. Análise de correlação

Foi gerada uma matriz de correlação entre os atributos selecionados e a variável alvo.

Alguns resultados observados:

- **Magnitude absoluta e diâmetro médio: −0,56.** Magnitudes menores tendem a acompanhar diâmetros maiores.
- **Magnitude absoluta e classe perigosa: −0,37.** Nesta base, os registros perigosos tendem a apresentar magnitudes menores.
- **Distância de passagem e classe perigosa: 0,04.** A associação linear isolada é baixa, mas isso não descarta relações não lineares ou interações com outros atributos.

### 3. Balanceamento das classes

A distribuição original apresentava aproximadamente:

- **90,3%** de registros não perigosos.
- **9,7%** de registros potencialmente perigosos.

Foi aplicada subamostragem aleatória à classe majoritária, conforme solicitado no trabalho.

Todos os **8.840 registros perigosos** foram mantidos, e **8.840 registros não perigosos** foram sorteados com `random_state=42`.

O conjunto balanceado ficou com:

| Classe | Registros |
|---|---:|
| Não perigoso | 8.840 |
| Potencialmente perigoso | 8.840 |
| **Total** | **17.680** |

Essa etapa reduz o número de exemplos da classe majoritária e o custo dos experimentos, mas também descarta parte dos dados disponíveis.

### 4. Transformação dos atributos

Os histogramas e as estatísticas de treino mostraram diferenças de escala entre os atributos e forte assimetria no diâmetro médio.

Foi utilizado o **PowerTransformer com Yeo-Johnson**, com padronização habilitada.

No conjunto de treino do Exercício 2, foram observadas as seguintes mudanças:

| Atributo | Assimetria antes | Assimetria depois |
|---|---:|---:|
| `relative_velocity` | 0,8457 | −0,0086 |
| `miss_distance` | −0,0638 | −0,3144 |
| `absolute_magnitude` | 0,4940 | 0,0018 |
| `diameter_mean` | 6,2705 | 0,2163 |

A maior redução ocorreu no diâmetro médio. A distância de passagem já apresentava pouca assimetria antes da transformação.

O transformador foi ajustado apenas no conjunto de treino de cada experimento. Validação e teste receberam a transformação já aprendida.

## Modelos clássicos

### Divisão dos dados

Para KNN e Regressão Logística, o conjunto balanceado foi dividido de forma estratificada:

| Conjunto | Proporção | Registros |
|---|---:|---:|
| Treino | 80% | 14.144 |
| Teste | 20% | 3.536 |

Foi utilizado `random_state=42`.

### Busca de hiperparâmetros

A busca foi feita com `GridSearchCV`, utilizando:

- **5 divisões de validação cruzada**.
- **F1 da classe positiva** como critério de seleção.
- Reajuste automático do melhor modelo em todo o conjunto de treino.

#### KNN

Parâmetros avaliados:

```python
{
    "n_neighbors": [3, 5, 7, 9, 11, 15],
    "weights": ["uniform", "distance"],
    "metric": ["euclidean", "manhattan"]
}
```

Foram testadas **24 combinações**, totalizando **120 ajustes** na validação cruzada.

Melhor configuração:

```python
{
    "n_neighbors": 15,
    "weights": "uniform",
    "metric": "manhattan"
}
```

F1 médio na validação cruzada: **0,8832**.

#### Regressão Logística

Parâmetros avaliados:

```python
{
    "C": [0.01, 0.1, 1, 10, 100],
    "penalty": ["l1", "l2"],
    "solver": ["liblinear"]
}
```

Foram testadas **10 combinações**, totalizando **50 ajustes** na validação cruzada.

Melhor configuração:

```python
{
    "C": 0.01,
    "penalty": "l1",
    "solver": "liblinear"
}
```

F1 médio na validação cruzada: **0,8545**.

### Resultados dos modelos clássicos

| Modelo | Acurácia — treino | F1 — treino | Acurácia — teste | F1 — teste |
|---|---:|---:|---:|---:|
| KNN | 0,8790 | 0,8907 | 0,8719 | 0,8838 |
| Regressão Logística | 0,8437 | 0,8533 | 0,8462 | 0,8556 |

O **KNN** apresentou o melhor resultado no teste entre os modelos clássicos.

As diferenças pequenas entre treino e teste não indicaram overfitting forte nas métricas avaliadas.

## Redes neurais MLP com PyTorch

### Divisão dos dados

Para as redes neurais, foi realizada uma nova divisão estratificada:

| Conjunto | Proporção | Registros |
|---|---:|---:|
| Treino | 70% | 12.376 |
| Validação | 15% | 2.652 |
| Teste | 15% | 2.652 |

Um novo `PowerTransformer` foi ajustado exclusivamente nesse conjunto de treino.

Os dados foram convertidos para tensores `torch.float32` e organizados em `TensorDataset` e `DataLoader`.

### Configuração de treinamento

Todas as redes utilizaram:

| Configuração | Valor |
|---|---|
| Otimizador | Adam |
| Taxa de aprendizado | 0,001 |
| Função de perda | `BCEWithLogitsLoss` |
| Tamanho do lote de treino | 128 |
| Tamanho do lote de validação e teste | 256 |
| Limite de épocas | 200 |
| Paciência do early stopping | 15 épocas |
| Melhora mínima para reiniciar a paciência | 0,0001 |
| Limiar de classificação | 0,5 |

A camada de saída produz um único valor sem ativação sigmoide explícita durante o treinamento, pois `BCEWithLogitsLoss` já incorpora essa operação.

Na previsão, a sigmoide converte a saída em um valor entre 0 e 1, e o limiar de 0,5 define a classe.

### Early stopping

O treinamento foi interrompido quando a perda de validação deixou de apresentar melhora suficiente durante 15 épocas consecutivas.

Os pesos da menor perda de validação observada foram armazenados e recuperados antes da avaliação final.

Assim, a **época de parada** pode ser diferente da **melhor época**.

### Arquiteturas e resultados

O número de camadas na tabela corresponde às **camadas ocultas**.

| Arquitetura | Camadas ocultas | Neurônios | Ativação | Época de parada | Melhor época | Acurácia — teste | F1 — teste |
|---|---:|---|---|---:|---:|---:|---:|
| MLP 1 | 1 | [32] | ReLU | 110 | 95 | 0,8801 | 0,8918 |
| MLP 2 | 2 | [64, 32] | ReLU | 33 | 18 | 0,8812 | 0,8928 |
| MLP 3 | 3 | [128, 64, 32] | Tanh | 44 | 29 | 0,8801 | 0,8918 |
| MLP 4 | 4 | [128, 64, 32, 16] | ReLU | 49 | 34 | 0,8797 | 0,8913 |
| MLP 5 | 5 | [128, 64, 32, 16, 8] | Sigmoid | 86 | 83 | 0,8759 | 0,8887 |

O early stopping foi ativado nas cinco arquiteturas, com média de **64,4 épocas** até a parada.

A **MLP 2** apresentou:

- Menor perda de validação: aproximadamente **0,2873**.
- Melhor acurácia de teste entre as redes: **0,8812**.
- Melhor F1 de teste entre as redes: **0,8928**.

As diferenças entre as redes foram pequenas. Aumentar a profundidade não trouxe melhora nos experimentos realizados.

### Matriz de confusão da MLP 2

| Classe real / Classe prevista | Não perigoso | Perigoso |
|---|---:|---:|
| Não perigoso | 1.025 | 301 |
| Perigoso | 14 | 1.312 |

A MLP 2 identificou **1.312 dos 1.326 registros perigosos** do teste, com recall de aproximadamente **98,94%**.

Também classificou **301 registros não perigosos como perigosos**, mostrando a importância de analisar os tipos de erro além da acurácia.

### Curvas de aprendizado

O notebook apresenta curvas de perda de treino e validação para cada arquitetura.

As curvas diminuíram no início e depois ficaram próximas e estáveis, sem sinal forte de overfitting nos experimentos observados.

Cada gráfico também marca:

- A melhor época pela perda de validação.
- A época em que o early stopping encerrou o treinamento.

## Métricas de avaliação

Foram utilizadas:

- **Acurácia:** proporção total de previsões corretas.
- **Precisão:** entre os classificados como perigosos, quantos realmente pertencem a essa classe.
- **Recall:** entre os perigosos, quantos foram identificados.
- **F1-score:** combinação entre precisão e recall.
- **Matriz de confusão:** contagem dos acertos e dos diferentes tipos de erro.

O recall da classe perigosa merece atenção porque um falso negativo representa um registro perigoso classificado como não perigoso.

## Limitações dos experimentos

- **Comparação entre exercícios:** os modelos clássicos e as redes utilizaram divisões diferentes de treino e teste. Os resultados podem ser comparados de forma descritiva, mas não demonstram, sozinhos, uma superioridade definitiva das redes.

- **Distribuição balanceada:** as métricas foram calculadas em conjuntos com 50% de cada classe. Elas não representam diretamente o desempenho em uma população com a proporção original de registros perigosos.

- **Validação cruzada:** seguindo a sequência do enunciado, a transformação dos modelos clássicos foi ajustada antes do `GridSearchCV`. Isso permite que os dados das divisões internas de validação influenciem os parâmetros da transformação. Uma avaliação mais rigorosa ajustaria o transformador dentro de cada divisão. O teste externo permaneceu separado.

- **Asteroides repetidos:** a divisão foi feita por registros, conforme solicitado. Um mesmo asteroide pode aparecer em conjuntos diferentes. Portanto, o resultado não mede exclusivamente a capacidade de prever asteroides nunca vistos.

- **Variação de execução:** versões das bibliotecas e diferenças entre CPU e GPU podem alterar os resultados, mesmo com sementes definidas.

## Como executar

### Pelo Google Colab

1. Baixe o notebook `Trabalho_IAA_FernandaMarques.ipynb`.
2. Acesse o [Google Colab](https://colab.research.google.com/).
3. Selecione **Arquivo → Fazer upload de notebook**.
4. Envie o arquivo `.ipynb`.
5. Execute as células na ordem, do início ao fim.

O dataset é baixado automaticamente com:

```python
kagglehub.dataset_download(
    "sameepvani/nasa-nearest-earth-objects"
)
```

Se a biblioteca não estiver disponível, execute antes:

```python
!pip install -q kagglehub
```

É necessário acesso à internet para baixar os dados. O código das redes utiliza GPU quando disponível e CPU caso contrário.

### Em ambiente local

Instale as dependências em um ambiente Python:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn torch kagglehub jupyter
```

Depois, abra o notebook com:

```bash
jupyter notebook
```

Execute as células na ordem para criar as variáveis utilizadas nas etapas seguintes.

## Organização do notebook

1. Identificação e apresentação do problema.
2. Carregamento do dataset.
3. Inspeção, limpeza e engenharia de atributos.
4. Matriz de correlação e balanceamento.
5. Treinamento e avaliação dos modelos clássicos.
6. Treinamento e avaliação das cinco MLPs.
7. Tabelas e análise dos resultados.
8. Questões teóricas de QA.

## Conteúdos abordados nas questões QA

- Efeito do desbalanceamento na acurácia.
- Uso de métricas complementares e pesos na função de perda.
- Vazamento de dados.
- Overfitting e regularização.
- Desaparecimento de gradientes.
- Diferenças de custo, memória e previsão entre KNN e MLP.

## Referências

- [Dataset NASA — Nearest Earth Objects](https://www.kaggle.com/datasets/sameepvani/nasa-nearest-earth-objects)
- [Documentação do scikit-learn](https://scikit-learn.org/stable/)
- [Documentação do PowerTransformer](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PowerTransformer.html)
- [Documentação do PyTorch](https://pytorch.org/docs/stable/index.html)
- Guia e enunciado fornecidos na disciplina.
