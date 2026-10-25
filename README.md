# Electric Vehicle Purchase Prediction

## Introdução

Este projeto foi desenvolvido no contexto de uma competição do Kaggle, com o objetivo de construir um modelo de Machine Learning capaz de prever se um consumidor possui intenção de comprar um veículo elétrico.

O problema foi tratado como uma classificação binária, utilizando a variável `Will_Buy_EV` como target. O desenvolvimento passou pelas etapas de análise dos dados, preparação das variáveis, normalização, treinamento, otimização e validação do modelo.

## Tecnologias utilizadas

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- XGBoost

## Análise da base

A base de treinamento possui **668.665 registros e 15 variáveis**, contendo informações demográficas, financeiras e comportamentais dos consumidores, além de características relacionadas à infraestrutura para veículos elétricos.

Entre as variáveis analisadas estão:

- idade;
- renda anual;
- distância percorrida diariamente;
- quantidade de carros;
- estações de carregamento próximas;
- preocupação ambiental;
- tipo de cidade e veículo atual;
- possibilidade de carregamento residencial;
- disponibilidade de subsídio;
- nível de preocupação com autonomia.

A variável alvo apresentou desbalanceamento entre as classes:

- `0 - Não compraria`: 551.886 registros;
- `1 - Compraria`: 116.779 registros.

Durante a análise exploratória foram avaliadas as distribuições das variáveis numéricas e categóricas e sua relação com a probabilidade de compra de um veículo elétrico.

## Feature Engineering

As variáveis categóricas foram preparadas para utilização no modelo. Algumas variáveis binárias e ordinais foram convertidas para representações numéricas, como:

- `Gender`;
- `Range_Anxiety_Level`;
- `Home_Charging_Possible`;
- `Subsidy_Available`;
- `Will_Buy_EV`.

As demais variáveis categóricas foram tratadas dentro do pipeline utilizando `OneHotEncoder`, com suporte para categorias não observadas durante o treinamento.

Valores ausentes também foram tratados automaticamente com `SimpleImputer`.

## Normalização das variáveis

As variáveis numéricas passaram por um pipeline composto por:

```text
SimpleImputer(strategy="median")
        ↓
StandardScaler()
```

Dessa forma, possíveis valores ausentes são preenchidos pela mediana e as variáveis numéricas são padronizadas antes do treinamento.

O processamento foi centralizado utilizando `ColumnTransformer` e `Pipeline`, garantindo que as mesmas transformações sejam aplicadas durante treinamento e inferência.

## Treinamento e validação

A base foi dividida em:

- 80% para treinamento;
- 20% para holdout.

A divisão foi realizada de forma estratificada para preservar a proporção das classes.

O modelo utilizado foi o **XGBoost Classifier**. A otimização dos hiperparâmetros foi realizada com `RandomizedSearchCV`, utilizando validação cruzada com `StratifiedKFold` e `ROC-AUC` como métrica principal.

O melhor modelo apresentou:

| Avaliação | ROC-AUC |
|---|---:|
| Cross Validation | 0.9414 |
| Holdout | 0.9413 |

A proximidade entre os resultados de validação cruzada e holdout indica boa consistência do modelo nos dados avaliados.

## Resultado

O modelo final obteve **ROC-AUC de 0.9413 no conjunto de holdout**, demonstrando boa capacidade de separar consumidores com maior e menor propensão à compra de veículos elétricos.
