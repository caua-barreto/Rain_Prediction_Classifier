#  Dados — weatherAUS

## Origem

O dataset **[weatherAUS](https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package)** foi obtido via **Kaggle** e contém observações meteorológicas diárias de **49 cidades australianas** entre **2008 e 2017**.

- **Formato:** CSV (~145.000 linhas × 23 colunas)
- **Fonte original:** pacote R [`rattle`](https://rattle.togaware.com/) (Bureau of Meteorology — Australia)
- **Licença:** domínio público (Kaggle)

## Como Obter

1. Acesse o link acima no Kaggle
2. Faça login (conta gratuita)
3. Clique em **"Download"** para obter o arquivo `weatherAUS.csv`
4. Salve o arquivo nesta pasta:

```
data/
└── weatherAUS.csv
```

>  O arquivo **não está incluído** neste repositório por questões de tamanho e licença. É necessário baixá-lo manualmente.

## Descrição das Variáveis

O dataset contém **23 colunas** de dados meteorológicos diários:

### Variáveis Numéricas (14)

| Variável | Descrição |
|----------|-----------|
| `MinTemp` | Temperatura mínima do dia (°C) |
| `MaxTemp` | Temperatura máxima do dia (°C) |
| `Rainfall` | Precipitação acumulada (mm) |
| `Evaporation` | Evaporação (mm) |
| `Sunshine` | Horas de sol |
| `WindGustSpeed` | Velocidade máxima da rajada de vento (km/h) |
| `WindSpeed9am` | Velocidade do vento às 9h (km/h) |
| `WindSpeed3pm` | Velocidade do vento às 15h (km/h) |
| `Humidity9am` | Umidade relativa às 9h (%) |
| `Humidity3pm` | Umidade relativa às 15h (%) |
| `Pressure9am` | Pressão atmosférica às 9h (hPa) |
| `Pressure3pm` | Pressão atmosférica às 15h (hPa) |
| `Temp9am` | Temperatura às 9h (°C) |
| `Temp3pm` | Temperatura às 15h (°C) |

### Variáveis Categóricas Nominais (4)

| Variável | Descrição |
|----------|-----------|
| `Location` | Cidade (49 cidades distintas) |
| `WindGustDir` | Direção da rajada de vento mais forte |
| `WindDir9am` | Direção do vento às 9h |
| `WindDir3pm` | Direção do vento às 15h |

### Variáveis Categóricas Ordinais (2)

| Variável | Descrição |
|----------|-----------|
| `Cloud9am` | Cobertura de nuvens às 9h (escala 0–9 oitavos) |
| `Cloud3pm` | Cobertura de nuvens às 15h (escala 0–9 oitavos) |

### Variáveis Binárias (2)

| Variável | Descrição |
|----------|-----------|
| `RainToday` | Choveu hoje? (`Yes` / `No`) |
| `RainTomorrow` | Choverá amanhã? (`Yes` / `No`) — **variável alvo** |

### Colunas Auxiliares

| Variável | Descrição |
|----------|-----------|
| `Date` | Data da observação |

> `Date` é convertida para `datetime` no pipeline e usada apenas para o corte temporal (histórico vs OOT). A coluna `Month` é derivada de `Date` para análise de sazonalidade, mas **não entra como feature do modelo**.

## Qualidade dos Dados

O dataset possui valores nulos significativos em algumas variáveis:

| Variável | % Nulos |
|----------|---------|
| `Sunshine` | ~48% |
| `Evaporation` | ~43% |
| `Cloud3pm` | ~41% |
| `Cloud9am` | ~38% |

O pipeline (`src/pipeline.py`) trata automaticamente os nulos via imputação com mediana (numéricas) ou moda (categóricas).

## Corte Temporal

O pipeline divide os dados por corte temporal:

| Conjunto | Período | Uso |
|----------|---------|-----|
| Histórico | Nov/2007 – Dez/2016 | Treino e teste (~134k linhas, 94%) |
| OOT | Jan/2017 – Jun/2017 | Validação final (~8.5k linhas, 6%) |

A separação garante que o modelo seja avaliado contra dados **futuros** que nunca foram vistos durante o treinamento.

## Tratamentos Aplicados

Antes de chegar ao modelo, o `_tratamento_base` em `src/pipeline.py` aplica os seguintes tratamentos ao CSV bruto:

1. **Remove linhas sem `RainTomorrow`** — registros onde a variável alvo está ausente são descartados
2. **Converte `Date` para `datetime`** — permite operações temporais e corte por data
3. **Cria a coluna `Month`** — extraída de `Date` (1–12), usada para análise de sazonalidade mas **não entra como feature do modelo**
4. **Remove duplicatas** — registros idênticos são eliminados
5. **Ordena por data** — garante sequência temporal nos dados

Após esses tratamentos, os dados seguem para as pipelines de pré-processamento (imputação, encoding, escalonamento) definidas em `src/pipeline.py`.

## Executando sem Notebooks

O projeto pode ser executado **inteiramente via scripts**, sem necessidade de abrir notebooks:

### Pré-requisitos

```bash
# Clone o repositório
git clone https://github.com/caua-barreto/Rain_Prediction_Model.git
cd Rain_Prediction_Model

# Crie ambiente virtual e instale dependências
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Baixe o weatherAUS.csv e salve em data/
```

### Treinar o modelo

```bash
python scripts/train.py
```

O script:
1. Carrega o `weatherAUS.csv` via `src/pipeline.py`
2. Aplica todos os tratamentos automaticamente (nulos, encoding, escalonamento)
3. Treina a **Regressão Logística L1** (pipeline Completa c/ Nulos)
4. Otimiza o threshold via curva Precision-Recall (max F1)
5. Avalia nas bases histórica e OOT
6. Salva os artefatos em `artifacts/`:
   - `modelo.joblib` — pipeline treinado
   - `metrics.json` — métricas (accuracy, precision, recall, F1, ROC-AUC, PR-AUC, KS)
   - `metadata.json` — hiperparâmetros e metadata do treino

### Gerar predições

```bash
python scripts/predict.py --input data/new_data.csv --output reports/predictions.csv
```

O script:
1. Carrega o modelo de `artifacts/modelo.joblib`
2. Lê o CSV de entrada (mesmo formato do `weatherAUS.csv`)
3. Aplica o pré-processamento automaticamente
4. Gera o arquivo de saída com colunas: `Date`, `Location`, `rain_probability`

### Estrutura esperada do CSV de entrada

O CSV passado para o `predict.py` deve conter as mesmas colunas do `weatherAUS.csv` original:

```
Date, Location, MinTemp, MaxTemp, Rainfall, Evaporation, Sunshine,
WindGustDir, WindGustSpeed, WindDir9am, WindDir3pm, WindSpeed9am,
WindSpeed3pm, Humidity9am, Humidity3pm, Pressure9am, Pressure3pm,
Cloud9am, Cloud3pm, Temp9am, Temp3pm, RainToday
```

> A coluna `RainTomorrow` **não é necessária** no CSV de entrada — é ela que o modelo vai prever.
