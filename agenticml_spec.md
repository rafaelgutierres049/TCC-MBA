# AgenticML — System Specification

> Versão: 0.1 (em construção)  
> Autor: Rafael Ponte Gutierres  
> Contexto: TCC — MBA em Data Science e Analytics

---

## 1. Identidade do Sistema

**Nome:** AgenticML  
**Objetivo:** Receber um dataset e um prompt em linguagem natural e retornar um pipeline de ML treinado com explicações.

---

## 2. Agentes

| # | Nome | Responsabilidade |
|---|------|-----------------|
| 1 | Ingestão e Inspeção | Leitura, parsing e inspeção estrutural do dataset |
| 2 | Diagnóstico de Qualidade | Detecção de missing values, desbalanceamento, leakage e outros problemas |
| 3 | Feature Engineering | Transformações e criação de variáveis com justificativas |
| 4 | Interpretação de Linguagem Natural | Inferência do tipo de problema a partir do prompt e do dataset |
| 5 | Seleção e Treinamento | Comparação de algoritmos e treinamento do modelo mais adequado |
| 6 | Previsão e Explicação | Geração de previsões e relatório explicativo em linguagem natural |

---

## 3. Interface do Sistema

### 3.1 Entrada

| Campo | Tipo | Obrigatório | Descrição |
|-------|------|-------------|-----------|
| `dataset` | Arquivo (CSV, XLSX, XLS) | Sim | Dataset a ser analisado |
| `prompt` | String | Sim | Objetivo do usuário em linguagem natural |
| `problem_type` | Enum (opcional) | Não | Sugestão do tipo de problema: `classification`, `regression`, `clustering`, `time_series` |

**Restrições de entrada:**
- Tamanho máximo: 50MB / 500.000 linhas
- Formatos aceitos: `.csv`, `.xlsx`, `.xls`
- Encoding: UTF-8 e Latin-1
- Separador CSV: auto-detectado

> ⚠️ O campo `problem_type` é uma **sugestão**. O Agente 4 (Interpretação) tem autonomia para divergir e informará o usuário com justificativa caso adote uma abordagem diferente.

### 3.2 Saída

| Campo | Formato | Descrição |
|-------|---------|-----------|
| `model` | Arquivo `.joblib` + JSON | Modelo treinado serializado |
| `report` | JSON + Markdown | Métricas de avaliação (ex: accuracy, F1, RMSE — conforme o problema) |
| `explanation` | JSON + Markdown | Explicação em linguagem natural (português) das decisões e resultados |
| `predictions` | JSON + CSV | Previsões sobre novos dados fornecidos pelo usuário (quando aplicável) |

---

## 4. Interfaces de Uso

O sistema é agnóstico de interface. A lógica dos agentes é exposta via **FastAPI** e pode ser consumida por:

- **Jupyter Notebooks** — interface primária para o TCC (reprodutibilidade e evidências dos experimentos)
- **Frontend web** — interface opcional, consome a mesma API sem alterações na lógica dos agentes

---

## 5. Invariantes Globais

Regras que se aplicam a **todos os agentes** sem exceção:

- **Falha explícita:** se um agente encontrar um problema crítico (dado corrompido, tipo incompatível, erro irrecuperável), ele **interrompe o fluxo** e retorna ao usuário uma mensagem explicando o problema encontrado e o motivo da interrupção.
- **Comunicação via State:** a troca de informações entre agentes é feita exclusivamente através do State do LangGraph — nenhum agente acessa diretamente o output de outro.
- **Rastreabilidade:** cada agente registra suas decisões no State para compor o relatório final de explicação.

---

## 6. Especificação dos Agentes

### Agente 1 — Ingestão e Inspeção

**Responsabilidade:** Receber o dataset e identificar sua estrutura — colunas, tipos de variáveis, distribuições, valores ausentes e possíveis relações entre os dados.

**Input (do usuário via API):**
- `dataset` — arquivo CSV, XLSX ou XLS
- `prompt` — objetivo em linguagem natural
- `problem_type` *(opcional)* — sugestão do tipo de problema

**Output (para o State):**
```json
{
  "schema": {
    "columns": ["col1", "col2"],
    "dtypes": {"col1": "float64", "col2": "object"},
    "shape": [1000, 10]
  },
  "descriptive_stats": {
    "col1": {"mean": 3.5, "std": 1.2, "min": 0.0, "max": 10.0, "nulls": 5}
  },
  "sample": [["val1", "val2"], ["val3", "val4"]],
  "encoding_detected": "utf-8",
  "separator_detected": ","
}
```

**Invariantes:**
- Dataset vazio, corrompido ou ilegível → interrompe o fluxo com mensagem explicativa ao usuário
- Nunca passa adiante um dataset com 0 linhas válidas

---

### Agente 2 — Diagnóstico de Qualidade

**Responsabilidade:** Detectar e reportar autonomamente problemas de qualidade nos dados, gerando alertas explicados em linguagem natural.

**Input (do State):** Output do Agente 1

**Problemas detectados:**
- Missing values
- Desbalanceamento de classes
- Data leakage
- Viés nos dados
- Risco de overfitting

**Output (para o State):**
```json
{
  "issues": [
    {
      "type": "class_imbalance",
      "severity": "medium",
      "alert": "A coluna target possui distribuição desbalanceada (90/10). Recomenda-se aplicar técnicas de balanceamento como SMOTE."
    },
    {
      "type": "data_leakage",
      "severity": "high",
      "alert": "A coluna 'resultado_final' apresenta correlação perfeita com o target, indicando possível vazamento de dados."
    }
  ],
  "overall_severity": "high"
}
```

**Invariantes:**
- Severidade `high` em qualquer issue → interrompe o fluxo com mensagem explicativa ao usuário
- Severidade `none`, `low` ou `medium` → registra alertas no State e continua o fluxo
- Todo problema detectado deve ter obrigatoriamente: `type`, `severity` e `alert` em linguagem natural

---

### Agente 3 — Feature Engineering

**Responsabilidade:** Recomendar e executar transformações e criação de variáveis semanticamente relevantes com base no contexto do problema e nas características do dataset, fornecendo justificativas em linguagem natural.

**Input (do State):** Outputs dos Agentes 1 e 2

**Transformações suportadas:**
- Encoding de variáveis categóricas (Label Encoding, One-Hot Encoding)
- Normalização / padronização (MinMax, StandardScaler)
- Imputação de missing values
- Criação de novas features (interações, agregações)
- Remoção de features irrelevantes ou redundantes

**Output (para o State):**
```json
{
  "transformations_applied": [
    {
      "type": "one_hot_encoding",
      "columns": ["categoria"],
      "justification": "A coluna 'categoria' é nominal sem ordem natural, por isso foi aplicado One-Hot Encoding."
    },
    {
      "type": "standard_scaler",
      "columns": ["idade", "renda"],
      "justification": "Variáveis numéricas com escalas distintas foram padronizadas para evitar dominância em algoritmos sensíveis a magnitude."
    }
  ],
  "dataset_transformed": "<dataset processado>"
}
```

**Invariantes:**
- Toda transformação executada deve ter obrigatoriamente: `type`, `columns` e `justification` em linguagem natural
- Justificativas são registradas no State **e** exibidas ao usuário em tempo real durante a execução
- Executa as transformações diretamente — não apenas recomenda

---

### Agente 4 — Interpretação de Linguagem Natural

**Responsabilidade:** Interpretar o prompt do usuário e os dados inspecionados para inferir o tipo de problema analítico, alinhando a compreensão do sistema à intenção do usuário.

**Input (do State):** Prompt do usuário + outputs dos Agentes 1, 2 e 3

**Tipos de problema suportados:**
- `classification`
- `regression`
- `clustering`
- `time_series`
- `anomaly_detection`

**Output (para o State):**
```json
{
  "problem_type": "classification",
  "confidence": "high",
  "justification": "O prompt solicita prever se um cliente irá cancelar o contrato. A variável target 'churn' é binária, caracterizando um problema de classificação.",
  "diverged_from_user": false
}
```

**Invariantes:**
- Se divergir da sugestão do usuário → registra `diverged_from_user: true`, explica a decisão e continua o fluxo sem bloquear
- Se não conseguir identificar o tipo de problema com confiança → interrompe o fluxo e solicita mais informações ao usuário
- O campo `justification` é obrigatório em qualquer cenário

---

### Agente 5 — Seleção e Treinamento

**Responsabilidade:** Selecionar os algoritmos candidatos com base no tipo de problema identificado, compará-los e treinar o modelo mais adequado.

**Input (do State):** Outputs dos Agentes 3 e 4 (dataset transformado + tipo de problema)

**Algoritmos candidatos por tipo de problema:**

| Tipo | Candidatos |
|---|---|
| `classification` | Logistic Regression, Random Forest, XGBoost, SVM |
| `regression` | Linear Regression, Random Forest, XGBoost, SVR |
| `clustering` | K-Means, DBSCAN, Agglomerative |
| `time_series` | ARIMA, Prophet, LSTM |
| `anomaly_detection` | Isolation Forest, LOF, One-Class SVM |

**Métrica de seleção:**
- Se o usuário definir uma métrica preferida → usa a métrica definida
- Caso contrário → o sistema adota a métrica padrão do tipo de problema (ex: F1 para `classification`, RMSE para `regression`)

**Output (para o State):**
```json
{
  "model_selected": "XGBoost",
  "metric_used": "f1",
  "metric_value": 0.91,
  "justification": "XGBoost obteve o melhor F1-score entre os candidatos avaliados para o problema de classificação.",
  "model_artifact": "<modelo serializado via joblib>"
}
```

**Invariantes:**
- Apenas o modelo vencedor é registrado no State — os demais candidatos são descartados
- O campo `justification` é obrigatório, explicando por que o modelo foi selecionado
- Se nenhum candidato atingir desempenho mínimo aceitável → interrompe o fluxo e informa o usuário

---

### Agente 6 — Previsão e Explicação

**Responsabilidade:** Gerar previsões sobre novos dados e produzir o relatório final explicando todas as decisões tomadas ao longo do pipeline, em dois níveis de detalhe.

**Input (do State):** Outputs de todos os agentes anteriores + novos dados para previsão (quando disponíveis)

**Modos de previsão:**
- **Junto na requisição inicial** — campo `prediction_data` opcional no input da API
- **Requisição separada** — endpoint `/predict` recebe o modelo treinado + novos dados

**Output (para o State e usuário):**
```json
{
  "predictions": {
    "values": [1, 0, 1],
    "format": "json+csv"
  },
  "explanation": {
    "technical": "O modelo XGBoost foi selecionado com F1-score de 0.91. Durante o Feature Engineering, foram aplicados One-Hot Encoding em variáveis categóricas e StandardScaler em variáveis numéricas. O Diagnóstico de Qualidade identificou desbalanceamento de classes com severidade medium, sem interrupção do fluxo.",
    "accessible": "O sistema analisou seus dados, organizou as informações e treinou um modelo que acerta 91% dos casos. Os principais fatores que influenciam o resultado são: idade, renda e categoria do cliente."
  }
}
```

**Invariantes:**
- O relatório deve sempre conter as duas versões: `technical` e `accessible`
- A explicação técnica deve referenciar as decisões de todos os agentes anteriores
- A explicação acessível deve ser em linguagem clara, sem jargões técnicos
- Previsões devem ser entregues em JSON e CSV

---

## 7. Fluxo LangGraph

### 7.1 Caminho Feliz

```
[Agente 1: Ingestão e Inspeção]
          │
          ▼
[Agente 2: Diagnóstico de Qualidade]
          │
          ▼
[Agente 3: Feature Engineering]
          │
          ▼
[Agente 4: Interpretação de Linguagem Natural]
          │
          ▼
[Agente 5: Seleção e Treinamento]
          │
          ▼
[Agente 6: Previsão e Explicação] ──► Output final ao usuário
```

### 7.2 Curto-Circuito (Erro Crítico)

Qualquer agente que encontre um problema crítico interrompe o fluxo e encaminha diretamente para o Agente 6 no **modo erro**:

```
[Agente N] ──► erro crítico ──► [Agente 6: modo erro] ──► Output de erro ao usuário
```

**Exemplos de gatilhos de curto-circuito:**

| Agente | Gatilho |
|--------|---------|
| Agente 1 | Dataset corrompido, vazio ou ilegível |
| Agente 2 | Issue com severidade `high` |
| Agente 4 | Tipo de problema não identificado com confiança |
| Agente 5 | Nenhum modelo atinge desempenho mínimo aceitável |

### 7.3 Agente 6 — Modo Erro

Quando acionado por curto-circuito, o Agente 6 não gera previsões. Em vez disso, entrega ao usuário:

```json
{
  "status": "error",
  "failed_at": "Agente 2 — Diagnóstico de Qualidade",
  "explanation": {
    "technical": "O pipeline foi interrompido no Agente 2. Foi detectado data leakage com severidade high na coluna 'resultado_final', indicando vazamento do target.",
    "accessible": "O sistema identificou um problema sério nos seus dados: uma coluna parece 'entregar' a resposta antes do modelo aprender, o que tornaria os resultados inválidos. Corrija o dataset e tente novamente."
  }
}
```

### 7.4 Estrutura do State (LangGraph)

O State é o objeto compartilhado entre todos os agentes ao longo do fluxo:

```python
class AgenticMLState(TypedDict):
    # Input
    dataset: Any                  # Dataset carregado
    prompt: str                   # Objetivo do usuário
    problem_type_hint: str | None # Sugestão do usuário

    # Outputs por agente
    inspection: dict | None       # Agente 1
    quality: dict | None          # Agente 2
    features: dict | None         # Agente 3
    interpretation: dict | None   # Agente 4
    training: dict | None         # Agente 5
    output: dict | None           # Agente 6

    # Controle de fluxo
    status: str                   # "running" | "error" | "done"
    error: dict | None            # Preenchido em caso de curto-circuito
    failed_at: str | None         # Nome do agente que falhou
```

---

## 8. Edge Cases Globais

Cenários que não são erro crítico mas possuem comportamento definido:

| # | Cenário | Comportamento |
|---|---------|---------------|
| 1 | Dataset com uma única coluna | Agente 1 interrompe o fluxo com mensagem explicativa |
| 2 | Prompt vazio ou muito vago (ex: "analise isso") | Agente 4 interrompe e solicita mais informações ao usuário |
| 3 | Todas as colunas são categóricas | Agente 3 aplica encoding sem normalização |
| 4 | Dataset sem missing values e sem problemas | Agente 2 passa direto, registra `issues: []` no State |
| 5 | `prediction_data` com colunas diferentes do dataset de treino | Agente 6 interrompe e informa incompatibilidade de schema |
| 6 | Tipo de problema é `clustering` | Agente 5 não utiliza coluna target; Agente 6 não gera métricas supervisionadas |
| 7 | Prompt injection no dataset | Sistema sanitiza o conteúdo das células antes de enviá-lo ao LLM; conteúdo do dataset nunca é interpretado como instrução |

---

## 9. Métricas de Avaliação do Sistema

### 9.1 Métricas Quantitativas

| Métrica | Descrição |
|---------|-----------|
| Desempenho do modelo | Métricas padrão por tipo de problema (F1, Accuracy, RMSE, Silhouette, etc.) |
| Comparação com baseline | Resultado do AgenticML vs. Auto-sklearn nos mesmos datasets |
| Tempo de execução | Tempo total do pipeline do input até o output final |

### 9.2 Métricas Qualitativas

Avaliação manual pelo autor com base em rubrica estruturada. Escala: **Insatisfatório / Regular / Bom / Excelente**

| Critério | Descrição |
|----------|-----------|
| Clareza | A explicação é compreensível para o público-alvo (técnico e não-técnico) |
| Correção técnica | A explicação está alinhada com o que o modelo realmente executou |
| Completude | A explicação cobre todas as etapas do pipeline |
| Ausência de alucinações | O sistema não afirmou algo falso ou inventou justificativas |

---

## 10. Stack Tecnológica

| Categoria | Tecnologia |
|-----------|-----------|
| Linguagem | Python |
| Orquestração de agentes | LangChain, LangGraph |
| LLMs | GPT-4o (OpenAI), Claude Sonnet (Anthropic) |
| API | FastAPI |
| Manipulação e análise de dados | Pandas, NumPy |
| Treinamento de ML | scikit-learn |
| Baseline de comparação | Auto-sklearn |
| Serialização de modelo | joblib |
| Ambiente de desenvolvimento | Jupyter Notebooks (reprodutibilidade dos experimentos) |
| Controle de versão | Git |

---

## 11. Limitações Conhecidas

| # | Limitação | Descrição |
|---|-----------|-----------|
| 1 | Dependência de qualidade do prompt | Prompts vagos ou ambíguos podem levar a interpretações incorretas do tipo de problema |
| 2 | Datasets muito grandes | Limitado a 50MB / 500k linhas; datasets maiores não são suportados |
| 3 | Séries temporais complexas | Suporte básico via ARIMA e Prophet; arquiteturas avançadas (ex: Transformers temporais) fora do escopo |
| 4 | Explicações dependem do LLM | Qualidade das explicações está sujeita às limitações e alucinações do modelo utilizado |
| 5 | Sem suporte a dados não-tabulares | Imagens, texto livre e áudio estão fora do escopo do sistema |
