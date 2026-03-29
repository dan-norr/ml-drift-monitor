# ML Drift Monitor

Sistema de monitoramento contínuo de um modelo de detecção de fraude em cartão de crédito.
Treina, simula degradação, detecta alertas e serve tudo via API e dashboard visual.

---

Continuous monitoring system for a credit card fraud detection model.
Trains, simulates degradation, detects alerts, and serves everything via API and visual dashboard.

---

## Índice / Index

1. [Para quem não é da área / For non-technical readers](#1-para-quem-não-é-da-área--for-non-technical-readers)
2. [Para desenvolvedores e cientistas de dados / For developers and data scientists](#2-para-desenvolvedores-e-cientistas-de-dados--for-developers-and-data-scientists)

---

## 1. Para quem não é da área / For non-technical readers

---

### Português

#### Por que isso existe

Um modelo de inteligência artificial não é estático. Ele aprende com dados do passado, e o mundo continua mudando depois que ele foi treinado.

No caso de fraude em cartão de crédito, golpistas mudam o comportamento constantemente: novos padrões, novos valores, novas estratégias. Se ninguém monitorar o modelo, ele começa a falhar em silêncio. As fraudes passam. O prejuízo cresce. E ninguém percebe até ser tarde.

Este projeto resolve exatamente esse problema.

#### Qual problema resolve

Imagine um cachorro treinado para latir toda vez que vê um gato. No começo, funciona perfeitamente. Mas com o tempo, os gatos do bairro mudam: chegam gatos de outras raças, com cores diferentes, que se movem de outro jeito. O cachorro não reconhece mais o que aprendeu e para de latir. Do lado de fora, ele parece estar bem. Por dentro, deixou de cumprir a função.

Esse fenômeno tem nome: **degradação de modelo**. É um dos problemas mais silenciosos e caros do mundo de IA em produção.

Este sistema observa o cachorro semana a semana. Se ele para de latir quando deveria, ou se os gatos que chegam estão muito diferentes dos que ele conhecia, um alerta é disparado.

#### Como funciona

Sem entrar em código, o processo é simples:

1. **Treinamento:** o modelo aprende com centenas de milhares de transações reais (284 mil transações do dataset Kaggle Credit Card Fraud).
2. **Simulação:** o sistema gera 12 semanas de dados com desvios crescentes. As primeiras semanas são normais. A partir da semana 4, os dados começam a mudar. Na semana 8, a mudança é severa e o modelo começa a errar.
3. **Monitoramento:** a cada semana, o sistema compara os dados novos com os dados originais e calcula métricas de qualidade. Se algo está fora do esperado, um alerta é gerado.
4. **Visualização:** um painel mostra tudo em gráficos: como o desempenho evoluiu semana a semana, quais características dos dados mudaram, e em quais semanas os alertas foram disparados.
5. **API:** os mesmos dados estão disponíveis via interface técnica para integração com outros sistemas.

O resultado: em vez de descobrir a falha do modelo em uma auditoria trimestral, você vê em qual semana exata ele começou a degradar e pode agir antes que isso se torne prejuízo real.

---

### English

#### Why it exists

A machine learning model is not static. It learns from past data, and the world keeps changing after it was trained.

In credit card fraud, fraudsters constantly adapt: new patterns, new amounts, new strategies. If no one monitors the model, it starts failing silently. Fraud slips through. Losses grow. No one notices until it is too late.

This project solves exactly that problem.

#### What problem it solves

Picture a dog trained to bark every time it sees a cat. At first, it works perfectly. But over time, the neighborhood changes: new breeds show up, different colors, different ways of moving. The dog no longer recognizes what it learned and stops barking. From the outside, it looks fine. On the inside, it stopped doing its job.

This is called **model degradation**. It is one of the most silent and costly problems in production AI.

This system watches the dog week by week. If it stops barking when it should, or if the cats arriving look too different from the ones it knows, an alert fires.

#### How it works

Without touching code, the process is straightforward:

1. **Training:** the model learns from hundreds of thousands of real transactions (284k transactions from the Kaggle Credit Card Fraud dataset).
2. **Simulation:** the system generates 12 weeks of synthetic data with increasing deviations. The first weeks are normal. From week 4, the data starts shifting. By week 8, the shift is severe and the model starts making errors.
3. **Monitoring:** each week, the system compares new data against the original data and computes quality metrics. If something is out of bounds, an alert is generated.
4. **Visualization:** a dashboard shows it all in charts: how performance evolved week by week, which data features changed, and which weeks triggered alerts.
5. **API:** the same data is available via a technical interface for integration with other systems.

The result: instead of discovering model failure in a quarterly audit, you see the exact week it started degrading and can act before it becomes real financial loss.

---

## 2. Para desenvolvedores e cientistas de dados / For developers and data scientists

---

### Português

#### Por que isso existe

MLOps fecha o gap entre experimentação e produção. Treinar um modelo com F1 alto é a parte fácil. Mantê-lo confiável em produção, onde a distribuição dos dados muda, os rótulos ficam escassos e o comportamento do mundo não avisa quando muda, é onde a maioria dos projetos falha.

O problema específico que este sistema ataca é **data drift e concept drift em produção**:

- **Data drift (covariate shift)**: a distribuição de entrada P(X) muda. O modelo nunca viu esse padrão de features durante o treino.
- **Concept drift**: a relação P(Y|X) muda. Os mesmos inputs passam a ter outputs diferentes. No caso de fraude, isso acontece quando golpistas adaptam o comportamento para driblar o modelo.

Sem monitoramento ativo, esses fenômenos são invisíveis até que a performance caia o suficiente para aparecer em uma métrica de negócio, geralmente tarde demais.

#### Qual problema resolve

O pipeline padrão de ML para fraude termina no deploy. Este projeto estende o ciclo com:

- Baseline snapshot (reference data) capturado no momento do treino
- Comparação estatística semanal entre produção e baseline via Evidently
- Alertas baseados em PSI por feature e share de features drifted
- Métricas de classificação recalculadas batch a batch (F1, Precision, Recall)
- Visualização interativa e API REST para consumo downstream

#### Como funciona

**Stack**

| Camada | Tecnologia |
|---|---|
| Modelo | XGBoost 2.0 + SMOTE (imbalanced-learn) |
| Drift detection | Evidently 0.7.21 (legacy API) |
| API | FastAPI + Uvicorn |
| Dashboard | Vanilla JS + Plotly CDN (servido pelo FastAPI) |
| Serialização | Parquet (dados), Pickle (modelo), JSON (métricas) |
| Config | YAML centralizado |
| Qualidade | pytest + ruff + mypy |
| Deploy | Docker Compose |

**Pipeline**

```
creditcard.csv
    │
    ▼
src/train.py            F1=0.88, ROC-AUC=0.98, threshold otimizado via PR curve
    │                   → model.pkl, reference_data.parquet, feature_importance.json
    ▼
src/simulate_drift.py   Semanas 1-3: sutil (PSI 0.05-0.10)
    │                   Semanas 4-7: detectável (PSI 0.10-0.20)
    │                   Semanas 8-12: crítico (PSI > 0.20 + concept drift)
    │                   → data/simulated/week_01..12.parquet
    ▼
src/monitor.py          Evidently DataDriftPreset + ClassificationPreset
    │                   Alertas: PSI > 0.2 por feature, share_drifted > 0.30
    │                   → metrics/week_NN.json, reports/week_NN.html
    ▼
src/infrastructure/api.py     GET /metrics  /alerts  /report/{week}  /health
dashboard/index.html +        F1 timeline, drift heatmap, distribuições, painel de alertas
assets/dashboard.js           (servido em /dashboard/ como static files pelo FastAPI)
```

**Decisões técnicas**

*SMOTE + scale_pos_weight:* O dataset tem ratio de 577:1 (legítimo:fraude). SMOTE foi aplicado no treino para balancear as classes. Quando SMOTE está ativo, `scale_pos_weight` deve ser 1, caso contrário o XGBoost duplo-penaliza a classe negativa e a precision colapsa.

*Threshold via PR curve:* O threshold padrão de 0.5 é subótimo para dados desbalanceados. O threshold ótimo é encontrado varrendo a curva precision-recall e maximizando F1, resultando em threshold=0.9752 neste dataset.

*Stratified batch sampling:* Com fraud rate de 0.17%, batches aleatórios de 3000 linhas podem ter zero casos de fraude, causando F1=0. A solução é garantir `min_fraud_per_batch=15` via amostragem estratificada explícita.

*Evidently legacy API:* Evidently 0.7.21 moveu a API v1 para `evidently.legacy.*`. Os imports corretos são `evidently.legacy.report`, `evidently.legacy.metric_preset` e `evidently.legacy.pipeline.column_mapping`.

*PSI em vez de KS test:* O teste Kolmogorov-Smirnov é assimétrico (o resultado muda dependendo de qual distribuição é referência), sensível ao tamanho da amostra e reporta apenas se há drift, sem indicar magnitude. O PSI (Population Stability Index) é simétrico, agnóstico ao tamanho de amostra, funciona igualmente para variáveis contínuas e categóricas, e mapeia diretamente para bandas de severidade de negócio: PSI < 0.1 = estável, 0.1-0.2 = monitorar, > 0.2 = alerta. Isso elimina a necessidade de calibrar um threshold p-value por feature.

*Dashboard Vanilla JS em vez de Streamlit:* Streamlit exige um processo Python separado rodando na porta 8501, o que implica um proxy reverso adicional, WebSocket tunneling e rerendering full-page a cada interação. Com Vanilla JS + Plotly CDN, o dashboard é servido como arquivos estáticos diretamente pelo FastAPI em `/dashboard/` (porta 8000), sem dependências de processo extra e com carregamento instantâneo.

**Arquitetura**

O projeto segue Clean Architecture com inversão de dependência explícita:

```
domain/          entidades puras (WeeklyMetrics, TrainingResult, DriftAlert)
                 + protocolos de I/O (IDataRepository, IDriftAnalyser, ...)
use_cases/       lógica de negócio (TrainModelUseCase, SimulateDriftUseCase, MonitorDriftUseCase)
adapters/        implementações concretas (Parquet, Pickle, JSON, YAML, Evidently)
infrastructure/  FastAPI, routing thin, sem lógica de negócio
train.py etc.    entrypoints: wire + execute
```

A regra de dependência é unidirecional: `infrastructure -> adapters -> use_cases -> domain`. Nenhuma camada interna conhece a externa. Os use cases são testáveis com mocks sem tocar em disco ou Evidently.

**Executar**

```bash
# Pipeline completo
make all

# Serviços individuais
make api        # http://localhost:8000
                # dashboard em http://localhost:8000/dashboard/

# Docker
make docker-up

# Qualidade
make test
make lint
make typecheck
```

---

### English

#### Why it exists

MLOps closes the gap between experimentation and production. Training a model with high F1 is the easy part. Keeping it reliable in production, where data distributions shift, labels become sparse, and the world changes without warning, is where most projects fail.

The specific problem this system targets is **data drift and concept drift in production**:

- **Data drift (covariate shift)**: the input distribution P(X) changes. The model never saw this feature pattern during training.
- **Concept drift**: the relationship P(Y|X) changes. The same inputs now produce different outputs. In fraud, this happens when fraudsters adapt their behavior to evade the model.

Without active monitoring, these phenomena are invisible until performance degrades enough to surface in a business metric, usually too late.

#### What problem it solves

The standard ML pipeline for fraud typically ends at deployment. This project extends the cycle with:

- Baseline snapshot (reference data) captured at training time
- Weekly statistical comparison between production and baseline via Evidently
- Alerts based on per-feature PSI and share of drifted features
- Classification metrics recomputed batch by batch (F1, Precision, Recall)
- Interactive visualization and REST API for downstream consumption

#### How it works

**Stack**

| Layer | Technology |
|---|---|
| Model | XGBoost 2.0 + SMOTE (imbalanced-learn) |
| Drift detection | Evidently 0.7.21 (legacy API) |
| API | FastAPI + Uvicorn |
| Dashboard | Vanilla JS + Plotly CDN (served by FastAPI) |
| Serialization | Parquet (data), Pickle (model), JSON (metrics) |
| Config | Centralized YAML |
| Quality | pytest + ruff + mypy |
| Deploy | Docker Compose |

**Pipeline**

```
creditcard.csv
    │
    ▼
src/train.py            F1=0.88, ROC-AUC=0.98, threshold optimized via PR curve
    │                   → model.pkl, reference_data.parquet, feature_importance.json
    ▼
src/simulate_drift.py   Weeks 1-3: subtle (PSI 0.05-0.10)
    │                   Weeks 4-7: detectable (PSI 0.10-0.20)
    │                   Weeks 8-12: critical (PSI > 0.20 + concept drift)
    │                   → data/simulated/week_01..12.parquet
    ▼
src/monitor.py          Evidently DataDriftPreset + ClassificationPreset
    │                   Alerts: PSI > 0.2 per feature, share_drifted > 0.30
    │                   → metrics/week_NN.json, reports/week_NN.html
    ▼
src/infrastructure/api.py     GET /metrics  /alerts  /report/{week}  /health
dashboard/index.html +        F1 timeline, drift heatmap, distributions, alert panel
assets/dashboard.js           (served at /dashboard/ as FastAPI static files)
```

**Key technical decisions**

*SMOTE + scale_pos_weight:* The dataset has a 577:1 ratio (legitimate:fraud). SMOTE was applied during training to balance classes. When SMOTE is active, `scale_pos_weight` must be 1, otherwise XGBoost double-penalizes the negative class and precision collapses.

*Threshold via PR curve:* The default 0.5 threshold is suboptimal for imbalanced data. The optimal threshold is found by sweeping the precision-recall curve and maximizing F1, resulting in threshold=0.9752 on this dataset.

*Stratified batch sampling:* With a 0.17% fraud rate, random 3000-row batches can contain zero fraud cases, causing F1=0. The fix is explicit stratified sampling guaranteeing `min_fraud_per_batch=15`.

*Evidently legacy API:* Evidently 0.7.21 moved the v1 API to `evidently.legacy.*`. Correct imports: `evidently.legacy.report`, `evidently.legacy.metric_preset`, `evidently.legacy.pipeline.column_mapping`.

*PSI over KS test:* The Kolmogorov-Smirnov test is asymmetric (result changes depending on which distribution is the reference), sensitive to sample size, and only reports whether drift exists without indicating magnitude. PSI (Population Stability Index) is symmetric, sample-size agnostic, works equally for continuous and categorical variables, and maps directly to business severity bands: PSI < 0.1 = stable, 0.1-0.2 = monitor, > 0.2 = alert. This eliminates the need to calibrate a p-value threshold per feature.

*Vanilla JS dashboard over Streamlit:* Streamlit requires a separate Python process running on port 8501, which implies an additional reverse proxy, WebSocket tunneling, and full-page re-rendering on every interaction. With Vanilla JS + Plotly CDN, the dashboard is served as static files directly by FastAPI at `/dashboard/` (port 8000), with no extra process dependency and instant load.

**Architecture**

The project follows Clean Architecture with explicit dependency inversion:

```
domain/          pure entities (WeeklyMetrics, TrainingResult, DriftAlert)
                 + I/O protocols (IDataRepository, IDriftAnalyser, ...)
use_cases/       business logic (TrainModelUseCase, SimulateDriftUseCase, MonitorDriftUseCase)
adapters/        concrete implementations (Parquet, Pickle, JSON, YAML, Evidently)
infrastructure/  FastAPI, thin routing, no business logic
train.py etc.    entrypoints: wire + execute
```

The dependency rule is one-directional: `infrastructure -> adapters -> use_cases -> domain`. No inner layer knows about the outer ones. Use cases are fully testable with mocks without touching disk or Evidently.

**Run it**

```bash
# Full pipeline
make all

# Individual services
make api        # http://localhost:8000
                # dashboard at http://localhost:8000/dashboard/

# Docker
make docker-up

# Quality
make test
make lint
make typecheck
```
