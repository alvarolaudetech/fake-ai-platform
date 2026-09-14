# fake-ai-platform

Internal gateway service that exposes a single, unified chat-completion API
in front of multiple upstream LLM providers (OpenAI, Anthropic, and
self-hosted OpenAI-compatible servers). Teams register a model once via the
admin API and every internal client can call it through one consistent
`/api/v1/inference/chat` endpoint, with authentication, per-user quotas,
usage logging, and cost estimation handled centrally.

> **Note:** this repository contains synthetic/sample code generated for a
> demo and does not represent a real production system.

## Features

- **Provider abstraction** — `LLMService` normalizes OpenAI, Anthropic, and
  local vLLM/OpenAI-compatible backends behind one interface, including a
  streaming (SSE) mode.
- **Model catalog** — LLM backends are registered as `ModelConfig` rows
  (slug, upstream model name, context window, pricing) so callers reference
  a stable `model_slug` instead of hardcoding provider details.
- **Auth** — dual credential support: JWT bearer tokens for the dashboard
  and long-lived API keys (SHA-256 hashed at rest) for service-to-service
  calls.
- **Quotas & usage tracking** — per-user daily token quotas enforced before
  each call, with every request logged to `inference_logs` for billing and
  audit.
- **Observability** — structured logging (`structlog`), Prometheus metrics
  at `/metrics`, and a request-id propagated through every log line.

## Project layout

```
fake-ai-platform/
├── api/
│   ├── routes/
│   │   ├── models.py        # CRUD for the registered model catalog
│   │   └── inference.py     # Chat completion + streaming endpoints
│   ├── services/
│   │   ├── llm_service.py   # Provider routing, retries, token/cost math
│   │   └── auth_service.py  # JWT + API key authentication
│   └── main.py               # FastAPI app, middleware, metrics
├── core/
│   ├── config.py              # pydantic-settings environment config
│   └── database.py            # SQLAlchemy engine, session, ORM models
├── README.md
└── requirements.txt
```

## Running locally

```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt

cp .env.example .env  # set JWT_SECRET_KEY, DATABASE_URL, provider keys, etc.
uvicorn api.main:app --reload
```

The API will be available at `http://localhost:8000`, with interactive docs
at `http://localhost:8000/docs` (non-production environments only).

## Key environment variables

| Variable                 | Description                                   |
|---------------------------|------------------------------------------------|
| `JWT_SECRET_KEY`           | HMAC secret used to sign access/refresh tokens |
| `DATABASE_URL`             | Postgres connection string                    |
| `REDIS_URL`                | Used for rate limiting                        |
| `OPENAI_API_KEY`           | Upstream OpenAI credential                    |
| `ANTHROPIC_API_KEY`        | Upstream Anthropic credential                 |
| `LOCAL_INFERENCE_BASE_URL` | Base URL for a self-hosted OpenAI-compatible server |

## Example request

```bash
curl -X POST http://localhost:8000/api/v1/inference/chat \
  -H "X-API-Key: fak_..." \
  -H "Content-Type: application/json" \
  -d '{
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "Summarize this repo"}]
      }'
```
