"""FastAPI application entrypoint: wires up routers, middleware, and
startup/shutdown hooks. Run locally with:

    uvicorn api.main:app --reload
"""
import time
import uuid

import structlog
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Histogram, make_asgi_app

from api.routes import inference, models
from core.config import get_settings
from core.database import init_db

settings = get_settings()
logger = structlog.get_logger(settings.app_name)

REQUEST_COUNT = Counter(
    "fake_ai_platform_requests_total", "Total HTTP requests", ["method", "path", "status_code"]
)
REQUEST_LATENCY = Histogram(
    "fake_ai_platform_request_latency_seconds", "Request latency in seconds", ["method", "path"]
)

app = FastAPI(
    title=settings.app_name,
    description="Internal gateway for routing chat completion requests across multiple LLM providers.",
    version="1.4.0",
    docs_url="/docs" if settings.environment != "production" else None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.environment == "local" else [],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_context_middleware(request: Request, call_next):
    """Attach a request id for tracing and record Prometheus metrics."""
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    structlog.contextvars.bind_contextvars(request_id=request_id)

    started_at = time.perf_counter()
    response = await call_next(request)
    elapsed = time.perf_counter() - started_at

    route_path = request.scope.get("route").path if request.scope.get("route") else request.url.path
    REQUEST_COUNT.labels(request.method, route_path, response.status_code).inc()
    REQUEST_LATENCY.labels(request.method, route_path).observe(elapsed)

    response.headers["X-Request-ID"] = request_id
    logger.info(
        "request_completed",
        method=request.method,
        path=route_path,
        status_code=response.status_code,
        duration_ms=round(elapsed * 1000, 2),
    )
    return response


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("unhandled_exception", path=request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


app.include_router(models.router, prefix=settings.api_v1_prefix)
app.include_router(inference.router, prefix=settings.api_v1_prefix)

if settings.enable_prometheus:
    app.mount("/metrics", make_asgi_app())


@app.get("/health", tags=["health"])
def health_check():
    return {"status": "ok", "environment": settings.environment}


@app.on_event("startup")
def on_startup():
    logger.info("starting_up", environment=settings.environment)
    if settings.environment == "local":
        init_db()


@app.on_event("shutdown")
def on_shutdown():
    logger.info("shutting_down")
