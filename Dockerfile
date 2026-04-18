FROM python:3.11-slim

# Install uv directly from Astral
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set the working directory
WORKDIR /app

# Enable bytecode compilation for faster startup
ENV UV_COMPILE_BYTECODE=1

# Copy the monorepo configuration
COPY pyproject.toml uv.lock ./

# Copy the core packages and the API server
COPY packages/ ./packages/
COPY apps/ ./apps/

# Sync the workspace (this builds a highly optimized .venv)
RUN uv sync --frozen --no-dev

# Expose the standard web port
EXPOSE 8000

# Run Uvicorn from the uv virtual environment
CMD ["/app/.venv/bin/uvicorn", "qgate_gateway.main:app", "--host", "0.0.0.0", "--port", "8000", "--app-dir", "apps/api-server/src"]