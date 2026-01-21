FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY pyproject.toml ./
COPY src ./src
COPY api ./api
COPY configs ./configs

# Install Python dependencies
RUN pip install --no-cache-dir -e . && \
    pip install --no-cache-dir \
    fastapi \
    uvicorn[standard] \
    structlog \
    apscheduler \
    pyyaml

# Expose port
EXPOSE 8001

# Run API server
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8001"]
