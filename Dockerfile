FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl && \
    rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
RUN pip install --no-cache-dir -e .

COPY . .

RUN mkdir -p /app/data/chroma

ENV PYTHONPATH=/app/src
ENV LOG_LEVEL=INFO

ENTRYPOINT ["matresearcher"]
CMD ["--help"]
