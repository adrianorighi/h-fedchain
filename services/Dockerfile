FROM python:3.11-slim

WORKDIR /app

# Install system deps for cryptography
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libffi-dev && rm -rf /var/lib/apt/lists/*

# Copy project package
COPY pyproject.toml /app/h-fedchain/
COPY hfc_types/ /app/h-fedchain/hfc_types/
COPY core/ /app/h-fedchain/core/
COPY zkp/ /app/h-fedchain/zkp/
COPY proto/ /app/h-fedchain/proto/

# Install deps
RUN pip install --no-cache-dir "grpcio>=1.60" "protobuf>=4.25" "paho-mqtt>=1.6" \
    "numpy>=1.24" "cryptography>=41.0" "py_ecc>=8.0.0"
RUN pip install --no-cache-dir -e /app/h-fedchain

# Copy service entrypoints
COPY services/ /app/

# Expose gRPC ports
EXPOSE 50051 50052

# Entrypoint reads SERVICE env var
CMD ["python", "-m", "services.entrypoint"]
