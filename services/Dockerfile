FROM python:3.11-slim

WORKDIR /app

# Build fallback for sdist-only packages (cryptography ships wheels on
# common platforms; gcc/libffi-dev kept as a safety net)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libffi-dev && rm -rf /var/lib/apt/lists/*

# Dependency manifest first so this layer caches across source changes
COPY services/requirements.txt /app/services/requirements.txt
RUN pip install --no-cache-dir -r /app/services/requirements.txt

# Full repo source: hfc_types, core, zkp, dataset, simulator, monitoring,
# proto (pb2 files are checked in — no protoc needed at build time), services
COPY . /app

# Non-root runtime user; /var/lib/hfc is the shared volume mount point for
# node keys, WORM db and edge device keys (created with hfc ownership so the
# named volume inherits it on first use)
RUN useradd --uid 10001 --create-home hfc \
    && mkdir -p /var/lib/hfc \
    && chown -R hfc:hfc /var/lib/hfc

USER hfc

EXPOSE 50051 50052

# Individual services override this command:
#   python -m services.fog_service | services.cloud_service | services.edge_runner
CMD ["python", "-m", "services.entrypoint"]
