FROM python:3.11-slim

WORKDIR /app

# Copy the h-fedchain package first (for pip install -e)
COPY pyproject.toml /app/h-fedchain/
RUN mkdir -p /app/h-fedchain/hfc_types /app/h-fedchain/core /app/h-fedchain/zkp
COPY hfc_types/ /app/h-fedchain/hfc_types/
COPY core/ /app/h-fedchain/core/
COPY zkp/ /app/h-fedchain/zkp/

# Copy service requirements and install dependencies
COPY services/requirements.txt /app/
RUN pip install --no-cache-dir -r /app/requirements.txt

# Install the local package
RUN pip install --no-cache-dir -e /app/h-fedchain

COPY services/fog_node_service.py /app/

EXPOSE 8000

CMD ["uvicorn", "fog_node_service:app", "--host", "0.0.0.0", "--port", "8000"]
