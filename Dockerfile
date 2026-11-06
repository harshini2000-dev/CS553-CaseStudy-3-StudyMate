FROM python:3.10-slim

WORKDIR /opt/app

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

# System deps + Prometheus Node Exporter
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        bash \
        ca-certificates \
        prometheus-node-exporter && \
    rm -rf /var/lib/apt/lists/*

# CPU-only PyTorch (avoids multi-GB CUDA libraries on a CPU VM)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Inside the container: app 6550, Python metrics 8000, node exporter 9100
ENV GRADIO_SERVER_NAME=0.0.0.0 \
    PORT=6550 \
    METRICS_PORT=8000 \
    MODE=both

EXPOSE 6550
EXPOSE 8000
EXPOSE 9100

CMD ["bash", "-c", "prometheus-node-exporter --web.listen-address=:9100 & exec python /opt/app/app.py"]