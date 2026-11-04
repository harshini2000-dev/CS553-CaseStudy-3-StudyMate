
FROM python:3.10-slim

WORKDIR /opt/app

# Copy project files
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install Prometheus Node Exporter
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ca-certificates \
    prometheus-node-exporter && \
    rm -rf /var/lib/apt/lists/*

# Application and monitoring ports
EXPOSE 6550
EXPOSE 8000
EXPOSE 9100

# Gradio configuration
ENV GRADIO_SERVER_NAME="0.0.0.0"
ENV PORT=6550

# Start Node Exporter and StudyMate
CMD ["bash", "-c", "prometheus-node-exporter --web.listen-address=:9100 & exec python /opt/app/app.py"]
