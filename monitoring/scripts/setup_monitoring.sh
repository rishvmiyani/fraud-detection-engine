#!/bin/bash
# Setup Monitoring Infrastructure for Fraud Detection Engine

set -e

echo "Setting up monitoring infrastructure..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    print_error "Docker is not running. Please start Docker first."
    exit 1
fi

# Create monitoring network if it doesn't exist
docker network inspect monitoring-network >/dev/null 2>&1 || {
    print_status "Creating monitoring network..."
    docker network create monitoring-network
}

# Create directories for persistent data
print_status "Creating persistent storage directories..."
mkdir -p ./data/prometheus
mkdir -p ./data/grafana
mkdir -p ./data/alertmanager
mkdir -p ./data/elasticsearch
mkdir -p ./data/kibana

# Set proper permissions
sudo chown -R 472:472 ./data/grafana
sudo chown -R 1000:1000 ./data/prometheus
sudo chown -R 1000:1000 ./data/alertmanager

# Start Prometheus
print_status "Starting Prometheus..."
docker run -d \
  --name prometheus \
  --network monitoring-network \
  -p 9090:9090 \
  -v $(pwd)/prometheus/prometheus.yml:/etc/prometheus/prometheus.yml \
  -v $(pwd)/prometheus/rules:/etc/prometheus/rules \
  -v $(pwd)/data/prometheus:/prometheus \
  -u root \
  --restart unless-stopped \
  prom/prometheus:latest \
  --config.file=/etc/prometheus/prometheus.yml \
  --storage.tsdb.path=/prometheus \
  --web.console.libraries=/etc/prometheus/console_libraries \
  --web.console.templates=/etc/prometheus/consoles \
  --storage.tsdb.retention.time=30d \
  --web.enable-lifecycle

# Start AlertManager
print_status "Starting AlertManager..."
docker run -d \
  --name alertmanager \
  --network monitoring-network \
  -p 9093:9093 \
  -v $(pwd)/alertmanager/alertmanager.yml:/etc/alertmanager/alertmanager.yml \
  -v $(pwd)/data/alertmanager:/alertmanager \
  --restart unless-stopped \
  prom/alertmanager:latest

# Start Grafana
print_status "Starting Grafana..."
docker run -d \
  --name grafana \
  --network monitoring-network \
  -p 3000:3000 \
  -v $(pwd)/grafana/provisioning:/etc/grafana/provisioning \
  -v $(pwd)/grafana/dashboards:/var/lib/grafana/dashboards \
  -v $(pwd)/data/grafana:/var/lib/grafana \
  -e GF_SECURITY_ADMIN_PASSWORD=admin123 \
  -e GF_USERS_ALLOW_SIGN_UP=false \
  -e GF_INSTALL_PLUGINS=grafana-piechart-panel,grafana-worldmap-panel \
  --restart unless-stopped \
  grafana/grafana:latest

# Start Node Exporter
print_status "Starting Node Exporter..."
docker run -d \
  --name node-exporter \
  --network monitoring-network \
  -p 9100:9100 \
  -v "/proc:/host/proc:ro" \
  -v "/sys:/host/sys:ro" \
  -v "/:/rootfs:ro" \
  --restart unless-stopped \
  prom/node-exporter:latest \
  --path.procfs=/host/proc \
  --path.rootfs=/rootfs \
  --path.sysfs=/host/sys \
  --collector.filesystem.mount-points-exclude='^/(sys|proc|dev|host|etc)($$|/)'

# Start cAdvisor
print_status "Starting cAdvisor..."
docker run -d \
  --name cadvisor \
  --network monitoring-network \
  -p 8080:8080 \
  -v /:/rootfs:ro \
  -v /var/run:/var/run:ro \
  -v /sys:/sys:ro \
  -v /var/lib/docker/:/var/lib/docker:ro \
  -v /dev/disk/:/dev/disk:ro \
  --privileged \
  --device=/dev/kmsg \
  --restart unless-stopped \
  gcr.io/cadvisor/cadvisor:latest

# Start Elasticsearch
print_status "Starting Elasticsearch..."
docker run -d \
  --name elasticsearch \
  --network monitoring-network \
  -p 9200:9200 \
  -p 9300:9300 \
  -v $(pwd)/data/elasticsearch:/usr/share/elasticsearch/data \
  -e "discovery.type=single-node" \
  -e "xpack.security.enabled=false" \
  -e "ES_JAVA_OPTS=-Xms512m -Xmx512m" \
  --restart unless-stopped \
  docker.elastic.co/elasticsearch/elasticsearch:8.9.0

# Start Kibana
print_status "Starting Kibana..."
docker run -d \
  --name kibana \
  --network monitoring-network \
  -p 5601:5601 \
  -v $(pwd)/data/kibana:/usr/share/kibana/data \
  -e "ELASTICSEARCH_HOSTS=http://elasticsearch:9200" \
  --restart unless-stopped \
  docker.elastic.co/kibana/kibana:8.9.0

# Wait for services to start
print_status "Waiting for services to start..."
sleep 30

# Check service health
print_status "Checking service health..."

services=("prometheus:9090" "grafana:3000" "alertmanager:9093" "elasticsearch:9200" "kibana:5601")

for service in "${services[@]}"; do
    name=${service%:*}
    port=${service#*:}
    if curl -s "http://localhost:$port" > /dev/null; then
        print_status "$name is healthy on port $port"
    else
        print_warning "$name might not be ready yet on port $port"
    fi
done

print_status "Monitoring stack setup complete!"
print_status "Access URLs:"
print_status "  - Prometheus: http://localhost:9090"
print_status "  - Grafana: http://localhost:3000 (admin/admin123)"
print_status "  - AlertManager: http://localhost:9093"
print_status "  - Elasticsearch: http://localhost:9200"
print_status "  - Kibana: http://localhost:5601"

print_warning "Please wait a few more minutes for all services to fully initialize."
