#!/bin/bash
# Egyptian ID OCR - Oracle Cloud VM Setup Script
# Run this script on your Oracle Cloud VM after provisioning
#
# Usage:
#   curl -o setup-oracle.sh <script-url>
#   chmod +x setup-oracle.sh
#   sudo ./setup-oracle.sh

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Egyptian ID OCR - Oracle VM Setup${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    print_error "Please run as root (sudo ./setup-oracle.sh)"
    exit 1
fi

# System update
print_info "Updating system packages..."
apt-get update
apt-get upgrade -y

# Install Docker
print_info "Installing Docker..."
if ! command -v docker &> /dev/null; then
    curl -fsSL https://get.docker.com -o get-docker.sh
    sh get-docker.sh
    rm get-docker.sh
else
    print_warning "Docker already installed"
fi

# Add user to docker group
print_info "Adding ubuntu user to docker group..."
usermod -aG docker ubuntu

# Install Docker Compose
print_info "Installing Docker Compose..."
DOCKER_CONFIG=${DOCKER_CONFIG:-$HOME/.docker}
mkdir -p $DOCKER_CONFIG/cli-plugins
curl -SL https://github.com/docker/compose/releases/download/v2.24.0/docker-compose-linux-aarch64 -o /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose

# Install additional dependencies
print_info "Installing system dependencies..."
apt-get install -y \
    nginx \
    certbot \
    python3-certbot-nginx \
    git \
    curl \
    wget \
    htop \
    vim \
    ufw \
    fail2ban \
    unzip

# Configure UFW firewall
print_info "Configuring firewall..."
ufw --force enable
ufw allow 22/tcp    # SSH
ufw allow 80/tcp    # HTTP
ufw allow 443/tcp   # HTTPS
ufw allow 8000/tcp  # API (direct access)
ufw logging on

# Configure fail2ban
print_info "Configuring fail2ban..."
cat > /etc/fail2ban/jail.local << 'EOF'
[DEFAULT]
bantime = 3600
findtime = 600
maxretry = 5

[sshd]
enabled = true
port = 22
filter = sshd
logpath = /var/log/auth.log
maxretry = 3
EOF

systemctl restart fail2ban

# Create application directory
print_info "Creating application directory..."
mkdir -p /opt/egyptian-ocr
cd /opt/egyptian-ocr

# Create docker-compose.yml
print_info "Creating Docker Compose configuration..."
cat > docker-compose.yml << 'EOF'
version: '3.8'

services:
  egyptian-ocr:
    build:
      context: .
      dockerfile: Dockerfile.oracle
    container_name: egyptian-ocr-api
    restart: unless-stopped
    ports:
      - "127.0.0.1:8000:8000"
    environment:
      - APP_ENV=production
      - APP_HOST=0.0.0.0
      - APP_PORT=8000
      - APP_WORKERS=2
      - YOLO_CONF_THRESHOLD=0.50
      - OCR_CPU_THREADS=4
      - MAX_IMAGE_SIZE_MB=10
      - LOG_LEVEL=INFO
    volumes:
      - ./logs:/app/logs
      - ./models_cache:/app/models_cache
      - ./weights:/app/weights
    cpus: "4"
    memory: "20g"
    memswap_limit: "24g"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/api/v1/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s
    networks:
      - ocr-network

  # Optional: Add Redis for caching
  # redis:
  #   image: redis:7-alpine
  #   container_name: egyptian-ocr-redis
  #   restart: unless-stopped
  #   volumes:
  #     - redis-data:/data
  #   networks:
  #     - ocr-network

networks:
  ocr-network:
    driver: bridge

# volumes:
#   redis-data:
EOF

# Create systemd service
print_info "Creating systemd service..."
cat > /etc/systemd/system/egyptian-ocr.service << 'EOF'
[Unit]
Description=Egyptian ID OCR API
Requires=docker.service
After=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=/opt/egyptian-ocr
ExecStart=/usr/local/bin/docker-compose up -d
ExecStop=/usr/local/bin/docker-compose down
ExecReload=/usr/local/bin/docker-compose restart

[Install]
WantedBy=multi-user.target
EOF

# Create logrotate configuration
print_info "Creating logrotate configuration..."
cat > /etc/logrotate.d/egyptian-ocr << 'EOF'
/opt/egyptian-ocr/logs/*.log {
    daily
    rotate 7
    compress
    delaycompress
    notifempty
    create 0644 ubuntu ubuntu
    missingok
}
EOF

# Create Nginx configuration
print_info "Creating Nginx configuration..."
cat > /etc/nginx/sites-available/egyptian-ocr << 'EOF'
# HTTP - Redirect to HTTPS
server {
    listen 80;
    listen [::]:80;
    server_name _;

    # Let's Encrypt validation
    location /.well-known/acme-challenge/ {
        root /var/www/certbot;
    }

    # Redirect all HTTP to HTTPS
    location / {
        return 301 https://$server_name$request_uri;
    }
}

# HTTPS Server
server {
    listen 443 ssl http2;
    listen [::]:443 ssl http2;
    server_name _;

    # SSL Configuration (update after certbot)
    ssl_certificate /etc/letsencrypt/live/YOUR_DOMAIN/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/YOUR_DOMAIN/privkey.pem;

    # SSL Settings
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_prefer_server_ciphers on;
    ssl_ciphers ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256;
    ssl_session_cache shared:SSL:10m;
    ssl_session_timeout 10m;

    # Security Headers
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;

    # Rate Limiting
    limit_req_zone $binary_remote_addr zone=api_limit:10m rate=10r/s;

    # Client body size (for image uploads)
    client_max_body_size 15M;

    # API Proxy
    location / {
        limit_req zone=api_limit burst=20 nodelay;

        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_cache_bypass $http_upgrade;

        # Timeouts for long-running OCR requests
        proxy_connect_timeout 60s;
        proxy_send_timeout 120s;
        proxy_read_timeout 120s;
    }

    # Health check endpoint (no rate limit)
    location /api/v1/health {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    # Static files (if needed)
    location /static {
        alias /opt/egyptian-ocr/static;
        expires 30d;
        add_header Cache-Control "public, immutable";
    }

    # Logs
    access_log /var/log/nginx/egyptian-ocr-access.log;
    error_log /var/log/nginx/egyptian-ocr-error.log;
}
EOF

# Enable Nginx site
ln -sf /etc/nginx/sites-available/egyptian-ocr /etc/nginx/sites-enabled/
rm -f /etc/nginx/sites-enabled/default

# Test Nginx configuration
nginx -t

# Create startup script
print_info "Creating startup script..."
cat > /opt/egyptian-ocr/start.sh << 'EOF'
#!/bin/bash
cd /opt/egyptian-ocr

# Pull or build Docker image
if [ ! -f ".built" ]; then
    echo "Building Docker image..."
    docker-compose build
    touch .built
fi

# Start services
docker-compose up -d

# Show status
docker-compose ps
EOF
chmod +x /opt/egyptian-ocr/start.sh

# Create backup script
print_info "Creating backup script..."
cat > /opt/egyptian-ocr/backup.sh << 'EOF'
#!/bin/bash
BACKUP_DIR="/opt/backups/egyptian-ocr"
DATE=$(date +%Y%m%d_%H%M%S)

mkdir -p "$BACKUP_DIR"

echo "Creating backup: $DATE"

# Backup weights and cache
tar -czf "$BACKUP_DIR/weights_$DATE.tar.gz" /opt/egyptian-ocr/weights
tar -czf "$BACKUP_DIR/cache_$DATE.tar.gz" /opt/egyptian-ocr/models_cache

# Backup logs
tar -czf "$BACKUP_DIR/logs_$DATE.tar.gz" /opt/egyptian-ocr/logs

# Keep only last 7 backups
find "$BACKUP_DIR" -name "*.tar.gz" -mtime +7 -delete

echo "Backup complete: $BACKUP_DIR"
EOF
chmod +x /opt/egyptian-ocr/backup.sh

# Create monitoring script
print_info "Creating monitoring script..."
cat > /opt/egyptian-ocr/monitor.sh << 'EOF'
#!/bin/bash
echo "=== Egyptian ID OCR - System Status ==="
echo ""

echo "Container Status:"
docker-compose ps
echo ""

echo "Memory Usage:"
docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}"
echo ""

echo "Disk Usage:"
df -h /opt
echo ""

echo "Recent Logs:"
docker-compose logs --tail=20
EOF
chmod +x /opt/egyptian-ocr/monitor.sh

# Enable and start services
print_info "Enabling systemd service..."
systemctl daemon-reload
systemctl enable egyptian-ocr

print_info "Starting Nginx..."
systemctl enable nginx
systemctl start nginx

# Print summary
echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Setup Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Next steps:"
echo ""
echo "1. Clone your application code:"
echo "   cd /opt/egyptian-ocr"
echo "   git clone <your-repo> ."
echo ""
echo "2. Copy your model weights to /opt/egyptian-ocr/weights/"
echo ""
echo "3. Build and start the application:"
echo "   /opt/egyptian-ocr/start.sh"
echo ""
echo "4. Set up SSL (replace with your domain):"
echo "   certbot --nginx -d your-domain.com"
echo ""
echo "5. Monitor the application:"
echo "   /opt/egyptian-ocr/monitor.sh"
echo ""
echo "Useful commands:"
echo "  systemctl status egyptian-ocr  # Check service status"
echo "  docker-compose logs -f         # View logs"
echo "  docker-compose restart         # Restart application"
echo "  /opt/egyptian-ocr/backup.sh    # Create backup"
echo ""
print_success "Oracle Cloud VM setup complete!"
