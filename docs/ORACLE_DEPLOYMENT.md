# Egyptian ID OCR - Oracle Cloud Free Tier Deployment Guide

Complete step-by-step guide for deploying the Egyptian ID OCR API to Oracle Cloud Always Free tier.

## Overview

**Oracle Cloud Free Tier** offers the best free VM resources:
- **4 OCPUs** (ARM Ampere A1)
- **24GB RAM** (excellent for our models)
- **200GB storage**
- **Always Free** (no time limit)
- **Full VM control**

**Requirements:**
- Credit card for verification (not charged)
- Manual setup via Console or Terraform
- Basic Linux knowledge

## Prerequisites

1. **Oracle Cloud Account**
   - Visit [cloud.oracle.com](https://cloud.oracle.com)
   - Sign up for Free Tier
   - Complete identity verification

2. **SSH Key Pair**
   ```bash
   # Generate SSH key (Windows: use Git Bash or PowerShell)
   ssh-keygen -t ed25519 -f ~/.ssh/oci_key -C "egyptian-ocr"
   
   # View public key
   cat ~/.ssh/oci_key.pub
   ```

3. **Terraform** (optional, for automated provisioning)
   - Download: [terraform.io](https://www.terraform.io/downloads)

## Step 1: Create VM Instance (Manual Method)

### 1.1 Sign in to Oracle Cloud Console

1. Go to [cloud.oracle.com](https://cloud.oracle.com)
2. Sign in with your account
3. Select your home region

### 1.2 Create Compute Instance

1. **Navigate to Compute**
   - Click "Create instance"

2. **Configure Instance**
   ```
   Name: egyptian-ocr-vm
   Compartment: root (or your compartment)
   Availability Domain: Any (pick one)
   ```

3. **Choose Image**
   ```
   Image: Ubuntu 22.04 aarch64 (ARM)
   ```

4. **Choose Shape**
   ```
   Shape: VM.Standard.A1.Flex
   OCPUs: 4
   Memory: 24 GB
   ```

5. **Networking**
   ```
   VCN: Create new VCN (default settings)
   Subnet: Public subnet
   Assign public IPv4: Yes
   ```

6. **Add SSH Key**
   - Select "Paste public keys"
   - Paste content of `~/.ssh/oci_key.pub`

7. **Boot Volume**
   ```
   Size: 200 GB (Always Free limit)
   ```

8. **Click "Create"**
   - Wait 2-5 minutes for provisioning
   - Note the Public IP address

### 1.3 Configure Security List

1. **Go to Virtual Cloud Networks**
2. **Click your VCN**
3. **Click Security Lists**
4. **Add Ingress Rules:**
   ```
   Port 22: SSH (already added)
   Port 80: HTTP
   Port 443: HTTPS
   Port 8000: API
   ```

## Step 2: Connect to VM

```bash
# SSH into the VM
ssh -i ~/.ssh/oci_key ubuntu@YOUR_PUBLIC_IP

# Test connection
uptime
free -h
df -h
```

## Step 3: Run Setup Script

### Option A: Automated Setup (Recommended)

1. **Download and run setup script:**
   ```bash
   # On the VM
   cd ~
   curl -O https://raw.githubusercontent.com/YOUR_REPO/scripts/main/setup-oracle.sh
   chmod +x setup-oracle.sh
   sudo ./setup-oracle.sh
   ```

2. **Wait for setup to complete** (~5-10 minutes)

### Option B: Manual Setup

1. **Update system:**
   ```bash
   sudo apt update && sudo apt upgrade -y
   ```

2. **Install Docker:**
   ```bash
   curl -fsSL https://get.docker.com -o get-docker.sh
   sudo sh get-docker.sh
   sudo usermod -aG docker ubuntu
   ```

3. **Install Docker Compose:**
   ```bash
   sudo curl -L "https://github.com/docker/compose/releases/download/v2.24.0/docker-compose-$(uname -s)-$(uname -m)" \
     -o /usr/local/bin/docker-compose
   sudo chmod +x /usr/local/bin/docker-compose
   ```

4. **Install additional tools:**
   ```bash
   sudo apt install -y nginx certbot python3-certbot-nginx git curl wget htop ufw fail2ban
   ```

5. **Configure firewall:**
   ```bash
   sudo ufw --force enable
   sudo ufw allow 22/tcp
   sudo ufw allow 80/tcp
   sudo ufw allow 443/tcp
   sudo ufw allow 8000/tcp
   ```

## Step 4: Deploy Application

1. **Create application directory:**
   ```bash
   sudo mkdir -p /opt/egyptian-ocr
   sudo chown ubuntu:ubuntu /opt/egyptian-ocr
   cd /opt/egyptian-ocr
   ```

2. **Clone your repository:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/egyptian-id-ocr.git .
   ```

3. **Copy Dockerfile:**
   ```bash
   cp Dockerfile.oracle Dockerfile
   ```

4. **Build Docker image:**
   ```bash
   docker-compose build
   ```

5. **Start the application:**
   ```bash
   docker-compose up -d
   ```

6. **Check status:**
   ```bash
   docker-compose ps
   docker-compose logs -f
   ```

## Step 5: Configure Nginx Reverse Proxy

1. **Copy nginx configuration:**
   ```bash
   sudo cp /path/to/nginx.conf /etc/nginx/sites-available/egyptian-ocr
   ```

2. **Update domain in nginx.conf:**
   ```bash
   sudo nano /etc/nginx/sites-available/egyptian-ocr
   # Replace YOUR_DOMAIN with your actual domain
   ```

3. **Enable the site:**
   ```bash
   sudo ln -s /etc/nginx/sites-available/egyptian-ocr /etc/nginx/sites-enabled/
   sudo rm /etc/nginx/sites-enabled/default
   sudo nginx -t
   sudo systemctl reload nginx
   ```

## Step 6: Set Up SSL Certificate

1. **Point domain to VM:**
   - Add A record: `your-domain.com → YOUR_PUBLIC_IP`

2. **Get Let's Encrypt certificate:**
   ```bash
   sudo certbot --nginx -d your-domain.com
   ```

3. **Auto-renewal is configured automatically**

## Step 7: Configure Systemd Service

1. **Copy service file:**
   ```bash
   sudo cp egyptian-ocr.service /etc/systemd/system/
   ```

2. **Enable and start:**
   ```bash
   sudo systemctl daemon-reload
   sudo systemctl enable egyptian-ocr
   sudo systemctl start egyptian-ocr
   ```

3. **Check status:**
   ```bash
   sudo systemctl status egyptian-ocr
   ```

## Step 8: Test the API

### Without Domain (Direct IP)

```bash
# Health check
curl http://YOUR_PUBLIC_IP:8000/api/v1/health

# Extract
curl -X POST "http://YOUR_PUBLIC_IP:8000/api/v1/extract" \
  -F "file=@id_card.jpg"
```

### With Domain (HTTPS)

```bash
# Health check
curl https://your-domain.com/api/v1/health

# Extract
curl -X POST "https://your-domain.com/api/v1/extract" \
  -F "file=@id_card.jpg"
```

## Step 9: Set Up Monitoring

### Create monitoring script

```bash
cat > /opt/egyptian-ocr/monitor.sh << 'EOF'
#!/bin/bash
echo "=== Egyptian ID OCR - Status ==="
echo ""
echo "Container Status:"
docker-compose ps
echo ""
echo "Memory Usage:"
docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}"
echo ""
echo "Disk Usage:"
df -h /opt
EOF
chmod +x /opt/egyptian-ocr/monitor.sh
```

### Run monitoring

```bash
/opt/egyptian-ocr/monitor.sh
```

## Step 10: Set Up Backups

### Create backup script

```bash
cat > /opt/egyptian-ocr/backup.sh << 'EOF'
#!/bin/bash
BACKUP_DIR="/opt/backups/egyptian-ocr"
DATE=$(date +%Y%m%d_%H%M%S)
mkdir -p "$BACKUP_DIR"

tar -czf "$BACKUP_DIR/weights_$DATE.tar.gz" /opt/egyptian-ocr/weights
tar -czf "$BACKUP_DIR/cache_$DATE.tar.gz" /opt/egyptian-ocr/models_cache

# Keep only last 7 backups
find "$BACKUP_DIR" -name "*.tar.gz" -mtime +7 -delete
echo "Backup complete: $BACKUP_DIR"
EOF
chmod +x /opt/egyptian-ocr/backup.sh
```

### Add to crontab

```bash
# Edit crontab
crontab -e

# Add daily backup at 3 AM
0 3 * * * /opt/egyptian-ocr/backup.sh
```

## Terraform Deployment (Optional)

For automated infrastructure:

1. **Configure OCI CLI:**
   ```bash
   oci setup config
   ```

2. **Initialize Terraform:**
   ```bash
   cd terraform/
   terraform init
   ```

3. **Create terraform.tfvars:**
   ```bash
   cp variables.tf terraform.tfvars
   # Edit with your values
   ```

4. **Deploy:**
   ```bash
   terraform plan
   terraform apply
   ```

## Troubleshooting

### Cannot Connect via SSH

**Problem:** SSH connection refused

**Solutions:**
1. Check security list allows port 22
2. Verify SSH key permissions: `chmod 600 ~/.ssh/oci_key`
3. Check VM status in Oracle Console

### Docker Build Fails

**Problem:** Docker build fails on ARM

**Solutions:**
1. Use `Dockerfile.oracle` (ARM-optimized)
2. Check available memory: `free -h`
3. Add swap if needed:
   ```bash
   sudo fallocate -l 4G /swapfile
   sudo chmod 600 /swapfile
   sudo mkswap /swapfile
   sudo swapon /swapfile
   ```

### Application Crashes

**Problem:** Container keeps restarting

**Solutions:**
1. Check logs: `docker-compose logs`
2. Verify model files exist in `weights/`
3. Check memory: `docker stats`

### SSL Certificate Fails

**Problem:** Certbot cannot obtain certificate

**Solutions:**
1. Verify domain points to VM IP
2. Check port 80 is open
3. Stop nginx temporarily: `sudo systemctl stop nginx`

## Performance Tuning

### Optimize for ARM

The `Dockerfile.oracle` is already optimized for ARM:
- Uses ARM-compatible base images
- Configures CPU threads for ARM cores

### Memory Management

```bash
# In docker-compose.yml
memory: "20g"        # Limit container memory
memswap_limit: "24g" # Allow swap usage
```

### CPU Optimization

```bash
# Set environment variables
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
```

## Cost Breakdown

**Oracle Cloud Always Free:**
- ✅ Compute (4 OCPUs, 24GB RAM): $0/month
- ✅ Storage (200GB): $0/month
- ✅ Network egress (10TB/month): $0/month
- ✅ Public IP: $0/month
- **Total: $0/month**

**Potential Costs:**
- Domain name: ~$10-15/year
- If you exceed free tier limits (unlikely for testing)

## Security Checklist

- [ ] SSH key authentication only (disable password)
- [ ] UFW firewall configured
- [ ] Fail2ban installed
- [ ] SSL certificate valid
- [ ] Regular system updates
- [ ] Backup script running
- [ ] Monitoring in place

## Useful Commands

```bash
# View logs
docker-compose logs -f

# Restart application
docker-compose restart

# Check resource usage
docker stats
htop

# View system logs
journalctl -u docker -f

# Check nginx logs
tail -f /var/log/nginx/egyptian-ocr-error.log

# Renew SSL certificate
sudo certbot renew --dry-run

# Update system
sudo apt update && sudo apt upgrade -y
```

## Support

- **Oracle Docs:** [docs.oracle.com](https://docs.oracle.com/en-us/iaas/)
- **Always Free:** [oracle.com/cloud/free](https://www.oracle.com/cloud/free/)
- **Community:** [Oracle Cloud Community](https://community.oracle.com/)
