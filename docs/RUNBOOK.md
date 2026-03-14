# Egyptian ID OCR - Operational Runbooks

Complete operational procedures for deploying, monitoring, and maintaining the Egyptian ID OCR API.

## Table of Contents

1. [Deployment Runbook](#deployment-runbook)
2. [Monitoring Runbook](#monitoring-runbook)
3. [Incident Response](#incident-response)
4. [Backup & Recovery](#backup--recovery)
5. [Scaling Guide](#scaling-guide)

---

## Deployment Runbook

### Pre-Deployment Checklist

- [ ] Code changes reviewed and tested locally
- [ ] Docker build successful
- [ ] All tests passing
- [ ] Model weights verified
- [ ] Environment variables documented
- [ ] Rollback plan prepared

### Hugging Face Spaces Deployment

#### Step 1: Prepare Files

```bash
# Verify files exist
ls -la Dockerfile.hf README-hf.md .gitattributes
ls -la app/ weights/ models_cache/
```

#### Step 2: Run Deployment Script

```bash
# Linux/Mac
./scripts/deploy-hf.sh

# Windows
scripts\deploy-hf.bat
```

#### Step 3: Monitor Build

1. Go to: `https://huggingface.co/spaces/YOUR_USERNAME/egyptian-id-ocr`
2. Click "Logs" tab
3. Wait for "Running" status (5-10 minutes)

#### Step 4: Verify Deployment

```bash
# Health check
curl https://YOUR_USERNAME-egyptian-id-ocr.hf.space/api/v1/health

# Expected response:
# {"status": "healthy", "models_loaded": true, ...}
```

#### Step 5: Update Documentation

- Update README with new Space URL
- Notify team of deployment
- Update any API consumers

### Oracle Cloud Deployment

#### Step 1: Connect to VM

```bash
ssh -i ~/.ssh/oci_key ubuntu@YOUR_PUBLIC_IP
```

#### Step 2: Pull Latest Code

```bash
cd /opt/egyptian-ocr
sudo -u ubuntu git pull origin main
```

#### Step 3: Rebuild Docker Image

```bash
sudo docker-compose build --no-cache
```

#### Step 4: Rolling Restart

```bash
# Zero-downtime restart
sudo docker-compose up -d --force-recreate

# Wait for health check
sudo docker-compose ps
```

#### Step 5: Verify Deployment

```bash
# Health check
curl http://localhost:8000/api/v1/health

# Check logs
sudo docker-compose logs --tail=50

# Test extraction
curl -X POST http://localhost:8000/api/v1/extract \
  -F "file=@test_image.jpg"
```

#### Step 6: Monitor

```bash
# Watch resource usage
sudo docker stats

# Check system resources
htop
df -h
```

### Rollback Procedure

#### Hugging Face Spaces

```bash
# Revert to previous commit
git revert HEAD
git push

# Or reset to specific commit
git reset --hard <previous-commit-hash>
git push -f
```

#### Oracle Cloud

```bash
# Revert code
cd /opt/egyptian-ocr
sudo -u ubuntu git reset --hard <previous-commit>

# Rebuild and restart
sudo docker-compose build
sudo docker-compose up -d
```

---

## Monitoring Runbook

### Daily Checks

#### 1. Health Check

```bash
# Automated health check
curl -f https://YOUR_DOMAIN/api/v1/health

# Expected: HTTP 200 with models_loaded: true
```

#### 2. Resource Usage

```bash
# Docker stats
docker stats --no-stream

# System resources
free -h
df -h
uptime
```

#### 3. Log Review

```bash
# Recent errors
docker-compose logs --tail=100 | grep -i error

# Application logs
tail -100 /opt/egyptian-ocr/logs/app.log
```

### Weekly Checks

#### 1. Performance Metrics

```bash
# Run verification script
python scripts/verify-endpoints.py https://YOUR_DOMAIN --benchmark

# Check response times
# Target: <15s for extraction
```

#### 2. Disk Usage

```bash
# Check log rotation
ls -lh /opt/egyptian-ocr/logs/

# Clean old logs if needed
find /opt/egyptian-ocr/logs -name "*.log" -mtime +7 -delete
```

#### 3. Backup Verification

```bash
# Run backup
/opt/egyptian-ocr/backup.sh

# Verify backup exists
ls -lh /opt/backups/egyptian-ocr/
```

### Alerting Thresholds

| Metric | Warning | Critical | Action |
|--------|---------|----------|--------|
| Memory Usage | >80% | >95% | Scale up or optimize |
| CPU Usage | >80% | >95% | Check for loops |
| Disk Usage | >70% | >90% | Clean logs/backups |
| Response Time | >15s | >30s | Investigate bottleneck |
| Error Rate | >5% | >10% | Check logs immediately |
| Uptime | <99% | <95% | Review incidents |

---

## Incident Response

### Incident: API Not Responding

#### Symptoms
- Health check returns 503 or timeout
- Requests fail with connection error

#### Diagnosis

```bash
# Check container status
docker-compose ps

# Check logs
docker-compose logs --tail=100

# Check system resources
free -h
df -h
```

#### Resolution

```bash
# Restart container
docker-compose restart

# If still failing, rebuild
docker-compose down
docker-compose build
docker-compose up -d
```

#### Post-Incident

- Document root cause
- Update runbook if needed
- Consider adding monitoring alert

---

### Incident: High Memory Usage

#### Symptoms
- Memory >90%
- Slow responses
- OOM kills

#### Diagnosis

```bash
# Check memory usage
docker stats

# Check system memory
free -h
cat /proc/meminfo
```

#### Resolution

```bash
# Restart to clear memory
docker-compose restart

# Reduce workers if needed
# Edit docker-compose.yml: APP_WORKERS=1

# Add swap (Oracle Cloud)
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

---

### Incident: Slow Response Times

#### Symptoms
- Response time >15s
- Timeout errors

#### Diagnosis

```bash
# Run benchmark
python scripts/verify-endpoints.py https://YOUR_DOMAIN --benchmark

# Check CPU usage
docker stats
htop

# Check logs for slow operations
docker-compose logs | grep -i "processing_ms"
```

#### Resolution

1. **Check model loading:**
   ```bash
   curl https://YOUR_DOMAIN/api/v1/models
   ```

2. **Optimize settings:**
   - Reduce OCR_CPU_THREADS
   - Disable PaddleOCR if not needed

3. **Scale resources** (Oracle Cloud):
   - Increase CPU/RAM if on paid tier

---

### Incident: Model Loading Failure

#### Symptoms
- Health check shows models_loaded: false
- Extraction returns errors

#### Diagnosis

```bash
# Check model files
ls -lh /opt/egyptian-ocr/weights/

# Check logs for model errors
docker-compose logs | grep -i "model\|weight"
```

#### Resolution

```bash
# Re-download models
# (Add model download script to container)

# Or rebuild with correct weights
docker-compose build --no-cache
docker-compose up -d
```

---

## Backup & Recovery

### Daily Backup

```bash
# Automated via cron
0 3 * * * /opt/egyptian-ocr/backup.sh
```

### Manual Backup

```bash
# Run backup script
/opt/egyptian-ocr/backup.sh

# Verify backup
ls -lh /opt/backups/egyptian-ocr/

# Download backup locally
scp ubuntu@YOUR_IP:/opt/backups/egyptian-ocr/*.tar.gz ./backups/
```

### Recovery Procedure

#### 1. Stop Current Service

```bash
docker-compose down
```

#### 2. Restore Weights

```bash
# Find backup
ls /opt/backups/egyptian-ocr/

# Extract weights
tar -xzf /opt/backups/egyptian-ocr/weights_YYYYMMDD_HHMMSS.tar.gz -C /
```

#### 3. Restore Application

```bash
cd /opt/egyptian-ocr
git reset --hard <known-good-commit>
```

#### 4. Restart

```bash
docker-compose build
docker-compose up -d
```

#### 5. Verify

```bash
curl http://localhost:8000/api/v1/health
```

---

## Scaling Guide

### Vertical Scaling (Oracle Cloud)

#### Increase Resources

```bash
# Stop instance in Oracle Console
# Change shape to larger instance
# Start instance
```

**Note:** Always Free tier limited to 4 OCPUs, 24GB RAM

### Horizontal Scaling

#### Add More Instances

```bash
# Deploy to multiple VMs
# Use load balancer (Oracle LB or nginx)

# Example nginx upstream:
upstream egyptian_ocr {
    server instance1:8000;
    server instance2:8000;
    server instance3:8000;
}
```

### Optimization

#### Reduce Memory Usage

```bash
# In docker-compose.yml
environment:
  - OCR_CPU_THREADS=2
  - APP_WORKERS=1
```

#### Reduce CPU Usage

```bash
# Limit Docker CPU
docker-compose up -d --cpus="2"
```

#### Enable Caching

```bash
# Add Redis to docker-compose.yml
# Cache OCR results for duplicate images
```

---

## Maintenance Schedule

### Daily
- [ ] Health check passes
- [ ] No errors in logs
- [ ] Resource usage normal

### Weekly
- [ ] Performance benchmark
- [ ] Log rotation check
- [ ] Backup verification

### Monthly
- [ ] System updates
- [ ] Security patches
- [ ] Dependency updates
- [ ] Documentation review

### Quarterly
- [ ] Disaster recovery test
- [ ] Capacity planning
- [ ] Cost review
- [ ] Performance optimization

---

## Contact & Escalation

### Support Channels

- **GitHub Issues:** [Your Repo]
- **Email:** your-email@example.com
- **On-Call:** [Your on-call rotation]

### Escalation Path

1. **Level 1:** Check runbook, restart service
2. **Level 2:** Contact on-call engineer
3. **Level 3:** Escalate to infrastructure team

---

## Appendix

### Useful Commands

```bash
# View all containers
docker ps -a

# View logs
docker-compose logs -f

# Restart service
docker-compose restart

# Rebuild
docker-compose build --no-cache

# Clean up
docker system prune -a

# Check disk
df -h

# Check memory
free -h

# Check CPU
htop

# Network stats
iftop
```

### Configuration Files

- `docker-compose.yml` - Container orchestration
- `nginx.conf` - Reverse proxy config
- `egyptian-ocr.service` - Systemd service
- `.env` - Environment variables

### Monitoring Tools

- **Built-in:** `/api/v1/health`, `/api/v1/models`
- **Docker:** `docker stats`
- **System:** `htop`, `iftop`, `iotop`
- **Logs:** `/var/log/nginx/`, `docker-compose logs`
