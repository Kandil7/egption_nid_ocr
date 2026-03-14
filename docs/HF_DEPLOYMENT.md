# Egyptian ID OCR - Hugging Face Spaces Deployment Guide

Complete step-by-step guide for deploying the Egyptian ID OCR API to Hugging Face Spaces for free.

## Overview

**Hugging Face Spaces** offers free Docker-based hosting with:
- **2 vCPU** cores
- **16GB RAM** (sufficient for our models)
- **50GB storage**
- **No credit card required**
- **Public or private spaces**

**Limitations:**
- Cold start: ~30-60 seconds
- Sleeps after inactivity (wakes on next request)
- Public by default (can be private)

## Prerequisites

1. **Hugging Face Account**
   - Visit [huggingface.co/join](https://huggingface.co/join)
   - Create a free account (email or GitHub login)

2. **Git Installed**
   - Windows: [git-scm.com](https://git-scm.com/download/win)
   - Mac: `brew install git`
   - Linux: `sudo apt install git`

3. **Python 3.11+** (for local testing)

## Step 1: Prepare Your Files

The deployment package includes:

```
egyptian-id-ocr/
├── Dockerfile.hf          # HF Spaces Docker config
├── README-hf.md           # Space README
├── .gitattributes         # Git LFS config
├── requirements.txt       # Python dependencies
├── app/                   # Application code
├── weights/               # YOLO model weights
├── models_cache/          # OCR model cache
└── scripts/
    ├── deploy-hf.sh       # Linux/Mac deployment script
    └── deploy-hf.bat      # Windows deployment script
```

## Step 2: Create Hugging Face Space

### Option A: Manual Creation (Recommended for first time)

1. **Go to Hugging Face Spaces**
   - Visit: [https://huggingface.co/new-space](https://huggingface.co/new-space)

2. **Fill in Space Details**
   ```
   Space name: egyptian-id-ocr
   License: MIT
   SDK: Docker
   Visibility: Public (or Private)
   ```

3. **Click "Create Space"**

### Option B: Automatic Creation via Script

Run the deployment script which creates the space automatically:

**Windows:**
```bash
cd K:\business\projects_v2\egption_nid_ocr
scripts\deploy-hf.bat
```

**Linux/Mac:**
```bash
cd /path/to/egyptian-id-ocr
chmod +x scripts/deploy-hf.sh
./scripts/deploy-hf.sh
```

## Step 3: Deploy Your Code

### Method 1: Using the Deployment Script (Recommended)

The script handles everything automatically:

1. **Run the script:**
   ```bash
   # Windows
   scripts\deploy-hf.bat
   
   # Linux/Mac
   ./scripts/deploy-hf.sh
   ```

2. **Follow the prompts:**
   - Login to Hugging Face (if not already)
   - Enter your username
   - Confirm deployment

3. **Wait for deployment:**
   - Script pushes code to HF Spaces
   - Docker image builds automatically (~5-10 minutes)

### Method 2: Manual Git Push

1. **Clone your space:**
   ```bash
   git clone https://huggingface.co/spaces/YOUR_USERNAME/egyptian-id-ocr
   cd egyptian-id-ocr
   ```

2. **Copy deployment files:**
   ```bash
   # From your project directory
   cp /path/to/project/Dockerfile.hf ./Dockerfile
   cp /path/to/project/README-hf.md ./README.md
   cp /path/to/project/.gitattributes .
   cp /path/to/project/requirements.txt .
   cp -r /path/to/project/app .
   cp -r /path/to/project/scripts .
   cp -r /path/to/project/weights .
   cp -r /path/to/project/models_cache .
   ```

3. **Push to Hugging Face:**
   ```bash
   git add .
   git commit -m "Initial deployment"
   git push -u origin main
   ```

## Step 4: Monitor Build Progress

1. **Go to your Space page:**
   ```
   https://huggingface.co/spaces/YOUR_USERNAME/egyptian-id-ocr
   ```

2. **Check the "Logs" tab**
   - Watch Docker build progress
   - Look for any errors

3. **Wait for "Running" status**
   - Build takes 5-10 minutes first time
   - Subsequent deploys are faster

## Step 5: Test Your API

### Using the Browser

1. **Visit your Space:**
   ```
   https://YOUR_USERNAME-egyptian-id-ocr.hf.space
   ```

2. **Open API docs:**
   ```
   https://YOUR_USERNAME-egyptian-id-ocr.hf.space/docs
   ```

3. **Test the health endpoint:**
   - Click on `GET /api/v1/health`
   - Click "Try it out"
   - Click "Execute"

### Using cURL

```bash
# Health check
curl https://YOUR_USERNAME-egyptian-id-ocr.hf.space/api/v1/health

# Extract from image
curl -X POST "https://YOUR_USERNAME-egyptian-id-ocr.hf.space/api/v1/extract" \
  -F "file=@path/to/your/id_card.jpg"
```

### Using Python

```python
import requests

BASE_URL = "https://YOUR_USERNAME-egyptian-id-ocr.hf.space"

# Health check
response = requests.get(f"{BASE_URL}/api/v1/health")
print(response.json())

# Extract from image
with open("id_card.jpg", "rb") as f:
    response = requests.post(
        f"{BASE_URL}/api/v1/extract",
        files={"file": f}
    )
    print(response.json())
```

## Step 6: Configure Secrets (Optional)

For sensitive configuration:

1. **Go to Space Settings**
   - Click "Settings" tab in your Space

2. **Add Repository Secrets**
   - Scroll to "Repository secrets"
   - Add variables like:
     ```
     APP_ENV=production
     LOG_LEVEL=INFO
     ```

3. **Restart the Space**
   - Go to "Settings" → "Factory reboot"

## Troubleshooting

### Build Fails

**Problem:** Docker build fails with error

**Solutions:**
1. Check the "Logs" tab for specific error
2. Verify all files are copied correctly
3. Check `requirements.txt` for typos
4. Try rebuilding: Settings → Factory reboot

### Cold Start Too Long

**Problem:** First request takes >60 seconds

**Solutions:**
1. Enable "Pre-warm models" in settings
2. Set up a uptime monitor to ping periodically
3. Consider Oracle Cloud for better performance

### Out of Memory

**Problem:** Container crashes with OOM error

**Solutions:**
1. Reduce model loading (disable PaddleOCR if not needed)
2. Optimize image preprocessing
3. Consider upgrading to paid HF Spaces

### Model Download Fails

**Problem:** EasyOCR/PaddleOCR models fail to download

**Solutions:**
1. Models are pre-downloaded in Dockerfile
2. Check network connectivity in logs
3. Increase build timeout in HF settings

## Performance Optimization

### Reduce Image Size

The Dockerfile already includes optimizations:
- Uses `python:3.11-slim` base image
- Excludes dev dependencies
- Clears pip cache

### Faster Startup

1. **Pre-download models** (already done in Dockerfile)
2. **Use lazy loading** for optional models
3. **Enable model caching** in `/tmp`

### Memory Optimization

1. **Limit workers:** Set `APP_WORKERS=1`
2. **Reduce threads:** Set `OCR_CPU_THREADS=2`
3. **Disable unused engines:** Set environment variables

## Cost

**Hugging Face Spaces Free Tier:**
- ✅ $0/month
- ✅ No credit card required
- ✅ 2 vCPU, 16GB RAM, 50GB storage
- ✅ Unlimited requests
- ⚠️ Public by default
- ⚠️ Sleeps after inactivity

## Next Steps

1. **Share your API:**
   - Share the Space URL with testers
   - Add API documentation link

2. **Monitor usage:**
   - Check Space analytics
   - Monitor response times

3. **Iterate:**
   - Make code changes
   - Push to git
   - Auto-deploy happens automatically

## Support

- **HF Docs:** [https://huggingface.co/docs/hub/spaces](https://huggingface.co/docs/hub/spaces)
- **Docker Spaces:** [https://huggingface.co/docs/hub/spaces-sdks-docker](https://huggingface.co/docs/hub/spaces-sdks-docker)
- **Community:** [Hugging Face Forums](https://discuss.huggingface.co/)
