#!/bin/bash
# Egyptian ID OCR - Hugging Face Spaces Deployment Script
# This script helps you deploy to Hugging Face Spaces

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
HF_USERNAME=""
SPACE_NAME="egyptian-id-ocr"
SPACE_ID=""
REPO_URL=""

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Egyptian ID OCR - HF Spaces Deployer${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Function to print colored messages
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    print_info "Checking prerequisites..."
    
    # Check if git is installed
    if ! command -v git &> /dev/null; then
        print_error "Git is not installed. Please install git first."
        exit 1
    fi
    
    # Check if huggingface-cli is installed
    if ! command -v huggingface-cli &> /dev/null; then
        print_warning "huggingface-cli not found. Installing..."
        pip install huggingface-hub -q
    fi
    
    # Check if Docker is installed (optional, for local testing)
    if command -v docker &> /dev/null; then
        print_success "Docker is available for local testing"
    else
        print_warning "Docker not found. Skipping local build test."
    fi
    
    print_success "Prerequisites check passed"
}

# Login to Hugging Face
hf_login() {
    echo ""
    print_info "Logging in to Hugging Face..."
    echo "You'll need a Hugging Face account. Create one at: https://huggingface.co/join"
    echo ""
    
    huggingface-cli login
    
    if [ $? -eq 0 ]; then
        print_success "Logged in to Hugging Face"
    else
        print_error "Failed to login to Hugging Face"
        exit 1
    fi
}

# Get username from HF
get_hf_username() {
    print_info "Getting your Hugging Face username..."
    
    # Try to get username from whoami
    HF_WHOAMI=$(huggingface-cli whoami 2>/dev/null || echo "")
    
    if [ -n "$HF_WHOAMI" ]; then
        HF_USERNAME=$(echo "$HF_WHOAMI" | cut -d':' -f1 | tr -d ' ')
        print_success "Username: $HF_USERNAME"
    else
        echo ""
        print_warning "Could not auto-detect username."
        read -p "Enter your Hugging Face username: " HF_USERNAME
    fi
    
    SPACE_ID="${HF_USERNAME}/${SPACE_NAME}"
    REPO_URL="https://huggingface.co/spaces/${SPACE_ID}"
}

# Create the Space if it doesn't exist
create_space() {
    echo ""
    print_info "Creating Hugging Face Space: ${SPACE_ID}..."
    
    # Check if space exists
    if curl -s -o /dev/null -w "%{http_code}" "https://huggingface.co/spaces/${SPACE_ID}" | grep -q "200"; then
        print_warning "Space already exists: ${SPACE_ID}"
        return 0
    fi
    
    # Create the space using API
    curl -X POST "https://huggingface.co/api/spaces" \
        -H "Authorization: Bearer $(huggingface-cli token 2>/dev/null)" \
        -H "Content-Type: application/json" \
        -d "{
            \"repoId\": \"${SPACE_ID}\",
            \"sdk\": \"docker\",
            \"visibility\": \"public\"
        }" 2>/dev/null
    
    if [ $? -eq 0 ]; then
        print_success "Space created: ${REPO_URL}"
    else
        print_warning "Could not create space via API. You may need to create it manually."
        echo "Visit: https://huggingface.co/new-space"
        echo "Space name: ${SPACE_NAME}"
        echo "SDK: Docker"
        echo "Visibility: Public"
    fi
}

# Prepare files for deployment
prepare_files() {
    echo ""
    print_info "Preparing files for deployment..."
    
    # Create a temporary directory for deployment
    DEPLOY_DIR="/tmp/hf-deploy-${SPACE_NAME}"
    rm -rf "$DEPLOY_DIR"
    mkdir -p "$DEPLOY_DIR"
    
    # Copy necessary files
    cp Dockerfile.hf "$DEPLOY_DIR/Dockerfile"
    cp README-hf.md "$DEPLOY_DIR/README.md"
    cp .gitattributes "$DEPLOY_DIR/"
    cp requirements.txt "$DEPLOY_DIR/"
    cp .env.example "$DEPLOY_DIR/.env"
    
    # Copy app directory
    cp -r app "$DEPLOY_DIR/"
    
    # Copy scripts directory
    cp -r scripts "$DEPLOY_DIR/"
    
    # Copy weights directory (if exists and not too large)
    if [ -d "weights" ]; then
        WEIGHTS_SIZE=$(du -sm weights | cut -f1)
        if [ "$WEIGHTS_SIZE" -lt 100 ]; then
            cp -r weights "$DEPLOY_DIR/"
            print_success "Copied weights directory (${WEIGHTS_SIZE}MB)"
        else
            print_warning "Weights directory too large (${WEIGHTS_SIZE}MB). Consider using Git LFS."
            cp -r weights "$DEPLOY_DIR/"
        fi
    fi
    
    # Copy models_cache directory (if exists)
    if [ -d "models_cache" ]; then
        cp -r models_cache "$DEPLOY_DIR/"
    fi
    
    # Copy PP-OCRv5_ar directory (if exists)
    if [ -d "PP-OCRv5_ar" ]; then
        cp -r PP-OCRv5_ar "$DEPLOY_DIR/"
    fi
    
    # Create .dockerignore
    cat > "$DEPLOY_DIR/.dockerignore" << 'EOF'
.git
.gitignore
*.md
!README.md
.env.example
run.bat
run.sh
docs/
tests/
debug/
logs/
*.log
__pycache__
*.pyc
.pytest_cache
.ruff_cache
.eggs
*.egg-info
EOF
    
    print_success "Files prepared in: $DEPLOY_DIR"
}

# Deploy using git
deploy_with_git() {
    echo ""
    print_info "Deploying to Hugging Face Spaces..."
    
    cd "$DEPLOY_DIR"
    
    # Initialize git repo if not already
    if [ ! -d ".git" ]; then
        git init
        git config user.email "deploy@local"
        git config user.name "Deploy Script"
    fi
    
    # Add remote
    git remote remove origin 2>/dev/null || true
    git remote add origin "https://huggingface.co/spaces/${SPACE_ID}"
    
    # Add and commit files
    git add -A
    git commit -m "Deploy Egyptian ID OCR API"
    
    # Push to HF Spaces
    print_info "Pushing to Hugging Face Spaces (this may take a while)..."
    git push -u origin main --force
    
    if [ $? -eq 0 ]; then
        print_success "Deployment initiated!"
    else
        print_error "Deployment failed. Check your credentials and try again."
        exit 1
    fi
    
    cd - > /dev/null
}

# Test locally with Docker
test_locally() {
    echo ""
    print_info "Testing Docker build locally..."
    
    if ! command -v docker &> /dev/null; then
        print_warning "Docker not available. Skipping local test."
        return 0
    fi
    
    cd "$DEPLOY_DIR"
    
    # Build Docker image
    print_info "Building Docker image..."
    docker build -t "${SPACE_NAME}:test" -f Dockerfile .
    
    if [ $? -ne 0 ]; then
        print_error "Docker build failed!"
        cd - > /dev/null
        return 1
    fi
    
    print_success "Docker build successful!"
    
    # Optional: Run container for testing
    echo ""
    read -p "Run container for testing? (y/n): " run_test
    if [ "$run_test" = "y" ]; then
        print_info "Starting container on port 7860..."
        docker run -p 7860:7860 --rm --cpus="2" --memory="4g" "${SPACE_NAME}:test" &
        CONTAINER_PID=$!
        
        echo ""
        print_info "Container started. Test with:"
        echo "  curl http://localhost:7860/api/v1/health"
        echo ""
        echo "Press Enter to stop the container..."
        read
        
        kill $CONTAINER_PID 2>/dev/null || true
    fi
    
    cd - > /dev/null
}

# Print deployment summary
print_summary() {
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}  Deployment Summary${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    echo -e "Space URL: ${BLUE}${REPO_URL}${NC}"
    echo -e "API Docs:  ${BLUE}${REPO_URL}/docs${NC}"
    echo -e "Health:    ${BLUE}${REPO_URL}/api/v1/health${NC}"
    echo ""
    print_info "Your Space is building! This takes 5-10 minutes."
    print_info "Monitor the build at: ${REPO_URL}"
    echo ""
    print_info "Once deployed, test with:"
    echo ""
    echo "  curl -X POST \"${REPO_URL}/api/v1/extract\" \\"
    echo "    -F \"file=@your_id_card.jpg\""
    echo ""
}

# Cleanup
cleanup() {
    print_info "Cleaning up..."
    rm -rf "/tmp/hf-deploy-${SPACE_NAME}"
    print_success "Cleanup complete"
}

# Main execution
main() {
    check_prerequisites
    hf_login
    get_hf_username
    create_space
    prepare_files
    test_locally
    deploy_with_git
    print_summary
    cleanup
    
    echo ""
    print_success "Deployment complete! 🎉"
}

# Run main function
main "$@"
