#!/bin/bash
# Egyptian ID OCR - Local Docker Deployment Test
# Tests the Docker build and API endpoints locally

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

# Configuration
IMAGE_NAME="egyptian-id-ocr"
CONTAINER_NAME="egyptian-ocr-test"
PORT=8000
DOCKERFILE="${1:-Dockerfile.hf}"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Egyptian ID OCR - Docker Test${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check Docker
print_info "Checking Docker..."
if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed"
    exit 1
fi
print_success "Docker found: $(docker --version)"

# Stop and remove existing container
print_info "Cleaning up existing containers..."
docker stop $CONTAINER_NAME 2>/dev/null || true
docker rm $CONTAINER_NAME 2>/dev/null || true

# Build Docker image
print_info "Building Docker image from $DOCKERFILE..."
START_TIME=$(date +%s)

docker build -f "$DOCKERFILE" -t "$IMAGE_NAME:test" .

if [ $? -ne 0 ]; then
    print_error "Docker build failed!"
    exit 1
fi

BUILD_TIME=$(($(date +%s) - START_TIME))
print_success "Docker build completed in ${BUILD_TIME}s"

# Get image size
IMAGE_SIZE=$(docker images $IMAGE_NAME:test --format "{{.Size}}")
print_info "Image size: $IMAGE_SIZE"

# Run container
print_info "Starting container..."
docker run -d \
    --name $CONTAINER_NAME \
    -p $PORT:7860 \
    --cpus="2" \
    --memory="4g" \
    --health-cmd="curl -f http://localhost:7860/api/v1/health || exit 1" \
    --health-interval=10s \
    --health-timeout=5s \
    --health-retries=3 \
    --health-start-period=60s \
    $IMAGE_NAME:test

print_success "Container started"

# Wait for container to be healthy
print_info "Waiting for container to be healthy (max 120s)..."
HEALTHY=false
for i in {1..24}; do
    HEALTH_STATUS=$(docker inspect -f '{{.State.Health.Status}}' $CONTAINER_NAME 2>/dev/null || echo "unknown")
    
    if [ "$HEALTH_STATUS" = "healthy" ]; then
        HEALTHY=true
        break
    fi
    
    echo -n "."
    sleep 5
done
echo ""

if [ "$HEALTHY" = false ]; then
    print_warning "Container health check did not pass, checking manually..."
fi

# Test endpoints
echo ""
print_info "Testing API endpoints..."

# Test root endpoint
print_info "Testing GET /..."
RESPONSE=$(curl -s -w "\n%{http_code}" http://localhost:$PORT/ 2>/dev/null || echo "000")
HTTP_CODE=$(echo "$RESPONSE" | tail -n1)
BODY=$(echo "$RESPONSE" | head -n-1)

if [ "$HTTP_CODE" = "200" ]; then
    print_success "GET / - HTTP $HTTP_CODE"
    echo "Response: $BODY" | head -c 200
    echo "..."
else
    print_error "GET / - HTTP $HTTP_CODE"
fi

# Test health endpoint
echo ""
print_info "Testing GET /api/v1/health..."
RESPONSE=$(curl -s -w "\n%{http_code}" http://localhost:$PORT/api/v1/health 2>/dev/null || echo "000")
HTTP_CODE=$(echo "$RESPONSE" | tail -n1)
BODY=$(echo "$RESPONSE" | head -n-1)

if [ "$HTTP_CODE" = "200" ]; then
    print_success "GET /api/v1/health - HTTP $HTTP_CODE"
    echo "Response: $BODY"
else
    print_error "GET /api/v1/health - HTTP $HTTP_CODE"
fi

# Test models endpoint
echo ""
print_info "Testing GET /api/v1/models..."
RESPONSE=$(curl -s -w "\n%{http_code}" http://localhost:$PORT/api/v1/models 2>/dev/null || echo "000")
HTTP_CODE=$(echo "$RESPONSE" | tail -n1)
BODY=$(echo "$RESPONSE" | head -n-1)

if [ "$HTTP_CODE" = "200" ]; then
    print_success "GET /api/v1/models - HTTP $HTTP_CODE"
    echo "Response: $BODY" | head -c 300
    echo "..."
else
    print_error "GET /api/v1/models - HTTP $HTTP_CODE"
fi

# Test docs endpoint
echo ""
print_info "Testing GET /docs..."
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:$PORT/docs 2>/dev/null || echo "000")

if [ "$HTTP_CODE" = "200" ]; then
    print_success "GET /docs - HTTP $HTTP_CODE"
else
    print_error "GET /docs - HTTP $HTTP_CODE"
fi

# Measure startup time
echo ""
STARTUP_LOG=$(docker logs $CONTAINER_NAME 2>&1 | grep -i "startup\|loaded\|initialized" | tail -5)
print_info "Startup log entries:"
echo "$STARTUP_LOG"

# Get container stats
echo ""
print_info "Container resource usage:"
docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}" $CONTAINER_NAME

# Performance benchmark (if test image exists)
echo ""
if [ -f "debug/test_id.jpg" ] || [ -f "tests/test_id.jpg" ]; then
    print_info "Running performance benchmark..."
    TEST_IMAGE="debug/test_id.jpg"
    [ -f "tests/test_id.jpg" ] && TEST_IMAGE="tests/test_id.jpg"
    
    START_TIME=$(date +%s%N)
    RESPONSE=$(curl -s -w "\n%{http_code}" -X POST \
        -F "file=@$TEST_IMAGE" \
        http://localhost:$PORT/api/v1/extract 2>/dev/null || echo "000")
    END_TIME=$(date +%s%N)
    
    HTTP_CODE=$(echo "$RESPONSE" | tail -n1)
    BODY=$(echo "$RESPONSE" | head -n-1)
    DURATION=$(( (END_TIME - START_TIME) / 1000000 ))
    
    if [ "$HTTP_CODE" = "200" ]; then
        print_success "POST /api/v1/extract - HTTP $HTTP_CODE (${DURATION}ms)"
        echo "Response: $BODY" | head -c 500
        echo "..."
    else
        print_error "POST /api/v1/extract - HTTP $HTTP_CODE"
    fi
else
    print_warning "No test image found. Skipping extraction benchmark."
    print_info "Add a test image to debug/test_id.jpg for full benchmark"
fi

# Summary
echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Test Summary${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Image: $IMAGE_NAME:test"
echo "Container: $CONTAINER_NAME"
echo "Port: $PORT"
echo "Build Time: ${BUILD_TIME}s"
echo "Image Size: $IMAGE_SIZE"
echo ""
echo "Useful commands:"
echo "  docker logs -f $CONTAINER_NAME     # View logs"
echo "  docker stats $CONTAINER_NAME       # Live stats"
echo "  docker stop $CONTAINER_NAME        # Stop container"
echo "  docker rm $CONTAINER_NAME          # Remove container"
echo ""
echo "API Endpoints:"
echo "  http://localhost:$PORT/            # Root"
echo "  http://localhost:$PORT/docs        # Swagger UI"
echo "  http://localhost:$PORT/api/v1/health  # Health check"
echo ""

# Ask to keep container running
read -p "Keep container running for manual testing? (y/n): " keep_running
if [ "$keep_running" != "y" ]; then
    print_info "Stopping container..."
    docker stop $CONTAINER_NAME
    docker rm $CONTAINER_NAME
    print_success "Cleanup complete"
fi

print_success "Test complete!"
