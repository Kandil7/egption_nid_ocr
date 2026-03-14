@echo off
REM Egyptian ID OCR - Local Docker Deployment Test (Windows)
REM Tests the Docker build and API endpoints locally

setlocal enabledelayedexpansion

echo ========================================
echo   Egyptian ID OCR - Docker Test
echo ========================================
echo.

REM Configuration
set IMAGE_NAME=egyptian-id-ocr
set CONTAINER_NAME=egyptian-ocr-test
set PORT=8000
set DOCKERFILE=%1
if "%DOCKERFILE%"=="" set DOCKERFILE=Dockerfile.hf

REM Check Docker
echo [INFO] Checking Docker...
where docker >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Docker is not installed or not running
    echo Please start Docker Desktop and try again
    pause
    exit /b 1
)

for /f "tokens=*" %%i in ('docker --version') do set DOCKER_VERSION=%%i
echo [SUCCESS] Docker found: %DOCKER_VERSION%

REM Stop and remove existing container
echo [INFO] Cleaning up existing containers...
docker stop %CONTAINER_NAME% 2>nul
docker rm %CONTAINER_NAME% 2>nul

REM Build Docker image
echo [INFO] Building Docker image from %DOCKERFILE%...
echo This may take 10-20 minutes on first build...
echo.

docker build -f %DOCKERFILE% -t %IMAGE_NAME%:test .

if %errorlevel% neq 0 (
    echo [ERROR] Docker build failed!
    echo Check the error messages above
    pause
    exit /b 1
)

echo [SUCCESS] Docker build completed

REM Get image size
for /f "tokens=*" %%i in ('docker images %IMAGE_NAME%:test --format "{{.Size}}"') do set IMAGE_SIZE=%%i
echo [INFO] Image size: %IMAGE_SIZE%

REM Run container
echo [INFO] Starting container...
docker run -d ^
    --name %CONTAINER_NAME% ^
    -p %PORT%:7860 ^
    --cpus="2" ^
    --memory="4g" ^
    %IMAGE_NAME%:test

echo [SUCCESS] Container started

REM Wait for container to be ready
echo [INFO] Waiting for container to start (60 seconds)...
timeout /t 60 /nobreak >nul

REM Test endpoints
echo.
echo [INFO] Testing API endpoints...

REM Test root endpoint
echo [INFO] Testing GET /...
curl -s http://localhost:%PORT%/ >nul 2>&1
if %errorlevel% equ 0 (
    echo [SUCCESS] GET / - Responding
) else (
    echo [WARNING] GET / - Not responding yet
)

REM Test health endpoint
echo [INFO] Testing GET /api/v1/health...
curl -s http://localhost:%PORT%/api/v1/health >nul 2>&1
if %errorlevel% equ 0 (
    echo [SUCCESS] GET /api/v1/health - Responding
) else (
    echo [WARNING] GET /api/v1/health - Not responding yet
)

REM Test docs endpoint
echo [INFO] Testing GET /docs...
curl -s -o nul http://localhost:%PORT%/docs >nul 2>&1
if %errorlevel% equ 0 (
    echo [SUCCESS] GET /docs - Responding
) else (
    echo [WARNING] GET /docs - Not responding
)

REM Get container stats
echo.
echo [INFO] Container resource usage:
docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}" %CONTAINER_NAME%

REM Summary
echo.
echo ========================================
echo   Test Summary
echo ========================================
echo.
echo Image: %IMAGE_NAME%:test
echo Container: %CONTAINER_NAME%
echo Port: %PORT%
echo Image Size: %IMAGE_SIZE%
echo.
echo Useful commands:
echo   docker logs -f %CONTAINER_NAME%     - View logs
echo   docker stats %CONTAINER_NAME%       - Live stats
echo   docker stop %CONTAINER_NAME%        - Stop container
echo   docker rm %CONTAINER_NAME%          - Remove container
echo.
echo API Endpoints:
echo   http://localhost:%PORT%/            - Root
echo   http://localhost:%PORT%/docs        - Swagger UI
echo   http://localhost:%PORT%/api/v1/health - Health check
echo.

REM Ask to keep container running
set /p keep_running="Keep container running for manual testing? (y/n): "
if /i not "%keep_running%"=="y" (
    echo [INFO] Stopping container...
    docker stop %CONTAINER_NAME%
    docker rm %CONTAINER_NAME%
    echo [SUCCESS] Cleanup complete
)

echo.
echo [SUCCESS] Test complete!
pause
