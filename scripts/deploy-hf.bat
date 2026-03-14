@echo off
REM Egyptian ID OCR - Hugging Face Spaces Deployment Script (Windows)
REM This script helps you deploy to Hugging Face Spaces

setlocal enabledelayedexpansion

echo ========================================
echo   Egyptian ID OCR - HF Spaces Deployer
echo ========================================
echo.

REM Configuration
set HF_USERNAME=
set SPACE_NAME=egyptian-id-ocr
set SPACE_ID=
set REPO_URL=

REM Check prerequisites
echo [INFO] Checking prerequisites...

where git >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Git is not installed. Please install Git first.
    echo Visit: https://git-scm.com/download/win
    exit /b 1
)
echo [SUCCESS] Git found

where docker >nul 2>&1
if %errorlevel% neq 0 (
    echo [WARNING] Docker not found. Skipping local build test.
) else (
    echo [SUCCESS] Docker found
)

where huggingface-cli >nul 2>&1
if %errorlevel% neq 0 (
    echo [WARNING] huggingface-cli not found. Installing...
    pip install huggingface-hub -q
)

echo.
echo [INFO] Logging in to Hugging Face...
echo You'll need a Hugging Face account. Create one at: https://huggingface.co/join
echo.

huggingface-cli login
if %errorlevel% neq 0 (
    echo [ERROR] Failed to login to Hugging Face
    exit /b 1
)
echo [SUCCESS] Logged in to Hugging Face

echo.
echo [INFO] Getting your Hugging Face username...

for /f "tokens=*" %%i in ('huggingface-cli whoami 2^>nul') do set HF_WHOAMI=%%i

if not "%HF_WHOAMI%"=="" (
    for /f "tokens=1 delims=:" %%a in ("%HF_WHOAMI%") do set HF_USERNAME=%%a
    set HF_USERNAME=%HF_USERNAME: =%
    echo [SUCCESS] Username: %HF_USERNAME%
) else (
    echo [WARNING] Could not auto-detect username.
    set /p HF_USERNAME="Enter your Hugging Face username: "
)

set SPACE_ID=%HF_USERNAME%/%SPACE_NAME%
set REPO_URL=https://huggingface.co/spaces/%SPACE_ID%

echo.
echo [INFO] Space ID: %SPACE_ID%
echo [INFO] Space URL: %REPO_URL%

echo.
echo [INFO] Preparing files for deployment...

REM Create a temporary directory for deployment
set DEPLOY_DIR=%TEMP%\hf-deploy-%SPACE_NAME%
if exist "%DEPLOY_DIR%" rmdir /s /q "%DEPLOY_DIR%"
mkdir "%DEPLOY_DIR%"

REM Copy necessary files
copy Dockerfile.hf "%DEPLOY_DIR%\Dockerfile" >nul
copy README-hf.md "%DEPLOY_DIR%\README.md" >nul
copy .gitattributes "%DEPLOY_DIR%\" >nul
copy requirements.txt "%DEPLOY_DIR%\" >nul
copy .env.example "%DEPLOY_DIR%\.env" >nul

REM Copy directories
xcopy /E /I /Y app "%DEPLOY_DIR%\app" >nul
xcopy /E /I /Y scripts "%DEPLOY_DIR%\scripts" >nul

if exist weights (
    xcopy /E /I /Y weights "%DEPLOY_DIR%\weights" >nul
    echo [SUCCESS] Copied weights directory
)

if exist models_cache (
    xcopy /E /I /Y models_cache "%DEPLOY_DIR%\models_cache" >nul
)

if exist PP-OCRv5_ar (
    xcopy /E /I /Y PP-OCRv5_ar "%DEPLOY_DIR%\PP-OCRv5_ar" >nul
)

REM Create .dockerignore
(
echo .git
echo .gitignore
echo *.md
echo !README.md
echo .env.example
echo run.bat
echo run.sh
echo docs/
echo tests/
echo debug/
echo logs/
echo *.log
echo __pycache__
echo *.pyc
echo .pytest_cache
echo .ruff_cache
) > "%DEPLOY_DIR%\.dockerignore"

echo [SUCCESS] Files prepared in: %DEPLOY_DIR%

echo.
echo [INFO] Testing Docker build locally...

where docker >nul 2>&1
if %errorlevel% equ 0 (
    cd /d "%DEPLOY_DIR%"
    
    echo [INFO] Building Docker image...
    docker build -t %SPACE_NAME%:test -f Dockerfile .
    
    if %errorlevel% equ 0 (
        echo [SUCCESS] Docker build successful!
        
        echo.
        set /p run_test="Run container for testing? (y/n): "
        if /i "!run_test!"=="y" (
            echo [INFO] Starting container on port 7860...
            start docker run -p 7860:7860 --rm --cpus="2" --memory="4g" %SPACE_NAME%:test
            
            echo.
            echo [INFO] Container started. Test with:
            echo   curl http://localhost:7860/api/v1/health
            echo.
            echo Press any key to continue...
            pause >nul
        )
    ) else (
        echo [ERROR] Docker build failed!
    )
    
    cd /d "%~dp0"
) else (
    echo [WARNING] Docker not available. Skipping local test.
)

echo.
echo [INFO] Deploying to Hugging Face Spaces...

cd /d "%DEPLOY_DIR%"

REM Initialize git repo if not already
if not exist ".git" (
    git init
    git config user.email "deploy@local"
    git config user.name "Deploy Script"
)

REM Add remote
git remote remove origin 2>nul
git remote add origin "https://huggingface.co/spaces/%SPACE_ID%"

REM Add and commit files
git add -A
git commit -m "Deploy Egyptian ID OCR API"

REM Push to HF Spaces
echo [INFO] Pushing to Hugging Face Spaces (this may take a while)...
git push -u origin main --force

if %errorlevel% equ 0 (
    echo [SUCCESS] Deployment initiated!
) else (
    echo [ERROR] Deployment failed. Check your credentials and try again.
    cd /d "%~dp0"
    exit /b 1
)

cd /d "%~dp0"

echo.
echo ========================================
echo   Deployment Summary
echo ========================================
echo.
echo Space URL: %REPO_URL%
echo API Docs:  %REPO_URL%/docs
echo Health:    %REPO_URL%/api/v1/health
echo.
echo [INFO] Your Space is building! This takes 5-10 minutes.
echo [INFO] Monitor the build at: %REPO_URL%
echo.
echo [INFO] Once deployed, test with:
echo.
echo   curl -X POST "%REPO_URL%/api/v1/extract" ^
echo     -F "file=@your_id_card.jpg"
echo.

REM Cleanup
echo [INFO] Cleaning up...
rmdir /s /q "%DEPLOY_DIR%"
echo [SUCCESS] Cleanup complete

echo.
echo [SUCCESS] Deployment complete!
pause
