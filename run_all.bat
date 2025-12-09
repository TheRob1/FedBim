@echo off
:: Run script for FedBim federated learning
:: This script will:
:: 1. Check if virtual environment exists, create if needed
:: 2. Install dependencies if not already installed
:: 3. Start the federated learning process

echo FedBim Federated Learning - Starting...
echo ===================================

:: Check if virtual environment exists
if not exist "venv" (
    echo Virtual environment not found. Running setup...
    call setup_environment.bat
    if %ERRORLEVEL% NEQ 0 (
        echo Failed to set up environment
        pause
        exit /b 1
    )
)

:: Activate the virtual environment
call venv\Scripts\activate
if %ERRORLEVEL% NEQ 0 (
    echo Failed to activate virtual environment
    pause
    exit /b 1
)

:: Check if required packages are installed
echo Checking for required packages...
python -c "import torch, flwr, ultralytics" >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Installing required packages...
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
    pip install ultralytics flwr numpy opencv-python Pillow matplotlib seaborn pandas tqdm pyyaml
    if %ERRORLEVEL% NEQ 0 (
        echo Failed to install required packages
        pause
        exit /b 1
    )
)

:: Create logs directory if it doesn't exist
if not exist "logs" mkdir logs

:: Run the federated learning
echo.
echo Starting federated learning...
echo ===================================
echo [INFO] Press Ctrl+C to stop the process
echo.

python run_federated.py %*

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Error occurred during execution
    pause
    exit /b 1
)

echo.
echo Federated learning completed successfully!
echo ===================================
pause