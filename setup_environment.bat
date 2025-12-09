@echo off
:: Setup script for FedBim federated learning environment

echo Setting up FedBim environment...
echo ===================================

:: Check if Python is installed
where python >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo Error: Python is not installed or not in PATH. Please install Python 3.8 or later.
    pause
    exit /b 1
)

:: Create virtual environment if it doesn't exist
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
    if %ERRORLEVEL% NEQ 0 (
        echo Failed to create virtual environment
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

:: Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip
if %ERRORLEVEL% NEQ 0 (
    echo Failed to upgrade pip
    pause
    exit /b 1
)

:: Install PyTorch with CUDA support
echo Installing PyTorch with CUDA support...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
if %ERRORLEVEL% NEQ 0 (
    echo Failed to install PyTorch
    pause
    exit /b 1
)

:: Install other dependencies
echo Installing other dependencies...
pip install ultralytics flwr numpy opencv-python Pillow matplotlib seaborn pandas tqdm pyyaml
if %ERRORLEVEL% NEQ 0 (
    echo Failed to install dependencies
    pause
    exit /b 1
)

echo.
echo ===================================
echo Setup completed successfully!
echo.
echo To activate the virtual environment, run:
echo    venv\Scripts\activate
echo.
echo To start federated learning, run:
echo    python run_federated.py
echo.
pause
