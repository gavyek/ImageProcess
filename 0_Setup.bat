@echo off
setlocal
cd /d "%~dp0"

echo ========================================================
echo  Image Processing Environment Setup Tool
echo ========================================================

:: 1. Delete existing venv folder if it exists
if exist "venv" (
    echo [LOG] Found existing 'venv' folder.
    echo [LOG] Deleting old environment to prevent path errors...
    rmdir /s /q "venv"
    if exist "venv" (
        echo [ERROR] Failed to delete 'venv'. Please close any open files and try again.
        pause
        exit /b
    )
    echo [OK] Old venv deleted successfully.
)

:: 2. Create new virtual environment
echo.
echo [LOG] Creating new virtual environment for this PC...
python -m venv venv
if errorlevel 1 (
    echo [ERROR] Python not found or not in PATH.
    echo Please install Python and check "Add Python to PATH".
    pause
    exit /b
)

:: 3. Activate and Install dependencies
echo.
echo [LOG] Activating venv and installing packages...
call venv\Scripts\activate

:: Upgrade pip first
python -m pip install --upgrade pip

:: Install PyTorch (CUDA 11.8) explicitly
echo [LOG] Installing PyTorch with CUDA 11.8 support...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

:: Install requirements if file exists
if exist "requirements.txt" (
    echo [LOG] Installing dependencies from requirements.txt...
    pip install -r requirements.txt
) else (
    echo [WARNING] 'requirements.txt' not found. Created empty venv.
)

echo.
echo ========================================================
echo  Setup Complete! Please check the logs above.
echo ========================================================
pause