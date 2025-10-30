@echo off
echo ========================================
echo Installing dependencies for framer_cv
echo ========================================
echo.

REM Install required packages
pip install opencv-python numpy psutil tqdm

echo.
echo ========================================
echo Installation complete!
echo ========================================
echo.
echo Note: For GPU/CUDA support, you need:
echo - NVIDIA GPU with CUDA support
echo - OpenCV built with CUDA (requires building from source)
echo.
pause
