@echo off
echo ============================================================
echo Installing MTCNN Face Detection (CPU-Only, Lightweight)
echo ============================================================
echo.

echo Installing MTCNN...
pip install mtcnn

echo.
echo Installing dependencies...
pip install opencv-python
pip install pillow
pip install tqdm
pip install tensorflow

echo.
echo ============================================================
echo Installation Complete!
echo ============================================================
echo.
echo MTCNN is now installed for CPU-based face detection
echo This is faster than RetinaFace on CPU but slightly less accurate
echo.
pause
