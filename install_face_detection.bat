@echo off
echo ============================================================
echo Installing Face Detection Dependencies
echo ============================================================
echo.

echo Installing PyTorch with CUDA support...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo.
echo Installing RetinaFace...
pip install retina-face

echo.
echo Installing additional dependencies...
pip install opencv-python
pip install opencv-contrib-python
pip install pillow
pip install tqdm

echo.
echo ============================================================
echo Installation Complete!
echo ============================================================
echo.
echo To verify GPU support, run:
echo python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
echo.
pause
