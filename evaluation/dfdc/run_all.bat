@echo off
REM Run complete DFDC evaluation pipeline
REM Execute this from J:\DF directory

echo ====================================================================
echo                 DFDC EVALUATION PIPELINE
echo ====================================================================
echo.
echo This will run the complete evaluation in 4 steps:
echo   1. Extract frames from videos
echo   2. Detect and align faces (GPU)
echo   3. Parse metadata to CSV
echo   4. Run model inference
echo.
echo Total time: ~2-3 hours for 400 videos
echo Disk space needed: ~20 GB on H:\EVAL-1
echo.
pause

cd /d J:\DF

echo.
echo ====================================================================
echo STEP 1/4: Extracting Frames
echo ====================================================================
python evaluation\dfdc\01_extract_frames_dfdc.py
if %errorlevel% neq 0 (
    echo ERROR: Frame extraction failed
    pause
    exit /b 1
)

echo.
echo ====================================================================
echo STEP 2/4: Detecting Faces (GPU)
echo ====================================================================
python evaluation\dfdc\02_detect_faces_dfdc.py
if %errorlevel% neq 0 (
    echo ERROR: Face detection failed
    pause
    exit /b 1
)

echo.
echo ====================================================================
echo STEP 3/4: Parsing Metadata
echo ====================================================================
python evaluation\dfdc\03_parse_metadata.py
if %errorlevel% neq 0 (
    echo ERROR: Metadata parsing failed
    pause
    exit /b 1
)

echo.
echo ====================================================================
echo STEP 4/4: Running Inference
echo ====================================================================
python evaluation\dfdc\04_run_inference.py
if %errorlevel% neq 0 (
    echo ERROR: Inference failed
    pause
    exit /b 1
)

echo.
echo ====================================================================
echo              EVALUATION COMPLETE!
echo ====================================================================
echo.
echo Results saved to: evaluation\dfdc\dfdc_results.csv
echo.
echo Next steps:
echo   1. Check results CSV
echo   2. Compare with Celeb-DF results
echo   3. Generate comparison plots
echo.
pause
