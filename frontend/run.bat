@echo off
echo ================================================================================
echo POLYP DETECTION AI - WEB INTERFACE
echo ================================================================================
echo.
echo Choose an option:
echo.
echo [1] Demo Mode (No backend required - simulated results)
echo [2] Full Mode (With Flask backend - requires trained models)
echo [3] Exit
echo.
set /p choice="Enter your choice (1-3): "

if "%choice%"=="1" goto demo
if "%choice%"=="2" goto full
if "%choice%"=="3" goto end

:demo
echo.
echo Starting Demo Mode...
echo Opening in your default browser...
echo.
start index.html
echo.
echo [OK] Frontend opened in browser!
echo [INFO] This is demo mode - results are simulated
echo.
pause
goto end

:full
echo.
echo Starting Full Mode with Flask Backend...
echo.
echo [INFO] Checking dependencies...
python -c "import flask" 2>nul
if errorlevel 1 (
    echo [WARN] Flask not found. Installing...
    pip install flask flask-cors
)

echo [INFO] Starting Flask server...
echo [INFO] Access the app at: http://localhost:5000
echo [INFO] Press Ctrl+C to stop the server
echo.
python app.py
goto end

:end
echo.
echo Goodbye!
pause
