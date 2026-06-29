@echo off
echo Stopping current backend...
taskkill /f /im python.exe 2>nul

echo Waiting 3 seconds...
timeout /t 3 /nobreak >nul

echo Starting optimized backend...
cd backend
python app.py

pause