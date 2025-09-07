@echo off
setlocal
cd /d "C:\Users\khang\Desktop\SchooolFiles\extracurricular\2025-2026\AiPresidentialChallange\TrainingModel"
if not exist logs mkdir logs
echo [%date% %time%] START >> logs\collector.log
".\.venv\Scripts\python.exe" ".\scripts\collect_live.py" --lat 34.028 --lon -84.173 >> logs\collector.log 2>&1
echo [%date% %time%] END exitcode=%ERRORLEVEL% >> logs\collector.log
endlocal