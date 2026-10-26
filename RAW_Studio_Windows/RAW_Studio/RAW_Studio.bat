@echo off
chcp 65001 >nul
title RAW Studio Launcher
setlocal
cd /d "%~dp0"

rem --- Мінімальна версія Python: змінюйте тут І у перевірці нижче ---
set MINVER=3.9

set PY=
py -3 --version >nul 2>nul && set PY=py -3
if not defined PY (
  python --version >nul 2>nul && set PY=python
)

if not defined PY (
  echo.
  echo  [RAW Studio] Python не знайдено.
  echo  1. Відкриється сторінка https://www.python.org/downloads/
  echo  2. Встановіть Python %MINVER% або новіший.
  echo  3. У першому вікні інсталятора відзначте "Add python.exe to PATH".
  echo  4. Після установки запустіть RAW_Studio.bat ще раз.
  echo.
  start https://www.python.org/downloads/
  pause
  exit /b 1
)

rem --- Перевірка версії Python ---
set VER=
for /f "usebackq delims=" %%i in (`%PY% -c "import sys; print('%%d.%%d' %% sys.version_info[:2])"`) do set VER=%%i
if not defined VER set VER=невідома

%PY% -c "import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)" >nul 2>nul
if errorlevel 1 (
  echo.
  echo  [RAW Studio] Знайдено Python %VER% — ЗАСТАРІЛА ВЕРСІЯ.
  echo  Для роботи потрібен Python %MINVER% або новіший.
  echo  1. Відкриється сторінка https://www.python.org/downloads/
  echo  2. Встановіть актуальний Python і відзначте "Add python.exe to PATH".
  echo  3. Запустіть RAW_Studio.bat ще раз.
  echo.
  start https://www.python.org/downloads/
  pause
  exit /b 1
)

%PY% -c "import numpy, matplotlib" >nul 2>nul
if errorlevel 1 (
  echo  [RAW Studio] Перша установка бібліотек: numpy, matplotlib...
  %PY% -m pip install --user numpy matplotlib
  if errorlevel 1 (
    echo  [RAW Studio] Не вдалося встановити бібліотеки автоматично.
    echo  Виконайте вручну в командному рядку:
    echo      %PY% -m pip install numpy matplotlib
    pause
    exit /b 1
  )
)

echo  [RAW Studio] Python %VER% — OK. Запуск...
start "" RAW_Studio.pyw
endlocal
