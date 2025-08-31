@echo off
echo Drone Flight Path Analyzer - GUI Launcher
echo =========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3.8 or higher from https://python.org
    pause
    exit /b 1
)

REM Check if we're in the right directory
if not exist "gui_drone_analyzer.py" (
    echo Error: gui_drone_analyzer.py not found in current directory
    echo Please run this batch file from the directory containing the drone analyzer files
    pause
    exit /b 1
)

REM Try to run the GUI launcher
echo Starting GUI...
python run_gui.py

REM If the launcher fails, try running the GUI directly
if errorlevel 1 (
    echo Trying to run GUI directly...
    python gui_drone_analyzer.py
)

pause

