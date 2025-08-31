#!/bin/bash

echo "Drone Flight Path Analyzer - GUI Launcher"
echo "========================================="
echo

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed or not in PATH"
    echo "Please install Python 3.8 or higher"
    exit 1
fi

# Check if we're in the right directory
if [ ! -f "gui_drone_analyzer.py" ]; then
    echo "Error: gui_drone_analyzer.py not found in current directory"
    echo "Please run this script from the directory containing the drone analyzer files"
    exit 1
fi

# Make the script executable
chmod +x run_gui.py

# Try to run the GUI launcher
echo "Starting GUI..."
python3 run_gui.py

# If the launcher fails, try running the GUI directly
if [ $? -ne 0 ]; then
    echo "Trying to run GUI directly..."
    python3 gui_drone_analyzer.py
fi

