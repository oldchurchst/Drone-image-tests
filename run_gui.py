#!/usr/bin/env python3
"""
GUI Launcher for Drone Flight Path Analyzer
==========================================

Simple launcher script to start the GUI application with proper error handling.
"""

import sys
import os
import subprocess
import importlib.util

def check_dependencies():
    """Check if required dependencies are installed."""
    required_packages = [
        'tkinter',
        'matplotlib',
        'numpy',
        'cv2',
        'PIL',
        'torch'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        if package == 'tkinter':
            try:
                import tkinter
            except ImportError:
                missing_packages.append('tkinter (usually included with Python)')
        elif package == 'cv2':
            try:
                import cv2
            except ImportError:
                missing_packages.append('opencv-python')
        elif package == 'PIL':
            try:
                import PIL
            except ImportError:
                missing_packages.append('Pillow')
        else:
            try:
                importlib.import_module(package)
            except ImportError:
                missing_packages.append(package)
    
    return missing_packages

def install_dependencies():
    """Install missing dependencies."""
    print("Installing missing dependencies...")
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
        print("Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error installing dependencies: {e}")
        return False

def main():
    """Main launcher function."""
    print("Drone Flight Path Analyzer - GUI Launcher")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists('gui_drone_analyzer.py'):
        print("Error: gui_drone_analyzer.py not found in current directory.")
        print("Please run this script from the directory containing the drone analyzer files.")
        input("Press Enter to exit...")
        return
    
    # Check dependencies
    print("Checking dependencies...")
    missing = check_dependencies()
    
    if missing:
        print(f"Missing dependencies: {', '.join(missing)}")
        print("\nAttempting to install missing dependencies...")
        
        if not install_dependencies():
            print("\nFailed to install dependencies automatically.")
            print("Please install them manually:")
            print("pip install -r requirements.txt")
            input("Press Enter to exit...")
            return
        
        # Check again after installation
        missing = check_dependencies()
        if missing:
            print(f"Still missing: {', '.join(missing)}")
            print("Please install them manually and try again.")
            input("Press Enter to exit...")
            return
    
    print("All dependencies found!")
    
    # Try to import and run the GUI
    try:
        print("Starting GUI...")
        from gui_drone_analyzer import main as gui_main
        gui_main()
    except ImportError as e:
        print(f"Error importing GUI module: {e}")
        print("Make sure all required files are in the current directory.")
        input("Press Enter to exit...")
    except Exception as e:
        print(f"Error starting GUI: {e}")
        input("Press Enter to exit...")

if __name__ == "__main__":
    main()

