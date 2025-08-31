#!/usr/bin/env python3
"""
GUI Drone Flight Path Analyzer
==============================

Graphical user interface for the drone flight path analyzer application.
Provides an easy-to-use interface for selecting folders, configuring settings,
and running the analysis with real-time progress updates.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import os
import sys
from pathlib import Path
import json
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from datetime import datetime

# Import our analyzer modules
try:
    from drone_flight_analyzer import DroneFlightAnalyzer
    from enhanced_drone_analyzer import EnhancedDroneAnalyzer
    from ai_image_matcher import AIImageMatcher
except ImportError as e:
    print(f"Warning: Could not import analyzer modules: {e}")
    print("Make sure all required modules are in the same directory.")

class DroneAnalyzerGUI:
    """Main GUI class for the drone flight path analyzer."""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Drone Flight Path Analyzer")
        self.root.geometry("1200x800")
        self.root.minsize(1000, 700)
        
        # Variables
        self.reference_dir = tk.StringVar()
        self.drone_dir = tk.StringVar()
        self.output_dir = tk.StringVar(value="output")
        self.use_enhanced = tk.BooleanVar(value=True)
        self.use_ai = tk.BooleanVar(value=True)
        self.use_semantic = tk.BooleanVar(value=False)
        self.ai_model = tk.StringVar(value="resnet50")
        self.detector_type = tk.StringVar(value="SIFT")
        self.matcher_type = tk.StringVar(value="FLANN")
        self.ratio_threshold = tk.DoubleVar(value=0.75)
        self.filename_index_window = tk.IntVar(value=4)
        
        # Analysis state
        self.analyzer = None
        self.analysis_thread = None
        self.is_analyzing = False
        
        # Create GUI components
        self.create_widgets()
        self.create_menu()
        
        # Configure style
        self.configure_styles()
        
    def configure_styles(self):
        """Configure ttk styles for better appearance."""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure colors
        style.configure('Title.TLabel', font=('Arial', 14, 'bold'))
        style.configure('Heading.TLabel', font=('Arial', 12, 'bold'))
        style.configure('Success.TLabel', foreground='green')
        style.configure('Error.TLabel', foreground='red')
        style.configure('Warning.TLabel', foreground='orange')
        
    def create_menu(self):
        """Create the application menu."""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Load Configuration", command=self.load_config)
        file_menu.add_command(label="Save Configuration", command=self.save_config)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.show_about)
        help_menu.add_command(label="Documentation", command=self.show_documentation)
        
    def create_widgets(self):
        """Create and arrange all GUI widgets."""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="Drone Flight Path Analyzer", 
                               style='Title.TLabel')
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # Create notebook for tabs
        notebook = ttk.Notebook(main_frame)
        notebook.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        
        # Configure notebook grid weights
        main_frame.rowconfigure(1, weight=1)
        
        # Settings tab
        settings_frame = ttk.Frame(notebook)
        notebook.add(settings_frame, text="Settings")
        self.create_settings_tab(settings_frame)
        
        # Analysis tab
        analysis_frame = ttk.Frame(notebook)
        notebook.add(analysis_frame, text="Analysis")
        self.create_analysis_tab(analysis_frame)
        
        # Results tab
        results_frame = ttk.Frame(notebook)
        notebook.add(results_frame, text="Results")
        self.create_results_tab(results_frame)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, 
                              relief=tk.SUNKEN, anchor=tk.W)
        status_bar.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 0))
        
    def create_settings_tab(self, parent):
        """Create the settings tab with folder selection and parameters."""
        # Folder selection frame
        folder_frame = ttk.LabelFrame(parent, text="Folder Selection", padding="10")
        folder_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        folder_frame.columnconfigure(1, weight=1)
        
        # Reference images folder
        ttk.Label(folder_frame, text="Reference Images Folder:").grid(row=0, column=0, sticky=tk.W, pady=5)
        ttk.Entry(folder_frame, textvariable=self.reference_dir, width=50).grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(5, 5), pady=5)
        ttk.Button(folder_frame, text="Browse", command=self.browse_reference_folder).grid(row=0, column=2, pady=5)
        
        # Drone images folder
        ttk.Label(folder_frame, text="Drone Images Folder:").grid(row=1, column=0, sticky=tk.W, pady=5)
        ttk.Entry(folder_frame, textvariable=self.drone_dir, width=50).grid(row=1, column=1, sticky=(tk.W, tk.E), padx=(5, 5), pady=5)
        ttk.Button(folder_frame, text="Browse", command=self.browse_drone_folder).grid(row=1, column=2, pady=5)
        
        # Output folder
        ttk.Label(folder_frame, text="Output Folder:").grid(row=2, column=0, sticky=tk.W, pady=5)
        ttk.Entry(folder_frame, textvariable=self.output_dir, width=50).grid(row=2, column=1, sticky=(tk.W, tk.E), padx=(5, 5), pady=5)
        ttk.Button(folder_frame, text="Browse", command=self.browse_output_folder).grid(row=2, column=2, pady=5)
        
        # Analysis options frame
        options_frame = ttk.LabelFrame(parent, text="Analysis Options", padding="10")
        options_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        options_frame.columnconfigure(1, weight=1)
        
        # Enhanced analyzer option
        ttk.Checkbutton(options_frame, text="Use Enhanced Analyzer (AI-powered)", 
                       variable=self.use_enhanced, command=self.toggle_enhanced_options).grid(row=0, column=0, columnspan=2, sticky=tk.W, pady=5)
        
        # AI options (only visible when enhanced is selected)
        self.ai_frame = ttk.Frame(options_frame)
        self.ai_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Checkbutton(self.ai_frame, text="Use AI-powered matching", 
                       variable=self.use_ai).grid(row=0, column=0, sticky=tk.W, pady=2)
        
        ttk.Label(self.ai_frame, text="AI Model:").grid(row=1, column=0, sticky=tk.W, pady=2)
        ai_model_combo = ttk.Combobox(self.ai_frame, textvariable=self.ai_model, 
                                     values=["resnet50", "vgg16", "efficientnet"], 
                                     state="readonly", width=15)
        ai_model_combo.grid(row=1, column=1, sticky=tk.W, padx=(5, 0), pady=2)
        
        ttk.Checkbutton(self.ai_frame, text="Use Semantic Segmentation", 
                       variable=self.use_semantic).grid(row=2, column=0, sticky=tk.W, pady=2)
        
        # Traditional options
        traditional_frame = ttk.LabelFrame(options_frame, text="Traditional CV Options", padding="5")
        traditional_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        traditional_frame.columnconfigure(1, weight=1)
        
        ttk.Label(traditional_frame, text="Feature Detector:").grid(row=0, column=0, sticky=tk.W, pady=2)
        detector_combo = ttk.Combobox(traditional_frame, textvariable=self.detector_type, 
                                     values=["SIFT", "ORB", "AKAZE"], state="readonly", width=15)
        detector_combo.grid(row=0, column=1, sticky=tk.W, padx=(5, 0), pady=2)
        
        ttk.Label(traditional_frame, text="Feature Matcher:").grid(row=1, column=0, sticky=tk.W, pady=2)
        matcher_combo = ttk.Combobox(traditional_frame, textvariable=self.matcher_type, 
                                    values=["FLANN", "BF"], state="readonly", width=15)
        matcher_combo.grid(row=1, column=1, sticky=tk.W, padx=(5, 0), pady=2)
        
        ttk.Label(traditional_frame, text="Ratio Threshold:").grid(row=2, column=0, sticky=tk.W, pady=2)
        ratio_scale = ttk.Scale(traditional_frame, from_=0.1, to=1.0, 
                               variable=self.ratio_threshold, orient=tk.HORIZONTAL)
        ratio_scale.grid(row=2, column=1, sticky=(tk.W, tk.E), padx=(5, 0), pady=2)

        # Filename index window option (applies to both basic and enhanced analyzers)
        ttk.Label(options_frame, text="Filename index window (+/-):").grid(row=3, column=0, sticky=tk.W, pady=(10, 2))
        self.window_spin = tk.Spinbox(options_frame, from_=0, to=100, width=6, textvariable=self.filename_index_window)
        self.window_spin.grid(row=3, column=1, sticky=tk.W, padx=(5, 0), pady=(10, 2))
        
        # Validation frame
        validation_frame = ttk.LabelFrame(parent, text="Validation", padding="10")
        validation_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.validation_text = scrolledtext.ScrolledText(validation_frame, height=6, width=70)
        self.validation_text.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        ttk.Button(validation_frame, text="Validate Settings", 
                  command=self.validate_settings).grid(row=1, column=0, pady=(10, 0))
        
        # Configure grid weights
        parent.columnconfigure(0, weight=1)
        
    def create_analysis_tab(self, parent):
        """Create the analysis tab with progress tracking and controls."""
        # Control frame
        control_frame = ttk.Frame(parent)
        control_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.start_button = ttk.Button(control_frame, text="Start Analysis", 
                                      command=self.start_analysis, style='Accent.TButton')
        self.start_button.grid(row=0, column=0, padx=(0, 10))
        
        self.stop_button = ttk.Button(control_frame, text="Stop Analysis", 
                                     command=self.stop_analysis, state=tk.DISABLED)
        self.stop_button.grid(row=0, column=1)
        
        # Progress frame
        progress_frame = ttk.LabelFrame(parent, text="Progress", padding="10")
        progress_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        progress_frame.columnconfigure(0, weight=1)
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, 
                                           maximum=100, length=400)
        self.progress_bar.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 5))
        
        self.progress_label = ttk.Label(progress_frame, text="Ready to start analysis")
        self.progress_label.grid(row=1, column=0, sticky=tk.W)
        
        # Log frame
        log_frame = ttk.LabelFrame(parent, text="Analysis Log", padding="10")
        log_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, height=15, width=80)
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(2, weight=1)
        
    def create_results_tab(self, parent):
        """Create the results tab with visualization and summary."""
        # Results summary frame
        summary_frame = ttk.LabelFrame(parent, text="Analysis Summary", padding="10")
        summary_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        summary_frame.columnconfigure(1, weight=1)
        
        self.summary_text = scrolledtext.ScrolledText(summary_frame, height=8, width=80)
        self.summary_text.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Button(summary_frame, text="Load Results", 
                  command=self.load_results).grid(row=1, column=0, sticky=tk.W)
        ttk.Button(summary_frame, text="Export Results", 
                  command=self.export_results).grid(row=1, column=1, sticky=tk.E)
        
        # Visualization frame
        viz_frame = ttk.LabelFrame(parent, text="Flight Path Visualization", padding="10")
        viz_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        viz_frame.columnconfigure(0, weight=1)
        viz_frame.rowconfigure(0, weight=1)
        
        # Create matplotlib figure
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, viz_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(1, weight=1)
        
    def toggle_enhanced_options(self):
        """Toggle visibility of AI options based on enhanced analyzer selection."""
        if self.use_enhanced.get():
            self.ai_frame.grid()
        else:
            self.ai_frame.grid_remove()
            
    def browse_reference_folder(self):
        """Browse for reference images folder."""
        folder = filedialog.askdirectory(title="Select Reference Images Folder")
        if folder:
            self.reference_dir.set(folder)
            
    def browse_drone_folder(self):
        """Browse for drone images folder."""
        folder = filedialog.askdirectory(title="Select Drone Images Folder")
        if folder:
            self.drone_dir.set(folder)
            
    def browse_output_folder(self):
        """Browse for output folder."""
        folder = filedialog.askdirectory(title="Select Output Folder")
        if folder:
            self.output_dir.set(folder)
            
    def validate_settings(self):
        """Validate the current settings and display results."""
        self.validation_text.delete(1.0, tk.END)
        
        errors = []
        warnings = []
        info = []
        
        # Check reference directory
        ref_dir = self.reference_dir.get()
        if not ref_dir:
            errors.append("Reference images folder is not selected")
        elif not os.path.exists(ref_dir):
            errors.append("Reference images folder does not exist")
        else:
            ref_images = list(Path(ref_dir).glob("*.jpg")) + list(Path(ref_dir).glob("*.jpeg"))
            if not ref_images:
                warnings.append("No JPEG images found in reference folder")
            else:
                info.append(f"Found {len(ref_images)} reference images")
                
        # Check drone directory
        drone_dir = self.drone_dir.get()
        if not drone_dir:
            errors.append("Drone images folder is not selected")
        elif not os.path.exists(drone_dir):
            errors.append("Drone images folder does not exist")
        else:
            drone_images = list(Path(drone_dir).glob("*.jpg")) + list(Path(drone_dir).glob("*.jpeg"))
            if not drone_images:
                warnings.append("No JPEG images found in drone folder")
            else:
                info.append(f"Found {len(drone_images)} drone images")
                
        # Check output directory
        output_dir = self.output_dir.get()
        if not output_dir:
            errors.append("Output folder is not specified")
        else:
            if not os.path.exists(output_dir):
                info.append("Output folder will be created")
            else:
                info.append("Output folder exists")
                
        # Display results
        if errors:
            self.validation_text.insert(tk.END, "ERRORS:\n", "error")
            for error in errors:
                self.validation_text.insert(tk.END, f"• {error}\n", "error")
            self.validation_text.insert(tk.END, "\n")
            
        if warnings:
            self.validation_text.insert(tk.END, "WARNINGS:\n", "warning")
            for warning in warnings:
                self.validation_text.insert(tk.END, f"• {warning}\n", "warning")
            self.validation_text.insert(tk.END, "\n")
            
        if info:
            self.validation_text.insert(tk.END, "INFO:\n", "info")
            for item in info:
                self.validation_text.insert(tk.END, f"• {item}\n", "info")
                
        # Configure text tags for colors
        self.validation_text.tag_configure("error", foreground="red")
        self.validation_text.tag_configure("warning", foreground="orange")
        self.validation_text.tag_configure("info", foreground="blue")
        
    def start_analysis(self):
        """Start the analysis in a separate thread."""
        if self.is_analyzing:
            return
            
        # Validate settings first
        if not self.reference_dir.get() or not self.drone_dir.get():
            messagebox.showerror("Error", "Please select both reference and drone image folders")
            return
            
        # Update UI state
        self.is_analyzing = True
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.progress_var.set(0)
        self.progress_label.config(text="Initializing analysis...")
        self.log_text.delete(1.0, tk.END)
        
        # Start analysis thread
        self.analysis_thread = threading.Thread(target=self.run_analysis)
        self.analysis_thread.daemon = True
        self.analysis_thread.start()
        
    def stop_analysis(self):
        """Stop the analysis."""
        self.is_analyzing = False
        self.log_message("Analysis stopped by user")
        self.update_ui_after_analysis()
        
    def run_analysis(self):
        """Run the analysis in a separate thread."""
        try:
            # Create output directory
            output_dir = self.output_dir.get()
            os.makedirs(output_dir, exist_ok=True)
            
            # Initialize analyzer
            if self.use_enhanced.get():
                self.analyzer = EnhancedDroneAnalyzer(
                    use_ai=self.use_ai.get(),
                    use_semantic=self.use_semantic.get(),
                    model_name=self.ai_model.get()
                )
                # Apply filename window optimization
                try:
                    self.analyzer.filename_index_window = int(self.filename_index_window.get())
                except Exception:
                    self.analyzer.filename_index_window = 4
                self.log_message("Initialized Enhanced Drone Analyzer")
            else:
                self.analyzer = DroneFlightAnalyzer(
                    feature_detector=self.detector_type.get(),
                    matcher=self.matcher_type.get()
                )
                # Apply filename window optimization
                try:
                    self.analyzer.filename_index_window = int(self.filename_index_window.get())
                except Exception:
                    self.analyzer.filename_index_window = 4
                self.log_message("Initialized Basic Drone Analyzer")
            self.log_message(f"Filename index window set to +/- {self.analyzer.filename_index_window}")
                
            # Load images
            self.log_message("Loading reference images...")
            self.analyzer.load_reference_images(self.reference_dir.get())
            self.update_progress(20)
            
            self.log_message("Loading drone images...")
            self.analyzer.load_drone_images(self.drone_dir.get())
            self.update_progress(40)
            
            # Run analysis
            self.log_message("Starting flight path analysis...")
            if self.use_enhanced.get():
                flight_path = self.analyzer.analyze_flight_path_enhanced()
            else:
                flight_path = self.analyzer.analyze_flight_path()
            self.update_progress(80)
            
            # Save results
            self.log_message("Saving results...")
            if self.use_enhanced.get():
                self.analyzer.save_enhanced_results(os.path.join(output_dir, 'enhanced_analysis_results.json'))
                # Do not open a matplotlib GUI from worker thread
                self.analyzer.visualize_enhanced_results(os.path.join(output_dir, 'enhanced_flight_path.png'), show=False)
            else:
                self.analyzer.save_results(os.path.join(output_dir, 'flight_analysis_results.json'))
                self.analyzer.export_gpx(os.path.join(output_dir, 'drone_flight_path.gpx'))
                # Do not open a matplotlib GUI from worker thread
                self.analyzer.visualize_flight_path(os.path.join(output_dir, 'flight_path.png'), show=False)
                
            self.update_progress(100)
            self.log_message("Analysis completed successfully!")
            
            # Update results
            self.update_results_summary(flight_path)
            self.plot_flight_path(flight_path)
            
        except Exception as e:
            self.log_message(f"Error during analysis: {str(e)}")
            messagebox.showerror("Analysis Error", f"An error occurred during analysis:\n{str(e)}")
            
        finally:
            self.update_ui_after_analysis()
            
    def update_progress(self, value):
        """Update progress bar and label."""
        self.progress_var.set(value)
        self.progress_label.config(text=f"Progress: {value}%")
        
    def log_message(self, message):
        """Add message to log with timestamp."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"
        
        # Use after() to update UI from thread
        self.root.after(0, lambda: self.log_text.insert(tk.END, log_entry))
        self.root.after(0, lambda: self.log_text.see(tk.END))
        
    def update_ui_after_analysis(self):
        """Update UI state after analysis completes."""
        self.is_analyzing = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.progress_label.config(text="Analysis completed")
        
    def update_results_summary(self, flight_path):
        """Update the results summary text."""
        self.summary_text.delete(1.0, tk.END)
        
        if not flight_path:
            self.summary_text.insert(tk.END, "No flight path data available.")
            return
            
        summary = f"Analysis Summary:\n"
        summary += f"================\n\n"
        summary += f"Total positions estimated: {len(flight_path)}\n"
        
        if hasattr(flight_path[0], 'combined_confidence'):
            # Enhanced analyzer results
            avg_ai_conf = np.mean([pos.ai_confidence for pos in flight_path])
            avg_trad_conf = np.mean([pos.traditional_confidence for pos in flight_path])
            avg_combined_conf = np.mean([pos.combined_confidence for pos in flight_path])
            
            summary += f"Average AI confidence: {avg_ai_conf:.3f}\n"
            summary += f"Average traditional confidence: {avg_trad_conf:.3f}\n"
            summary += f"Average combined confidence: {avg_combined_conf:.3f}\n"
        else:
            # Basic analyzer results
            avg_conf = np.mean([pos['confidence'] for pos in flight_path])
            summary += f"Average confidence: {avg_conf:.3f}\n"
            
        summary += f"\nResults saved to: {self.output_dir.get()}\n"
        
        self.summary_text.insert(tk.END, summary)
        
    def plot_flight_path(self, flight_path):
        """Plot the flight path in the results tab."""
        if not flight_path:
            return
            
        self.ax.clear()
        
        if hasattr(flight_path[0], 'longitude'):
            # Enhanced analyzer results
            lons = [pos.longitude for pos in flight_path]
            lats = [pos.latitude for pos in flight_path]
            confidences = [pos.combined_confidence for pos in flight_path]
        else:
            # Basic analyzer results
            lons = [pos['longitude'] for pos in flight_path]
            lats = [pos['latitude'] for pos in flight_path]
            confidences = [pos['confidence'] for pos in flight_path]
            
        # Plot flight path
        scatter = self.ax.scatter(lons, lats, c=confidences, cmap='viridis', s=50, alpha=0.7)
        self.ax.plot(lons, lats, 'b-', linewidth=2, alpha=0.5)
        
        # Add colorbar
        plt.colorbar(scatter, ax=self.ax, label='Confidence')
        
        self.ax.set_xlabel('Longitude')
        self.ax.set_ylabel('Latitude')
        self.ax.set_title('Drone Flight Path')
        self.ax.grid(True, alpha=0.3)
        
        self.canvas.draw()
        
    def load_results(self):
        """Load and display previously saved results."""
        file_path = filedialog.askopenfilename(
            title="Select Results File",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                with open(file_path, 'r') as f:
                    results = json.load(f)
                    
                flight_path = results.get('flight_path', [])
                self.update_results_summary(flight_path)
                self.plot_flight_path(flight_path)
                
                messagebox.showinfo("Success", "Results loaded successfully!")
                
            except Exception as e:
                messagebox.showerror("Error", f"Could not load results: {str(e)}")
                
    def export_results(self):
        """Export results to different formats."""
        if not hasattr(self, 'analyzer') or not self.analyzer:
            messagebox.showwarning("Warning", "No analysis results to export")
            return
            
        # Export options
        export_window = tk.Toplevel(self.root)
        export_window.title("Export Results")
        export_window.geometry("300x200")
        export_window.transient(self.root)
        export_window.grab_set()
        
        ttk.Label(export_window, text="Select export format:").pack(pady=10)
        
        export_format = tk.StringVar(value="gpx")
        ttk.Radiobutton(export_window, text="GPX (for mapping applications)", 
                       variable=export_format, value="gpx").pack(anchor=tk.W, padx=20)
        ttk.Radiobutton(export_window, text="CSV (for spreadsheet applications)", 
                       variable=export_format, value="csv").pack(anchor=tk.W, padx=20)
        ttk.Radiobutton(export_window, text="KML (for Google Earth)", 
                       variable=export_format, value="kml").pack(anchor=tk.W, padx=20)
        
        def do_export():
            try:
                output_path = filedialog.asksaveasfilename(
                    title="Save Export File",
                    defaultextension=f".{export_format.get()}",
                    filetypes=[(f"{export_format.get().upper()} files", f"*.{export_format.get()}")]
                )
                
                if output_path:
                    if export_format.get() == "gpx" and hasattr(self.analyzer, 'export_gpx'):
                        self.analyzer.export_gpx(output_path)
                    elif export_format.get() == "csv":
                        self.export_to_csv(output_path)
                    elif export_format.get() == "kml":
                        self.export_to_kml(output_path)
                        
                    messagebox.showinfo("Success", f"Results exported to {output_path}")
                    export_window.destroy()
                    
            except Exception as e:
                messagebox.showerror("Error", f"Export failed: {str(e)}")
                
        ttk.Button(export_window, text="Export", command=do_export).pack(pady=20)
        
    def export_to_csv(self, output_path):
        """Export results to CSV format."""
        import csv
        
        with open(output_path, 'w', newline='') as csvfile:
            if hasattr(self.analyzer, 'flight_path') and self.analyzer.flight_path:
                if hasattr(self.analyzer.flight_path[0], 'longitude'):
                    # Enhanced analyzer results
                    fieldnames = ['latitude', 'longitude', 'altitude', 'ai_confidence', 
                                'traditional_confidence', 'combined_confidence', 'match_count']
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    
                    for pos in self.analyzer.flight_path:
                        writer.writerow({
                            'latitude': pos.latitude,
                            'longitude': pos.longitude,
                            'altitude': pos.altitude,
                            'ai_confidence': pos.ai_confidence,
                            'traditional_confidence': pos.traditional_confidence,
                            'combined_confidence': pos.combined_confidence,
                            'match_count': pos.match_count
                        })
                else:
                    # Basic analyzer results
                    fieldnames = ['latitude', 'longitude', 'altitude', 'confidence', 'match_count']
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    
                    for pos in self.analyzer.flight_path:
                        writer.writerow(pos)
                        
    def export_to_kml(self, output_path):
        """Export results to KML format for Google Earth."""
        kml_content = """<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Document>
    <name>Drone Flight Path</name>
    <Placemark>
      <name>Flight Path</name>
      <LineString>
        <coordinates>
"""
        
        if hasattr(self.analyzer, 'flight_path') and self.analyzer.flight_path:
            for pos in self.analyzer.flight_path:
                if hasattr(pos, 'longitude'):
                    kml_content += f"          {pos.longitude},{pos.latitude},{pos.altitude}\n"
                else:
                    kml_content += f"          {pos['longitude']},{pos['latitude']},{pos['altitude']}\n"
                    
        kml_content += """        </coordinates>
      </LineString>
    </Placemark>
  </Document>
</kml>"""
        
        with open(output_path, 'w') as f:
            f.write(kml_content)
            
    def load_config(self):
        """Load configuration from file."""
        file_path = filedialog.askopenfilename(
            title="Load Configuration",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                with open(file_path, 'r') as f:
                    config = json.load(f)
                    
                # Update variables
                self.reference_dir.set(config.get('reference_dir', ''))
                self.drone_dir.set(config.get('drone_dir', ''))
                self.output_dir.set(config.get('output_dir', 'output'))
                self.use_enhanced.set(config.get('use_enhanced', True))
                self.use_ai.set(config.get('use_ai', True))
                self.use_semantic.set(config.get('use_semantic', False))
                self.ai_model.set(config.get('ai_model', 'resnet50'))
                self.detector_type.set(config.get('detector_type', 'SIFT'))
                self.matcher_type.set(config.get('matcher_type', 'FLANN'))
                self.ratio_threshold.set(config.get('ratio_threshold', 0.75))
                self.filename_index_window.set(config.get('filename_index_window', 4))
                
                self.toggle_enhanced_options()
                messagebox.showinfo("Success", "Configuration loaded successfully!")
                
            except Exception as e:
                messagebox.showerror("Error", f"Could not load configuration: {str(e)}")
                
    def save_config(self):
        """Save current configuration to file."""
        file_path = filedialog.asksaveasfilename(
            title="Save Configuration",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                config = {
                    'reference_dir': self.reference_dir.get(),
                    'drone_dir': self.drone_dir.get(),
                    'output_dir': self.output_dir.get(),
                    'use_enhanced': self.use_enhanced.get(),
                    'use_ai': self.use_ai.get(),
                    'use_semantic': self.use_semantic.get(),
                    'ai_model': self.ai_model.get(),
                    'detector_type': self.detector_type.get(),
                    'matcher_type': self.matcher_type.get(),
                    'ratio_threshold': self.ratio_threshold.get(),
                    'filename_index_window': int(self.filename_index_window.get())
                }
                
                with open(file_path, 'w') as f:
                    json.dump(config, f, indent=2)
                    
                messagebox.showinfo("Success", "Configuration saved successfully!")
                
            except Exception as e:
                messagebox.showerror("Error", f"Could not save configuration: {str(e)}")
                
    def show_about(self):
        """Show about dialog."""
        about_text = """Drone Flight Path Analyzer
Version 1.0

A Python application for analyzing drone flight paths using computer vision and AI techniques.

Features:
• Traditional OpenCV-based image matching
• AI-powered deep learning matching
• Semantic segmentation analysis
• Multiple export formats (JSON, GPX, CSV, KML)
• Interactive GUI with real-time progress tracking

For more information, see the README.md file."""
        
        messagebox.showinfo("About", about_text)
        
    def show_documentation(self):
        """Show documentation."""
        doc_text = """Documentation

1. SETUP:
   - Install required packages: pip install -r requirements.txt
   - Ensure you have reference images with GPS metadata
   - Prepare drone images for analysis

2. USAGE:
   - Select reference images folder (with GPS metadata)
   - Select drone images folder
   - Choose output directory
   - Configure analysis options
   - Click "Start Analysis"

3. OUTPUT:
   - JSON results file
   - Flight path visualization
   - GPX file for mapping applications
   - Analysis summary and statistics

4. TROUBLESHOOTING:
   - Ensure images are in JPEG format
   - Check that reference images have GPS metadata
   - Verify folder permissions
   - Check log messages for detailed error information"""
        
        # Create a new window for documentation
        doc_window = tk.Toplevel(self.root)
        doc_window.title("Documentation")
        doc_window.geometry("600x500")
        
        text_widget = scrolledtext.ScrolledText(doc_window, wrap=tk.WORD)
        text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        text_widget.insert(tk.END, doc_text)


def main():
    """Main function to run the GUI application."""
    root = tk.Tk()
    app = DroneAnalyzerGUI(root)
    
    # Center the window
    root.update_idletasks()
    x = (root.winfo_screenwidth() // 2) - (root.winfo_width() // 2)
    y = (root.winfo_screenheight() // 2) - (root.winfo_height() // 2)
    root.geometry(f"+{x}+{y}")
    
    root.mainloop()


if __name__ == "__main__":
    main()

