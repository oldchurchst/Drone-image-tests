# Drone Flight Path Analyzer

An advanced Python application that analyzes drone images to determine flight paths using AI-powered computer vision techniques. The application compares drone images with reference images containing GPS metadata to estimate the drone's position without using the drone images' own GPS data.

## Features

- **AI-Powered Image Matching**: Uses deep learning models (ResNet50, VGG16, EfficientNet) for advanced feature extraction and matching
- **Traditional Computer Vision**: OpenCV-based feature detection and matching as fallback
- **Semantic Segmentation**: Optional semantic understanding for improved matching accuracy
- **Multiple Confidence Metrics**: Combines AI, traditional, and semantic confidence scores
- **Enhanced Triangulation**: Sophisticated position estimation using homography and geometric transformations
- **Comprehensive Visualization**: Detailed plots and analysis of flight paths, confidence scores, and matching statistics
- **Export Capabilities**: Results exported as JSON, GPX, and high-quality visualizations

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (optional, for faster AI processing)

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Manual Installation (if needed)

```bash
pip install opencv-python numpy matplotlib scipy Pillow pathlib2
pip install torch torchvision
```

## Usage

### Graphical User Interface (Recommended)

For the easiest experience, use the GUI application:

```bash
python gui_drone_analyzer.py
```

The GUI provides:
- **Settings Tab**: Select folders, configure analysis options, validate settings
- **Analysis Tab**: Start/stop analysis with real-time progress tracking and logging
- **Results Tab**: View analysis summary, flight path visualization, and export results

Features:
- Folder selection with browse dialogs
- Configuration validation
- Real-time progress tracking
- Interactive flight path visualization
- Multiple export formats (JSON, GPX, CSV, KML)
- Configuration save/load functionality
- Comprehensive logging

### Command Line Interface

#### Basic Usage

```bash
python drone_flight_analyzer.py --reference-dir /path/to/reference/images --drone-dir /path/to/drone/images
```

#### Enhanced AI-Powered Analysis

```bash
python enhanced_drone_analyzer.py --reference-dir /path/to/reference/images --drone-dir /path/to/drone/images --use-ai --model resnet50
```

#### Advanced Options

```bash
python enhanced_drone_analyzer.py \
    --reference-dir /path/to/reference/images \
    --drone-dir /path/to/drone/images \
    --output-dir results \
    --use-ai \
    --use-semantic \
    --model efficientnet
```

## Command Line Arguments

### Basic Analyzer (`drone_flight_analyzer.py`)

- `--reference-dir`: Directory containing reference images with GPS metadata (required)
- `--drone-dir`: Directory containing drone images to analyze (required)
- `--output-dir`: Output directory for results (default: 'output')
- `--detector`: Feature detector to use ('SIFT', 'ORB', 'AKAZE') (default: 'SIFT')
- `--matcher`: Feature matcher to use ('FLANN', 'BF') (default: 'FLANN')
- `--ratio-threshold`: Lowe's ratio test threshold (default: 0.75)

### Enhanced Analyzer (`enhanced_drone_analyzer.py`)

- `--reference-dir`: Directory containing reference images with GPS metadata (required)
- `--drone-dir`: Directory containing drone images to analyze (required)
- `--output-dir`: Output directory for results (default: 'enhanced_output')
- `--use-ai`: Enable AI-powered matching (default: True)
- `--use-semantic`: Enable semantic segmentation matching
- `--model`: AI model to use ('resnet50', 'vgg16', 'efficientnet') (default: 'resnet50')

## Input Requirements

### Reference Images
- Must contain GPS metadata in EXIF format
- Supported formats: JPG, JPEG, PNG, TIFF, BMP
- Should cover the area where the drone flew
- Higher quality and more reference images improve accuracy

### Drone Images
- Should be in chronological order (sorted by filename)
- Supported formats: JPG, JPEG, PNG, TIFF, BMP
- Should have sufficient overlap between consecutive images
- Higher resolution images provide better feature matching

## Output Files

### Basic Analyzer Output
- `flight_analysis_results.json`: Analysis results in JSON format
- `drone_flight_path.gpx`: Flight path in GPX format for mapping applications
- `flight_path.png`: Visualization of the flight path and statistics

### Enhanced Analyzer Output
- `enhanced_analysis_results.json`: Detailed analysis with confidence metrics
- `enhanced_flight_path.png`: Comprehensive visualization with multiple plots

## Technical Details

### AI-Powered Matching

The enhanced analyzer uses pre-trained deep learning models to extract high-level features from images:

1. **Feature Extraction**: Uses CNN backbones (ResNet50, VGG16, EfficientNet) to extract 512-dimensional feature vectors
2. **Similarity Computation**: Calculates cosine similarity between feature vectors
3. **Attention Mechanism**: Optional attention module to focus on important image regions
4. **Hybrid Matching**: Combines AI and traditional OpenCV matching for robust results

### Traditional Computer Vision

Fallback matching using OpenCV techniques:

1. **Feature Detection**: SIFT, ORB, or AKAZE keypoint detection
2. **Feature Matching**: FLANN or Brute Force matcher with Lowe's ratio test
3. **Homography Estimation**: RANSAC-based homography matrix calculation
4. **Position Triangulation**: Geometric transformation to estimate drone position

### Semantic Segmentation

Optional semantic understanding for improved matching:

1. **Segmentation Model**: FCN-ResNet50 pre-trained on COCO dataset
2. **Semantic Features**: Class probability distributions as feature vectors
3. **Semantic Similarity**: Correlation-based similarity between semantic features

### Confidence Metrics

The enhanced analyzer provides multiple confidence measures:

- **AI Confidence**: Based on deep learning feature similarity
- **Traditional Confidence**: Based on OpenCV feature match count
- **Semantic Similarity**: Based on semantic segmentation correlation
- **Combined Confidence**: Weighted combination of all metrics

## Example Workflow

1. **Prepare Data**:
   ```bash
   mkdir reference_images drone_images
   # Copy reference images with GPS metadata to reference_images/
   # Copy drone images to drone_images/
   ```

2. **Run Basic Analysis**:
   ```bash
   python drone_flight_analyzer.py --reference-dir reference_images --drone-dir drone_images
   ```

3. **Run Enhanced Analysis**:
   ```bash
   python enhanced_drone_analyzer.py --reference-dir reference_images --drone-dir drone_images --use-ai
   ```

4. **View Results**:
   - Check the output directory for results
   - Open the PNG visualization files
   - Import GPX files into mapping applications

## Performance Considerations

### GPU Acceleration
- Install CUDA-compatible PyTorch for GPU acceleration
- AI matching is significantly faster with GPU support
- Traditional OpenCV matching works well on CPU

### Memory Usage
- Large image datasets may require significant RAM
- Consider processing in batches for very large datasets
- Feature extraction is cached to avoid recomputation

### Processing Time
- Traditional matching: ~1-5 seconds per image pair
- AI matching: ~2-10 seconds per image pair (CPU), ~0.5-2 seconds (GPU)
- Semantic matching: ~3-15 seconds per image pair

## Troubleshooting

### Common Issues

1. **No GPS metadata found**: Ensure reference images contain GPS EXIF data
2. **Low matching confidence**: Try different feature detectors or increase reference image quality
3. **Memory errors**: Reduce image resolution or process in smaller batches
4. **CUDA errors**: Install CPU-only PyTorch version if GPU is not available

### Performance Tips

1. **Reference Images**: Use high-quality, well-distributed reference images
2. **Image Overlap**: Ensure drone images have sufficient overlap
3. **Image Resolution**: Higher resolution images provide better feature matching
4. **Model Selection**: ResNet50 provides good balance of speed and accuracy

## Advanced Usage

### Custom AI Models

You can extend the AI matcher with custom models:

```python
from ai_image_matcher import AIImageMatcher

# Use custom model
matcher = AIImageMatcher(model_name='custom_model')
```

### Batch Processing

For large datasets, consider processing in batches:

```python
from enhanced_drone_analyzer import EnhancedDroneAnalyzer

analyzer = EnhancedDroneAnalyzer(use_ai=True)
# Process images in batches
batch_size = 10
for i in range(0, len(images), batch_size):
    batch = images[i:i+batch_size]
    # Process batch
```

### Custom Confidence Weights

Adjust the confidence weighting in the enhanced analyzer:

```python
# In match_images_enhanced method
weights = {
    'traditional': 0.2,  # Reduce traditional weight
    'ai': 0.6,          # Increase AI weight
    'semantic': 0.2     # Keep semantic weight
}
```

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for:

- Bug fixes
- Performance improvements
- New AI models
- Additional visualization options
- Documentation improvements

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenCV community for computer vision algorithms
- PyTorch team for deep learning framework
- DJI for drone technology inspiration
- Computer vision research community for feature matching techniques
