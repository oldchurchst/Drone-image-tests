#!/usr/bin/env python3
"""
Enhanced Drone Flight Path Analyzer
===================================

Advanced drone flight path analysis using AI-powered image matching
and sophisticated triangulation techniques.
"""

import cv2
import numpy as np
import os
import json
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import matplotlib.pyplot as plt
import logging
from dataclasses import dataclass
from scipy.spatial import Delaunay
from scipy.optimize import least_squares
import pickle
from ai_image_matcher import AIImageMatcher, SemanticSegmentationMatcher
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class EnhancedPosition:
    """Enhanced position estimation with multiple confidence metrics."""
    latitude: float
    longitude: float
    altitude: float
    ai_confidence: float
    traditional_confidence: float
    combined_confidence: float
    match_count: int
    image_path: str
    reference_image: str
    semantic_similarity: Optional[float] = None

class EnhancedDroneAnalyzer:
    """Enhanced drone flight analyzer with AI-powered matching."""
    
    def __init__(self, use_ai=True, use_semantic=False, model_name='resnet50'):
        """
        Initialize the enhanced analyzer.
        
        Args:
            use_ai: Whether to use AI-powered matching
            use_semantic: Whether to use semantic segmentation
            model_name: AI model to use
        """
        self.use_ai = use_ai
        self.use_semantic = use_semantic
        
        # Initialize AI matcher if requested
        if use_ai:
            self.ai_matcher = AIImageMatcher(model_name=model_name)
            logger.info(f"AI matcher initialized with {model_name}")
        
        # Initialize semantic matcher if requested
        if use_semantic:
            self.semantic_matcher = SemanticSegmentationMatcher()
            logger.info("Semantic segmentation matcher initialized")
        
        # Traditional OpenCV components
        self.feature_detector = cv2.SIFT_create()
        self.matcher = cv2.FlannBasedMatcher(
            dict(algorithm=1, trees=5),
            dict(checks=50)
        )
        
        # Data storage
        self.reference_images = {}
        self.drone_images = []
        self.flight_path = []
        self.match_cache = {}
        self.filename_index_window = 4  # Only compare to references within +/- this index window
        
        logger.info("Enhanced Drone Analyzer initialized")
    
    def load_reference_images(self, reference_dir: str):
        """Load reference images with GPS metadata."""
        logger.info(f"Loading reference images from {reference_dir}")
        
        for image_path in Path(reference_dir).glob("*.jpg"):
            gps_data = self._extract_gps_metadata(str(image_path))
            if gps_data:
                self.reference_images[str(image_path)] = gps_data
                logger.info(f"Loaded reference image: {image_path} with GPS: {gps_data}")
        
        logger.info(f"Loaded {len(self.reference_images)} reference images")

    def _extract_filename_index(self, image_path: str) -> Optional[int]:
        """Extract trailing numeric index from filename stem (e.g., DJI_0122.jpg -> 122)."""
        try:
            import re
            stem = Path(image_path).stem
            numbers = re.findall(r"(\d+)", stem)
            if not numbers:
                return None
            return int(numbers[-1])
        except Exception:
            return None
    
    def load_drone_images(self, drone_dir: str):
        """Load drone images for analysis."""
        logger.info(f"Loading drone images from {drone_dir}")
        
        image_extensions = ['.jpg', '.jpeg', '.png', '.tiff', '.bmp']
        for ext in image_extensions:
            for image_path in Path(drone_dir).glob(f"*{ext}"):
                self.drone_images.append(str(image_path))
        
        self.drone_images.sort()
        logger.info(f"Loaded {len(self.drone_images)} drone images")
    
    def _extract_gps_metadata(self, image_path: str) -> Optional[Dict]:
        """Extract GPS metadata from image EXIF data."""
        try:
            from PIL import Image
            from PIL.ExifTags import TAGS, GPSTAGS
            
            image = Image.open(image_path)
            exif = image._getexif()
            
            if not exif:
                return None
            
            gps_data = {}
            for tag_id in exif:
                tag = TAGS.get(tag_id, tag_id)
                data = exif[tag_id]
                
                if tag == 'GPSInfo':
                    for gps_tag_id in data:
                        gps_tag = GPSTAGS.get(gps_tag_id, gps_tag_id)
                        gps_data[gps_tag] = data[gps_tag_id]
            
            if gps_data:
                lat = self._convert_to_degrees(gps_data.get('GPSLatitude', [0, 0, 0]))
                lon = self._convert_to_degrees(gps_data.get('GPSLongitude', [0, 0, 0]))
                alt = gps_data.get('GPSAltitude', 0)
                
                if gps_data.get('GPSLatitudeRef') == 'S':
                    lat = -lat
                if gps_data.get('GPSLongitudeRef') == 'W':
                    lon = -lon
                
                return {
                    'latitude': lat,
                    'longitude': lon,
                    'altitude': alt
                }
        
        except Exception as e:
            logger.warning(f"Could not extract GPS metadata from {image_path}: {e}")
        
        return None
    
    def _convert_to_degrees(self, dms: List) -> float:
        """Convert GPS coordinates from degrees, minutes, seconds to decimal degrees."""
        degrees = float(dms[0])
        minutes = float(dms[1])
        seconds = float(dms[2])
        return degrees + (minutes / 60.0) + (seconds / 3600.0)
    
    def match_images_enhanced(self, drone_image: str, reference_image: str) -> Dict:
        """
        Enhanced image matching using multiple techniques.
        
        Args:
            drone_image: Path to drone image
            reference_image: Path to reference image
            
        Returns:
            Dictionary with matching results
        """
        # Check cache first
        cache_key = f"{drone_image}_{reference_image}"
        if cache_key in self.match_cache:
            return self.match_cache[cache_key]
        
        results = {
            'traditional_matches': 0,
            'traditional_confidence': 0.0,
            'ai_confidence': 0.0,
            'semantic_similarity': 0.0,
            'combined_confidence': 0.0
        }
        
        # Traditional OpenCV matching
        try:
            t_match = time.perf_counter()
            img1 = cv2.imread(drone_image)
            img2 = cv2.imread(reference_image)
            kp1, des1 = self.feature_detector.detectAndCompute(img1, None)
            kp2, des2 = self.feature_detector.detectAndCompute(img2, None)
            
            if des1 is not None and des2 is not None:
                matches = self.matcher.knnMatch(des1, des2, k=2)
                
                good_matches = []
                for match_pair in matches:
                    if len(match_pair) == 2:
                        m, n = match_pair
                        if m.distance < 0.75 * n.distance:
                            good_matches.append(m)
                
                results['traditional_matches'] = len(good_matches)
                results['traditional_confidence'] = min(1.0, len(good_matches) / 50.0)
                logger.info(
                    f"Traditional matching | drone={Path(drone_image).name} ref={Path(reference_image).name} "
                    f"kp1={len(kp1) if kp1 else 0} kp2={len(kp2) if kp2 else 0} "
                    f"good_matches={len(good_matches)} elapsed_ms={(time.perf_counter()-t_match)*1000:.1f}"
                )
        except Exception as e:
            logger.warning(f"Traditional matching failed: {e}")
        
        # AI-based matching
        if self.use_ai:
            try:
                t_ai = time.perf_counter()
                ai_result = self.ai_matcher.match_images_hybrid(drone_image, reference_image)
                results['ai_confidence'] = ai_result['combined_confidence']
                logger.info(
                    f"AI matching | drone={Path(drone_image).name} ref={Path(reference_image).name} "
                    f"combined_confidence={results['ai_confidence']:.3f} elapsed_ms={(time.perf_counter()-t_ai)*1000:.1f}"
                )
            except Exception as e:
                logger.warning(f"AI matching failed: {e}")
        
        # Semantic matching
        if self.use_semantic:
            try:
                sem1 = self.semantic_matcher.extract_semantic_features(drone_image)
                sem2 = self.semantic_matcher.extract_semantic_features(reference_image)
                results['semantic_similarity'] = self.semantic_matcher.compare_semantic_features(sem1, sem2)
            except Exception as e:
                logger.warning(f"Semantic matching failed: {e}")
        
        # Combine confidences
        weights = {
            'traditional': 0.3,
            'ai': 0.5,
            'semantic': 0.2
        }
        
        combined = (weights['traditional'] * results['traditional_confidence'] +
                   weights['ai'] * results['ai_confidence'] +
                   weights['semantic'] * results['semantic_similarity'])
        
        results['combined_confidence'] = combined
        
        # Cache results
        self.match_cache[cache_key] = results
        
        return results
    
    def find_best_reference_match(self, drone_image: str) -> Tuple[str, Dict]:
        """Find the best matching reference image for a drone image."""
        t0 = time.perf_counter()
        best_match = None
        best_score = 0.0
        best_info = {}
        
        # Candidate filtering by filename index window
        all_refs = list(self.reference_images.keys())
        drone_idx = self._extract_filename_index(drone_image)
        if drone_idx is not None:
            ref_candidates = []
            for ref_image in all_refs:
                ref_idx = self._extract_filename_index(ref_image)
                if ref_idx is not None and abs(ref_idx - drone_idx) <= self.filename_index_window:
                    ref_candidates.append(ref_image)
        else:
            ref_candidates = all_refs

        logger.info(
            f"Reference candidate filtering | drone={Path(drone_image).name} index={drone_idx} "
            f"candidates={len(ref_candidates)}/{len(all_refs)} window=+/-{self.filename_index_window}"
        )

        for ref_image in ref_candidates:
            match_info = self.match_images_enhanced(drone_image, ref_image)
            
            if match_info['combined_confidence'] > best_score:
                best_score = match_info['combined_confidence']
                best_match = ref_image
                best_info = match_info
        
        logger.info(
            f"Best reference selection | drone={Path(drone_image).name} refs={len(ref_candidates)}(filtered from {len(all_refs)}) "
            f"best_ref={Path(best_match).name if best_match else None} score={best_info.get('combined_confidence', 0):.3f} "
            f"elapsed_ms={(time.perf_counter()-t0)*1000:.1f}"
        )
        return best_match, best_info
    
    def estimate_position_enhanced(self, drone_image: str, reference_image: str, 
                                 match_info: Dict) -> Optional[EnhancedPosition]:
        """
        Enhanced position estimation using multiple techniques.
        
        Args:
            drone_image: Path to drone image
            reference_image: Path to reference image
            match_info: Matching information
            
        Returns:
            Enhanced position estimation
        """
        if match_info['combined_confidence'] < 0.3:
            logger.warning(f"Low confidence match ({match_info['combined_confidence']:.3f}) for {drone_image}")
            return None
        
        # Get reference GPS data
        ref_gps = self.reference_images[reference_image]
        
        # Extract traditional features for triangulation
        try:
            kp1, des1 = self.feature_detector.detectAndCompute(cv2.imread(drone_image), None)
            kp2, des2 = self.feature_detector.detectAndCompute(cv2.imread(reference_image), None)
            
            if des1 is not None and des2 is not None:
                matches = self.matcher.knnMatch(des1, des2, k=2)
                
                good_matches = []
                for match_pair in matches:
                    if len(match_pair) == 2:
                        m, n = match_pair
                        if m.distance < 0.75 * n.distance:
                            good_matches.append(m)
                
                if len(good_matches) >= 4:
                    # Extract matched points
                    drone_points = np.array([kp1[m.queryIdx].pt for m in good_matches])
                    ref_points = np.array([kp2[m.trainIdx].pt for m in good_matches])
                    
                    # Calculate homography
                    t_h = time.perf_counter()
                    H, mask = cv2.findHomography(ref_points, drone_points, cv2.RANSAC, 5.0)
                    
                    if H is not None:
                        # Estimate position using enhanced triangulation
                        position = self._triangulate_position_enhanced(H, ref_gps, good_matches, match_info)
                        logger.info(
                            f"Homography + triangulation | drone={Path(drone_image).name} ref={Path(reference_image).name} "
                            f"matches={len(good_matches)} homography_ms={(time.perf_counter()-t_h)*1000:.1f}"
                        )
                        
                        return EnhancedPosition(
                            latitude=position['latitude'],
                            longitude=position['longitude'],
                            altitude=position['altitude'],
                            ai_confidence=match_info['ai_confidence'],
                            traditional_confidence=match_info['traditional_confidence'],
                            combined_confidence=match_info['combined_confidence'],
                            match_count=match_info['traditional_matches'],
                            image_path=drone_image,
                            reference_image=reference_image,
                            semantic_similarity=match_info.get('semantic_similarity', 0.0)
                        )
        
        except Exception as e:
            logger.error(f"Position estimation failed for {drone_image}: {e}")
        
        return None
    
    def _triangulate_position_enhanced(self, homography: np.ndarray, ref_gps: Dict, 
                                     matches: List, match_info: Dict) -> Dict:
        """Enhanced triangulation using multiple techniques."""
        # Calculate center of matched points
        drone_center_x = np.mean([m.queryIdx for m in matches])
        drone_center_y = np.mean([m.queryIdx for m in matches])
        
        # Transform using homography
        center_point = np.array([[drone_center_x, drone_center_y, 1]], dtype=np.float32)
        transformed_point = homography @ center_point.T
        
        # Adaptive scale factor based on confidence
        base_scale = 0.0001
        confidence_factor = match_info['combined_confidence']
        scale_factor = base_scale * confidence_factor
        
        # Estimate position with uncertainty
        estimated_lat = ref_gps['latitude'] + (transformed_point[0] - center_point[0, 0]) * scale_factor
        estimated_lon = ref_gps['longitude'] + (transformed_point[1] - center_point[0, 1]) * scale_factor
        estimated_alt = ref_gps['altitude']
        
        return {
            'latitude': float(estimated_lat),
            'longitude': float(estimated_lon),
            'altitude': float(estimated_alt)
        }
    
    def analyze_flight_path_enhanced(self) -> List[EnhancedPosition]:
        """Analyze flight path using enhanced techniques."""
        logger.info("Starting enhanced flight path analysis...")
        
        flight_path = []
        
        for i, drone_image in enumerate(self.drone_images):
            t_img = time.perf_counter()
            logger.info(
                f"Processing drone image {i+1}/{len(self.drone_images)}: {drone_image} | "
                f"references_available={len(self.reference_images)} ai={self.use_ai} semantic={self.use_semantic}"
            )
            
            # Find best reference match
            t_best = time.perf_counter()
            best_ref, match_info = self.find_best_reference_match(drone_image)
            logger.info(
                f"Best reference found | image={Path(drone_image).name} best_ref={Path(best_ref).name if best_ref else None} "
                f"score={match_info.get('combined_confidence', 0):.3f} elapsed_ms={(time.perf_counter()-t_best)*1000:.1f}"
            )
            
            if best_ref and match_info['combined_confidence'] > 0.3:
                # Estimate position
                t_est = time.perf_counter()
                position = self.estimate_position_enhanced(drone_image, best_ref, match_info)
                
                if position:
                    flight_path.append(position)
                    logger.info(
                        f"Estimated position | image={Path(drone_image).name} lat={position.latitude:.6f} "
                        f"lon={position.longitude:.6f} conf={position.combined_confidence:.3f} "
                        f"est_ms={(time.perf_counter()-t_est)*1000:.1f}"
                    )
                else:
                    logger.warning(f"Could not estimate position for {drone_image}")
            else:
                logger.warning(f"No good matches found for {drone_image}")
            logger.info(
                f"Finished processing image {i+1}/{len(self.drone_images)} | total_ms={(time.perf_counter()-t_img)*1000:.1f}"
            )
        
        self.flight_path = flight_path
        logger.info(f"Enhanced flight path analysis complete. Estimated {len(flight_path)} positions.")
        
        return flight_path
    
    def visualize_enhanced_results(self, output_path: str = "enhanced_flight_path.png", show: bool = True):
        """Visualize enhanced analysis results."""
        if not self.flight_path:
            logger.warning("No flight path data to visualize")
            return
        
        # Extract data
        lats = [pos.latitude for pos in self.flight_path]
        lons = [pos.longitude for pos in self.flight_path]
        ai_confidences = [pos.ai_confidence for pos in self.flight_path]
        traditional_confidences = [pos.traditional_confidence for pos in self.flight_path]
        combined_confidences = [pos.combined_confidence for pos in self.flight_path]
        match_counts = [pos.match_count for pos in self.flight_path]
        
        # Create visualization
        plt.figure(figsize=(16, 12))
        
        # Flight path
        plt.subplot(3, 3, 1)
        plt.plot(lons, lats, 'b-', linewidth=2, label='Flight Path')
        scatter = plt.scatter(lons, lats, c=combined_confidences, cmap='viridis', s=50, alpha=0.7)
        plt.colorbar(scatter, label='Combined Confidence')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.title('Enhanced Drone Flight Path')
        plt.legend()
        plt.grid(True)
        
        # Confidence comparison
        plt.subplot(3, 3, 2)
        x = range(len(combined_confidences))
        plt.plot(x, ai_confidences, 'r-', label='AI Confidence', linewidth=2)
        plt.plot(x, traditional_confidences, 'g-', label='Traditional Confidence', linewidth=2)
        plt.plot(x, combined_confidences, 'b-', label='Combined Confidence', linewidth=2)
        plt.xlabel('Image Index')
        plt.ylabel('Confidence')
        plt.title('Confidence Comparison')
        plt.legend()
        plt.grid(True)
        
        # Match counts
        plt.subplot(3, 3, 3)
        plt.plot(x, match_counts, 'm-', linewidth=2)
        plt.xlabel('Image Index')
        plt.ylabel('Feature Matches')
        plt.title('Traditional Feature Matches')
        plt.grid(True)
        
        # Confidence distribution
        plt.subplot(3, 3, 4)
        plt.hist(combined_confidences, bins=20, alpha=0.7, color='blue')
        plt.xlabel('Combined Confidence')
        plt.ylabel('Frequency')
        plt.title('Confidence Distribution')
        plt.grid(True)
        
        # AI vs Traditional scatter
        plt.subplot(3, 3, 5)
        plt.scatter(traditional_confidences, ai_confidences, alpha=0.6)
        plt.xlabel('Traditional Confidence')
        plt.ylabel('AI Confidence')
        plt.title('AI vs Traditional Confidence')
        plt.grid(True)
        
        # Altitude profile
        altitudes = [pos.altitude for pos in self.flight_path]
        plt.subplot(3, 3, 6)
        plt.plot(x, altitudes, 'g-', linewidth=2)
        plt.xlabel('Image Index')
        plt.ylabel('Altitude (m)')
        plt.title('Drone Altitude Profile')
        plt.grid(True)
        
        # Semantic similarity (if available)
        semantic_similarities = [pos.semantic_similarity for pos in self.flight_path if pos.semantic_similarity is not None]
        if semantic_similarities:
            plt.subplot(3, 3, 7)
            plt.plot(range(len(semantic_similarities)), semantic_similarities, 'c-', linewidth=2)
            plt.xlabel('Image Index')
            plt.ylabel('Semantic Similarity')
            plt.title('Semantic Similarity')
            plt.grid(True)
        
        # Success rate analysis
        plt.subplot(3, 3, 8)
        success_rate = len(self.flight_path) / len(self.drone_images) * 100
        plt.pie([success_rate, 100-success_rate], labels=['Successful', 'Failed'], 
                autopct='%1.1f%%', colors=['lightgreen', 'lightcoral'])
        plt.title('Position Estimation Success Rate')
        
        # Average confidences
        plt.subplot(3, 3, 9)
        avg_ai = np.mean(ai_confidences)
        avg_traditional = np.mean(traditional_confidences)
        avg_combined = np.mean(combined_confidences)
        
        categories = ['AI', 'Traditional', 'Combined']
        values = [avg_ai, avg_traditional, avg_combined]
        colors = ['red', 'green', 'blue']
        
        plt.bar(categories, values, color=colors, alpha=0.7)
        plt.ylabel('Average Confidence')
        plt.title('Average Confidence by Method')
        plt.ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close()
        
        logger.info(f"Enhanced visualization saved to {output_path}")
    
    def save_enhanced_results(self, output_path: str = "enhanced_analysis_results.json"):
        """Save enhanced analysis results."""
        results = {
            'flight_path': [
                {
                    'latitude': pos.latitude,
                    'longitude': pos.longitude,
                    'altitude': pos.altitude,
                    'ai_confidence': pos.ai_confidence,
                    'traditional_confidence': pos.traditional_confidence,
                    'combined_confidence': pos.combined_confidence,
                    'match_count': pos.match_count,
                    'image_path': pos.image_path,
                    'reference_image': pos.reference_image,
                    'semantic_similarity': pos.semantic_similarity
                }
                for pos in self.flight_path
            ],
            'analysis_summary': {
                'total_images_processed': len(self.drone_images),
                'successful_estimations': len(self.flight_path),
                'success_rate': len(self.flight_path) / len(self.drone_images) if self.drone_images else 0,
                'average_ai_confidence': np.mean([pos.ai_confidence for pos in self.flight_path]) if self.flight_path else 0,
                'average_traditional_confidence': np.mean([pos.traditional_confidence for pos in self.flight_path]) if self.flight_path else 0,
                'average_combined_confidence': np.mean([pos.combined_confidence for pos in self.flight_path]) if self.flight_path else 0,
                'ai_enabled': self.use_ai,
                'semantic_enabled': self.use_semantic
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Enhanced analysis results saved to {output_path}")


def main():
    """Main function for enhanced drone analyzer."""
    parser = argparse.ArgumentParser(description='Enhanced drone flight path analysis')
    parser.add_argument('--reference-dir', required=True, help='Directory containing reference images with GPS metadata')
    parser.add_argument('--drone-dir', required=True, help='Directory containing drone images to analyze')
    parser.add_argument('--output-dir', default='enhanced_output', help='Output directory for results')
    parser.add_argument('--use-ai', action='store_true', default=True, help='Use AI-powered matching')
    parser.add_argument('--use-semantic', action='store_true', help='Use semantic segmentation')
    parser.add_argument('--model', default='resnet50', choices=['resnet50', 'vgg16', 'efficientnet'], help='AI model to use')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize enhanced analyzer
    analyzer = EnhancedDroneAnalyzer(
        use_ai=args.use_ai,
        use_semantic=args.use_semantic,
        model_name=args.model
    )
    
    # Load images
    analyzer.load_reference_images(args.reference_dir)
    analyzer.load_drone_images(args.drone_dir)
    
    # Analyze flight path
    flight_path = analyzer.analyze_flight_path_enhanced()
    
    # Save results
    analyzer.save_enhanced_results(os.path.join(args.output_dir, 'enhanced_analysis_results.json'))
    analyzer.visualize_enhanced_results(os.path.join(args.output_dir, 'enhanced_flight_path.png'))
    
    print(f"\nEnhanced analysis complete! Results saved to {args.output_dir}/")
    print(f"Processed {len(analyzer.drone_images)} drone images")
    print(f"Successfully estimated {len(flight_path)} positions")
    if flight_path:
        avg_ai_conf = np.mean([pos.ai_confidence for pos in flight_path])
        avg_trad_conf = np.mean([pos.traditional_confidence for pos in flight_path])
        avg_combined_conf = np.mean([pos.combined_confidence for pos in flight_path])
        print(f"Average AI confidence: {avg_ai_conf:.3f}")
        print(f"Average traditional confidence: {avg_trad_conf:.3f}")
        print(f"Average combined confidence: {avg_combined_conf:.3f}")


if __name__ == "__main__":
    main()

