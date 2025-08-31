#!/usr/bin/env python3
"""
Drone Flight Path Analyzer
==========================

This application analyzes a series of drone images to determine the flight path
by comparing them with reference images that contain GPS metadata.
Uses AI and computer vision techniques for image matching and triangulation.
"""

import cv2
import numpy as np
import os
import json
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import logging
from dataclasses import dataclass
from scipy.spatial import Delaunay
from scipy.optimize import least_squares
import pickle
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ImagePoint:
    """Represents a point in an image with its coordinates and metadata."""
    x: float
    y: float
    image_path: str
    gps_lat: Optional[float] = None
    gps_lon: Optional[float] = None
    gps_alt: Optional[float] = None

@dataclass
class MatchPoint:
    """Represents a matched point between two images."""
    point1: ImagePoint
    point2: ImagePoint
    confidence: float

class DroneFlightAnalyzer:
    """Main class for analyzing drone flight paths from image sequences."""
    
    def __init__(self, feature_detector='SIFT', matcher='FLANN'):
        """
        Initialize the analyzer with specified feature detection and matching algorithms.
        
        Args:
            feature_detector: Feature detector to use ('SIFT', 'ORB', 'AKAZE')
            matcher: Matcher to use ('FLANN', 'BF')
        """
        self.feature_detector = self._initialize_detector(feature_detector)
        self.matcher = self._initialize_matcher(matcher)
        self.reference_images = {}
        self.drone_images = []
        self.matches = []
        self.flight_path = []
        self.filename_index_window = 4  # Only compare to references within +/- this index window
        
        logger.info(f"Initialized DroneFlightAnalyzer with {feature_detector} detector and {matcher} matcher")
    
    def _initialize_detector(self, detector_type: str):
        """Initialize the feature detector based on type."""
        if detector_type == 'SIFT':
            return cv2.SIFT_create()
        elif detector_type == 'ORB':
            return cv2.ORB_create()
        elif detector_type == 'AKAZE':
            return cv2.AKAZE_create()
        else:
            raise ValueError(f"Unsupported detector type: {detector_type}")
    
    def _initialize_matcher(self, matcher_type: str):
        """Initialize the feature matcher based on type."""
        if matcher_type == 'FLANN':
            if self.feature_detector.getDefaultName() == 'Feature2D.SIFT':
                index_params = dict(algorithm=1, trees=5)
            else:
                index_params = dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1)
            search_params = dict(checks=50)
            return cv2.FlannBasedMatcher(index_params, search_params)
        elif matcher_type == 'BF':
            if self.feature_detector.getDefaultName() == 'Feature2D.SIFT':
                return cv2.BFMatcher()
            else:
                return cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        else:
            raise ValueError(f"Unsupported matcher type: {matcher_type}")
    
    def load_reference_images(self, reference_dir: str):
        """
        Load reference images with GPS metadata.
        
        Args:
            reference_dir: Directory containing reference images with GPS data
        """
        logger.info(f"Loading reference images from {reference_dir}")
        
        for image_path in Path(reference_dir).glob("*.jpg"):
            gps_data = self._extract_gps_metadata(str(image_path))
            if gps_data:
                # Store GPS data; index filtering is computed on the fly per request
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
            # Use the last numeric group as index (common for camera naming)
            return int(numbers[-1])
        except Exception:
            return None
    
    def load_drone_images(self, drone_dir: str):
        """
        Load drone images for analysis.
        
        Args:
            drone_dir: Directory containing drone images
        """
        logger.info(f"Loading drone images from {drone_dir}")
        
        image_extensions = ['.jpg', '.jpeg', '.png', '.tiff', '.bmp']
        for ext in image_extensions:
            for image_path in Path(drone_dir).glob(f"*{ext}"):
                self.drone_images.append(str(image_path))
        
        # Sort images by filename to maintain temporal order
        self.drone_images.sort()
        logger.info(f"Loaded {len(self.drone_images)} drone images")
    
    def _extract_gps_metadata(self, image_path: str) -> Optional[Dict]:
        """
        Extract GPS metadata from image EXIF data.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary containing GPS coordinates or None if not found
        """
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
    
    def extract_features(self, image_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract features from an image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Tuple of (keypoints, descriptors)
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        start = time.perf_counter()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.feature_detector.detectAndCompute(gray, None)
        logger.debug(f"Feature extraction for {image_path}: keypoints={len(keypoints) if keypoints else 0}, "
                     f"descriptors_shape={(descriptors.shape if descriptors is not None else None)}, "
                     f"elapsed_ms={(time.perf_counter()-start)*1000:.1f}")
        
        return keypoints, descriptors
    
    def match_images(self, img1_path: str, img2_path: str, ratio_threshold: float = 0.75) -> List[MatchPoint]:
        """
        Match features between two images.
        
        Args:
            img1_path: Path to first image
            img2_path: Path to second image
            ratio_threshold: Threshold for Lowe's ratio test
            
        Returns:
            List of matched points
        """
        # Extract features from both images
        t0 = time.perf_counter()
        kp1, des1 = self.extract_features(img1_path)
        kp2, des2 = self.extract_features(img2_path)
        
        if des1 is None or des2 is None:
            return []
        
        # Match features
        t_match_start = time.perf_counter()
        matches = self.matcher.knnMatch(des1, des2, k=2)
        
        # Apply Lowe's ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < ratio_threshold * n.distance:
                    good_matches.append(m)
        
        # Convert to MatchPoint objects
        match_points = []
        for match in good_matches:
            point1 = ImagePoint(
                x=kp1[match.queryIdx].pt[0],
                y=kp1[match.queryIdx].pt[1],
                image_path=img1_path
            )
            
            point2 = ImagePoint(
                x=kp2[match.trainIdx].pt[0],
                y=kp2[match.trainIdx].pt[1],
                image_path=img2_path
            )
            
            # Calculate confidence based on match distance
            confidence = 1.0 - (match.distance / 1000.0)  # Normalize distance
            confidence = max(0.0, min(1.0, confidence))
            
            match_points.append(MatchPoint(point1, point2, confidence))
        
        elapsed_total = (time.perf_counter() - t0) * 1000
        elapsed_match = (time.perf_counter() - t_match_start) * 1000
        logger.info(
            f"Matched images | img1={Path(img1_path).name} img2={Path(img2_path).name} "
            f"kp1={len(kp1) if kp1 else 0} kp2={len(kp2) if kp2 else 0} "
            f"good_matches={len(good_matches)} ratio={ratio_threshold} "
            f"extract+match_ms={elapsed_total:.1f} (match_ms={elapsed_match:.1f})"
        )
        return match_points
    
    def find_best_reference_match(self, drone_image_path: str) -> Tuple[str, List[MatchPoint]]:
        """
        Find the best matching reference image for a drone image.
        
        Args:
            drone_image_path: Path to the drone image
            
        Returns:
            Tuple of (best_reference_path, match_points)
        """
        t0 = time.perf_counter()
        best_match_count = 0
        best_reference = None
        best_matches = []
        
        # Build candidate list limited by filename index window
        ref_list_all = list(self.reference_images.keys())
        drone_idx = self._extract_filename_index(drone_image_path)
        if drone_idx is not None:
            candidates = []
            for ref_path in ref_list_all:
                ref_idx = self._extract_filename_index(ref_path)
                if ref_idx is not None and abs(ref_idx - drone_idx) <= self.filename_index_window:
                    candidates.append(ref_path)
        else:
            candidates = ref_list_all

        logger.info(
            f"Reference candidate filtering | drone={Path(drone_image_path).name} index={drone_idx} "
            f"candidates={len(candidates)}/{len(ref_list_all)} window=+/-{self.filename_index_window}"
        )

        ref_list = candidates
        for ref_path in ref_list:
            matches = self.match_images(drone_image_path, ref_path)
            
            if len(matches) > best_match_count:
                best_match_count = len(matches)
                best_reference = ref_path
                best_matches = matches
        
        logger.info(
            f"Best reference selected | drone={Path(drone_image_path).name} "
            f"refs_tried={len(ref_list)} best_ref={Path(best_reference).name if best_reference else None} "
            f"best_matches={best_match_count} elapsed_ms={(time.perf_counter()-t0)*1000:.1f}"
        )
        return best_reference, best_matches
    
    def estimate_position(self, drone_image_path: str, reference_path: str, matches: List[MatchPoint]) -> Optional[Dict]:
        """
        Estimate the position of the drone based on image matches.
        
        Args:
            drone_image_path: Path to the drone image
            reference_path: Path to the reference image
            matches: List of matched points
            
        Returns:
            Estimated position dictionary or None if estimation fails
        """
        if len(matches) < 4:
            logger.warning(f"Insufficient matches ({len(matches)}) for position estimation")
            return None
        
        # Get reference GPS data
        ref_gps = self.reference_images[reference_path]
        
        # Extract matched points
        drone_points = np.array([[m.point1.x, m.point1.y] for m in matches])
        ref_points = np.array([[m.point2.x, m.point2.y] for m in matches])
        
        # Calculate homography matrix
        t_h = time.perf_counter()
        H, mask = cv2.findHomography(ref_points, drone_points, cv2.RANSAC, 5.0)
        
        if H is None:
            logger.warning("Could not compute homography matrix")
            return None
        
        # Estimate position using triangulation
        # This is a simplified approach - in practice, you'd need more sophisticated 3D reconstruction
        estimated_position = self._triangulate_position(H, ref_gps, matches)
        logger.info(
            f"Position estimation | ref=({ref_gps.get('latitude'):.6f},{ref_gps.get('longitude'):.6f}) "
            f"matches_used={len(matches)} homography_ms={(time.perf_counter()-t_h)*1000:.1f}"
        )
        
        return estimated_position
    
    def _triangulate_position(self, homography: np.ndarray, ref_gps: Dict, matches: List[MatchPoint]) -> Dict:
        """
        Triangulate drone position using homography and reference GPS.
        
        Args:
            homography: Homography matrix
            ref_gps: Reference GPS coordinates
            matches: Matched points
            
        Returns:
            Estimated position dictionary
        """
        # This is a simplified triangulation approach
        # In a real implementation, you'd use proper 3D reconstruction techniques
        
        # Calculate the center of matched points in drone image
        drone_center_x = np.mean([m.point1.x for m in matches])
        drone_center_y = np.mean([m.point1.y for m in matches])
        
        # Transform center point using homography
        center_point = np.array([[drone_center_x, drone_center_y, 1]], dtype=np.float32)
        transformed_point = homography @ center_point.T
        
        # Estimate relative position based on transformation
        # This is a heuristic approach - real implementation would use proper 3D geometry
        scale_factor = 0.0001  # Adjust based on your specific use case
        
        estimated_lat = ref_gps['latitude'] + (transformed_point[0] - center_point[0, 0]) * scale_factor
        estimated_lon = ref_gps['longitude'] + (transformed_point[1] - center_point[0, 1]) * scale_factor
        estimated_alt = ref_gps['altitude']  # Assume similar altitude for now
        
        return {
            'latitude': float(estimated_lat),
            'longitude': float(estimated_lon),
            'altitude': float(estimated_alt),
            'confidence': np.mean([m.confidence for m in matches])
        }
    
    def analyze_flight_path(self) -> List[Dict]:
        """
        Analyze the complete flight path from drone images.
        
        Returns:
            List of estimated positions for each drone image
        """
        logger.info("Starting flight path analysis...")
        
        flight_path = []
        
        for i, drone_image in enumerate(self.drone_images):
            t_img = time.perf_counter()
            logger.info(
                f"Processing drone image {i+1}/{len(self.drone_images)}: {drone_image} | "
                f"references_available={len(self.reference_images)}"
            )
            
            # Find best reference match
            t_best = time.perf_counter()
            best_ref, matches = self.find_best_reference_match(drone_image)
            logger.info(
                f"Best reference found | image={Path(drone_image).name} best_ref={Path(best_ref).name if best_ref else None} "
                f"match_count={len(matches) if matches else 0} elapsed_ms={(time.perf_counter()-t_best)*1000:.1f}"
            )
            
            if best_ref and matches:
                # Estimate position
                t_est = time.perf_counter()
                position = self.estimate_position(drone_image, best_ref, matches)
                
                if position:
                    position['image_path'] = drone_image
                    position['reference_image'] = best_ref
                    position['match_count'] = len(matches)
                    flight_path.append(position)
                    
                    logger.info(
                        f"Estimated position computed | image={Path(drone_image).name} "
                        f"lat={position['latitude']:.6f} lon={position['longitude']:.6f} "
                        f"confidence={position['confidence']:.3f} est_ms={(time.perf_counter()-t_est)*1000:.1f}"
                    )
                else:
                    logger.warning(f"Could not estimate position for {drone_image}")
            else:
                logger.warning(f"No good matches found for {drone_image}")

            logger.info(
                f"Finished processing image {i+1}/{len(self.drone_images)} | total_ms={(time.perf_counter()-t_img)*1000:.1f}"
            )
        
        self.flight_path = flight_path
        logger.info(f"Flight path analysis complete. Estimated {len(flight_path)} positions.")
        
        return flight_path
    
    def visualize_flight_path(self, output_path: str = "flight_path.png", show: bool = True):
        """
        Visualize the estimated flight path.
        
        Args:
            output_path: Path to save the visualization
            show: Whether to display the plot window (use False when called from GUI threads)
        """
        if not self.flight_path:
            logger.warning("No flight path data to visualize")
            return
        
        # Extract coordinates
        lats = [pos['latitude'] for pos in self.flight_path]
        lons = [pos['longitude'] for pos in self.flight_path]
        confidences = [pos['confidence'] for pos in self.flight_path]
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        
        # Plot flight path
        plt.subplot(2, 2, 1)
        plt.plot(lons, lats, 'b-', linewidth=2, label='Flight Path')
        plt.scatter(lons, lats, c=confidences, cmap='viridis', s=50, alpha=0.7)
        plt.colorbar(label='Confidence')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.title('Drone Flight Path')
        plt.legend()
        plt.grid(True)
        
        # Plot confidence over time
        plt.subplot(2, 2, 2)
        plt.plot(range(len(confidences)), confidences, 'r-', linewidth=2)
        plt.xlabel('Image Index')
        plt.ylabel('Confidence')
        plt.title('Position Estimation Confidence')
        plt.grid(True)
        
        # Plot altitudes
        altitudes = [pos['altitude'] for pos in self.flight_path]
        plt.subplot(2, 2, 3)
        plt.plot(range(len(altitudes)), altitudes, 'g-', linewidth=2)
        plt.xlabel('Image Index')
        plt.ylabel('Altitude (m)')
        plt.title('Drone Altitude')
        plt.grid(True)
        
        # Plot match counts
        match_counts = [pos['match_count'] for pos in self.flight_path]
        plt.subplot(2, 2, 4)
        plt.plot(range(len(match_counts)), match_counts, 'm-', linewidth=2)
        plt.xlabel('Image Index')
        plt.ylabel('Feature Matches')
        plt.title('Feature Match Count')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close()
        
        logger.info(f"Flight path visualization saved to {output_path}")
    
    def save_results(self, output_path: str = "flight_analysis_results.json"):
        """
        Save analysis results to JSON file.
        
        Args:
            output_path: Path to save the results
        """
        results = {
            'flight_path': self.flight_path,
            'analysis_summary': {
                'total_images_processed': len(self.drone_images),
                'successful_estimations': len(self.flight_path),
                'success_rate': len(self.flight_path) / len(self.drone_images) if self.drone_images else 0,
                'average_confidence': np.mean([pos['confidence'] for pos in self.flight_path]) if self.flight_path else 0
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Analysis results saved to {output_path}")
    
    def export_gpx(self, output_path: str = "drone_flight_path.gpx"):
        """
        Export flight path as GPX file for use in mapping applications.
        
        Args:
            output_path: Path to save the GPX file
        """
        if not self.flight_path:
            logger.warning("No flight path data to export")
            return
        
        gpx_content = """<?xml version="1.0" encoding="UTF-8"?>
<gpx version="1.1" creator="DroneFlightAnalyzer" xmlns="http://www.topografix.com/GPX/1/1">
  <trk>
    <name>Drone Flight Path</name>
    <trkseg>
"""
        
        for position in self.flight_path:
            gpx_content += f"""      <trkpt lat="{position['latitude']}" lon="{position['longitude']}">
        <ele>{position['altitude']}</ele>
        <extensions>
          <confidence>{position['confidence']}</confidence>
          <match_count>{position['match_count']}</match_count>
        </extensions>
      </trkpt>
"""
        
        gpx_content += """    </trkseg>
  </trk>
</gpx>"""
        
        with open(output_path, 'w') as f:
            f.write(gpx_content)
        
        logger.info(f"GPX file exported to {output_path}")


def main():
    """Main function to run the drone flight analyzer."""
    parser = argparse.ArgumentParser(description='Analyze drone flight path from images')
    parser.add_argument('--reference-dir', required=True, help='Directory containing reference images with GPS metadata')
    parser.add_argument('--drone-dir', required=True, help='Directory containing drone images to analyze')
    parser.add_argument('--output-dir', default='output', help='Output directory for results')
    parser.add_argument('--detector', default='SIFT', choices=['SIFT', 'ORB', 'AKAZE'], help='Feature detector to use')
    parser.add_argument('--matcher', default='FLANN', choices=['FLANN', 'BF'], help='Feature matcher to use')
    parser.add_argument('--ratio-threshold', type=float, default=0.75, help='Lowe\'s ratio test threshold')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize analyzer
    analyzer = DroneFlightAnalyzer(detector=args.detector, matcher=args.matcher)
    
    # Load images
    analyzer.load_reference_images(args.reference_dir)
    analyzer.load_drone_images(args.drone_dir)
    
    # Analyze flight path
    flight_path = analyzer.analyze_flight_path()
    
    # Save results
    analyzer.save_results(os.path.join(args.output_dir, 'flight_analysis_results.json'))
    analyzer.export_gpx(os.path.join(args.output_dir, 'drone_flight_path.gpx'))
    analyzer.visualize_flight_path(os.path.join(args.output_dir, 'flight_path.png'))
    
    print(f"\nAnalysis complete! Results saved to {args.output_dir}/")
    print(f"Processed {len(analyzer.drone_images)} drone images")
    print(f"Successfully estimated {len(flight_path)} positions")
    if flight_path:
        avg_confidence = np.mean([pos['confidence'] for pos in flight_path])
        print(f"Average confidence: {avg_confidence:.3f}")


if __name__ == "__main__":
    main()

