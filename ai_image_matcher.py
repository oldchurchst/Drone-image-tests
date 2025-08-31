#!/usr/bin/env python3
"""
AI-Powered Image Matcher
========================

Advanced image matching using deep learning techniques for improved
feature detection and matching in drone flight path analysis.
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from typing import List, Tuple, Dict, Optional
import logging
from dataclasses import dataclass
from PIL import Image
import os

logger = logging.getLogger(__name__)

@dataclass
class DeepMatch:
    """Represents a deep learning-based match between two images."""
    confidence: float
    similarity_score: float
    feature_vector1: np.ndarray
    feature_vector2: np.ndarray
    spatial_distance: float

class DeepFeatureExtractor(nn.Module):
    """Deep learning-based feature extractor using pre-trained CNN.

    Dynamically infers the backbone output feature dimension to avoid
    shape mismatches across different model families (ResNet, VGG, EfficientNet).
    """
    
    def __init__(self, model_name: str = 'resnet50', device: torch.device | None = None):
        super(DeepFeatureExtractor, self).__init__()
        self.device = device if device is not None else torch.device('cpu')
        
        # Load pre-trained model (backbone without classifier head)
        if model_name == 'resnet50':
            backbone = models.resnet50(pretrained=True)
            backbone = nn.Sequential(*list(backbone.children())[:-1])  # (B, 2048, 1, 1)
        elif model_name == 'vgg16':
            # For VGG, use features block followed by avgpool to standardize spatial dims
            vgg = models.vgg16(pretrained=True)
            backbone = nn.Sequential(
                vgg.features,
                vgg.avgpool,  # (B, 512, 7, 7) for 224x224 input
            )
        elif model_name == 'efficientnet':
            eff = models.efficientnet_b0(pretrained=True)
            # Use all layers up to, but excluding, the classifier
            backbone = nn.Sequential(*list(eff.children())[:-1])  # (B, 1280, 1, 1)
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        self.backbone = backbone
        
        # Infer the flattened feature dimension with a dummy forward
        self.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            out = self.backbone(dummy)
            out = out.view(out.size(0), -1)
            inferred_feature_dim = out.size(1)
        
        # Add feature projection layer
        self.feature_projection = nn.Linear(inferred_feature_dim, 512)
        self.dropout = nn.Dropout(0.3)
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def forward(self, x):
        """Forward pass through the network."""
        features = self.backbone(x)
        features = features.view(features.size(0), -1)
        features = self.feature_projection(features)
        features = F.relu(features)
        features = self.dropout(features)
        return F.normalize(features, p=2, dim=1)
    
    def extract_features(self, image_path: str) -> np.ndarray:
        """Extract deep features from an image."""
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Extract features
            with torch.no_grad():
                # Ensure module is on the same device as input
                self.to(self.device)
                features = self.forward(image_tensor)
                return features.squeeze().cpu().numpy()
        
        except Exception as e:
            logger.error(f"Error extracting features from {image_path}: {e}")
            return None

class AttentionModule(nn.Module):
    """Attention mechanism for focusing on important image regions."""
    
    def __init__(self, feature_dim=512):
        super(AttentionModule, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 4),
            nn.ReLU(),
            nn.Linear(feature_dim // 4, 1),
            nn.Sigmoid()
        )
    
    def forward(self, features):
        """Apply attention weights to features."""
        attention_weights = self.attention(features)
        attended_features = features * attention_weights
        return attended_features, attention_weights

class AIImageMatcher:
    """AI-powered image matcher using deep learning techniques."""
    
    def __init__(self, model_name='resnet50', use_attention=True):
        """
        Initialize the AI image matcher.
        
        Args:
            model_name: Pre-trained model to use
            use_attention: Whether to use attention mechanism
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Initialize feature extractor
        self.feature_extractor = DeepFeatureExtractor(model_name)
        self.feature_extractor.to(self.device)
        self.feature_extractor.eval()
        
        # Initialize attention module if requested
        self.use_attention = use_attention
        if use_attention:
            self.attention_module = AttentionModule()
            self.attention_module.to(self.device)
            self.attention_module.eval()
        
        # Traditional OpenCV matcher as fallback
        self.opencv_matcher = cv2.SIFT_create()
        self.opencv_matcher_flann = cv2.FlannBasedMatcher(
            dict(algorithm=1, trees=5),
            dict(checks=50)
        )
        
        logger.info("AI Image Matcher initialized successfully")
    
    def extract_deep_features(self, image_path: str) -> Optional[np.ndarray]:
        """Extract deep features from an image."""
        return self.feature_extractor.extract_features(image_path)
    
    def extract_traditional_features(self, image_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Extract traditional OpenCV features as fallback."""
        image = cv2.imread(image_path)
        if image is None:
            return None, None
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.opencv_matcher.detectAndCompute(gray, None)
        return keypoints, descriptors
    
    def compute_similarity(self, features1: np.ndarray, features2: np.ndarray) -> float:
        """Compute cosine similarity between feature vectors."""
        if features1 is None or features2 is None:
            return 0.0
        
        # Normalize features
        features1_norm = features1 / np.linalg.norm(features1)
        features2_norm = features2 / np.linalg.norm(features2)
        
        # Compute cosine similarity
        similarity = np.dot(features1_norm, features2_norm)
        return float(similarity)
    
    def match_images_ai(self, img1_path: str, img2_path: str, 
                       similarity_threshold: float = 0.7) -> DeepMatch:
        """
        Match images using AI techniques.
        
        Args:
            img1_path: Path to first image
            img2_path: Path to second image
            similarity_threshold: Minimum similarity threshold
            
        Returns:
            DeepMatch object with matching information
        """
        # Extract deep features
        features1 = self.extract_deep_features(img1_path)
        features2 = self.extract_deep_features(img2_path)
        
        if features1 is None or features2 is None:
            logger.warning("Could not extract deep features from images")
            return None
        
        # Compute similarity
        similarity = self.compute_similarity(features1, features2)
        
        # Calculate confidence based on similarity
        confidence = max(0.0, min(1.0, similarity))
        
        # Calculate spatial distance (simplified)
        spatial_distance = np.linalg.norm(features1 - features2)
        
        return DeepMatch(
            confidence=confidence,
            similarity_score=similarity,
            feature_vector1=features1,
            feature_vector2=features2,
            spatial_distance=spatial_distance
        )
    
    def match_images_hybrid(self, img1_path: str, img2_path: str,
                           ai_weight: float = 0.7) -> Dict:
        """
        Hybrid matching combining AI and traditional techniques.
        
        Args:
            img1_path: Path to first image
            img2_path: Path to second image
            ai_weight: Weight for AI-based matching (0-1)
            
        Returns:
            Dictionary with combined matching results
        """
        # AI-based matching
        ai_match = self.match_images_ai(img1_path, img2_path)
        
        # Traditional OpenCV matching
        kp1, des1 = self.extract_traditional_features(img1_path)
        kp2, des2 = self.extract_traditional_features(img2_path)
        
        traditional_score = 0.0
        match_count = 0
        
        if des1 is not None and des2 is not None:
            matches = self.opencv_matcher_flann.knnMatch(des1, des2, k=2)
            
            # Apply Lowe's ratio test
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < 0.75 * n.distance:
                        good_matches.append(m)
            
            match_count = len(good_matches)
            traditional_score = min(1.0, match_count / 100.0)  # Normalize
        
        # Combine scores
        if ai_match:
            combined_confidence = (ai_weight * ai_match.confidence + 
                                 (1 - ai_weight) * traditional_score)
        else:
            combined_confidence = traditional_score
        
        return {
            'ai_match': ai_match,
            'traditional_score': traditional_score,
            'match_count': match_count,
            'combined_confidence': combined_confidence,
            'is_match': combined_confidence > 0.5
        }
    
    def find_best_match(self, query_image: str, reference_images: List[str]) -> Tuple[str, Dict]:
        """
        Find the best matching reference image for a query image.
        
        Args:
            query_image: Path to query image
            reference_images: List of reference image paths
            
        Returns:
            Tuple of (best_match_path, match_info)
        """
        best_match = None
        best_score = 0.0
        best_info = {}
        
        for ref_image in reference_images:
            match_info = self.match_images_hybrid(query_image, ref_image)
            
            if match_info['combined_confidence'] > best_score:
                best_score = match_info['combined_confidence']
                best_match = ref_image
                best_info = match_info
        
        return best_match, best_info
    
    def batch_match(self, query_images: List[str], 
                   reference_images: List[str]) -> Dict[str, Dict]:
        """
        Perform batch matching of multiple query images against reference images.
        
        Args:
            query_images: List of query image paths
            reference_images: List of reference image paths
            
        Returns:
            Dictionary mapping query images to their best matches
        """
        results = {}
        
        for i, query_image in enumerate(query_images):
            logger.info(f"Processing query image {i+1}/{len(query_images)}: {query_image}")
            
            best_match, match_info = self.find_best_match(query_image, reference_images)
            results[query_image] = {
                'best_match': best_match,
                'match_info': match_info
            }
        
        return results
    
    def visualize_matches(self, img1_path: str, img2_path: str, 
                         output_path: str = "match_visualization.png"):
        """
        Visualize matches between two images.
        
        Args:
            img1_path: Path to first image
            img2_path: Path to second image
            output_path: Path to save visualization
        """
        import matplotlib.pyplot as plt
        
        # Load images
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)
        
        if img1 is None or img2 is None:
            logger.error("Could not load images for visualization")
            return
        
        # Convert BGR to RGB
        img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        
        # Extract traditional features for visualization
        kp1, des1 = self.extract_traditional_features(img1_path)
        kp2, des2 = self.extract_traditional_features(img2_path)
        
        if des1 is not None and des2 is not None:
            matches = self.opencv_matcher_flann.knnMatch(des1, des2, k=2)
            
            # Apply Lowe's ratio test
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < 0.75 * n.distance:
                        good_matches.append(m)
            
            # Draw matches
            matched_img = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None,
                                        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            matched_img_rgb = cv2.cvtColor(matched_img, cv2.COLOR_BGR2RGB)
            
            # Create visualization
            plt.figure(figsize=(15, 10))
            
            plt.subplot(2, 2, 1)
            plt.imshow(img1_rgb)
            plt.title('Image 1')
            plt.axis('off')
            
            plt.subplot(2, 2, 2)
            plt.imshow(img2_rgb)
            plt.title('Image 2')
            plt.axis('off')
            
            plt.subplot(2, 2, 3)
            plt.imshow(matched_img_rgb)
            plt.title(f'Feature Matches ({len(good_matches)} matches)')
            plt.axis('off')
            
            # AI similarity score
            ai_match = self.match_images_ai(img1_path, img2_path)
            if ai_match:
                plt.subplot(2, 2, 4)
                plt.text(0.1, 0.5, f'AI Similarity: {ai_match.similarity_score:.3f}\n'
                         f'Confidence: {ai_match.confidence:.3f}\n'
                         f'Traditional Matches: {len(good_matches)}',
                         fontsize=12, transform=plt.gca().transAxes)
                plt.title('Matching Statistics')
                plt.axis('off')
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.show()
            
            logger.info(f"Match visualization saved to {output_path}")
        else:
            logger.warning("Could not extract features for visualization")

class SemanticSegmentationMatcher:
    """Advanced matcher using semantic segmentation for better understanding."""
    
    def __init__(self):
        """Initialize semantic segmentation matcher."""
        # Load pre-trained segmentation model
        self.segmentation_model = models.segmentation.fcn_resnet50(pretrained=True)
        self.segmentation_model.eval()
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        logger.info("Semantic Segmentation Matcher initialized")
    
    def extract_semantic_features(self, image_path: str) -> np.ndarray:
        """Extract semantic segmentation features from image."""
        try:
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0)
            
            with torch.no_grad():
                output = self.segmentation_model(image_tensor)['out']
                # Get class probabilities
                probabilities = F.softmax(output, dim=1)
                # Use as feature vector
                features = probabilities.squeeze().cpu().numpy()
                return features
        
        except Exception as e:
            logger.error(f"Error in semantic feature extraction: {e}")
            return None
    
    def compare_semantic_features(self, features1: np.ndarray, 
                                features2: np.ndarray) -> float:
        """Compare semantic features between two images."""
        if features1 is None or features2 is None:
            return 0.0
        
        # Compute correlation between semantic distributions
        correlation = np.corrcoef(features1.flatten(), features2.flatten())[0, 1]
        return max(0.0, correlation)  # Ensure non-negative

