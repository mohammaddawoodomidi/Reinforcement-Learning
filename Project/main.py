import os
import cv2
from sklearn.metrics import average_precision_score, precision_recall_curve
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import torch
import torch.nn as nn
from torch.optim import Adam
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.evaluation import evaluate_policy
from ultralytics import YOLO
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
import pandas as pd
import logging
import yaml
from stable_baselines3.common.callbacks import BaseCallback
import gc  


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Constants
TARGET_CLASS = "chair"  # Same as in your notebook
IMAGE_SIZE = 1280
MAX_STEPS = 30

class ValidationCallback(BaseCallback):
    # Replace the ValidationCallback __init__ method around line 54
    def __init__(self, eval_env, eval_freq=500, n_eval_episodes=10, verbose=1):
        super(ValidationCallback, self).__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq  # Consider reducing this for smaller datasets
        self.n_eval_episodes = n_eval_episodes
        self.best_mean_reward = -np.inf
        self.no_improvement_count = 0
        self.patience = 3  # Lower patience for small datasets
            
    def _on_step(self):
        if self.n_calls % self.eval_freq == 0:
            # Evaluate the agent
            mean_reward, _ = evaluate_policy(
                self.model, 
                self.eval_env, 
                n_eval_episodes=self.n_eval_episodes
            )
            
            self.logger.record("eval/mean_reward", mean_reward)
            
            # Check if performance improved
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                self.no_improvement_count = 0
                # Save best model
                self.model.save("best_rl_occlusion_model")
            else:
                self.no_improvement_count += 1
                
            # Early stopping
            if self.no_improvement_count >= 5:  # Stop after 5 evaluations without improvement
                if self.verbose > 0:
                    print("Stopping training early due to no improvement in validation")
                return False
                
        return True

# Callback for learning rate scheduling
class LRSchedulerCallback(BaseCallback):
    def __init__(self, initial_lr=3e-4, min_lr=1e-5, decay_factor=0.5, decay_steps=1000):
        super(LRSchedulerCallback, self).__init__()
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.decay_factor = decay_factor
        self.decay_steps = decay_steps
        
    def _on_step(self):
        if self.n_calls % self.decay_steps == 0:
            new_lr = max(self.initial_lr * (self.decay_factor ** (self.n_calls // self.decay_steps)), self.min_lr)
            self.model.learning_rate = new_lr
            if self.verbose > 0:
                print(f"Updating learning rate to {new_lr}")
        return True


# This environment is designed to optimize the viewpoint of an object in a 2D space
# to reduce occlusion using reinforcement learning.
# It uses a YOLO model to detect objects and calculate occlusion scores.
class OcclusionEnvironment(gym.Env):
    """
    A 2D environment for optimizing viewpoints to reduce occlusion
    """
    def __init__(self, yolo_model, images_dir, labels_dir=None, target_class=TARGET_CLASS):
        super().__init__()
        self.yolo_model = yolo_model
        self.target_class = target_class
        self.evaluation_mode = False  # to track if in evaluation mode
        
        # Define observation size for more efficient RL
        self.observation_size = 512  # More efficient size for RL

        # to confirm which model is being used:
        if hasattr(yolo_model, 'model') and hasattr(yolo_model.model, 'names'):
            logging.info(f"YOLO model loaded with classes: {yolo_model.model.names}")
        else:
            logging.info("Loaded YOLO model, but could not display class names")
        
        # Load images
        self.images_dir = images_dir
        self.image_paths = [os.path.join(images_dir, f) for f in os.listdir(images_dir) 
                        if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        assert len(self.image_paths) > 0, f"No images found in {images_dir}"
        logging.info(f"Loaded {len(self.image_paths)} images")
        
        # Set labels directory
        self.labels_dir = labels_dir
        logging.info(f"Using provided labels directory: {self.labels_dir}")
        
        # Define action space: [translate_x, translate_y, rotation, zoom]
        # All values normalized between -1 and 1
        self.action_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
        
        # Define observation space: RGB image + transformation parameters
        self.observation_space = spaces.Dict({
            'image': spaces.Box(low=0, high=255, shape=(self.observation_size, self.observation_size, 3), dtype=np.uint8),
            'transforms': spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)  # [translate_x, translate_y, rotate, zoom]
        })
        
        # Maximum transformation values
        self.max_translate = 0.2  # 20% of image size
        self.max_rotate = 30  # degrees
        self.max_zoom = 0.3  # 30% zoom in/out
        
        # Initialize variables that will be set in reset()
        self.current_image = None
        self.original_image = None
        self.current_step = 0
        self.previous_detection_score = 0
        self.previous_occlusion_score = 0
        self.previous_map_score = 0  # Added for mAP tracking
        self.best_image = None
        self.best_score = 0
        self.transformations = []
        self.current_ground_truth = None  # Added to store current image ground truth
        self.current_transforms = np.zeros(4, dtype=np.float32)  # Initialize transforms
        
        # Store class index
        with open("Augmented_Dataset/data.yaml", 'r') as file:
            data_yaml = yaml.safe_load(file)
        class_names = data_yaml['names']
        self.target_class_index = class_names.index(self.target_class)

        
    def reset(self, seed=None, options=None):
        """Reset the environment with optional seed and options parameters"""
        # Set seed if provided
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Load image on demand rather than keeping all in memory
        image_path = None
        if options and "image" in options:
            self.original_image = options["image"].copy()
            if "image_path" in options:
                image_path = options["image_path"]
        else:
            # Select a random image path
            image_path = random.choice(self.image_paths)
            # Only load when needed
            self.original_image = cv2.imread(image_path)
            self.original_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
            self.original_image = cv2.resize(self.original_image, (IMAGE_SIZE, IMAGE_SIZE))
        
        # Load ground truth annotations if path is available
        self.current_ground_truth = self._load_ground_truth_annotations(image_path)
        if self.current_ground_truth:
            logging.debug(f"Loaded {len(self.current_ground_truth)} ground truth annotations for {image_path}")
        else:
            logging.debug(f"No ground truth annotations found for {image_path}")
        
        # Make a copy of the original image
        self.current_image = self.original_image.copy()
        
        # Reset episode state
        self.current_step = 0
        self.transformations = []
        
        # Get initial detection score
        results = self.yolo_model(self.current_image)
        self.previous_detection_score = self._calculate_detection_score(results)
        self.previous_occlusion_score = self._calculate_occlusion_score(results)
        self.previous_map_score = self._calculate_map(results, self.current_ground_truth)
        
        self.best_image = self.current_image.copy()
        self.best_score = self.previous_map_score if self.current_ground_truth else self.previous_detection_score
        
        observation = cv2.resize(self.current_image, (self.observation_size, self.observation_size))
        # In the reset method around line 212, replace the return statement:
        # Initialize transformation parameters to zeros (no transformation)
        self.current_transforms = np.zeros(4, dtype=np.float32)

        # Return dict observation
        return {
            'image': observation,
            'transforms': self.current_transforms
        }, {}  # Adding empty info dictionary
        
    # In the step method around line 350, add tracking of transforms and modify return:
    def step(self, action):
        self.current_step += 1
        
        # Store normalized action as current transformation state
        self.current_transforms = action.copy()
        
        # Apply transformations to the image
        transformed_image = self._apply_transformations(self.original_image, action)
        self.current_image = transformed_image
        
        # Get detection results
        results = self.yolo_model(self.current_image)
        
        # Calculate metrics
        detection_score = self._calculate_detection_score(results)
        occlusion_score = self._calculate_occlusion_score(results)
        map_score = self._calculate_map(results, self.current_ground_truth)
        
        # Calculate reward
        reward = self._calculate_reward(detection_score, occlusion_score, map_score)
        
        # Update best image if better score
        if self.current_ground_truth and map_score > self.best_score:
            self.best_score = map_score
            self.best_image = self.current_image.copy()
        elif not self.current_ground_truth and detection_score > self.best_score:
            self.best_score = detection_score
            self.best_image = self.current_image.copy()
                
        # Store current scores
        self.previous_detection_score = detection_score
        self.previous_occlusion_score = occlusion_score
        self.previous_map_score = map_score
        
        # Check if episode is done
        terminated = self.current_step >= MAX_STEPS
        truncated = False  # No time limit truncation in this environment
        
        info = {
            'detection_score': detection_score,
            'occlusion_score': occlusion_score,
            'map_score': map_score,
            'step': self.current_step
        }
        
        observation = cv2.resize(self.current_image, (self.observation_size, self.observation_size))
        return {
            'image': observation,
            'transforms': self.current_transforms
        }, reward, terminated, truncated, info

        
    def _apply_transformations(self, image, action):
        # Unpack actions
        translate_x, translate_y, rotate, zoom = action
        
        # Scale actions to actual values
        translate_x = translate_x * self.max_translate * IMAGE_SIZE
        translate_y = translate_y * self.max_translate * IMAGE_SIZE
        rotate = rotate * self.max_rotate
        zoom = 1 + (zoom * self.max_zoom)
        
        # If first step, initialize transformation matrices
        if len(self.transformations) == 0:
            self.cumulative_translate = np.float32([[1, 0, 0], [0, 1, 0]])
            self.cumulative_rotate = cv2.getRotationMatrix2D((IMAGE_SIZE/2, IMAGE_SIZE/2), 0, 1)
        
        # Update cumulative transformations
        self.cumulative_translate[0, 2] += translate_x
        self.cumulative_translate[1, 2] += translate_y
        
        # For rotation and zoom, create new matrix and combine with previous
        new_rotate = cv2.getRotationMatrix2D((IMAGE_SIZE/2, IMAGE_SIZE/2), rotate, zoom)
        
        # Combine rotation matrices (simplified approach)
        self.cumulative_rotate = new_rotate.dot(np.vstack([self.cumulative_rotate, [0, 0, 1]]))[:2]
        
        # Store current transformation
        self.transformations.append({
            'translate_x': translate_x,
            'translate_y': translate_y,
            'rotate': rotate,
            'zoom': zoom
        })
        
        # Apply transformations
        transformed = cv2.warpAffine(image, self.cumulative_translate, (IMAGE_SIZE, IMAGE_SIZE))
        transformed = cv2.warpAffine(transformed, self.cumulative_rotate, (IMAGE_SIZE, IMAGE_SIZE))
        
        return transformed
    
    def _calculate_detection_score(self, results):
        """Calculate detection quality score based on confidence and count"""
        # Extract boxes related to target class
        boxes = results[0].boxes
        if len(boxes) == 0:
            return 0
            
        # Filter boxes for target class
        target_boxes = [box for box in boxes if int(box.cls[0]) == self.target_class_index]
        if not target_boxes:
            return 0
            
        # Calculate score based on confidence and count
        confidences = [float(box.conf[0]) for box in target_boxes]
        avg_confidence = sum(confidences) / len(confidences)
        
        # Penalize too many detections (likely false positives)
        count_factor = min(len(target_boxes), 5) / 5  # Cap at 5 objects
        
        return avg_confidence * count_factor
    
    def _calculate_occlusion_score(self, results):
        """
        Calculate occlusion score based on overlap of bounding boxes
        Higher score means more occlusion
        """
        boxes = results[0].boxes
        if len(boxes) < 2:  # Need at least 2 boxes for occlusion
            return 0
            
        # Calculate IoU (Intersection over Union) between all box pairs
        box_ious = []
        boxes_xyxy = boxes.xyxy.cpu().numpy()
        
        for i in range(len(boxes_xyxy)):
            for j in range(i+1, len(boxes_xyxy)):
                iou = self._calculate_iou(boxes_xyxy[i], boxes_xyxy[j])
                box_ious.append(iou)
        
        # Return average IoU if any, else 0
        return sum(box_ious) / len(box_ious) if box_ious else 0
    
    def _calculate_iou(self, box1, box2):
        """Calculate IoU between two boxes"""
        # Calculate intersection area
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        
        # Calculate union area
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = box1_area + box2_area - intersection
        
        # Return IoU
        return intersection / union if union > 0 else 0
    
    def _calculate_map(self, results, ground_truth=None):
        """Calculate mean Average Precision for detections"""
        # Get the boxes for target class
        boxes = results[0].boxes
        target_boxes = [box for box in boxes if int(box.cls[0]) == self.target_class_index]
        
        if not target_boxes:
            return 0.0
        
        if ground_truth is None or len(ground_truth) == 0:
            # If no ground truth, return 0 to differentiate from confidence score
            # This change will make it clear when we're falling back to confidence-based eval
            return 0.0  # Changed from confidence calculation
        
        # With ground truth, calculate precision at different IoU thresholds
        iou_thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        aps = []
        
        for threshold in iou_thresholds:
            # Convert detections to format [confidence, x1, y1, x2, y2]
            detections = []
            for box in target_boxes:
                conf = float(box.conf[0])
                xyxy = box.xyxy.cpu().numpy()[0]
                detections.append([conf, *xyxy])
            
            # Sort by confidence (highest first)
            detections.sort(reverse=True)
            
            # Calculate precision and recall
            tp = np.zeros(len(detections))
            fp = np.zeros(len(detections))
            matched_gt = set()
            
            for i, det in enumerate(detections):
                best_iou = 0
                best_gt_idx = -1
                
                # Find best matching ground truth
                for gt_idx, gt_box in enumerate(ground_truth):
                    if gt_idx in matched_gt:
                        continue
                    
                    # The detected box is [conf, x1, y1, x2, y2] but we only need [x1, y1, x2, y2]
                    # The ground truth box is already in [x1, y1, x2, y2] format
                    iou = self._calculate_iou(det[1:], gt_box)
                    
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx
                
                # If IoU above threshold, count as true positive
                if best_iou >= threshold and best_gt_idx not in matched_gt:
                    tp[i] = 1
                    matched_gt.add(best_gt_idx)
                else:
                    fp[i] = 1
            
            # Calculate cumulative precision and recall
            cum_tp = np.cumsum(tp)
            cum_fp = np.cumsum(fp)
            
            precision = cum_tp / (cum_tp + cum_fp)
            recall = cum_tp / max(len(ground_truth), 1)
            
            # Calculate AP using 11-point interpolation (as per Pascal VOC)
            ap = 0
            for r in np.arange(0, 1.1, 0.1):
                if len(precision) == 0:
                    p_interp = 0
                else:
                    p_interp = np.max(precision[recall >= r]) if np.any(recall >= r) else 0
                ap += p_interp / 11
            
            aps.append(ap)
        
        # Return mean of APs (mAP)
        return sum(aps) / len(aps)

    
    def _calculate_reward(self, detection_score, occlusion_score, map_score=None):
        """Calculate reward with adaptive weighting"""
        
        # mAP improvement (primary metric with ground truth)
        map_improvement = map_score - self.previous_map_score
            
        # Occlusion reduction (negative is good)
        occlusion_reduction = self.previous_occlusion_score - occlusion_score
            
        # Calculate adaptive weights - emphasize what needs improvement
        map_weight = 2.0 + max(0, 1.0 - map_score)  # Higher weight when mAP is low
        occlusion_weight = 0.5 + min(1.5, occlusion_score)  # Higher weight when occlusion is high
            
        # Calculate reward with adaptive weights
        reward = map_weight * map_improvement + occlusion_weight * occlusion_reduction
            
        # Additional reward for significant improvements
        if map_improvement > 0.05:
            reward += 1.0
                
        # Penalty for no improvement to encourage exploration
        if abs(map_improvement) < 0.01 and abs(occlusion_reduction) < 0.01:
            reward -= 0.1
                
        return reward
    
    def _normalize_rewards(self, reward):
        """Normalize rewards to a reasonable range for stable training"""
        # Clip rewards to avoid extreme values
        reward = np.clip(reward, -5.0, 5.0)
        return reward
    
    # Function to load ground truth annotations from YOLO format
    def _load_ground_truth_annotations(self, image_path):
        """
        Load ground truth annotations for a given image from YOLO format labels
        Returns a list of bounding boxes in [x1, y1, x2, y2] format
        """
        # Convert image path to corresponding label path
        image_name = os.path.basename(image_path)
        base_name = os.path.splitext(image_name)[0]
        label_path = os.path.join(self.labels_dir, f"{base_name}.txt")
        
        if not os.path.exists(label_path):
            return []
        
        ground_truth = []
        img_height, img_width = IMAGE_SIZE, IMAGE_SIZE  # Assuming square images
        
        # Read annotations
        with open(label_path, 'r') as file:
            for line in file:
                parts = line.strip().split()
                if len(parts) >= 5:  # class, x_center, y_center, width, height
                    class_id = int(parts[0])
                    
                    # Only include the target class
                    if class_id == self.target_class_index:
                        x_center = float(parts[1]) * img_width
                        y_center = float(parts[2]) * img_height
                        width = float(parts[3]) * img_width
                        height = float(parts[4]) * img_height
                        
                        # Convert to xyxy format
                        x1 = x_center - width / 2
                        y1 = y_center - height / 2
                        x2 = x_center + width / 2
                        y2 = y_center + height / 2
                        
                        # No need to include class_id in the boxes
                        ground_truth.append([x1, y1, x2, y2])
        
        return ground_truth
        
    def render(self, mode='human'):
        if mode == 'human':
            plt.figure(figsize=(10, 5))
            
            plt.subplot(1, 2, 1)
            plt.title("Original Image")
            plt.imshow(self.original_image)
            plt.axis('off')
            
            plt.subplot(1, 2, 2)
            plt.title(f"Transformed (Step {self.current_step})")
            plt.imshow(self.current_image)
            plt.axis('off')
            
            plt.tight_layout()
            plt.show()
            
    def get_best_image(self):
        """Return the best image found during the episode"""
        return self.best_image


# Feature extractor for the combined image and transformation parameters
class CombinedFeaturesExtractor(BaseFeaturesExtractor):
    """
    Feature extractor that handles both image and transformation parameters
    """
    def __init__(self, observation_space, features_dim=256):
        # We need to initialize with the complete observation space
        super(CombinedFeaturesExtractor, self).__init__(observation_space, features_dim)
        
        # Extract the image and transform spaces
        self.image_space = observation_space.spaces['image']
        self.transform_space = observation_space.spaces['transforms']
        
        # Get observation size from the image space shape
        self.observation_size = self.image_space.shape[0]
        
        # Add padding so that kernel sizes never exceed the (possibly small) input spatial dims
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(4, 16),
            nn.ReLU(),
            nn.Dropout2d(0.2),

            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 32),
            nn.ReLU(),
            nn.Dropout2d(0.2),

            nn.AdaptiveAvgPool2d((6, 6)),
            nn.Flatten()
        )
        
        # Re-calculate CNN output shape with the new layer
        with torch.no_grad():
            sample = torch.zeros(1, 3, self.image_space.shape[0], self.image_space.shape[1])
            cnn_output = self.cnn(sample)
            cnn_features = cnn_output.shape[1]
        
        # Transform parameters feature extractor
        self.transform_net = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU()
        )
        
        # Combine both features
        self.combined = nn.Sequential(
            nn.Linear(cnn_features + 32, features_dim),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        
    def forward(self, observations):
        # Process image
        img = observations['image']
        # Ensure we're working with the right size
        if img.shape[-1] == 3:  # If HWC format
            img = img.permute(0, 3, 1, 2).float() / 255.0
        else:
            img = img.float() / 255.0
        
        # Don't assume a specific size, just pass through the CNN
        img_features = self.cnn(img)
        
        # Process transform parameters
        trans_features = self.transform_net(observations['transforms'].float())
        
        # Combine features
        combined = torch.cat([img_features, trans_features], dim=1)
        
        return self.combined(combined)

# Function to run evaluation and compare results
def evaluate_occlusion_reduction(model_path, test_images_dir, test_labels_dir, output_dir):
    """
    Evaluate the RL model's performance in reducing occlusion
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load YOLO model - use your trained model for evaluation
    yolo_model = YOLO(model_path)  # This is your trained model from best.pt
    print(f"Using trained YOLO model from {model_path} for evaluation")
    
    # Create environment with your trained model and labels directory
    env = OcclusionEnvironment(yolo_model, test_images_dir, labels_dir=test_labels_dir)
    
    # Load best RL model from validation
    try:
        rl_model = PPO.load("best_rl_occlusion_model", env=env)
        print("Using best model from validation for evaluation")
    except:
        rl_model = PPO.load("rl_occlusion_model", env=env)
        print("Using final model for evaluation")
        
    # Load test images
    test_images = [os.path.join(test_images_dir, f) for f in os.listdir(test_images_dir) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    results = []
    
    for img_path in tqdm(test_images, desc="Evaluating images"):
        # Load and preprocess image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        
        # Reset environment with the image and path to load ground truth
        obs, _ = env.reset(options={"image": img.copy(), "image_path": img_path})
        
        # Get baseline results
        baseline_results = yolo_model(img)
        baseline_boxes = baseline_results[0].boxes
        baseline_count = sum(1 for box in baseline_boxes if int(box.cls[0]) == env.target_class_index)
        baseline_confidences = [float(box.conf[0]) for box in baseline_boxes 
                              if int(box.cls[0]) == env.target_class_index]
        baseline_conf = sum(baseline_confidences) / len(baseline_confidences) if baseline_confidences else 0
        
        # Get baseline mAP using ground truth - load ground truth once
        ground_truth = env.current_ground_truth  # Use already loaded ground truth
        baseline_map = env._calculate_map(baseline_results, ground_truth)
        
        # Don't reset again, just optimize from current state
        done = False
        
        while not done:
            action, _ = rl_model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
        
        # Get optimized image
        optimized_img = env.get_best_image()
        
        # Get results on optimized image
        optimized_results = yolo_model(optimized_img)
        optimized_boxes = optimized_results[0].boxes
        optimized_count = sum(1 for box in optimized_boxes if int(box.cls[0]) == env.target_class_index)
        optimized_confidences = [float(box.conf[0]) for box in optimized_boxes 
                            if int(box.cls[0]) == env.target_class_index]
        optimized_conf = sum(optimized_confidences) / len(optimized_confidences) if optimized_confidences else 0

        # Calculate mAP using ground truth
        baseline_map = env._calculate_map(baseline_results, env.current_ground_truth)
        optimized_map = env._calculate_map(optimized_results, env.current_ground_truth)

        # Save comparison image
        plt.figure(figsize=(12, 8))
        
        plt.subplot(1, 2, 1)
        plt.title(f"Baseline: {baseline_count} chairs, Conf: {baseline_conf:.2f}")
        baseline_annotated = baseline_results[0].plot()
        plt.imshow(cv2.cvtColor(baseline_annotated, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.title(f"Optimized: {optimized_count} chairs, Conf: {optimized_conf:.2f}")
        optimized_annotated = optimized_results[0].plot()
        plt.imshow(cv2.cvtColor(optimized_annotated, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        
        plt.tight_layout()
        img_name = os.path.basename(img_path)
        plt.savefig(os.path.join(output_dir, f"comparison_{img_name}"))
        plt.close()
        
        # Store results
        results.append({
            'image': img_name,
            'baseline_count': baseline_count,
            'baseline_confidence': baseline_conf,
            'baseline_map': baseline_map,
            'optimized_count': optimized_count,
            'optimized_confidence': optimized_conf,
            'optimized_map': optimized_map,
            'count_improvement': optimized_count - baseline_count,
            'confidence_improvement': optimized_conf - baseline_conf,
            'map_improvement': optimized_map - baseline_map
        })
    
    # Create summary dataframe
    df = pd.DataFrame(results)
    
   # Calculate summary statistics
    summary = {
        'avg_baseline_count': df['baseline_count'].mean(),
        'avg_optimized_count': df['optimized_count'].mean(),
        'avg_baseline_conf': df['baseline_confidence'].mean(),
        'avg_optimized_conf': df['optimized_confidence'].mean(),
        'avg_baseline_map': df['baseline_map'].mean(),
        'avg_optimized_map': df['optimized_map'].mean(),
        'avg_count_improvement': df['count_improvement'].mean(),
        'avg_conf_improvement': df['confidence_improvement'].mean(),
        'avg_map_improvement': df['map_improvement'].mean(),
        'pct_improved_images': (df['confidence_improvement'] > 0).mean() * 100,
        'pct_improved_map': (df['map_improvement'] > 0).mean() * 100
    }
    
    # Save results
    df.to_csv(os.path.join(output_dir, "detailed_results.csv"), index=False)
    
    # Plot summary results
    plt.figure(figsize=(15, 12))

    # Count comparison
    plt.subplot(2, 3, 1)
    plt.bar(['Baseline', 'Optimized'], [summary['avg_baseline_count'], summary['avg_optimized_count']])
    plt.title('Average Object Count')
    plt.ylabel('Count')

    # Confidence comparison
    plt.subplot(2, 3, 2)
    plt.bar(['Baseline', 'Optimized'], [summary['avg_baseline_conf'], summary['avg_optimized_conf']])
    plt.title('Average Confidence')
    plt.ylabel('Confidence')

    # mAP comparison
    plt.subplot(2, 3, 3)
    
    plt.bar(['Baseline', 'Optimized'], [summary['avg_baseline_map'], summary['avg_optimized_map']])
    plt.title('Average mAP (with Ground Truth)')
    plt.ylabel('mAP')
    
    # Improvement distribution
    plt.subplot(2, 3, 4)
    plt.hist(df['confidence_improvement'], bins=20)
    plt.title('Confidence Improvement Distribution')
    plt.xlabel('Improvement')
    plt.ylabel('Count')

    # mAP improvement distribution
    plt.subplot(2, 3, 5)
    plt.hist(df['map_improvement'], bins=20)
    plt.title('mAP Improvement Distribution')
    plt.xlabel('Improvement')
    plt.ylabel('Count')

    # Success rate
    plt.subplot(2, 3, 6)
    plt.pie([summary['pct_improved_images'], 100-summary['pct_improved_images']], 
            labels=['Improved', 'Not Improved'], autopct='%1.1f%%')
    plt.title('Percentage of Images Improved')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "summary_results.png"))
    
    # Print summary
    print("\nSummary Results:")
    for key, value in summary.items():
        print(f"{key}: {value:.4f}")
    
    # Memory cleanup
    plt.close('all')  # Close all plots
    gc.collect()  # Force garbage collection

    return summary, df


def evaluate_model_generalization(model, train_env, val_env, test_env, n_episodes=20):
    """Compare performance on training, validation and test environments to detect overfitting"""
    # Set environments to evaluation mode
    for env in [train_env, val_env, test_env]:
        env.evaluation_mode = True
    
    # Evaluate on training set
    print("Evaluating on training set...")
    mean_reward_train, _ = evaluate_policy(model, train_env, n_eval_episodes=n_episodes)
    
    # Evaluate on validation set
    print("Evaluating on validation set...")
    mean_reward_val, _ = evaluate_policy(model, val_env, n_eval_episodes=n_episodes)
    
    # Evaluate on test set
    print("Evaluating on test set...")
    mean_reward_test, _ = evaluate_policy(model, test_env, n_eval_episodes=n_episodes)
    
    # Calculate performance gaps
    train_val_gap = mean_reward_train - mean_reward_val
    train_test_gap = mean_reward_train - mean_reward_test
    
    print(f"Training reward: {mean_reward_train:.4f}")
    print(f"Validation reward: {mean_reward_val:.4f}")
    print(f"Test reward: {mean_reward_test:.4f}")
    print(f"Train-validation gap: {train_val_gap:.4f} ({train_val_gap/mean_reward_train:.2%})")
    print(f"Train-test gap: {train_test_gap:.4f} ({train_test_gap/mean_reward_train:.2%})")
    
    # Check for overfitting
    if train_val_gap > 0.3 * mean_reward_train or train_test_gap > 0.3 * mean_reward_train:
        print("WARNING: Large performance gap detected! Model may be overfitting.")
    
    return mean_reward_train, mean_reward_val, mean_reward_test

# Main function
def main():
    # Paths
    data_yaml_path = "Augmented_Dataset/data.yaml"
    best_model_path = "train_results/train/weights/best.pt"  
    train_images_dir = "Augmented_Dataset/train/images"
    train_labels_dir = "Augmented_Dataset/train/labels"  
    valid_images_dir = "Augmented_Dataset/valid/images"
    valid_labels_dir = "Augmented_Dataset/valid/labels"  
    test_images_dir = "Augmented_Dataset/test/images"
    test_labels_dir = "Augmented_Dataset/test/labels" 
    output_dir = "occlusion_reduction_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load the pre-trained YOLO model
    yolo_model = YOLO(best_model_path)
    print(f"Loaded custom trained YOLO model from {best_model_path}")
    
    # 2. Create the environments - one for each dataset split with explicit labels directories
    train_env = OcclusionEnvironment(yolo_model, train_images_dir, labels_dir=train_labels_dir)
    valid_env = OcclusionEnvironment(yolo_model, valid_images_dir, labels_dir=valid_labels_dir)
    test_env = OcclusionEnvironment(yolo_model, test_images_dir, labels_dir=test_labels_dir)
    
    # Add evaluation_mode attribute to environments
    train_env.evaluation_mode = False
    valid_env.evaluation_mode = True
    test_env.evaluation_mode = True
    
    # Check if ground truth is available
    if os.path.exists(train_labels_dir):
        print("Using ground truth annotations for reward calculation")
    else:
        print("Warning: No ground truth annotations found for reward calculation")
    
    # Add evaluation_mode attribute to environments
    train_env.evaluation_mode = False  # Training environment shouldn't be in evaluation mode
    valid_env.evaluation_mode = True   # Validation should be in evaluation mode
    test_env.evaluation_mode = True    # Test should be in evaluation mode
    
    # 3. Create a policy network with regularization
    policy_kwargs=dict(
        features_extractor_class=CombinedFeaturesExtractor,
        features_extractor_kwargs=dict(features_dim=128),  # Reduced features
        net_arch=dict(pi=[64, 32], vf=[64, 32]),  # Smaller network
        activation_fn=nn.ReLU,
        ortho_init=True
    )

    # 4. Initialize the RL agent with increased entropy coefficient
    model = PPO(
    "MultiInputPolicy",  
    train_env,
    policy_kwargs=policy_kwargs,
    verbose=1,
    learning_rate=3e-4,     # Matches LRScheduler's initial_lr
    n_steps=256,            # More steps per update
    batch_size=64,          # n_steps must be divisible by batch_size (256/64=4)
    n_epochs=4,             # Fewer epochs to prevent overfitting
    gamma=0.99,             # Longer reward horizon
    ent_coef=0.2,           # Increased exploration
    clip_range=0.2,         # Standard value for stability
    target_kl=0.03,         # Slightly more flexibility
    tensorboard_log="./occlusion_rl_tensorboard/"
)
    
    # 5. Create validation callback for early stopping
    val_callback = ValidationCallback(valid_env, eval_freq=500, n_eval_episodes=10)
    lr_callback = LRSchedulerCallback(initial_lr=3e-4, min_lr=1e-5, decay_factor=0.5, 
                                      decay_steps=500)  # Faster decay for small dataset
    # Combine callbacks
    callbacks = [val_callback, lr_callback]
    
    # 6. Train the RL agent with validation monitoring
    print("Training RL agent...")
    model.learn(total_timesteps=10000, callback=callbacks)  # Adjust timesteps based on dataset size
    
    # 7. Load best model (saved by callback during training)
    try:
        best_model = PPO.load("best_rl_occlusion_model", env=train_env)
        print("Loaded best model from validation checkpoints")
    except:
        best_model = model
        print("Using final model as best model")
     
    # 8. Evaluate generalization capabilities
    print("Evaluating model generalization...")
    evaluate_model_generalization(best_model, train_env, valid_env, test_env)
    
    # 9. Save the final model
    best_model.save("rl_occlusion_model")
    print("Best model saved as rl_occlusion_model")
    
    # 10. Evaluate performance and compare with baseline
    print("Evaluating performance using the trained YOLO model...")
    # In the main function:
    summary, results = evaluate_occlusion_reduction(
        best_model_path,
        test_images_dir,
        test_labels_dir, 
        output_dir
    )
    
    print(f"Evaluation complete. Results saved to {output_dir}")

if __name__ == "__main__":
    main()