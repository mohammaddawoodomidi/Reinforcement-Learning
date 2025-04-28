import os
import cv2
from sklearn.metrics import average_precision_score, precision_recall_curve
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.evaluation import evaluate_policy
from ultralytics import YOLO
import matplotlib
# Use 'Agg' backend which doesn't require GUI
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import random
from tqdm import tqdm
import pandas as pd
from PIL import Image, ImageEnhance
import logging
import yaml

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Constants
TARGET_CLASS = "chair"  # Same as in your notebook
IMAGE_SIZE = 1280
MAX_STEPS = 30

class OcclusionEnvironment(gym.Env):
    """
    A 2D environment for optimizing viewpoints to reduce occlusion
    """
    def __init__(self, yolo_model, images_dir, target_class=TARGET_CLASS):
        super().__init__()
        self.yolo_model = yolo_model
        self.target_class = target_class

        # Add this to confirm which model is being used:
        if hasattr(yolo_model, 'model') and hasattr(yolo_model.model, 'names'):
            logging.info(f"YOLO model loaded with classes: {yolo_model.model.names}")
        else:
            logging.info("Loaded YOLO model, but could not display class names")
        
        # Load images
        self.image_paths = [os.path.join(images_dir, f) for f in os.listdir(images_dir) 
                           if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        assert len(self.image_paths) > 0, f"No images found in {images_dir}"
        logging.info(f"Loaded {len(self.image_paths)} images")
        
        # Define action space: [translate_x, translate_y, rotation, zoom]
        # All values normalized between -1 and 1
        self.action_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
        
        # Define observation space: RGB image
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8
        )
        
        # Maximum transformation values
        self.max_translate = 0.2  # 20% of image size
        self.max_rotate = 30  # degrees
        self.max_zoom = 0.3  # 30% zoom in/out
        
        # Current episode state
        self.current_image = None
        self.original_image = None
        self.current_step = 0
        self.previous_detection_score = 0
        self.previous_occlusion_score = 0
        self.best_image = None
        self.best_score = 0
        self.transformations = []
        
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
        
        # If specific image is provided in options, use it
        if options and "image" in options:
            self.original_image = options["image"].copy()
        else:
            # Otherwise select a random image
            image_path = random.choice(self.image_paths)
            self.original_image = cv2.imread(image_path)
            self.original_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
            self.original_image = cv2.resize(self.original_image, (IMAGE_SIZE, IMAGE_SIZE))
        
        # Make a copy of the original image
        self.current_image = self.original_image.copy()
        
        # Reset episode state
        self.current_step = 0
        self.transformations = []
        
        # Get initial detection score
        results = self.yolo_model(self.current_image)
        self.previous_detection_score = self._calculate_detection_score(results)
        self.previous_occlusion_score = self._calculate_occlusion_score(results)
        
        self.best_image = self.current_image.copy()
        self.best_score = self.previous_detection_score
        
        # Return just the observation, not a tuple with info
        return self.current_image, {}  # Adding empty info dictionary
    
    def step(self, action):
        self.current_step += 1
        
        # Apply transformations to the image
        transformed_image = self._apply_transformations(self.original_image, action)
        self.current_image = transformed_image
        
        # Get detection results
        results = self.yolo_model(self.current_image)
        
        # Calculate metrics
        detection_score = self._calculate_detection_score(results)
        occlusion_score = self._calculate_occlusion_score(results)
        
        # Calculate reward
        reward = self._calculate_reward(detection_score, occlusion_score)
        
        # Update best image if better score
        if detection_score > self.best_score:
            self.best_score = detection_score
            self.best_image = self.current_image.copy()
            
        # Store current score
        self.previous_detection_score = detection_score
        self.previous_occlusion_score = occlusion_score
        
        # Check if episode is done
        terminated = self.current_step >= MAX_STEPS
        truncated = False  # No time limit truncation in this environment
        
        info = {
            'detection_score': detection_score,
            'occlusion_score': occlusion_score,
            'step': self.current_step
        }
        
        return self.current_image, reward, terminated, truncated, info
    
    def _apply_transformations(self, image, action):
        """Apply a series of transformations to the image based on the action"""
        # Unpack actions
        translate_x, translate_y, rotate, zoom = action
        
        # Scale actions to actual values
        translate_x = translate_x * self.max_translate * IMAGE_SIZE
        translate_y = translate_y * self.max_translate * IMAGE_SIZE
        rotate = rotate * self.max_rotate
        zoom = 1 + (zoom * self.max_zoom)  # 0.7 to 1.3
        
        # Store transformation
        self.transformations.append({
            'translate_x': translate_x,
            'translate_y': translate_y,
            'rotate': rotate,
            'zoom': zoom
        })
        
        # Create transformation matrix
        M_translate = np.float32([[1, 0, translate_x], [0, 1, translate_y]])
        M_rotate = cv2.getRotationMatrix2D((IMAGE_SIZE/2, IMAGE_SIZE/2), rotate, zoom)
        
        # Apply transformations
        transformed = cv2.warpAffine(image, M_translate, (IMAGE_SIZE, IMAGE_SIZE))
        transformed = cv2.warpAffine(transformed, M_rotate, (IMAGE_SIZE, IMAGE_SIZE))
        
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
        # If no ground truth is provided, this is for comparison purposes only
        # We'll use a simplified mAP calculation based on confidence scores and IoU
        
        # Get the boxes for target class
        boxes = results[0].boxes
        target_boxes = [box for box in boxes if int(box.cls[0]) == self.target_class_index]
        
        if not target_boxes or (ground_truth is not None and not ground_truth):
            return 0.0
        
        if ground_truth is None:
            # If no ground truth, just use a confidence-weighted score
            return sum(float(box.conf[0]) for box in target_boxes) / len(target_boxes)
        
        # With ground truth, calculate precision at different IoU thresholds
        iou_thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        aps = []
        
        for threshold in iou_thresholds:
            tp, fp = 0, 0
            matched_gt = set()
            
            # Sort detections by confidence
            sorted_boxes = sorted(target_boxes, key=lambda box: float(box.conf[0]), reverse=True)
            
            for det_box in sorted_boxes:
                best_iou = 0
                best_gt_idx = -1
                
                # Find best matching ground truth
                for gt_idx, gt_box in enumerate(ground_truth):
                    if gt_idx in matched_gt:
                        continue
                    
                    iou = self._calculate_iou(
                        det_box.xyxy.cpu().numpy()[0], 
                        gt_box
                    )
                    
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx
                
                # If IoU above threshold, count as true positive
                if best_iou >= threshold and best_gt_idx not in matched_gt:
                    tp += 1
                    matched_gt.add(best_gt_idx)
                else:
                    fp += 1
            
            # Calculate precision
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            aps.append(precision)
        
        # Return mean of APs (mAP)
        return sum(aps) / len(aps)

    
    def _calculate_reward(self, detection_score, occlusion_score):
        """Calculate reward based on detection improvement and occlusion reduction"""
        # Detection improvement
        detection_improvement = detection_score - self.previous_detection_score
        
        # Occlusion reduction (negative is good)
        occlusion_reduction = self.previous_occlusion_score - occlusion_score
        
        # Calculate reward with weights
        reward = 2.0 * detection_improvement + 1.0 * occlusion_reduction
        
        # Additional reward for significant improvements
        if detection_improvement > 0.1:
            reward += 0.5
            
        # Penalty for no improvement to encourage exploration
        if abs(detection_improvement) < 0.01 and abs(occlusion_reduction) < 0.01:
            reward -= 0.1
            
        return reward
        
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

# Custom CNN feature extractor for images
class ImageCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        super().__init__(observation_space, features_dim)
        
        # CNN layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Calculate output shape
        with torch.no_grad():
            sample = torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE)
            x = F.relu(self.conv1(sample))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
            flat_size = x.flatten(1).shape[1]
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(flat_size, features_dim),
            nn.ReLU()
        )
        
    def forward(self, observations):
        # Reshape and normalize
        # Change from:
        # x = observations.permute(0, 3, 1, 2).float() / 255.0
        
        # To:
        if observations.shape[-1] == 3:  # If the last dimension is 3 (HWC format)
            x = observations.permute(0, 3, 1, 2).float() / 255.0  # Convert to NCHW format
        else:  # If already in NCHW format or other format
            x = observations.float() / 255.0  # Just normalize
            
        # Apply CNN layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.flatten(1)
        
        return self.fc(x)

# Function to run evaluation and compare results
def evaluate_occlusion_reduction(model_path, test_images_dir, output_dir):
    """
    Evaluate the RL model's performance in reducing occlusion
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load YOLO model - use your trained model for evaluation
    yolo_model = YOLO(model_path)  # This is your trained model from best.pt
    print(f"Using trained YOLO model from {model_path} for evaluation")
    
    # Create environment with your trained model
    env = OcclusionEnvironment(yolo_model, test_images_dir)
    
    # Load RL model that was trained with your custom YOLO model
    rl_model = PPO.load("rl_occlusion_model", env=env)
    
    # Load test images
    test_images = [os.path.join(test_images_dir, f) for f in os.listdir(test_images_dir) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    results = []
    
    for img_path in tqdm(test_images, desc="Evaluating images"):
        # Load and preprocess image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        
        # Get baseline results
        baseline_results = yolo_model(img)
        baseline_boxes = baseline_results[0].boxes
        baseline_count = sum(1 for box in baseline_boxes if int(box.cls[0]) == env.target_class_index)
        baseline_confidences = [float(box.conf[0]) for box in baseline_boxes 
                              if int(box.cls[0]) == env.target_class_index]
        baseline_conf = sum(baseline_confidences) / len(baseline_confidences) if baseline_confidences else 0
        
        # Use RL agent to optimize image
        # Use RL agent to optimize image
        env.original_image = img.copy()
        env.current_image = img.copy()  # Ensure current image is also set
        obs, _ = env.reset(options={"image": img.copy()})  # Pass the image through options
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

        # Calculate mAP
        baseline_map = env._calculate_map(baseline_results)
        optimized_map = env._calculate_map(optimized_results)
        
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
    plt.title('Average mAP')
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
        
    return summary, df

# Main function
def main():
    # Paths
    data_yaml_path = "Augmented_Dataset/data.yaml"
    best_model_path = "train_results/train/weights/best.pt"  # This is correct - your trained model
    train_images_dir = "Augmented_Dataset/train/images"
    test_images_dir = "Augmented_Dataset/test/images"
    output_dir = "occlusion_reduction_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load the pre-trained YOLO model - your custom trained model
    yolo_model = YOLO(best_model_path)  # This loads your trained model
    print(f"Loaded custom trained YOLO model from {best_model_path}")
    
    # 2. Create the environment with your trained model
    env = OcclusionEnvironment(yolo_model, train_images_dir)
    
    # 3. Create a policy network using stable-baselines3 PPO
    policy_kwargs = dict(
        features_extractor_class=ImageCNN,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=[dict(pi=[128, 64], vf=[128, 64])]
    )
    
    # 4. Initialize the RL agent
    model = PPO(
        "CnnPolicy",
        env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        learning_rate=3e-4,
        n_steps=128,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        ent_coef=0.01,
        tensorboard_log="./occlusion_rl_tensorboard/"
    )
    
    # 5. Train the RL agent
    print("Training RL agent...")
    model.learn(total_timesteps=10000)  # Reduce for faster training
    
    # 6. Save the model
    model.save("rl_occlusion_model")
    print("Model saved as rl_occlusion_model")
    
    # 7. Evaluate performance and compare with baseline
    print("Evaluating performance using the trained YOLO model...")
    summary, results = evaluate_occlusion_reduction(
        best_model_path,  # This is your trained model path
        test_images_dir, 
        output_dir
    )
    
    print(f"Evaluation complete. Results saved to {output_dir}")

if __name__ == "__main__":
    main()