"""
Скрипт для аугментации изображений с созданием полного набора вариаций
Работает как в Jupyter Notebook, так и в командной строке
"""

import os
import cv2
import numpy as np
import random
import shutil
from pathlib import Path
from typing import Tuple, List, Optional, Dict
import argparse
import sys
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

if not hasattr(np, 'float'):
    np.float = float
if not hasattr(np, 'int'):
    np.int = int
if not hasattr(np, 'bool'):
    np.bool = bool
if not hasattr(np, 'complex'):
    np.complex = complex
if not hasattr(np, 'object'):
    np.object = object
if not hasattr(np, 'str'):
    np.str = str

class FullAugmentationPipeline:
    def __init__(self, 
                 input_image_path: str,
                 input_label_path: Optional[str] = None,
                 output_dir: str = "full_augmented_dataset"):
        self.input_image_path = Path(input_image_path)
        self.input_label_path = Path(input_label_path) if input_label_path else None
        self.output_dir = Path(output_dir)
        
        self.images_dir = self.output_dir / "images"
        self.labels_dir = self.output_dir / "labels"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.labels_dir.mkdir(parents=True, exist_ok=True)
        
        self.original_image = cv2.imread(str(self.input_image_path))
        if self.original_image is None:
            raise ValueError(f"Failed to load image: {input_image_path}")
        
        self.original_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
        self.height, self.width = self.original_image.shape[:2]
        
        self.labels = []
        if self.input_label_path and self.input_label_path.exists():
            self._load_labels()
        
        self.rotation_angles = [0, 45, 90, 135, 180, 225, 270, 315]
        
        self.augmentation_params = {
            'brightness': {
                'levels': [-50, -30, -10, 0, 10, 30, 50],
                'contrast': [0.6, 0.8, 1.0, 1.2, 1.4]
            },
            'gaussian_noise': {
                'sigma': [10, 20, 30, 40, 50]
            },
            'salt_pepper': {
                'prob': [0.01, 0.02, 0.03, 0.04, 0.05]
            },
            'gaussian_blur': {
                'kernel': [3, 5, 7, 9, 11]
            },
            'median_blur': {
                'kernel': [3, 5, 7, 9, 11]
            },
            'motion_blur': {
                'size': [5, 10, 15, 20, 25],
                'angle': [0, 45, 90, 135]
            },
            'color_jitter': {
                'hue': [-0.1, -0.05, 0, 0.05, 0.1],
                'saturation': [0.7, 0.85, 1.0, 1.15, 1.3],
                'value': [0.7, 0.85, 1.0, 1.15, 1.3]
            },
            'elastic': {
                'alpha': [30, 50, 70, 90, 110],
                'sigma': [3, 5, 7, 9, 11]
            },
            'resize': {
                'scale': [0.7, 0.8, 0.85, 0.9, 0.95]
            },
            'perspective': {
                'scale': [0.05, 0.1, 0.15, 0.2]
            }
        }
    
    def _load_labels(self):
        with open(self.input_label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    self.labels.append([class_id, x_center, y_center, width, height])
    
    def _save_labels(self, labels: List[List[float]], image_name: str):
        label_path = self.labels_dir / f"{image_name}.txt"
        with open(label_path, 'w') as f:
            for label in labels:
                f.write(f"{label[0]} {label[1]:.6f} {label[2]:.6f} "
                       f"{label[3]:.6f} {label[4]:.6f}\n")
    
    def _transform_bbox_rotation(self, bbox: List[float], angle: float) -> List[float]:
        class_id, x_center, y_center, width, height = bbox
        
        x1 = (x_center - width/2) * self.width
        y1 = (y_center - height/2) * self.height
        x2 = (x_center + width/2) * self.width
        y2 = (y_center + height/2) * self.height
        
        center_x, center_y = self.width / 2, self.height / 2
        
        def rotate_point(x, y, angle_deg):
            rad = np.radians(angle_deg)
            cos_a, sin_a = np.cos(rad), np.sin(rad)
            x_rot = center_x + (x - center_x) * cos_a - (y - center_y) * sin_a
            y_rot = center_y + (x - center_x) * sin_a + (y - center_y) * cos_a
            return x_rot, y_rot
        
        corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        rotated_corners = [rotate_point(x, y, angle) for x, y in corners]
        
        xs = [c[0] for c in rotated_corners]
        ys = [c[1] for c in rotated_corners]
        new_x1, new_x2 = min(xs), max(xs)
        new_y1, new_y2 = min(ys), max(ys)
        
        new_width = (new_x2 - new_x1) / self.width
        new_height = (new_y2 - new_y1) / self.height
        new_x_center = (new_x1 + new_x2) / 2 / self.width
        new_y_center = (new_y1 + new_y2) / 2 / self.height
        
        new_x_center = np.clip(new_x_center, 0.0, 1.0)
        new_y_center = np.clip(new_y_center, 0.0, 1.0)
        new_width = np.clip(new_width, 0.0, 1.0 - new_x_center)
        new_height = np.clip(new_height, 0.0, 1.0 - new_y_center)
        
        return [class_id, new_x_center, new_y_center, new_width, new_height]
    
    def rotate_image(self, image: np.ndarray, angle: float) -> np.ndarray:
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), 
                                 borderMode=cv2.BORDER_CONSTANT, 
                                 borderValue=(0, 0, 0))
        return rotated
    
    def adjust_brightness_contrast(self, image: np.ndarray, 
                                   brightness: float, 
                                   contrast: float) -> np.ndarray:
        return cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)
    
    def add_gaussian_noise(self, image: np.ndarray, sigma: float) -> np.ndarray:
        noise = np.random.normal(0, sigma, image.shape).astype(np.int16)
        noisy = image.astype(np.int16) + noise
        return np.clip(noisy, 0, 255).astype(np.uint8)
    
    def add_salt_pepper_noise(self, image: np.ndarray, prob: float) -> np.ndarray:
        output = image.copy()
        salt_mask = np.random.random(image.shape[:2]) < prob / 2
        pepper_mask = np.random.random(image.shape[:2]) < prob / 2
        output[salt_mask] = 255
        output[pepper_mask] = 0
        return output
    
    def gaussian_blur(self, image: np.ndarray, kernel: int) -> np.ndarray:
        kernel = kernel if kernel % 2 == 1 else kernel + 1
        return cv2.GaussianBlur(image, (kernel, kernel), 0)
    
    def median_blur(self, image: np.ndarray, kernel: int) -> np.ndarray:
        kernel = kernel if kernel % 2 == 1 else kernel + 1
        return cv2.medianBlur(image, kernel)
    
    def motion_blur(self, image: np.ndarray, size: int, angle: int) -> np.ndarray:
        size = size if size % 2 == 1 else size + 1
        kernel = np.zeros((size, size))
        kernel[int((size-1)/2), :] = np.ones(size)
        kernel = kernel / size
        M = cv2.getRotationMatrix2D((size/2, size/2), angle, 1)
        kernel = cv2.warpAffine(kernel, M, (size, size))
        return cv2.filter2D(image, -1, kernel)
    
    def apply_color_jitter(self, image: np.ndarray, hue: float, 
                          saturation: float, value: float) -> np.ndarray:
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 0] = (hsv[:, :, 0] + hue * 180) % 180
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation, 0, 255)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * value, 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    
    def elastic_transform(self, image: np.ndarray, alpha: int, sigma: int) -> np.ndarray:
        random_state = np.random.RandomState(None)
        shape = image.shape[:2]
        
        dx = random_state.randn(*shape) * sigma
        dy = random_state.randn(*shape) * sigma
        
        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        indices = (y + dy).reshape(-1).astype(np.float32), \
                  (x + dx).reshape(-1).astype(np.float32)
        
        return cv2.remap(image, indices[1], indices[0], 
                        cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    
    def resize_image(self, image: np.ndarray, scale: float) -> np.ndarray:
        h, w = image.shape[:2]
        new_h, new_w = int(h * scale), int(w * scale)
        resized = cv2.resize(image, (new_w, new_h))
        result = np.zeros_like(image)
        y_offset = (h - new_h) // 2
        x_offset = (w - new_w) // 2
        result[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        return result
    
    def perspective_transform(self, image: np.ndarray, scale: float) -> np.ndarray:
        h, w = image.shape[:2]
        
        pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        pts2 = np.float32([
            [scale * w, scale * h],
            [w - scale * w, scale * h],
            [scale * w, h - scale * h],
            [w - scale * w, h - scale * h]
        ])
        
        M = cv2.getPerspectiveTransform(pts1, pts2)
        return cv2.warpPerspective(image, M, (w, h))
    
    def generate_full_dataset(self):
        print("=" * 60)
        print("Starting full dataset generation")
        print("=" * 60)
        
        original_name = self.input_image_path.stem
        cv2.imwrite(str(self.images_dir / f"{original_name}_original.jpg"), 
                   cv2.cvtColor(self.original_image, cv2.COLOR_RGB2BGR))
        if self.labels:
            self._save_labels(self.labels, f"{original_name}_original")
        
        total_images = 1
        total_combinations = 0
        
        print("\nGenerating rotated images...")
        for angle in tqdm(self.rotation_angles, desc="Rotations"):
            if angle == 0:
                continue
                
            rotated_img = self.rotate_image(self.original_image, angle)
            rotated_labels = [self._transform_bbox_rotation(label, angle) for label in self.labels]
            
            img_name = f"{original_name}_rotate_{angle}"
            cv2.imwrite(str(self.images_dir / f"{img_name}.jpg"), 
                       cv2.cvtColor(rotated_img, cv2.COLOR_RGB2BGR))
            if rotated_labels:
                self._save_labels(rotated_labels, img_name)
            total_images += 1
        
        print("\nGenerating augmented images for all angles...")
        
        for aug_type, params in self.augmentation_params.items():
            print(f"\n  Processing {aug_type}...")
            
            param_combinations = self._get_param_combinations(aug_type, params)
            
            for param_combo in tqdm(param_combinations, desc=f"    {aug_type}", leave=False):
                for angle in self.rotation_angles:
                    if angle == 0:
                        base_img = self.original_image.copy()
                        base_labels = self.labels.copy()
                    else:
                        base_img = self.rotate_image(self.original_image, angle)
                        base_labels = [self._transform_bbox_rotation(label, angle) for label in self.labels]
                    
                    augmented_img, augmented_labels = self._apply_augmentation(
                        base_img, base_labels, aug_type, param_combo
                    )
                    
                    param_str = self._param_to_string(aug_type, param_combo)
                    img_name = f"{original_name}_{aug_type}_{param_str}_angle_{angle}"
                    
                    cv2.imwrite(str(self.images_dir / f"{img_name}.jpg"), 
                               cv2.cvtColor(augmented_img, cv2.COLOR_RGB2BGR))
                    if augmented_labels:
                        self._save_labels(augmented_labels, img_name)
                    
                    total_images += 1
                    total_combinations += 1
        
        print("\n" + "=" * 60)
        print(f"Generation completed")
        print(f"Statistics:")
        print(f"   - Source images: 1")
        print(f"   - Rotation angles: {len(self.rotation_angles)}")
        print(f"   - Augmentation types: {len(self.augmentation_params)}")
        print(f"   - Total images generated: {total_images}")
        print(f"   - Total combinations: {total_combinations}")
        print(f" Directory: {self.output_dir}")
        print("=" * 60)
    
    def _get_param_combinations(self, aug_type: str, params: Dict) -> List[Dict]:
        if aug_type == 'brightness':
            return [{'brightness': b, 'contrast': c} 
                   for b in params['levels'] 
                   for c in params['contrast']]
        elif aug_type == 'gaussian_noise':
            return [{'sigma': s} for s in params['sigma']]
        elif aug_type == 'salt_pepper':
            return [{'prob': p} for p in params['prob']]
        elif aug_type == 'gaussian_blur':
            return [{'kernel': k} for k in params['kernel']]
        elif aug_type == 'median_blur':
            return [{'kernel': k} for k in params['kernel']]
        elif aug_type == 'motion_blur':
            return [{'size': s, 'angle': a} 
                   for s in params['size'] 
                   for a in params['angle']]
        elif aug_type == 'color_jitter':
            return [{'hue': h, 'saturation': sat, 'value': val} 
                   for h in params['hue'] 
                   for sat in params['saturation'] 
                   for val in params['value']]
        elif aug_type == 'elastic':
            return [{'alpha': a, 'sigma': s} 
                   for a in params['alpha'] 
                   for s in params['sigma']]
        elif aug_type == 'resize':
            return [{'scale': s} for s in params['scale']]
        elif aug_type == 'perspective':
            return [{'scale': s} for s in params['scale']]
        return []
    
    def _apply_augmentation(self, image: np.ndarray, labels: List, 
                           aug_type: str, params: Dict) -> Tuple[np.ndarray, List]:
        if aug_type == 'brightness':
            augmented = self.adjust_brightness_contrast(
                image, params['brightness'], params['contrast']
            )
        elif aug_type == 'gaussian_noise':
            augmented = self.add_gaussian_noise(image, params['sigma'])
        elif aug_type == 'salt_pepper':
            augmented = self.add_salt_pepper_noise(image, params['prob'])
        elif aug_type == 'gaussian_blur':
            augmented = self.gaussian_blur(image, params['kernel'])
        elif aug_type == 'median_blur':
            augmented = self.median_blur(image, params['kernel'])
        elif aug_type == 'motion_blur':
            augmented = self.motion_blur(image, params['size'], params['angle'])
        elif aug_type == 'color_jitter':
            augmented = self.apply_color_jitter(
                image, params['hue'], params['saturation'], params['value']
            )
        elif aug_type == 'elastic':
            augmented = self.elastic_transform(image, params['alpha'], params['sigma'])
        elif aug_type == 'resize':
            augmented = self.resize_image(image, params['scale'])
        elif aug_type == 'perspective':
            augmented = self.perspective_transform(image, params['scale'])
        else:
            augmented = image
        
        return augmented, labels
    
    def _param_to_string(self, aug_type: str, params: Dict) -> str:
        parts = []
        for key, value in params.items():
            if isinstance(value, float):
                val_str = f"{value:.2f}".replace('.', '_').replace('-', 'neg')
                parts.append(f"{key}_{val_str}")
            else:
                parts.append(f"{key}_{value}")
        return '_'.join(parts)


def augment_image_jupyter(image_path, label_path=None, output_dir="augmented_dataset"):
    pipeline = FullAugmentationPipeline(
        input_image_path=image_path,
        input_label_path=label_path,
        output_dir=output_dir
    )
    pipeline.generate_full_dataset()
    return output_dir


def augment_batch_jupyter(input_dir, output_dir="augmented_dataset"):
    processor = BatchProcessor(input_dir, output_dir)
    processor.process_all()
    return output_dir


class BatchProcessor:
    def __init__(self, input_dir: str, output_dir: str):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        
    def process_all(self):
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        images = []
        for ext in image_extensions:
            images.extend(self.input_dir.glob(f"*{ext}"))
        
        print(f"Found images: {len(images)}")
        
        for img_path in tqdm(images, desc="Processing images"):
            label_path = self.input_dir / f"{img_path.stem}.txt"
            if not label_path.exists():
                print(f"Warning: label file not found for {img_path.name}")
                label_path = None
            
            img_output_dir = self.output_dir / img_path.stem
            img_output_dir.mkdir(parents=True, exist_ok=True)
            
            try:
                pipeline = FullAugmentationPipeline(
                    input_image_path=str(img_path),
                    input_label_path=str(label_path) if label_path else None,
                    output_dir=str(img_output_dir)
                )
                pipeline.generate_full_dataset()
            except Exception as e:
                print(f"Error processing {img_path.name}: {e}")


def main():
    argv = [arg for arg in sys.argv if arg != '-f']
    
    parser = argparse.ArgumentParser(description='Full dataset augmentation for YOLO with all rotation angles')
    parser.add_argument('--image', type=str, required=False,
                       help='Path to source image')
    parser.add_argument('--label', type=str, default=None,
                       help='Path to YOLO label file (.txt)')
    parser.add_argument('--output', type=str, default='full_augmented_dataset',
                       help='Directory to save dataset')
    parser.add_argument('--batch', action='store_true',
                       help='Batch processing (process all images in folder)')
    parser.add_argument('--input_dir', type=str, default=None,
                       help='Directory with images for batch processing')
    
    args = parser.parse_args(argv[1:])
    
    if args.batch:
        if not args.input_dir:
            print("Error: specify --input_dir for batch processing")
            return
        if not Path(args.input_dir).exists():
            print(f"Error: directory {args.input_dir} does not exist")
            return
        processor = BatchProcessor(args.input_dir, args.output)
        processor.process_all()
    else:
        if not args.image:
            print("Error: specify --image for single image processing or --batch for batch processing")
            return
        if not Path(args.image).exists():
            print(f"Error: file {args.image} does not exist")
            return
        pipeline = FullAugmentationPipeline(
            input_image_path=args.image,
            input_label_path=args.label,
            output_dir=args.output
        )
        pipeline.generate_full_dataset()


if __name__ == "__main__":
    if 'ipykernel' in sys.modules:
        print("=" * 60)
        print("Running in Jupyter Notebook")
        print("=" * 60)
        print("Use functions:")
        print("  augment_image_jupyter('path/to/image.jpg', 'path/to/label.txt')")
        print("  augment_batch_jupyter('path/to/images_dir', 'output_dir')")
        print("=" * 60)
    else:
        main()
