import os
import random
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from collections import Counter

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

# Although number 6 is indexed as "trash" in the dataset, 
# we will use it to index the additional "other" class
# which covers other types of trash that are not included in the dataset, 
# such as: organic, wood, etc.
DEFAULT_CLASS_DICT = {
    1: 'glass',
    2: 'paper',
    3: 'cardboard',
    4: 'plastic',
    5: 'metal',
    6: 'other'
}

class TrashDataset(Dataset):
    """Custom dataset for trash classification"""
    
    def __init__(self, data_list: List[Tuple[str, int]], transform=None, class_dict=None):
        """
        Args:
            data_list: List of (image_path, label) tuples
            transform: torchvision transforms to apply
            class_dict: Dictionary mapping class names to indices
        """
        self.data_list = data_list
        self.transform = transform
        self.class_dict = class_dict
        
        # Create class dictionary if not provided
        if class_dict is None:
            self.class_dict = DEFAULT_CLASS_DICT

        self.num_classes = len(self.class_dict)
    
    def __len__(self) -> int:
        return len(self.data_list)
    
    def __getitem__(self, idx) -> Tuple[torch.Tensor, int]:
        image_path, class_idx = self.data_list[idx]
        
        try:
            image = Image.open(image_path).convert('RGB')

            if self.transform:
                image = self.transform(image)
            
            return image, class_idx
            
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a dummy image and label in case of error
            dummy_image = Image.new('RGB', (224, 224), color='black')
            if self.transform:
                dummy_image = self.transform(dummy_image)
            
            # Use "other" class.
            # If any class is added to or removed from the dictionary, 
            # this index should be adjusted accordingly.
            return dummy_image, 6
        
class TrashDataLoader:
    """Data loader manager for trash classification dataset"""
    
    def __init__(self, data_dir: str, batch_size: int = 32, num_workers: int = 4, 
                 image_size: int = 224, augment_training: bool = True):
        """
        Args:
            data_dir: Root directory containing the dataset
            batch_size: Batch size for data loading
            num_workers: Number of worker processes for data loading
            image_size: Size to resize images to (square)
            augment_training: Whether to apply data augmentation to training set
        """
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        
        # Define transforms
        self.train_transform = self._get_train_transforms(augment_training)
        self.val_test_transform = self._get_val_test_transforms()
        
        # Class dictionary
        self.class_dict = DEFAULT_CLASS_DICT
        
        # Load data lists
        self.train_data = self._load_data_from_file('one-indexed-files-notrash_train.txt')
        self.val_data = self._load_data_from_file('one-indexed-files-notrash_val.txt')
        self.test_data = self._load_data_from_file('one-indexed-files-notrash_test.txt')
        
        print(f"Loaded {len(self.train_data)} training samples")
        print(f"Loaded {len(self.val_data)} validation samples") 
        print(f"Loaded {len(self.test_data)} test samples")
        
        # Print class distribution for training set
        self._print_class_distribution()
    
    def _get_train_transforms(self, augment: bool) -> transforms.Compose:
        """Get transforms for training data"""
        transform_list = [
            transforms.Resize((self.image_size, self.image_size)),
        ]
        
        if augment:
            transform_list.extend([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.RandomResizedCrop(self.image_size, scale=(0.8, 1.0)),
            ])
        
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        return transforms.Compose(transform_list)
    
    def _get_val_test_transforms(self) -> transforms.Compose:
        """Get transforms for validation and test data"""
        return transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def _load_data_from_file(self, filename: str) -> List[Tuple[str, int]]:
        """Load data from text file"""
        file_path = self.data_dir / filename
        data_list = []
        
        if not file_path.exists():
            print(f"Warning: {file_path} not found!")
            return data_list
        
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split()
                    if len(parts) >= 2:
                        # Assume format: image_name label
                        image_name = parts[0]
                        label = int(parts[1])
                        
                        # Extract category from image name (e.g., "glass3.jpg" -> "glass")
                        category_name = ''.join([c for c in image_name if not c.isdigit() and c != '.'])
                        category_name = category_name.replace('jpg', '').replace('png', '').replace('jpeg', '')
                        
                        # Construct full image path: data_dir/category/image_name
                        image_path = self.data_dir / "Garbage classification/Garbage classification" / category_name / image_name
                        
                        if image_path.exists():
                            data_list.append((str(image_path), label))
                        else:
                            print(f"Warning: Image not found: {image_path}")
                    else:
                        print(f"Warning: Invalid line format: {line}")
        
        return data_list
    
    def _print_class_distribution(self):
        """Print class distribution for training data"""
        if not self.train_data:
            return
            
        label_counts = Counter([label for _, label in self.train_data])
        
        print("\nTraining data class distribution:")
        print("-" * 40)
        for class_idx, count in sorted(label_counts.items()):
            class_name = self.class_dict.get(class_idx, f"class_{class_idx}")
            percentage = (count / len(self.train_data)) * 100
            print(f"{class_name:>10}: {count:>4} samples ({percentage:>5.1f}%)")
        print("-" * 40)
    
    def get_train_loader(self, shuffle: bool = True) -> DataLoader:
        """Get training data loader"""
        dataset = TrashDataset(
            data_list=self.train_data,
            transform=self.train_transform,
            class_dict=self.class_dict
        )
        
        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=True
        )
    
    def get_val_loader(self, shuffle: bool = False) -> DataLoader:
        """Get validation data loader"""
        dataset = TrashDataset(
            data_list=self.val_data,
            transform=self.val_test_transform,
            class_dict=self.class_dict
        )
        
        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=False
        )
    
    def get_test_loader(self, shuffle: bool = False) -> DataLoader:
        """Get test data loader"""
        dataset = TrashDataset(
            data_list=self.test_data,
            transform=self.val_test_transform,
            class_dict=self.class_dict
        )
        
        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=False
        )
    
    def get_single_image_loader(self, image_path: str) -> DataLoader:
        """Get data loader for a single image (for inference)"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Use dummy label (will be ignored during inference)
        data_list = [(image_path, 1)]
        
        dataset = TrashDataset(
            data_list=data_list,
            transform=self.val_test_transform,
            class_dict=self.class_dict
        )
        
        return DataLoader(
            dataset=dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
            drop_last=False
        )
    
    def get_class_names(self) -> Dict[int, str]:
        """Get class index to name mapping"""
        return self.class_dict.copy()
    
    def get_num_classes(self) -> int:
        """Get number of classes"""
        return len(self.class_dict)
    
    def visualize_batch(self, data_loader: DataLoader, num_samples: int = 8):
        """Visualize a batch of images with labels"""
        data_iter = iter(data_loader)
        images, labels = next(data_iter)
        
        # Denormalize images for visualization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        
        fig, axes = plt.subplots(2, 4, figsize=(12, 6))
        axes = axes.flatten()
        
        for i in range(min(num_samples, len(images))):
            # Denormalize
            img = images[i] * std + mean
            img = torch.clamp(img, 0, 1)
            
            # Convert to numpy and transpose
            img_np = img.permute(1, 2, 0).numpy()
            
            axes[i].imshow(img_np)
            axes[i].set_title(f"{self.class_dict[labels[i].item()]}")
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()