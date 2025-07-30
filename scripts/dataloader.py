from pathlib import Path
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


class MPDDDataset(Dataset):
    """Dataset for anomaly detection with mask support."""
    
    def __init__(self, root_path, category, split='train', image_size=288):
        self.root = Path(root_path) / category
        self.split = split
        self.image_size = image_size
        
        # Image transforms
        self.image_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Mask transforms (for ground truth masks)
        self.mask_transform = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()
        ])
        
        # Load samples
        self.samples = self._load_samples()
        print(f"Loaded {len(self.samples)} {split} samples for '{category}'")
    
    
    def _load_samples(self):
        """Load image paths with labels and mask paths"""
        samples = []
        
        if self.split == 'train':
            # Training: only normal images
            good_path = self.root / 'train' / 'good'
            for img_path in good_path.glob('*.png'):
                samples.append((img_path, 0, None))
                
        else:  # test
            # Normal test images
            good_path = self.root / 'test' / 'good'
            for img_path in good_path.glob('*.png'):
                samples.append((img_path, 0, None))
            
            # Anomaly images with masks
            test_path = self.root / 'test'
            for defect_dir in test_path.iterdir():
                if defect_dir.is_dir() and defect_dir.name != 'good':
                    mask_dir = self.root / 'ground_truth' / defect_dir.name
                    
                    for img_path in defect_dir.glob('*.png'):
                        # build the expected mask path (mask naming: 000.png -> 000_mask.png)
                        candidate_mask = mask_dir / f"{img_path.stem}_mask{img_path.suffix}"
                        # only use it if it actually exists
                        mask_path = candidate_mask if candidate_mask.exists() else None
                        # append as a 3‑tuple: (image_path, label=1, mask_path or None)
                        samples.append((img_path, 1, mask_path))
        
        return samples

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """Get a sample: image, label, and optionally mask"""
        img_path, label, mask_path = self.samples[idx]  # unpack the tuple

        # 1) load & normalize the image
        image = Image.open(img_path).convert("RGB")
        image = self.image_transform(image)

        # 2) load or create the mask
        if mask_path and mask_path.exists():
            mask = Image.open(mask_path).convert("L")
            mask = (self.mask_transform(mask) > 0.5).float()
        else:
            mask = torch.zeros(1, self.image_size, self.image_size)

        return {
            "image": image,
            "label": torch.tensor(label, dtype=torch.long),
            "mask":  mask
        }


def create_dataloaders(dataset_path, category, batch_size=8, image_size=288, num_workers=4):
    """Create train and test dataloaders."""
    
    # Create datasets
    train_dataset = MPDDDataset(dataset_path, category, 'train', image_size)
    test_dataset = MPDDDataset(dataset_path, category, 'test', image_size)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader

# ========== SANITY CHECK ==========
def main():

    train_loader, test_loader = create_dataloaders("data/mpdd", "bracket_white")

    train_batch = next(iter(train_loader))
    test_batch = next(iter(test_loader))

    print(f"\nTrain Batch")
    print(f"\tImage shape: {train_batch['image'].shape} \n\tLabel shape: {train_batch['label'].shape} \n\tMask shape: {train_batch['mask'].shape}")
    print(f"\nTest Batch")
    print(f"\tImage shape: {test_batch['image'].shape} \n\tLabel shape: {test_batch['label'].shape} \n\tMask shape: {test_batch['mask'].shape}")

if __name__ == "__main__":
    main()
    


