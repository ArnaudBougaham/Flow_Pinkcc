import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import nibabel as nib
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

class MedicalImageDataset(Dataset):
    def __init__(self, ct_paths, mask_paths, transform=None):
        self.ct_paths = ct_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.ct_paths)

    def __getitem__(self, idx):
        # Load CT image
        ct_img = nib.load(self.ct_paths[idx]).get_fdata()
        mask = nib.load(self.mask_paths[idx]).get_fdata()
        
        # Normalize CT image
        ct_img = (ct_img - ct_img.min()) / (ct_img.max() - ct_img.min())
        
        # Convert to torch tensors
        ct_img = torch.from_numpy(ct_img).float()
        mask = torch.from_numpy(mask).float()
        
        if self.transform:
            ct_img = self.transform(ct_img)
            mask = self.transform(mask)
            
        return ct_img, mask

def get_data_loaders(ct_dir, mask_dir, batch_size=8, num_folds=5):
    """
    Create data loaders for training, validation, and testing
    """
    # Get all file paths
    ct_files = sorted([os.path.join(ct_dir, f) for f in os.listdir(ct_dir) if f.endswith('.nii.gz')])
    mask_files = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.endswith('.nii.gz')])
    
    # Create KFold splitter
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    
    # Define transforms
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    
    # Create datasets and dataloaders for each fold
    fold_loaders = []
    for train_idx, val_idx in kf.split(ct_files):
        train_ct = [ct_files[i] for i in train_idx]
        train_mask = [mask_files[i] for i in train_idx]
        val_ct = [ct_files[i] for i in val_idx]
        val_mask = [mask_files[i] for i in val_idx]
        
        train_dataset = MedicalImageDataset(train_ct, train_mask, transform=transform)
        val_dataset = MedicalImageDataset(val_ct, val_mask, transform=transform)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        fold_loaders.append((train_loader, val_loader))
    
    return fold_loaders

def plot_sample(ct_img, mask, pred_mask=None):
    """
    Plot a sample CT image with its ground truth mask and optional prediction
    """
    plt.figure(figsize=(15, 5))
    
    plt.subplot(131)
    plt.imshow(ct_img.squeeze(), cmap='gray')
    plt.title('CT Image')
    plt.axis('off')
    
    plt.subplot(132)
    plt.imshow(mask.squeeze(), cmap='gray')
    plt.title('Ground Truth Mask')
    plt.axis('off')
    
    if pred_mask is not None:
        plt.subplot(133)
        plt.imshow(pred_mask.squeeze(), cmap='gray')
        plt.title('Predicted Mask')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def dice_coefficient(pred, target):
    """
    Calculate Dice coefficient
    """
    smooth = 1e-5
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

def save_model(model, path):
    """
    Save model state
    """
    torch.save(model.state_dict(), path)

def load_model(model, path):
    """
    Load model state
    """
    model.load_state_dict(torch.load(path))
    return model 