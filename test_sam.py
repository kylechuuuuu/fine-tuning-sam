import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from segment_anything import build_sam_vit_l
import logging
from datetime import datetime
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import argparse

def setup_logging():
    # Create logs directory if it doesn't exist
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    # Create a unique log file name with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = f'logs/test_{timestamp}.log'
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def calculate_f1_score(pred_masks, true_masks, threshold=0.5):
    # Convert to numpy arrays
    pred_masks = pred_masks.detach().cpu().numpy()
    true_masks = true_masks.detach().cpu().numpy()
    
    # Apply threshold
    pred_masks = (pred_masks > threshold).astype(np.uint8)
    true_masks = (true_masks > threshold).astype(np.uint8)
    
    # Flatten the arrays
    pred_masks = pred_masks.reshape(-1)
    true_masks = true_masks.reshape(-1)
    
    # Calculate F1 score
    return f1_score(true_masks, pred_masks)

class SegmentationDataset(Dataset):
    def __init__(self, root_dir, split='val'):
        self.root_dir = root_dir
        self.split = split
        self.image_dir = os.path.join(root_dir, split, 'image')
        self.mask_dir = os.path.join(root_dir, split, 'annotation_mask')
        self.image_files = sorted(os.listdir(self.image_dir))
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)
        
        # Load original image and mask
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        
        # Get original size
        original_size = image.size
        
        # Create transforms for model input (1024x1024)
        model_transform = transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
        ])
        
        # Create transforms for original size mask
        mask_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        
        # Transform images
        model_input = model_transform(image)
        original_mask = mask_transform(mask)
        
        return model_input, original_mask, original_size, img_name

def test(args):
    # Setup logging
    setup_logging()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    
    # Load model and trained weights
    sam = build_sam_vit_l(checkpoint=None)  # Don't load original weights
    sam.to(device)
    
    # Load trained weights
    checkpoint = torch.load(args.trained_weights, weights_only=True)
    sam.load_state_dict(checkpoint['model_state_dict'])
    logging.info(f"Loaded trained weights from {args.trained_weights}")
    logging.info(f"Trained F1 score: {checkpoint['f1_score']:.4f}")
    
    # Create dataset and dataloader
    val_dataset = SegmentationDataset(args.data_root, 'val')
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)
    
    logging.info(f"Validation dataset size: {len(val_dataset)}")
    
    # Create output directory
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    sam.eval()
    f1_scores = []
    
    with torch.no_grad():
        for model_input, original_mask, original_size, img_name in val_loader:
            model_input = model_input.to(device)
            original_mask = original_mask.to(device)
            
            # Forward pass with model input size (1024x1024)
            image_embeddings = sam.image_encoder(model_input)
            sparse_embeddings, dense_embeddings = sam.prompt_encoder(
                points=None,
                boxes=None,
                masks=None
            )
            low_res_masks, iou_predictions = sam.mask_decoder(
                image_embeddings=image_embeddings,
                image_pe=sam.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )
            
            # Resize prediction to original size
            low_res_masks = F.interpolate(low_res_masks, size=original_size[::-1], mode='bilinear', align_corners=False)
            
            # Calculate F1 score
            f1 = calculate_f1_score(low_res_masks, original_mask)
            f1_scores.append(f1)
            
            # Save prediction visualization
            pred_mask = (low_res_masks[0, 0] > 0.5).cpu().numpy().astype(np.uint8) * 255
            
            # Create visualization with exact original size
            width, height = original_size
            width = width.item() if torch.is_tensor(width) else width
            height = height.item() if torch.is_tensor(height) else height
            
            plt.figure(figsize=(width/100, height/100), dpi=100)
            plt.imshow(pred_mask, cmap='gray')
            plt.axis('off')
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
            
            plt.savefig(os.path.join(args.output_dir, f'{img_name[0]}_result.png'), 
                       bbox_inches='tight', 
                       pad_inches=0,
                       dpi=100)
            plt.close()
            
            logging.info(f"Image: {img_name[0]}, F1 Score: {f1:.4f}")
    
    # Calculate and log average F1 score
    avg_f1 = np.mean(f1_scores)
    logging.info(f"Average F1 Score: {avg_f1:.4f}")
    logging.info(f"F1 Score Std: {np.std(f1_scores):.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test trained SAM model')
    parser.add_argument('--data_root', type=str, default='/hy-tmp/sam/datasets/DRIVE', help='Path to dataset root directory')
    parser.add_argument('--trained_weights', type=str, default='best_model.pth', help='Path to trained weights')
    parser.add_argument('--output_dir', type=str, default='test_results', help='Directory to save test results')
    
    args = parser.parse_args()
    test(args) 