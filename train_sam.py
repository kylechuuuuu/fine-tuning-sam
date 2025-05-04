import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from segment_anything import build_sam_vit_l
from segment_anything.modeling import ImageEncoderViT
import argparse
import torch.nn.functional as F
import logging
from datetime import datetime
from sklearn.metrics import f1_score

def setup_logging(args):
    # Create logs directory if it doesn't exist
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    # Create a unique log file name with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = f'logs/training_{timestamp}.log'
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    # Log training configuration
    logging.info("Training Configuration:")
    for arg, value in vars(args).items():
        logging.info(f"{arg}: {value}")

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

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
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.image_dir = os.path.join(root_dir, split, 'image')
        self.mask_dir = os.path.join(root_dir, split, 'annotation_mask')
        self.image_files = sorted(os.listdir(self.image_dir))
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)
        
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
            
        return image, mask

def train(args):
    # Setup logging
    setup_logging(args)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    
    sam = build_sam_vit_l(checkpoint=args.checkpoint)
    sam.to(device)
    
    # Count and log parameters
    total_params, trainable_params = count_parameters(sam)
    logging.info(f"Total parameters: {total_params:,}")
    logging.info(f"Trainable parameters: {trainable_params:,}")
    logging.info(f"Trainable parameters percentage: {trainable_params/total_params*100:.2f}%")
    
    for name, param in sam.image_encoder.named_parameters():
        if 'adapter' not in name:
            param.requires_grad = False
    
    for param in sam.prompt_encoder.parameters():
        param.requires_grad = False
    
    # Count and log parameters after freezing
    total_params, trainable_params = count_parameters(sam)
    logging.info(f"After freezing - Total parameters: {total_params:,}")
    logging.info(f"After freezing - Trainable parameters: {trainable_params:,}")
    logging.info(f"After freezing - Trainable parameters percentage: {trainable_params/total_params*100:.2f}%")
    
    transform = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
    ])
    
    train_dataset = SegmentationDataset(args.data_root, 'train', transform)
    val_dataset = SegmentationDataset(args.data_root, 'val', transform)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    logging.info(f"Training dataset size: {len(train_dataset)}")
    logging.info(f"Validation dataset size: {len(val_dataset)}")
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam([
        {'params': sam.image_encoder.adapters.parameters()},
        {'params': sam.mask_decoder.parameters()}
    ], lr=args.learning_rate)
    
    best_f1_score = 0.0
    
    for epoch in range(args.epochs):
        sam.train()
        train_loss = 0.0
        
        for batch_idx, (images, masks) in enumerate(train_loader):
            images = images.to(device)
            masks = masks.to(device)
            
            image_embeddings = sam.image_encoder(images)
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
            
            masks = F.interpolate(masks, size=(1024, 1024), mode='bilinear', align_corners=False)
            low_res_masks = F.interpolate(low_res_masks, size=(1024, 1024), mode='bilinear', align_corners=False)
            
            loss = criterion(low_res_masks, masks)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            if batch_idx % 10 == 0:
                logging.info(f'Epoch: {epoch+1}/{args.epochs}, Batch: {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')
        
        avg_train_loss = train_loss/len(train_loader)
        logging.info(f'Epoch {epoch+1}/{args.epochs} - Average Train Loss: {avg_train_loss:.4f}')
        
        # Validate and calculate F1 score every 5 epochs
        if (epoch + 1) % 5 == 0:
            sam.eval()
            val_loss = 0.0
            all_pred_masks = []
            all_true_masks = []
            
            with torch.no_grad():
                for images, masks in val_loader:
                    images = images.to(device)
                    masks = masks.to(device)
                    
                    image_embeddings = sam.image_encoder(images)
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
                    
                    masks = F.interpolate(masks, size=(1024, 1024), mode='bilinear', align_corners=False)
                    low_res_masks = F.interpolate(low_res_masks, size=(1024, 1024), mode='bilinear', align_corners=False)
                    
                    loss = criterion(low_res_masks, masks)
                    val_loss += loss.item()
                    
                    # Collect predictions and ground truth for F1 calculation
                    all_pred_masks.append(low_res_masks)
                    all_true_masks.append(masks)
            
            # Calculate F1 score
            all_pred_masks = torch.cat(all_pred_masks, dim=0)
            all_true_masks = torch.cat(all_true_masks, dim=0)
            f1 = calculate_f1_score(all_pred_masks, all_true_masks)
            
            avg_val_loss = val_loss/len(val_loader)
            logging.info(f'Epoch {epoch+1}/{args.epochs} - Average Val Loss: {avg_val_loss:.4f}')
            logging.info(f'Epoch {epoch+1}/{args.epochs} - F1 Score: {f1:.4f}')
            
            # Save best model based on F1 score
            if f1 > best_f1_score:
                best_f1_score = f1
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': sam.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': avg_train_loss,
                    'val_loss': avg_val_loss,
                    'f1_score': f1,
                }, 'best_model.pth')
                logging.info(f'New best model saved with F1 score: {best_f1_score:.4f}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train SAM with adapters')
    parser.add_argument('--data_root', type=str, default='/hy-tmp/sam/datasets/DRIVE', help='Path to dataset root directory')
    parser.add_argument('--checkpoint', type=str, default='/hy-tmp/sam/sam_pth/sam_vit_l_0b3195.pth', help='Path to SAM checkpoint')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    
    args = parser.parse_args()
    train(args) 