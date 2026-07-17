import os
import json
import warnings
import random
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, random_split, Subset
from torchvision import transforms
import timm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr, pearsonr
from tqdm import tqdm
from PIL import Image

warnings.filterwarnings("ignore", category=UserWarning)

# -----------------------------------------------------------------------------
# 1. Utils
# -----------------------------------------------------------------------------
def set_seed(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# -----------------------------------------------------------------------------
# 2. Dataset
# -----------------------------------------------------------------------------
class UnderwaterIQADataset(Dataset):
    def __init__(self, json_path, image_dir, transform=None, normalize_target=True):
        self.image_dir = image_dir
        self.transform = transform
        self.normalize_target = normalize_target
        with open(json_path, 'r') as f:
            self.annotations = list(json.load(f).items())

        try:
            self.saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
        except AttributeError:
            self.saliency = None

    def get_saliency_map(self, image_np):
        if self.saliency:
            success, map_out = self.saliency.computeSaliency(image_np)
            map_out = (map_out * 255).astype("uint8") if success else np.zeros(image_np.shape[:2], dtype="uint8")
        else:
            h, w = image_np.shape[:2]
            y, x = np.ogrid[:h, :w]
            center_x, center_y = w / 2, h / 2
            map_out = np.exp(-((x - center_x)**2 + (y - center_y)**2) / (2 * (min(h, w) / 2)**2))
            map_out = (map_out * 255).astype("uint8")
        return map_out

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_name, score = self.annotations[idx]
        img_path = os.path.join(self.image_dir, img_name)
        
        image_cv = cv2.imread(img_path)
        if image_cv is None:
            return torch.zeros(3, 256, 256), torch.zeros(1, 256, 256), torch.tensor(0.0, dtype=torch.float32)

        image_cv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
        saliency_map = self.get_saliency_map(image_cv)

        image_pil = Image.fromarray(image_cv)
        saliency_pil = Image.fromarray(saliency_map)

        if self.transform:
            image_tensor = self.transform(image_pil)
            target_size = image_tensor.shape[1:]
            sal_transform = transforms.Compose([
                transforms.Resize(target_size),
                transforms.ToTensor()
            ])
            saliency_tensor = sal_transform(saliency_pil)
        else:
            to_tensor = transforms.ToTensor()
            image_tensor = to_tensor(image_pil)
            saliency_tensor = to_tensor(saliency_pil)

        final_score = score / 100.0 if self.normalize_target else score

        return image_tensor, saliency_tensor, torch.tensor(final_score, dtype=torch.float32)

# -----------------------------------------------------------------------------
# 3. Model
# -----------------------------------------------------------------------------
class SpatialCrossAttention(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x1, x2):
        B, N, C = x1.shape
        q = self.q(x1).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k(x2).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v(x2).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)

class SaliencyGuidedIQAModel(nn.Module):
    def __init__(self, cnn_model_name, swin_model_name, projection_dim=256, pretrained=True):
        super().__init__()
        self.cnn_backbone = timm.create_model(cnn_model_name, pretrained=pretrained, num_classes=0, global_pool='', in_chans=4)
        self.swin_backbone = timm.create_model(swin_model_name, pretrained=pretrained, num_classes=0, global_pool='', in_chans=4)
        
        with torch.no_grad():
            dummy = torch.randn(1, 4, 256, 256)
            cnn_feats = self.cnn_backbone(dummy)
            swin_feats = self.swin_backbone(dummy)
        
        self.cnn_dim = cnn_feats.shape[1]
        
        if swin_feats.shape[-1] == self.swin_backbone.num_features:
            self.swin_permute = True 
            self.swin_dim = swin_feats.shape[-1]
        else:
            self.swin_permute = False
            self.swin_dim = swin_feats.shape[1]
            if len(swin_feats.shape) == 3: 
                self.swin_dim = swin_feats.shape[2]

        self.cnn_proj = nn.Conv2d(self.cnn_dim, projection_dim, kernel_size=1)
        self.swin_proj = nn.Conv2d(self.swin_dim, projection_dim, kernel_size=1)
        self.cross_attention = SpatialCrossAttention(projection_dim, num_heads=4)
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.LayerNorm(projection_dim),
            nn.Linear(projection_dim, 128),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
            nn.Sigmoid() 
        )

    def forward(self, x, saliency_map):
        x_combined = torch.cat([x, saliency_map], dim=1)
        f_cnn = self.cnn_backbone(x_combined)        
        f_swin = self.swin_backbone(x_combined) 
        
        if self.swin_permute:
            f_swin = f_swin.permute(0, 3, 1, 2)
        elif len(f_swin.shape) == 3: 
            B, N, C = f_swin.shape
            side = int(N**0.5)
            f_swin = f_swin.transpose(1, 2).reshape(B, C, side, side)

        f_cnn = self.cnn_proj(f_cnn)
        f_swin = self.swin_proj(f_swin)

        if f_cnn.shape[2:] != f_swin.shape[2:]:
            f_swin = F.interpolate(f_swin, size=f_cnn.shape[2:], mode='bilinear', align_corners=False)

        b, c, h, w = f_cnn.shape
        f_cnn_flat = f_cnn.flatten(2).transpose(1, 2) 
        f_swin_flat = f_swin.flatten(2).transpose(1, 2)
        
        fused = self.cross_attention(f_cnn_flat, f_swin_flat)
        fused_spatial = fused.transpose(1, 2).reshape(b, c, h, w)
        pooled = self.avg_pool(fused_spatial).flatten(1) 
        score = self.head(pooled)
        return score.squeeze(-1)

# -----------------------------------------------------------------------------
# 4. Training Utils
# -----------------------------------------------------------------------------
class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha
        self.l1_loss = nn.L1Loss()

    def forward(self, y_pred, y_true):
        l1 = self.l1_loss(y_pred, y_true)
        vx = y_pred - torch.mean(y_pred)
        vy = y_true - torch.mean(y_true)
        plcc = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)) + 1e-8)
        plcc_loss = 1 - plcc
        return self.alpha * l1 + (1 - self.alpha) * plcc_loss

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.cnn_backbone.parameters():
            param.requires_grad = False
        for param in model.swin_backbone.parameters():
            param.requires_grad = False
    else:
        for param in model.parameters():
            param.requires_grad = True

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    
    progress_bar = tqdm(dataloader, desc="Training", leave=False)
    
    for images, saliency_maps, scores in progress_bar:
        images = images.to(device)
        saliency_maps = saliency_maps.to(device)
        scores = scores.to(device)
        
        optimizer.zero_grad()
        predictions = model(images, saliency_maps)
        loss = criterion(predictions, scores)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        all_preds.extend(predictions.detach().cpu().numpy())
        all_targets.extend(scores.detach().cpu().numpy())
        progress_bar.set_postfix(loss=loss.item())
        
    avg_loss = total_loss / len(dataloader)
    train_plcc = pearsonr(all_preds, all_targets)[0] if len(all_preds) > 1 else 0.0
    return avg_loss, train_plcc

def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_predictions, all_scores = [], []
    
    with torch.no_grad():
        for images, saliency_maps, scores in tqdm(dataloader, desc="Eval", leave=False):
            images = images.to(device)
            saliency_maps = saliency_maps.to(device)
            scores = scores.to(device)
            
            predictions = model(images, saliency_maps)
            loss = criterion(predictions, scores)
            
            total_loss += loss.item()
            all_predictions.extend(predictions.cpu().numpy())
            all_scores.extend(scores.cpu().numpy())
            
    return total_loss / len(dataloader), np.array(all_predictions), np.array(all_scores)

# -----------------------------------------------------------------------------
# 5. Visualization Functions
# -----------------------------------------------------------------------------
class DualInputGradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.feature_map = None
        self.gradient = None
        self.target_layer.register_forward_hook(self._save_feature_map)
        self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_feature_map(self, module, input, output):
        self.feature_map = output.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradient = grad_out[0].detach()

    def __call__(self, x, saliency_map):
        self.model.eval()
        output = self.model(x.unsqueeze(0), saliency_map.unsqueeze(0)).squeeze()
        self.model.zero_grad()
        output.backward(retain_graph=True)
        weights = torch.mean(self.gradient, dim=[2, 3], keepdim=True)
        cam = torch.sum(weights * self.feature_map, dim=1).squeeze(0)
        cam = torch.relu(cam)
        cam = cam.cpu().numpy()
        cam = cv2.resize(cam, (x.shape[2], x.shape[1]))
        cam -= np.min(cam)
        cam /= (np.max(cam) + 1e-8)
        return cam

def save_plots(train_losses, val_losses, train_plccs, val_plccs, predictions, scores):
    # 1. Save Data to CSV
    pd.DataFrame({
        'Epoch': range(1, len(train_losses)+1), 
        'Train_Loss': train_losses, 'Train_PLCC': train_plccs,
        'Val_Loss': val_losses, 'Val_PLCC': val_plccs
    }).to_csv('learning_curve.csv', index=False)
    
    pd.DataFrame({'True': scores, 'Pred': predictions}).to_csv('scatter_data.csv', index=False)

    # 2. Save Loss Curve
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label='Train Loss', linewidth=3, color='green')
    plt.plot(val_losses, label='Val Loss', linewidth=3, color='red')
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.savefig('loss_curve.png')
    plt.close()

    # 3. Save PLCC Curve
    plt.figure(figsize=(8, 5))
    plt.plot(train_plccs, label='Train PLCC', linewidth=3, color='green')
    plt.plot(val_plccs, label='Val PLCC', linewidth=3, color='red')
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('PLCC', fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.savefig('plcc_curve.png')
    plt.close()

    # 4. Save Scatter Plot
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=scores, y=predictions, alpha=0.6)
    
    
    min_val = min(scores.min(), predictions.min())
    max_val = max(scores.max(), predictions.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    plt.xlabel('True Score', fontsize=14)
    plt.ylabel('Predicted Score', fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.savefig('scatter_plot.png')
    plt.close()

def save_gradcam_samples(model, dataset, indices, device, img_size):
    print("Generating Grad-CAM samples...")
    base_model = model.module if isinstance(model, nn.DataParallel) else model
    target_layer = base_model.cnn_backbone.stages[-1].blocks[-1]
    grad_cam = DualInputGradCAM(base_model, target_layer)
    
    save_dir = 'grad_cam_results'
    os.makedirs(save_dir, exist_ok=True)
    
    for idx in indices:
        img_tensor, sal_tensor, _ = dataset[idx]
        
        # Get original filename
        orig_idx = dataset.indices[idx]
        img_name = dataset.dataset.annotations[orig_idx][0]
        img_path = os.path.join(dataset.dataset.image_dir, img_name)
        safe_name = img_name.replace('/', '_').replace('\\', '_')
        
        # Generate CAM
        cam = grad_cam(img_tensor.to(device), sal_tensor.to(device))
        
        # Overlay
        try:
            orig_img = Image.open(img_path).convert('RGB').resize((img_size, img_size))
            img_np = np.array(orig_img) / 255.0
            heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
            heatmap = np.float32(heatmap) / 255
            superimposed = heatmap * 0.4 + img_np
            superimposed = superimposed / np.max(superimposed)
            
            plt.figure(figsize=(5, 5))
            plt.imshow(superimposed)
            plt.axis('off')
            plt.savefig(f"{save_dir}/cam_{safe_name}.png", bbox_inches='tight', pad_inches=0)
            plt.close()
        except Exception as e:
            print(f"Failed to save CAM for {img_name}: {e}")

# -----------------------------------------------------------------------------
# 6. Main Execution
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    # Hyperparameters
    NUM_EPOCHS = 60
    BATCH_SIZE = 16
    LEARNING_RATE = 5e-5 
    IMG_SIZE = 256
    SEED = 42
    WARMUP_EPOCHS = 3
    EARLY_STOPPING_PATIENCE = 8 

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Paths
    data_root = '/kaggle/input/saud-dataset/SAUD_dataset/Enhanced'
    json_path = '/kaggle/input/saud-dataset/SAUD_dataset/saud_dataset.json'

    # Dataset Setup
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    train_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    val_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    if os.path.exists(json_path):
        full_dataset = UnderwaterIQADataset(json_path, data_root, transform=None, normalize_target=True)
        gen = torch.Generator().manual_seed(SEED)
        train_len = int(0.7 * len(full_dataset))
        val_len = int(0.15 * len(full_dataset))
        test_len = len(full_dataset) - train_len - val_len
        
        train_idx, val_idx, test_idx = random_split(range(len(full_dataset)), [train_len, val_len, test_len], generator=gen)

        train_ds = Subset(UnderwaterIQADataset(json_path, data_root, transform=train_transforms, normalize_target=True), train_idx.indices)
        val_ds = Subset(UnderwaterIQADataset(json_path, data_root, transform=val_transforms, normalize_target=True), val_idx.indices)
        test_ds = Subset(UnderwaterIQADataset(json_path, data_root, transform=val_transforms, normalize_target=True), test_idx.indices)

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

        print("Initializing Model...")
        model = SaliencyGuidedIQAModel(
            cnn_model_name='convnext_tiny.in12k_ft_in1k',
            swin_model_name='swinv2_tiny_window8_256.ms_in1k' 
        ).to(device)
        
        if torch.cuda.device_count() > 1: model = nn.DataParallel(model)

        criterion = CombinedLoss(alpha=0.5).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-2)
        scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

        best_val_loss = float('inf')
        early_stopping_counter = 0 
        train_losses, val_losses, train_plccs, val_plccs = [], [], [], []

        print("Starting Training...")
        for epoch in range(NUM_EPOCHS):
            # Warmup
            if epoch < WARMUP_EPOCHS:
                set_parameter_requires_grad(model.module if isinstance(model, nn.DataParallel) else model, feature_extracting=True)
                print(f"Epoch {epoch+1}: Warmup (Backbones Frozen)")
            else:
                set_parameter_requires_grad(model.module if isinstance(model, nn.DataParallel) else model, feature_extracting=False)
                if epoch == WARMUP_EPOCHS: print("Unfreezing Backbones...")

            t_loss, t_plcc = train_one_epoch(model, train_loader, criterion, optimizer, device)
            v_loss, v_preds, v_scores = evaluate_model(model, val_loader, criterion, device)
            v_plcc = pearsonr(v_preds.flatten(), v_scores.flatten())[0] if len(v_preds) > 1 else 0.0
            
            train_losses.append(t_loss)
            val_losses.append(v_loss)
            train_plccs.append(t_plcc)
            val_plccs.append(v_plcc)
            
            scheduler.step()
            
            print(f"Epoch {epoch+1} | T_Loss: {t_loss:.4f} T_PLCC: {t_plcc:.4f} | V_Loss: {v_loss:.4f} V_PLCC: {v_plcc:.4f}")
            
            # --- Early Stopping Logic ---
            if v_loss < best_val_loss:
                best_val_loss = v_loss
                torch.save(model.state_dict(), 'best_model.pth')
                early_stopping_counter = 0 
                print("  [Saved Best Model] - Counter Reset to 0/6")
            else:
                early_stopping_counter += 1
                print(f"  [No Improvement] - Early Stopping Counter: {early_stopping_counter}/{EARLY_STOPPING_PATIENCE}")
                
                if early_stopping_counter >= EARLY_STOPPING_PATIENCE:
                    print("Early Stopping Triggered! Stopping Training.")
                    break
            # ---------------------------

        # Final Test & Visualization
        if os.path.exists('best_model.pth'):
            model.load_state_dict(torch.load('best_model.pth'))
            
        _, preds, truths = evaluate_model(model, test_loader, criterion, device)
        
        # Scale back to 0-100 for final metrics
        preds_scaled = preds * 100
        truths_scaled = truths * 100
        
        rmse = np.sqrt(mean_squared_error(truths_scaled, preds_scaled))
        srocc = spearmanr(preds, truths)[0]
        plcc = pearsonr(preds, truths)[0]
        
        print(f"\nFinal Results (Scaled 0-100):\nRMSE: {rmse:.4f}\nSROCC: {srocc:.4f}\nPLCC: {plcc:.4f}")
        
        # Save All Plots
        print("Saving plots and heatmaps...")
        save_plots(train_losses, val_losses, train_plccs, val_plccs, preds_scaled, truths_scaled)
        
        # Save Grad-CAM Samples
        if len(test_ds) > 0:
            sample_indices = random.sample(range(len(test_ds)), min(4, len(test_ds)))
            save_gradcam_samples(model, test_ds, sample_indices, device, IMG_SIZE)
            
        print("Done. Check current directory for png and csv files.")
    else:
        print("Data path invalid.")