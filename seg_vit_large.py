import os
import json
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pycocotools import mask as mask_utils
from functools import partial
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast, GradScaler

import sys
sys.path = [p for p in sys.path if 'uninext/models' not in p]
sys.path.append('/root/autodl-tmp/UNINEXT')

from UNINEXT.projects.UNINEXT.uninext.backbone import ViT

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 计算 Rec@0.5 和 IoU
def calculate_metrics(pred_mask, target_mask):
    pred_bin = torch.sigmoid(pred_mask) > 0.5
    target_bin = target_mask > 0.5
    pred_bin = pred_bin.view(-1).cpu().numpy()
    target_bin = target_bin.view(-1).cpu().numpy()
    true_positives = np.sum(pred_bin * target_bin)
    false_negatives = np.sum((1 - pred_bin) * target_bin)
    false_positives = np.sum(pred_bin * (1 - target_bin))
    recall = true_positives / (true_positives + false_negatives + 1e-5)
    intersection = true_positives
    union = true_positives + false_negatives + false_positives
    iou = intersection / (union + 1e-5)
    return recall, iou

# 自定义 MaskHeadSmallConv
class MaskHeadSmallConv(nn.Module):
    def __init__(self, dim=768, context_dim=256):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, context_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(context_dim, context_dim // 2, 3, padding=1)
        self.conv3 = nn.Conv2d(context_dim // 2, 1, 1)
        self.upsample = nn.ConvTranspose2d(1, 1, kernel_size=4, stride=2, padding=1, output_padding=0)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)
    def forward(self, x, fpns=None):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = self.upsample(x)
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        return x

# 数据集
class RefCocoDataset(Dataset):
    def __init__(self, root="/root/autodl-tmp", split="train"):
        self.image_root = os.path.join(root, "datasets/coco/train/train2014/train2014")
        if split == "train":
            self.json_path = os.path.join(root, "datasets/refcoco-mixed/instances_train.json")
        elif split == "val":
            self.json_path = os.path.join(root, "datasets/refcoco-mixed/instances_val.json")
        else:
            raise ValueError("split must be 'train' or 'val'")
        with open(self.json_path, "r") as f:
            self.data = json.load(f)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.valid_samples = []
        image_ids = set(img["id"] for img in self.data["images"])
        for ann in self.data["annotations"]:
            if ann["image_id"] in image_ids:
                self.valid_samples.append(ann)
    def __len__(self):
        return len(self.valid_samples)
    def __getitem__(self, idx):
        ann = self.valid_samples[idx]
        image_id = ann["image_id"]
        image_info = next(img for img in self.data["images"] if img["id"] == image_id)
        image_path = os.path.join(self.image_root, image_info["file_name"])
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        try:
            if isinstance(ann["segmentation"], list):
                rles = mask_utils.frPyObjects(ann["segmentation"], image_info["height"], image_info["width"])
                mask = mask_utils.decode(rles)
            else:
                mask = mask_utils.decode(ann["segmentation"])
            if mask.ndim == 3:
                mask = np.max(mask, axis=2)
            mask = cv2.resize(mask, (224, 224), interpolation=cv2.INTER_NEAREST)
            mask = (mask > 0).astype(np.float32)
            target_mask = torch.tensor(mask, dtype=torch.float32)
        except Exception as e:
            print(f"Error processing mask for annotation {ann['id']}: {e}")
            target_mask = torch.zeros(224, 224, dtype=torch.float32)
        return image, target_mask

# UNINEXT 模型
class UNINEXT(nn.Module):
    def __init__(self, pretrained_path=None):
        super(UNINEXT, self).__init__()
        self.visual = ViT(
            img_size=224,
            patch_size=16,
            in_chans=3,
            embed_dim=768,
            depth=12,
            num_heads=12,
            drop_path_rate=0.1,
            window_size=14,
            mlp_ratio=4,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            window_block_indexes=[0, 1, 3, 4, 6, 7, 9, 10],
            use_rel_pos=True,
            out_feature="last_feat",
            pretrain_img_size=224,
            pretrain_use_cls_token=False
        )
        self.seg_head = MaskHeadSmallConv(dim=768, context_dim=256)
        if pretrained_path and os.path.exists(pretrained_path):
            checkpoint = torch.load(pretrained_path, map_location=device)
            state_dict = checkpoint if not isinstance(checkpoint, dict) else checkpoint.get('state_dict', checkpoint)
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            missing, unexpected = self.load_state_dict(state_dict, strict=False)
            print("Missing keys from pretrained checkpoint:", missing)
            print("Unexpected keys from pretrained checkpoint:", unexpected)
        else:
            print(f"Pretrained checkpoint {pretrained_path} not found, initializing randomly.")
    def forward(self, images):
        features = self.visual(images)
        last_feat = features["res4"] if isinstance(features, dict) else features
        seg_pred = self.seg_head(last_feat)
        return seg_pred

# 分割损失
def seg_loss(pred, target):
    bce_loss = F.binary_cross_entropy_with_logits(pred, target.unsqueeze(1), pos_weight=torch.tensor(5.0, device=device))
    pred_sigmoid = torch.sigmoid(pred)
    intersection = (pred_sigmoid * target.unsqueeze(1)).sum(dim=(2, 3))
    union = pred_sigmoid.sum(dim=(2, 3)) + target.unsqueeze(1).sum(dim=(2, 3))
    dice_loss = 1 - (2 * intersection + 1e-5) / (union + 1e-5)
    dice_loss = dice_loss.mean()
    return bce_loss + 2.0 * dice_loss

# 主函数
def main():
    train_dataset = RefCocoDataset(split="train")
    val_dataset = RefCocoDataset(split="val")
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    pretrained_path = "model_final.pth"
    uninext_model = UNINEXT(pretrained_path=pretrained_path).to(device).float()
    optimizer = torch.optim.Adam(uninext_model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    scaler = GradScaler()
    num_epochs = 50
    train_seg_loss_history = []
    val_seg_loss_history = []
    val_rec_at_0_5_history = []
    val_iou_history = []
    best_val_loss = float('inf')
    patience = 5
    counter = 0
    best_model_path = "best_model.pth"
    for epoch in range(num_epochs):
        uninext_model.train()
        total_train_seg_loss = 0
        for batch_idx, (images, target_mask) in enumerate(train_dataloader):
            images = images.to(device).float()
            target_mask = target_mask.to(device).float()
            optimizer.zero_grad()
            with autocast():
                seg_pred = uninext_model(images)
                loss_seg = seg_loss(seg_pred, target_mask)
            scaler.scale(loss_seg).backward()
            torch.nn.utils.clip_grad_norm_(uninext_model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            total_train_seg_loss += loss_seg.item()
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Train Seg Loss: {loss_seg.item()}")
            del images, target_mask, seg_pred
            torch.cuda.empty_cache()
        avg_train_seg_loss = total_train_seg_loss / len(train_dataloader)
        train_seg_loss_history.append(avg_train_seg_loss)
        print(f"Epoch {epoch}, Average Train Seg Loss: {avg_train_seg_loss}")
        uninext_model.eval()
        total_val_seg_loss = 0
        total_val_rec_at_0_5 = 0
        total_val_iou = 0
        with torch.no_grad():
            for images, target_mask in val_dataloader:
                images = images.to(device).float()
                target_mask = target_mask.to(device).float()
                with autocast():
                    seg_pred = uninext_model(images)
                    loss_seg = seg_loss(seg_pred, target_mask)
                total_val_seg_loss += loss_seg.item()
                rec_at_0_5, iou = calculate_metrics(seg_pred, target_mask)
                total_val_rec_at_0_5 += rec_at_0_5
                total_val_iou += iou
                print(f"Epoch {epoch}, Val Seg Loss: {loss_seg.item()}, Val Rec@0.5: {rec_at_0_5}, Val IoU: {iou}")
                del images, target_mask, seg_pred
                torch.cuda.empty_cache()
        avg_val_seg_loss = total_val_seg_loss / len(val_dataloader)
        avg_val_rec_at_0_5 = total_val_rec_at_0_5 / len(val_dataloader)
        avg_val_iou = total_val_iou / len(val_dataloader)
        val_seg_loss_history.append(avg_val_seg_loss)
        val_rec_at_0_5_history.append(avg_val_rec_at_0_5)
        val_iou_history.append(avg_val_iou)
        print(f"Epoch {epoch}, Average Val Seg Loss: {avg_val_seg_loss}, Average Val Rec@0.5: {avg_val_rec_at_0_5}, Average Val IoU: {avg_val_iou}")
        scheduler.step(avg_val_seg_loss)
        if avg_val_seg_loss < best_val_loss:
            best_val_loss = avg_val_seg_loss
            counter = 0
            torch.save({'uninext_state_dict': uninext_model.state_dict()}, best_model_path)
            print(f"Saved best model at epoch {epoch} with val loss: {best_val_loss}")
        else:
            counter += 1
            print(f"No improvement in val loss, counter: {counter}/{patience}")
            if counter >= patience:
                print("Early stopping triggered")
                break
        torch.save({'uninext_state_dict': uninext_model.state_dict()}, f"uninext_epoch_{epoch}.pth")
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.plot(range(len(train_seg_loss_history)), train_seg_loss_history, marker='o', linestyle='-', color='b', label='Train Seg Loss')
    plt.title('Train Segmentation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Average Seg Loss')
    plt.grid(True)
    plt.legend()
    plt.subplot(1, 3, 2)
    plt.plot(range(len(val_seg_loss_history)), val_seg_loss_history, marker='s', linestyle='--', color='r', label='Val Seg Loss')
    plt.plot(range(len(val_rec_at_0_5_history)), val_rec_at_0_5_history, marker='x', linestyle='-', color='g', label='Val Rec@0.5')
    plt.title('Validation Metrics')
    plt.xlabel('Epoch')
    plt.ylabel('Metrics')
    plt.grid(True)
    plt.legend()
    plt.subplot(1, 3, 3)
    plt.plot(range(len(val_iou_history)), val_iou_history, marker='^', linestyle='-.', color='m', label='Val IoU')
    plt.title('Validation IoU')
    plt.xlabel('Epoch')
    plt.ylabel('Average IoU')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('seg_metrics.png')

if __name__ == "__main__":
    main()
