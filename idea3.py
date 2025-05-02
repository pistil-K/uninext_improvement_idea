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
from transformers import BertTokenizer, BertModel

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

# 修改后的 TokenSelector（支持图像-文本特征融合）
class TokenSelector(nn.Module):
    def __init__(self, dim, keep_ratio=0.5):
        super().__init__()
        self.dim = dim
        self.keep_ratio = keep_ratio
        self.score_fc = nn.Linear(dim, 1)  # 计算每个 token 的重要性分数
        self.norm = nn.LayerNorm(dim)
        self.fusion_layer = nn.Linear(dim * 2, dim)  # 融合图像和文本特征
    
    def forward(self, img_tokens, text_tokens=None):
        # img_tokens: [batch, dim, h, w]（例如 [128, 768, 14, 14]）
        # text_tokens: [batch, num_text_tokens, dim]（例如 [128, 32, 768]）
        batch_size, dim, h, w = img_tokens.size()
        # 展平图像特征为 token 序列: [batch, num_img_tokens, dim]
        img_tokens = img_tokens.permute(0, 2, 3, 1).reshape(batch_size, h * w, dim)  # [128, 196, 768]
        
        if text_tokens is not None:
            # 融合图像和文本 token
            # 广播 text_tokens 到每个图像 token
            text_tokens = text_tokens.mean(dim=1, keepdim=True)  # [batch, 1, dim]
            text_tokens = text_tokens.expand(-1, img_tokens.size(1), -1)  # [batch, num_img_tokens, dim]
            fused_tokens = torch.cat([img_tokens, text_tokens], dim=-1)  # [batch, num_img_tokens, dim*2]
            fused_tokens = self.fusion_layer(fused_tokens)  # [batch, num_img_tokens, dim]
        else:
            fused_tokens = img_tokens
        
        # 归一化并计算分数
        scores = self.score_fc(self.norm(fused_tokens)).squeeze(-1)  # [batch, num_tokens]
        scores = F.softmax(scores, dim=-1)  # 归一化分数
        
        # 选择 top-k token
        num_keep = int(fused_tokens.size(1) * self.keep_ratio)
        _, indices = scores.topk(num_keep, dim=-1, sorted=True)
        indices = indices.sort(dim=-1)[0]
        
        # 提取选中的 token
        selected_tokens = torch.gather(fused_tokens, 1, indices.unsqueeze(-1).expand(-1, -1, fused_tokens.size(-1)))  # [batch, num_selected, dim]
        
        return selected_tokens

# 动态查询生成器（保持不变）
class QueryGenerator(nn.Module):
    def __init__(self, dim=768, num_queries=196, context_dim=256):
        super().__init__()
        self.num_queries = num_queries
        self.dim = dim
        self.context_dim = context_dim
        self.query_embed = nn.Parameter(torch.randn(1, num_queries, dim))
        self.cross_attn = nn.MultiheadAttention(dim, num_heads=8)
        self.mlp = nn.Sequential(
            nn.Linear(dim, context_dim),
            nn.ReLU(),
            nn.Linear(context_dim, dim)
        )
    
    def forward(self, features):
        batch_size = features.size(0)
        queries = self.query_embed.expand(batch_size, -1, -1)
        queries, _ = self.cross_attn(queries.transpose(0, 1), 
                                   features.transpose(0, 1), 
                                   features.transpose(0, 1))
        queries = queries.transpose(0, 1)
        queries = queries + self.mlp(queries)
        return queries

# 修改后的 UNINEXT 模型
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
        self.text_encoder = BertModel.from_pretrained('bert-base-uncased').to(device)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.token_selector = TokenSelector(dim=768, keep_ratio=0.5)
        self.query_generator = QueryGenerator(dim=768, num_queries=196, context_dim=256)
        self.seg_head = MaskHeadSmallConv(dim=768, context_dim=256)
        if pretrained_path and os.path.exists(pretrained_path):
            checkpoint = torch.load(pretrained_path, map_location=device)
            state_dict = checkpoint if not isinstance(checkpoint, dict) else checkpoint.get('state_dict', checkpoint)
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            missing, unexpected = self.load_state_dict(state_dict, strict=False)
            print("Missing keys:", missing)
            print("Unexpected keys:", unexpected)
        else:
            print(f"Pretrained checkpoint {pretrained_path} not found.")
    
    def forward(self, images, texts):
        # 图像特征
        features = self.visual(images)  # [batch, dim, h, w]
        last_feat = features["res4"] if isinstance(features, dict) else features
        
        # 文本特征
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=32).to(device)
        text_outputs = self.text_encoder(**inputs)
        text_tokens = text_outputs.last_hidden_state  # [batch, num_text_tokens, 768]
        
        # Early Fusion：融合图像和文本特征并选择 token
        selected_tokens = self.token_selector(last_feat, text_tokens)  # [batch, num_selected, dim]
        
        # Dynamic Queries
        queries = self.query_generator(selected_tokens)
        
        # 重塑查询为特征图
        batch_size = queries.size(0)
        queries = queries.view(batch_size, 196, 768).permute(0, 2, 1).view(batch_size, 768, 14, 14)
        seg_pred = self.seg_head(queries)
        
        return seg_pred

# 分割头（保持不变）
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
            if ann["image_id"] in image_ids and "sentence" in ann:
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
        
        # 加载参考表达式
        text = ann["sentence"]
        
        # 处理分割掩码
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
        
        return image, text, target_mask

# 分割损失（保持不变）
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
        pin_memory=True,
        collate_fn=lambda x: tuple(zip(*x))
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        collate_fn=lambda x: tuple(zip(*x))
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
        for batch_idx, (images, texts, target_mask) in enumerate(train_dataloader):
            images = torch.stack(images).to(device).float()
            target_mask = torch.stack(target_mask).to(device).float()
            optimizer.zero_grad()
            with autocast():
                seg_pred = uninext_model(images, texts)
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
            for images, texts, target_mask in val_dataloader:
                images = torch.stack(images).to(device).float()
                target_mask = torch.stack(target_mask).to(device).float()
                with autocast():
                    seg_pred = uninext_model(images, texts)
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
    
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 2, 1)
    plt.plot(range(len(train_seg_loss_history)), train_seg_loss_history, marker='o', linestyle='-', color='b', label='Train Seg Loss')
    plt.title('Train Segmentation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Average Seg Loss')
    plt.grid(True)
    plt.legend()
    plt.subplot(2, 2, 2)
    plt.plot(range(len(val_seg_loss_history)), val_seg_loss_history, marker='s', linestyle='--', color='r', label='Val Seg Loss')
    plt.title('Validation Segmentation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Average Seg Loss')
    plt.grid(True)
    plt.legend()
    plt.subplot(2, 2, 3)
    plt.plot(range(len(val_rec_at_0_5_history)), val_rec_at_0_5_history, marker='x', linestyle='-', color='g', label='Val Rec@0.5')
    plt.title('Validation Recall@0.5')
    plt.xlabel('Epoch')
    plt.ylabel('Average Rec@0.5')
    plt.grid(True)
    plt.legend()
    plt.subplot(2, 2, 4)
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
