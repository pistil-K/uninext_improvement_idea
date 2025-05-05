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
from transformers import CLIPProcessor, CLIPModel
import random

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

# 自适应 token 精炼模块 (ATRM, FineLIP)
class AdaptiveTokenRefinementModule(nn.Module):
    def __init__(self, dim=768, reduction_ratio=0.2):
        super().__init__()
        self.dim = dim
        self.reduced_dim = int(dim * 0.5)
        self.query = nn.Linear(dim, self.reduced_dim)
        self.key = nn.Linear(dim, self.reduced_dim)
        self.temperature = nn.Parameter(torch.ones(1) * 0.1)
        self.reduction_ratio = reduction_ratio

    def forward(self, x, num_tokens):
        batch_size, seq_len, feat_dim = x.size()
        assert feat_dim == self.dim, f"Expected feature dim {self.dim}, got {feat_dim}"
        reduced_num = int(num_tokens * self.reduction_ratio)
        
        q = self.query(x)  # [batch_size, seq_len, reduced_dim]
        k = self.key(x)    # [batch_size, seq_len, reduced_dim]
        attn = torch.bmm(q, k.transpose(1, 2)) / self.temperature  # [batch_size, seq_len, seq_len]
        attn = F.softmax(attn, dim=-1)
        
        # 选择最重要的 token
        attn_scores = attn.mean(dim=1)  # [batch_size, seq_len]
        _, indices = attn_scores.topk(reduced_num, dim=-1, largest=True)  # [batch_size, reduced_num]
        indices = indices.sort(dim=-1)[0]  # [batch_size, reduced_num]
        
        # 扩展 indices 以匹配 x 的特征维度
        indices = indices.unsqueeze(-1)  # [batch_size, reduced_num, 1]
        indices = indices.expand(batch_size, reduced_num, self.dim)  # [batch_size, reduced_num, dim]
        
        # 使用 torch.gather 提取精炼 token
        refined_x = torch.gather(x, 1, indices)  # [batch_size, reduced_num, dim]
        # print(f"ATRM output shape: {refined_x.shape}")
        return refined_x

# 跨模态后期交互模块 (CLIM, FineLIP)
class CrossModalLateInteraction(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, visual_tokens, text_tokens):
        visual_norm = F.normalize(visual_tokens, dim=-1)
        text_norm = F.normalize(text_tokens, dim=-1)
        similarity = torch.bmm(visual_norm, text_norm.transpose(1, 2))
        img_to_text = similarity.max(dim=2)[0].mean(dim=1)
        text_to_img = similarity.max(dim=1)[0].mean(dim=1)
        return (img_to_text + text_to_img) / 2

# Token 选择模块（用于 Early Fusion）
class TokenSelector(nn.Module):
    def __init__(self, dim, keep_ratio=0.5):
        super().__init__()
        self.dim = dim
        self.keep_ratio = keep_ratio
        self.score_fc = nn.Linear(dim, 1)
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, tokens):
        batch_size, dim, h, w = tokens.size()
        assert dim == self.dim, f"Expected dim {self.dim}, got {dim}"
        tokens = tokens.permute(0, 2, 3, 1).reshape(batch_size, h * w, dim)
        scores = self.score_fc(self.norm(tokens)).squeeze(-1)
        scores = F.softmax(scores, dim=-1)
        num_keep = int(tokens.size(1) * self.keep_ratio)
        _, indices = scores.topk(num_keep, dim=-1, sorted=True)
        indices = indices.sort(dim=-1)[0]
        selected_tokens = torch.gather(tokens, 1, indices.unsqueeze(-1).expand(-1, -1, self.dim))
        # print(f"TokenSelector output shape: {selected_tokens.shape}")
        return selected_tokens

# 动态查询生成器（用于 Dynamic Queries）
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
        # print(f"QueryGenerator output shape: {queries.shape}")
        return queries

# 分割头
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

# 修改后的数据集，从 images 中提取 expressions
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
        
        # 构建 image_id 到 expressions 的映射
        self.image_to_expressions = {}
        for img in self.data["images"]:
            image_id = img["id"]
            expressions = img.get("expressions", ["Unknown object"])
            self.image_to_expressions[image_id] = expressions
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.tokenizer = CLIPProcessor.from_pretrained("clip-vit-base-patch32")
        self.valid_samples = []
        image_ids = set(img["id"] for img in self.data["images"])
        for ann in self.data["annotations"]:
            if ann["image_id"] in image_ids:
                self.valid_samples.append(ann)
        # print(f"Total valid samples for {split}: {len(self.valid_samples)}")
    
    def __len__(self):
        return len(self.valid_samples)
    
    def __getitem__(self, idx):
        ann = self.valid_samples[idx]
        image_id = ann["image_id"]
        image_info = next(img for img in self.data["images"] if img["id"] == image_id)
        image_path = os.path.join(self.image_root, image_info["file_name"])
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        
        # 加载文本描述（随机选择一个 expression）
        expressions = self.image_to_expressions.get(image_id, ["Unknown object"])
        sentence = random.choice(expressions)
        text_inputs = self.tokenizer(
            text=sentence,
            return_tensors="pt",
            padding="max_length",
            max_length=248,
            truncation=True
        )
        text_ids = text_inputs["input_ids"].squeeze(0)
        # print(f"Selected expression: {sentence}, text_ids shape: {text_ids.shape}")
        
        # 加载掩码
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
        
        return image, text_ids, target_mask

# 修改后的 UNINEXTFineLIP 模型
class UNINEXTFineLIP(nn.Module):
    def __init__(self, pretrained_path=None, fineclip_path=None):
        super(UNINEXTFineLIP, self).__init__()
        # 视觉骨干
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
        # 文本编码器
        self.text_encoder = CLIPModel.from_pretrained("clip-vit-base-patch32").text_model
        self.text_encoder.config.max_position_embeddings = 248
        
        # 扩展位置嵌入
        old_pos_embed = self.text_encoder.embeddings.position_embedding.weight.data
        hidden_size = self.text_encoder.config.hidden_size
        new_pos_embed = nn.Embedding(248, hidden_size).to(device)
        with torch.no_grad():
            new_pos_embed.weight[:20] = old_pos_embed[:20]
            for i in range(20, 77):
                new_pos_embed.weight[20 + (i-20)*4:20 + (i-19)*4] = old_pos_embed[i:i+1]
            new_pos_embed.weight[228:] = old_pos_embed[-1:].expand(20, hidden_size)
        self.text_encoder.embeddings.position_embedding = new_pos_embed
        self.text_encoder.embeddings.position_ids = torch.arange(248).expand((1, -1)).to(device)
        # print(f"New position_embedding num_embeddings: {self.text_encoder.embeddings.position_embedding.num_embeddings}")
        
        # 现有模块
        self.token_selector = TokenSelector(dim=768, keep_ratio=0.5)
        self.query_generator = QueryGenerator(dim=768, num_queries=196, context_dim=256)
        # FineLIP 模块
        self.visual_atrm = AdaptiveTokenRefinementModule(dim=768, reduction_ratio=0.2)
        self.text_atrm = AdaptiveTokenRefinementModule(dim=768, reduction_ratio=0.2)  # 适配投影后的 768 维
        self.text_projection = nn.Linear(512, 768).to(device)  # 512 -> 768
        self.clim = CrossModalLateInteraction()
        self.seg_head = MaskHeadSmallConv(dim=768, context_dim=256)

        # 加载预训练权重
        if pretrained_path and os.path.exists(pretrained_path):
            checkpoint = torch.load(pretrained_path, map_location=device)
            state_dict = checkpoint if not isinstance(checkpoint, dict) else checkpoint.get('state_dict', checkpoint)
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            missing, unexpected = self.load_state_dict(state_dict, strict=False)
            print("Missing keys from pretrained checkpoint:", missing)
            print("Unexpected keys from pretrained checkpoint:", unexpected)
        else:
            print(f"Pretrained checkpoint {pretrained_path} not found, initializing randomly.")

        # 加载 FineCLIP 权重
        if fineclip_path and os.path.exists(fineclip_path):
            fineclip_checkpoint = torch.load(fineclip_path, map_location=device)
            visual_state_dict = {k.replace('visual.', ''): v for k, v in fineclip_checkpoint.items() if k.startswith('visual.')}
            missing, unexpected = self.visual.load_state_dict(visual_state_dict, strict=False)
            print("Missing keys from FineCLIP checkpoint:", missing)
            print("Unexpected keys from FineCLIP checkpoint:", unexpected)
        else:
            print(f"FineCLIP checkpoint {fineclip_path} not found.")

    def forward(self, images, text_ids):
        # 视觉特征
        features = self.visual(images)
        last_feat = features["res4"] if isinstance(features, dict) else features
        # print(f"last_feat shape: {last_feat.shape}")
        
        # Early Fusion
        selected_tokens = self.token_selector(last_feat)
        # print(f"selected_tokens shape: {selected_tokens.shape}")
        
        # Dynamic Queries
        queries = self.query_generator(selected_tokens)
        batch_size = queries.size(0)
        queries = queries.view(batch_size, 196, 768).permute(0, 2, 1).view(batch_size, 768, 14, 14)
        
        # FineLIP: ATRM for visual
        num_visual_tokens = selected_tokens.size(1)
        refined_visual = self.visual_atrm(selected_tokens, num_visual_tokens)
        
        # 文本特征
        text_features = self.text_encoder(input_ids=text_ids).last_hidden_state  # [batch, 248, 512]
        text_features = self.text_projection(text_features)  # [batch, 248, 768]
        # print(f"text_features shape: {text_features.shape}")
        num_text_tokens = text_features.size(1)
        refined_text = self.text_atrm(text_features, num_text_tokens)
        
        # FineLIP: CLIM
        clim_score = self.clim(refined_visual, refined_text)
        
        # 分割预测
        seg_pred = self.seg_head(queries)
        
        return seg_pred, clim_score

# 分割损失
def seg_loss(pred, target):
    bce_loss = F.binary_cross_entropy_with_logits(pred, target.unsqueeze(1), pos_weight=torch.tensor(5.0, device=device))
    pred_sigmoid = torch.sigmoid(pred)
    intersection = (pred_sigmoid * target.unsqueeze(1)).sum(dim=(2, 3))
    union = pred_sigmoid.sum(dim=(2, 3)) + target.unsqueeze(1).sum(dim=(2, 3))
    dice_loss = 1 - (2 * intersection + 1e-5) / (union + 1e-5)
    dice_loss = dice_loss.mean()
    return bce_loss + 2.0 * dice_loss

# 三元组边缘损失
def triplet_loss(clim_score, batch_size):
    margin = 0.2
    positive_scores = clim_score[:batch_size//2]
    negative_scores = clim_score[batch_size//2:]
    loss = F.relu(negative_scores - positive_scores + margin).mean()
    return loss

# 主函数
def main():
    # 调试数据集
    train_dataset = RefCocoDataset(split="train")
    print("Train dataset size:", len(train_dataset))
    if len(train_dataset) == 0:
        raise ValueError("Train dataset is empty, check JSON files and paths")
    val_dataset = RefCocoDataset(split="val")
    print("Val dataset size:", len(val_dataset))
    if len(val_dataset) == 0:
        raise ValueError("Val dataset is empty, check JSON files and paths")
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    pretrained_path = "model_final.pth"
    fineclip_path = "FineCLIP/checkpoints/FineCLIP_coco_vitb16.pt"
    uninext_model = UNINEXTFineLIP(pretrained_path=pretrained_path, fineclip_path=fineclip_path).to(device).float()
    optimizer = torch.optim.Adam(uninext_model.parameters(), lr=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    scaler = GradScaler()
    num_epochs = 20
    train_seg_loss_history = []
    train_triplet_loss_history = []
    val_seg_loss_history = []
    val_triplet_loss_history = []
    val_rec_at_0_5_history = []
    val_iou_history = []
    best_val_loss = float('inf')
    patience = 5
    counter = 0
    best_model_path = "best_model_finelip.pth"

    for epoch in range(num_epochs):
        uninext_model.train()
        total_train_seg_loss = 0
        total_train_triplet_loss = 0
        for batch_idx, (images, text_ids, target_mask) in enumerate(train_dataloader):
            images = images.to(device).float()
            text_ids = text_ids.to(device).long()
            target_mask = target_mask.to(device).float()
            
            # 创建负样本
            batch_size = images.size(0)
            neg_text_ids = torch.roll(text_ids, shifts=batch_size//2, dims=0)
            
            optimizer.zero_grad()
            with autocast():
                # 正样本前向
                seg_pred_pos, clim_score_pos = uninext_model(images, text_ids)
                # 负样本前向
                seg_pred_neg, clim_score_neg = uninext_model(images, neg_text_ids)
                # 拼接 clim_score
                clim_score = torch.cat([clim_score_pos, clim_score_neg], dim=0)
                loss_seg = seg_loss(seg_pred_pos, target_mask)
                loss_triplet = triplet_loss(clim_score, batch_size * 2)
                total_loss = loss_seg + 0.5 * loss_triplet
            
            scaler.scale(total_loss).backward()
            torch.nn.utils.clip_grad_norm_(uninext_model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            total_train_seg_loss += loss_seg.item()
            total_train_triplet_loss += loss_triplet.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Train Seg Loss: {loss_seg.item()}, Train Triplet Loss: {loss_triplet.item()}")
            
            del images, text_ids, target_mask, seg_pred_pos, seg_pred_neg, clim_score
            torch.cuda.empty_cache()
        
        avg_train_seg_loss = total_train_seg_loss / len(train_dataloader)
        avg_train_triplet_loss = total_train_triplet_loss / len(train_dataloader)
        train_seg_loss_history.append(avg_train_seg_loss)
        train_triplet_loss_history.append(avg_train_triplet_loss)
        print(f"Epoch {epoch}, Average Train Seg Loss: {avg_train_seg_loss}, Average Train Triplet Loss: {avg_train_triplet_loss}")
        
        uninext_model.eval()
        total_val_seg_loss = 0
        total_val_triplet_loss = 0
        total_val_rec_at_0_5 = 0
        total_val_iou = 0
        with torch.no_grad():
            for images, text_ids, target_mask in val_dataloader:
                images = images.to(device).float()
                text_ids = text_ids.to(device).long()
                target_mask = target_mask.to(device).float()
                
                batch_size = images.size(0)
                neg_text_ids = torch.roll(text_ids, shifts=batch_size//2, dims=0)
                
                with autocast():
                    seg_pred_pos, clim_score_pos = uninext_model(images, text_ids)
                    seg_pred_neg, clim_score_neg = uninext_model(images, neg_text_ids)
                    clim_score = torch.cat([clim_score_pos, clim_score_neg], dim=0)
                    loss_seg = seg_loss(seg_pred_pos, target_mask)
                    loss_triplet = triplet_loss(clim_score, batch_size * 2)
                
                total_val_seg_loss += loss_seg.item()
                total_val_triplet_loss += loss_triplet.item()
                rec_at_0_5, iou = calculate_metrics(seg_pred_pos, target_mask)
                total_val_rec_at_0_5 += rec_at_0_5
                total_val_iou += iou
                
                del images, text_ids, target_mask, seg_pred_pos, seg_pred_neg, clim_score
                torch.cuda.empty_cache()
        
        avg_val_seg_loss = total_val_seg_loss / len(val_dataloader)
        avg_val_triplet_loss = total_val_triplet_loss / len(val_dataloader)
        avg_val_rec_at_0_5 = total_val_rec_at_0_5 / len(val_dataloader)
        avg_val_iou = total_val_iou / len(val_dataloader)
        val_seg_loss_history.append(avg_val_seg_loss)
        val_triplet_loss_history.append(avg_val_triplet_loss)
        val_rec_at_0_5_history.append(avg_val_rec_at_0_5)
        val_iou_history.append(avg_val_iou)
        print(f"Epoch {epoch}, Average Val Seg Loss: {avg_val_seg_loss}, Average Val Triplet Loss: {avg_val_triplet_loss}, Average Val Rec@0.5: {avg_val_rec_at_0_5}, Average Val IoU: {avg_val_iou}")
        
        total_val_loss = avg_val_seg_loss + 0.5 * avg_val_triplet_loss
        scheduler.step(total_val_loss)
        if total_val_loss < best_val_loss:
            best_val_loss = total_val_loss
            counter = 0
            torch.save({'uninext_state_dict': uninext_model.state_dict()}, best_model_path)
            print(f"Saved best model at epoch {epoch} with val loss: {best_val_loss}")
        else:
            counter += 1
            print(f"No improvement in val loss, counter: {counter}/{patience}")
            if counter >= patience:
                print("Early stopping triggered")
                break
        
        # torch.save({'uninext_state_dict': uninext_model.state_dict()}, f"uninext_finelip_epoch_{epoch}.pth")
    
    plt.figure(figsize=(20, 10))
    plt.subplot(2, 3, 1)
    plt.plot(range(len(train_seg_loss_history)), train_seg_loss_history, marker='o', linestyle='-', color='b', label='Train Seg Loss')
    plt.title('Train Segmentation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Average Seg Loss')
    plt.grid(True)
    plt.legend()
    plt.subplot(2, 3, 2)
    plt.plot(range(len(val_seg_loss_history)), val_seg_loss_history, marker='s', linestyle='--', color='r', label='Val Seg Loss')
    plt.title('Validation Segmentation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Average Seg Loss')
    plt.grid(True)
    plt.legend()
    plt.subplot(2, 3, 3)
    plt.plot(range(len(val_rec_at_0_5_history)), val_rec_at_0_5_history, marker='x', linestyle='-', color='g', label='Val Rec@0.5')
    plt.title('Validation Recall@0.5')
    plt.xlabel('Epoch')
    plt.ylabel('Average Rec@0.5')
    plt.grid(True)
    plt.legend()
    plt.subplot(2, 3, 4)
    plt.plot(range(len(val_iou_history)), val_iou_history, marker='^', linestyle='-.', color='m', label='Val IoU')
    plt.title('Validation IoU')
    plt.xlabel('Epoch')
    plt.ylabel('Average IoU')
    plt.grid(True)
    plt.legend()
    plt.subplot(2, 3, 5)
    plt.plot(range(len(train_triplet_loss_history)), train_triplet_loss_history, marker='o', linestyle='-', color='c', label='Train Triplet Loss')
    plt.title('Train Triplet Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Average Triplet Loss')
    plt.grid(True)
    plt.legend()
    plt.subplot(2, 3, 6)
    plt.plot(range(len(val_triplet_loss_history)), val_triplet_loss_history, marker='s', linestyle='--', color='y', label='Val Triplet Loss')
    plt.title('Validation Triplet Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Average Triplet Loss')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('seg_metrics_finelip.png')

if __name__ == "__main__":
    main()
