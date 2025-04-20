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
import clip
from pycocotools import mask as mask_utils
from AlphaCLIP import alpha_clip
from functools import partial
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast, GradScaler

import sys
sys.path = [p for p in sys.path if 'uninext/models' not in p]
sys.path.append('/root/autodl-tmp/UNINEXT')

from UNINEXT.projects.UNINEXT.uninext.backbone import ViT

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 自定义 MaskHeadSmallConv
class MaskHeadSmallConv(nn.Module):
    def __init__(self, dim, fpn_dims, context_dim, use_raft=False, up_rate=4):
        super().__init__()
        self.use_raft = use_raft
        if use_raft:
            self.out_stride = up_rate
        else:
            self.out_stride = 2
        self.up_rate = up_rate
        inter_dims = [dim, context_dim, context_dim, context_dim, context_dim, context_dim]

        # 添加适配层，将输入通道从 dim 降到 context_dim
        self.adapter_input = nn.Conv2d(dim, context_dim, kernel_size=1)

        # 修改 lay1 和 lay2 的输入通道数
        self.lay1 = torch.nn.Conv2d(context_dim, dim//4, 3, padding=1)  # 输入通道从 dim 改为 context_dim
        self.lay2 = torch.nn.Conv2d(dim//4, dim//32, 3, padding=1)

        self.lay3 = torch.nn.Conv2d(inter_dims[1], inter_dims[2], 3, padding=1)
        self.lay4 = torch.nn.Conv2d(inter_dims[2], inter_dims[3], 3, padding=1)
        self.jia_dcn = torch.nn.Conv2d(inter_dims[3], inter_dims[4], 3, padding=1)
        self.dim = dim

        # 添加输出适配层
        self.output_conv = nn.Conv2d(dim//32, 1, kernel_size=1)
        self.upsample = nn.Upsample(size=(224, 224), mode='bilinear', align_corners=False)

        if fpn_dims is not None:
            self.adapter1 = torch.nn.Conv2d(fpn_dims[0], inter_dims[1], 1)
            self.adapter2 = torch.nn.Conv2d(fpn_dims[1], inter_dims[2], 1)
            self.adapter3 = torch.nn.Conv2d(fpn_dims[2], inter_dims[3], 1)

        for name, m in self.named_modules():
            if name == "conv_offset":
                nn.init.constant_(m.weight, 0)
                nn.init.constant_(m.bias, 0)
            else:
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_uniform_(m.weight, a=1)
                    nn.init.constant_(m.bias, 0)
        if self.use_raft:
            self.up_mask_layer = nn.Sequential(
                nn.Conv2d(context_dim, context_dim, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(context_dim, self.up_rate*self.up_rate*9, 1, padding=0))

    def forward(self, x, fpns):
        if not isinstance(x, (list, tuple)):
            x = [x] * 3

        # 适配输入通道数
        x = [self.adapter_input(feature) for feature in x]  # 将通道从 768 降到 256

        if fpns is not None:
            cur_fpn = self.adapter1(fpns[0])
            if cur_fpn.size(0) != x[-1].size(0):
                cur_fpn = _expand(cur_fpn, x[-1].size(0) // cur_fpn.size(0))
            fused_x = (cur_fpn + x[-1]) / 2
        else:
            fused_x = x[-1]
        fused_x = self.lay3(fused_x)
        fused_x = F.relu(fused_x)

        if fpns is not None:
            cur_fpn = self.adapter2(fpns[1])
            if cur_fpn.size(0) != x[-2].size(0):
                cur_fpn = _expand(cur_fpn, x[-2].size(0) // cur_fpn.size(0))
            fused_x = (cur_fpn + x[-2]) / 2 + F.interpolate(fused_x, size=cur_fpn.shape[-2:], mode="nearest")
        else:
            fused_x = x[-2] + F.interpolate(fused_x, size=x[-2].shape[-2:], mode="nearest")
        fused_x = self.lay4(fused_x)
        fused_x = F.relu(fused_x)

        if fpns is not None:
            cur_fpn = self.adapter3(fpns[2])
            if cur_fpn.size(0) != x[-3].size(0):
                cur_fpn = _expand(cur_fpn, x[-3].size(0) // cur_fpn.size(0))
            fused_x = (cur_fpn + x[-3]) / 2 + F.interpolate(fused_x, size=cur_fpn.shape[-2:], mode="nearest")
        else:
            fused_x = x[-3] + F.interpolate(fused_x, size=x[-3].shape[-2:], mode="nearest")
        fused_x = self.jia_dcn(fused_x)
        fused_x_fpn = F.relu(fused_x)

        fused_x = self.lay1(fused_x_fpn)
        fused_x = F.relu(fused_x)
        fused_x = self.lay2(fused_x)
        fused_x = F.relu(fused_x)

        fused_x = self.output_conv(fused_x)
        fused_x = self.upsample(fused_x)

        if self.use_raft:
            up_masks = self.up_mask_layer(fused_x_fpn)
            return fused_x, up_masks
        else:
            return fused_x

# 定义 _expand 函数
def _expand(tensor, factor):
    return tensor.repeat(factor, 1, 1, 1)

# 数据集
class RefCocoDataset(Dataset):
    def __init__(self, root="/root/autodl-tmp", split="train"):
        self.image_root = os.path.join(root, "datasets/coco/train/train2014")
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
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
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
            alpha = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
            target_mask = torch.tensor(mask, dtype=torch.float32)
        except Exception as e:
            print(f"Error processing mask for annotation {ann['id']}: {e}")
            alpha = torch.zeros(1, 224, 224, dtype=torch.float32)
            target_mask = torch.zeros(224, 224, dtype=torch.float32)
        return image, alpha, target_mask

# UNINEXT 模型
class UNINEXT(nn.Module):
    def __init__(self):
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
        self.seg_head = MaskHeadSmallConv(
            dim=768,
            fpn_dims=None,
            context_dim=256,
            use_raft=False,
            up_rate=4
        )
        clip_model, _ = clip.load("ViT-B/16", device=device)
        self.clip_model = clip_model
        clip_state_dict = clip_model.visual.state_dict()
        adapted_state_dict = {}
        for k, v in clip_state_dict.items():
            if k == 'conv1.weight':
                adapted_state_dict['patch_embed.proj.weight'] = v
            elif k == 'conv1.bias':
                adapted_state_dict['patch_embed.proj.bias'] = v
            elif k == 'ln_pre.weight':
                adapted_state_dict['pre_norm.weight'] = v
            elif k == 'ln_pre.bias':
                adapted_state_dict['pre_norm.bias'] = v
            elif k.startswith('transformer.resblocks'):
                i = int(k.split('.')[2])
                prefix = f'blocks.{i}.'
                if 'ln_1' in k:
                    adapted_state_dict[prefix + 'norm1.' + k.split('.')[-1]] = v
                elif 'ln_2' in k:
                    adapted_state_dict[prefix + 'norm2.' + k.split('.')[-1]] = v
                elif 'attn.in_proj_weight' in k:
                    qkv_weight = v.view(3, 768, 768).permute(1, 0, 2).reshape(-1, 768)
                    adapted_state_dict[prefix + 'attn.qkv.weight'] = qkv_weight
                elif 'attn.in_proj_bias' in k:
                    qkv_bias = v.view(3, 768)
                    adapted_state_dict[prefix + 'attn.qkv.bias'] = qkv_bias.flatten()
                elif 'attn.out_proj' in k:
                    adapted_state_dict[prefix + 'attn.proj.' + k.split('.')[-1]] = v
                elif 'mlp.c_fc' in k:
                    adapted_state_dict[prefix + 'mlp.fc1.' + k.split('.')[-1]] = v
                elif 'mlp.c_proj' in k:
                    adapted_state_dict[prefix + 'mlp.fc2.' + k.split('.')[-1]] = v
            elif k == 'ln_post.weight':
                adapted_state_dict['norm.weight'] = v
            elif k == 'ln_post.bias':
                adapted_state_dict['norm.bias'] = v
            elif k == 'pos_embed':
                adapted_state_dict['pos_embed'] = v[:, 1:]
        missing, unexpected = self.visual.load_state_dict(adapted_state_dict, strict=False)
        print("Missing keys:", missing)
        print("Unexpected keys:", unexpected)
        self.clip_proj_weight = clip_model.visual.proj

    def forward(self, images):
        features = self.visual(images)
        last_feat = features["res4"] if isinstance(features, dict) else features
        seg_pred = self.seg_head(last_feat, fpns=None)
        return last_feat, seg_pred

# 获取 UNINEXT 特征
def get_uninext_features(model, images, alpha):
    last_feat, _ = model(images)
    batch_size = last_feat.size(0)
    alpha_down = F.avg_pool2d(alpha, kernel_size=16, stride=16)
    alpha_mask = (alpha_down > 0.1).float()
    alpha_mask = alpha_mask.squeeze(1)
    local_features = last_feat.permute(0, 2, 3, 1)
    local_features = local_features.reshape(batch_size, -1, 768)
    alpha_mask = alpha_mask.reshape(batch_size, -1)
    pooled_features = torch.zeros(batch_size, 768, device=device)
    for i in range(batch_size):
        valid_tokens = local_features[i][alpha_mask[i] > 0]
        if valid_tokens.size(0) > 0:
            pooled_features[i] = valid_tokens.mean(dim=0)
        else:
            pooled_features[i] = local_features[i].mean(dim=0)
    del last_feat, alpha_down, alpha_mask, local_features, valid_tokens
    torch.cuda.empty_cache()
    return pooled_features

# 获取 Alpha-CLIP 嵌入
def get_alpha_clip_mask_emb(model, images, alpha):
    with torch.no_grad():
        embedding = model.visual(images, alpha)
    return embedding

# MSE 损失
def mse_loss(x, y):
    return F.mse_loss(x, y)

# 分割损失
def seg_loss(pred, target):
    bce_loss = F.binary_cross_entropy_with_logits(pred, target.unsqueeze(1))
    pred_sigmoid = torch.sigmoid(pred)
    intersection = (pred_sigmoid * target.unsqueeze(1)).sum(dim=(2, 3))
    union = pred_sigmoid.sum(dim=(2, 3)) + target.unsqueeze(1).sum(dim=(2, 3))
    dice_loss = 1 - (2 * intersection + 1e-5) / (union + 1e-5)
    dice_loss = dice_loss.mean()
    return bce_loss + dice_loss

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

    uninext_model = UNINEXT().to(device).float()
    alpha_clip_model, _ = alpha_clip.load(
        "ViT-B/16",
        alpha_vision_ckpt_pth="/root/autodl-tmp/checkpoints/clip_b16_grit+mim_fultune_4xe.pth",
        device=device
    )
    alpha_clip_model.eval()
    alpha_clip_model = alpha_clip_model.float()
    for param in alpha_clip_model.parameters():
        param.requires_grad = False

    image_projection = nn.Linear(768, 512).to(device).float()
    with torch.no_grad():
        image_projection.weight.copy_(uninext_model.clip_proj_weight.t())
        image_projection.bias.zero_()

    # 不冻结 ViT，使用原学习率
    optimizer = torch.optim.Adam([
        {'params': uninext_model.parameters(), 'lr': 1e-5},
        {'params': image_projection.parameters(), 'lr': 1e-4}
    ])
    
    # 添加 CosineAnnealingLR 调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=0)
    scaler = GradScaler()

    num_epochs = 50
    lambda_align = 0.02
    train_mse_loss_history = []
    val_mse_loss_history = []
    train_total_loss_history = []
    val_total_loss_history = []
    val_seg_loss_history = []

    best_val_loss = float('inf')
    patience = 5
    counter = 0
    best_model_path = "best_model.pth"

    for epoch in range(num_epochs):
        uninext_model.train()
        image_projection.train()
        total_train_mse_loss = 0
        total_train_total_loss = 0
        for batch_idx, (images, alpha, target_mask) in enumerate(train_dataloader):
            images = images.to(device).float()
            alpha = alpha.to(device).float()
            target_mask = target_mask.to(device).float()
            
            optimizer.zero_grad()
            with autocast():
                last_feat, seg_pred = uninext_model(images)
                loss_seg = seg_loss(seg_pred, target_mask)
                uninext_features = get_uninext_features(uninext_model, images, alpha)
                uninext_features = image_projection(uninext_features)
                alpha_clip_mask_emb = get_alpha_clip_mask_emb(alpha_clip_model, images, alpha)
                loss_align = mse_loss(uninext_features, alpha_clip_mask_emb)
                loss_total = loss_seg + lambda_align * loss_align
            
            scaler.scale(loss_total).backward()
            torch.nn.utils.clip_grad_norm_(uninext_model.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(image_projection.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            total_train_mse_loss += loss_align.item()
            total_train_total_loss += loss_total.item()
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Train MSE Loss: {loss_align.item()}, Train Total Loss: {loss_total.item()}")
            del images, alpha, target_mask, last_feat, seg_pred, uninext_features, alpha_clip_mask_emb
            torch.cuda.empty_cache()
        
        avg_train_mse_loss = total_train_mse_loss / len(train_dataloader)
        avg_train_total_loss = total_train_total_loss / len(train_dataloader)
        train_mse_loss_history.append(avg_train_mse_loss)
        train_total_loss_history.append(avg_train_total_loss)
        print(f"Epoch {epoch}, Average Train MSE Loss: {avg_train_mse_loss}, Average Train Total Loss: {avg_train_total_loss}")

        uninext_model.eval()
        image_projection.eval()
        total_val_mse_loss = 0
        total_val_total_loss = 0
        total_val_seg_loss = 0
        with torch.no_grad():
            for images, alpha, target_mask in val_dataloader:
                images = images.to(device).float()
                alpha = alpha.to(device).float()
                target_mask = target_mask.to(device).float()
                
                with autocast():
                    last_feat, seg_pred = uninext_model(images)
                    loss_seg = seg_loss(seg_pred, target_mask)
                    uninext_features = get_uninext_features(uninext_model, images, alpha)
                    uninext_features = image_projection(uninext_features)
                    alpha_clip_mask_emb = get_alpha_clip_mask_emb(alpha_clip_model, images, alpha)
                    loss_align = mse_loss(uninext_features, alpha_clip_mask_emb)
                    loss_total = loss_seg + lambda_align * loss_align
                
                total_val_mse_loss += loss_align.item()
                total_val_total_loss += loss_total.item()
                total_val_seg_loss += loss_seg.item()
                print(f"Epoch {epoch}, Val Seg Loss: {loss_seg.item()}, Val Align Loss: {loss_align.item()}, Val Total Loss: {loss_total.item()}")
                del images, alpha, target_mask, last_feat, seg_pred, uninext_features, alpha_clip_mask_emb
                torch.cuda.empty_cache()
        
        avg_val_mse_loss = total_val_mse_loss / len(val_dataloader)
        avg_val_total_loss = total_val_total_loss / len(val_dataloader)
        avg_val_seg_loss = total_val_seg_loss / len(val_dataloader)
        val_mse_loss_history.append(avg_val_mse_loss)
        val_total_loss_history.append(avg_val_total_loss)
        val_seg_loss_history.append(avg_val_seg_loss)
        print(f"Epoch {epoch}, Average Val MSE Loss: {avg_val_mse_loss}, Average Val Seg Loss: {avg_val_seg_loss}, Average Val Total Loss: {avg_val_total_loss}")

        # 学习率调度器更新
        scheduler.step()

        if avg_val_total_loss < best_val_loss:
            best_val_loss = avg_val_total_loss
            counter = 0
            torch.save({
                'uninext_state_dict': uninext_model.state_dict(),
                'projection_state_dict': image_projection.state_dict()
            }, best_model_path)
            print(f"Saved best model at epoch {epoch} with val loss: {best_val_loss}")
        else:
            counter += 1
            print(f"No improvement in val loss, counter: {counter}/{patience}")
            if counter >= patience:
                print("Early stopping triggered")
                break

        torch.save({
            'uninext_state_dict': uninext_model.state_dict(),
            'projection_state_dict': image_projection.state_dict()
        }, f"uninext_epoch_{epoch}.pth")

    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    plt.plot(range(len(train_mse_loss_history)), train_mse_loss_history, marker='o', linestyle='-', color='b', label='Train MSE Loss')
    plt.title('Train MSE Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Average MSE Loss')
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 3, 2)
    plt.plot(range(len(val_mse_loss_history)), val_mse_loss_history, marker='s', linestyle='--', color='r', label='Val MSE Loss')
    plt.title('Val MSE Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Average MSE Loss')
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 3, 3)
    plt.plot(range(len(val_seg_loss_history)), val_seg_loss_history, marker='^', linestyle='-.', color='g', label='Val Seg Loss')
    plt.title('Val Segmentation Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Average Seg Loss')
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 3, 4)
    plt.plot(range(len(train_total_loss_history)), train_total_loss_history, marker='o', linestyle='-', color='b', label='Train Total Loss')
    plt.title('Train Total Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Average Total Loss')
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 3, 5)
    plt.plot(range(len(val_total_loss_history)), val_total_loss_history, marker='s', linestyle='--', color='r', label='Val Total Loss')
    plt.title('Val Total Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Average Total Loss')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig('loss_curves.png')

if __name__ == "__main__":
    main()
