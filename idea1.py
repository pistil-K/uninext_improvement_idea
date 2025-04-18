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

import sys
sys.path = [p for p in sys.path if 'uninext/models' not in p]
sys.path.append('/root/autodl-tmp/UNINEXT')

# 导入 ViT
from UNINEXT.projects.UNINEXT.uninext.backbone import ViT

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据集
class RefCocoDataset(Dataset):
    def __init__(self, root="/root/autodl-tmp"):
        self.image_root = os.path.join(root, "datasets/coco/train/train2014")
        self.json_path = os.path.join(root, "datasets/refcoco-mixed/instances_train.json")
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
            alpha = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
        except Exception as e:
            print(f"Error processing mask for annotation {ann['id']}: {e}")
            alpha = torch.zeros(1, 224, 224, dtype=torch.float32)
        return image, alpha

# UNINEXT 模型（仅视觉编码）
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
        # 初始化 CLIP 权重
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
        # 保存 CLIP 投影层权重
        self.clip_proj_weight = clip_model.visual.proj  # [512, 768]

    def forward(self, images):
        features = self.visual(images)
        last_feat = features["res4"] if isinstance(features, dict) else features
        return last_feat

# 获取 UNINEXT 池化特征（只选择 mask 对应的 local token）
def get_uninext_features(model, images, alpha):
    last_feat = model(images)  # [batch_size, 768, 14, 14]
    batch_size = last_feat.size(0)
    # 下采样 alpha 到 14x14
    alpha_down = F.avg_pool2d(alpha, kernel_size=16, stride=16)  # [batch_size, 1, 14, 14]
    alpha_mask = (alpha_down > 0.1).float()  # 二值化，阈值 0.1
    alpha_mask = alpha_mask.squeeze(1)  # [batch_size, 14, 14]
    # 重塑特征和掩码
    local_features = last_feat.permute(0, 2, 3, 1)  # [batch_size, 14, 14, 768]
    local_features = local_features.reshape(batch_size, -1, 768)  # [batch_size, 196, 768]
    alpha_mask = alpha_mask.reshape(batch_size, -1)  # [batch_size, 196]
    # 选择掩码对应的 token
    pooled_features = torch.zeros(batch_size, 768, device=device)
    for i in range(batch_size):
        valid_tokens = local_features[i][alpha_mask[i] > 0]  # [N, 768]
        if valid_tokens.size(0) > 0:
            pooled_features[i] = valid_tokens.mean(dim=0)  # [768]
        else:
            # 若无有效 token，平均所有 token
            pooled_features[i] = local_features[i].mean(dim=0)
    return pooled_features

# 获取 Alpha-CLIP 嵌入
def get_alpha_clip_mask_emb(model, images, alpha):
    with torch.no_grad():
        embedding = model.visual(images, alpha)  # [batch_size, 512]
    return embedding

# MSE 损失
def mse_loss(x, y):
    # print("UNINEXT features shape:", x.shape)
    # print("Alpha-CLIP mask emb shape:", y.shape)
    return F.mse_loss(x, y)

# 主函数
def main():
    dataset = RefCocoDataset()
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=4)
    # dataloader = DataLoader(dataset, batch_size=10, shuffle=True, num_workers=4)

    uninext_model = UNINEXT().to(device).float()
    alpha_clip_model, _ = alpha_clip.load(
        "ViT-B/16",
        alpha_vision_ckpt_pth="/root/autodl-tmp/checkpoints/clip_b16_grit+mim_fultune_4xe.pth",
        device=device
    )
    alpha_clip_model.eval()
    for param in alpha_clip_model.parameters():
        param.requires_grad = False
    alpha_clip_model = alpha_clip_model.float()

    # 初始化投影层，使用 CLIP 投影权重
    image_projection = nn.Linear(768, 512).to(device).float()
    with torch.no_grad():
        image_projection.weight.copy_(uninext_model.clip_proj_weight.t())  # 转置 [512, 768] → [768, 512]
        image_projection.bias.zero_()  # 偏置初始化为零

    optimizer = torch.optim.Adam(
        list(uninext_model.parameters()) + list(image_projection.parameters()),
        lr=1e-5
    )

    num_epochs = 20
    loss_history = []  # 记录每个 epoch 的平均 loss

    for epoch in range(num_epochs):
        uninext_model.train()
        total_loss = 0
        for batch_idx, (images, alpha) in enumerate(dataloader):
            images = images.to(device).float()
            alpha = alpha.to(device).float()
            uninext_features = get_uninext_features(uninext_model, images, alpha)  # [batch_size, 768]
            uninext_features = image_projection(uninext_features)  # [batch_size, 512]
            alpha_clip_mask_emb = get_alpha_clip_mask_emb(alpha_clip_model, images, alpha)  # [batch_size, 512]
            loss_align = mse_loss(uninext_features, alpha_clip_mask_emb)
            optimizer.zero_grad()
            loss_align.backward()
            optimizer.step()
            total_loss += loss_align.item()
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, MSE Loss: {loss_align.item()}")
        avg_loss = total_loss / len(dataloader)
        loss_history.append(avg_loss)  # 记录平均 loss
        print(f"Epoch {epoch}, Average MSE Loss: {avg_loss}")
        torch.save({
            'uninext_state_dict': uninext_model.state_dict(),
            'projection_state_dict': image_projection.state_dict()
        }, f"uninext_epoch_{epoch}.pth")

    # 绘制 Loss 曲线
    plt.figure(figsize=(10, 5))
    plt.plot(range(num_epochs), loss_history, marker='o', linestyle='-', color='b', label='Average MSE Loss')
    plt.title('Training Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Average MSE Loss')
    plt.grid(True)
    plt.legend()
    plt.savefig('loss_curve.png')  
    plt.show() 

if __name__ == "__main__":
    main()