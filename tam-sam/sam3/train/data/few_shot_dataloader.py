import json
import os
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from sam3.train.utils.train_utils import makedir
from iopath.common.file_io import g_pathmgr

class FewShotSegDataset(Dataset):
    def __init__(self, json_path, img_root, shot=1, transform=None):
        self.shot = shot
        self.transform = transform
        self.img_root = img_root
        # 加载你已划分好的 JSON 标注（包含图像路径、掩码、几何提示）
        with g_pathmgr.open(json_path, "r") as f:
            self.annotations = json.load(f)
        # 按少样本逻辑筛选数据（1-shot 取每个类1个样本，16-shot 取16个）
        self.filter_few_shot_samples()

    def filter_few_shot_samples(self):
        # 假设 JSON 中包含 "category" 字段，按类别筛选少样本
        cat_to_samples = {}
        for ann in self.annotations:
            cat = ann["category"]
            if cat not in cat_to_samples:
                cat_to_samples[cat] = []
            if len(cat_to_samples[cat]) < self.shot:
                cat_to_samples[cat].append(ann)
        # 合并所有类别少样本
        self.few_shot_anns = [ann for cats in cat_to_samples.values() for ann in cats]

    def __len__(self):
        return len(self.few_shot_anns)

    def __getitem__(self, idx):
        ann = self.few_shot_anns[idx]
        img_path = os.path.join(self.img_root, ann["image_path"])
        mask_path = os.path.join(self.img_root, ann["mask_path"])
        # 加载图像和掩码（复用 SAM3 现有图像加载逻辑）
        img = self.load_image(img_path)
        mask = self.load_mask(mask_path)
        # 加载几何提示（论文中用边界框、随机点、粗糙掩码）
        prompts = {
            "box": torch.tensor(ann["bbox"]),
            "points": torch.tensor(ann["points"]),
            "coarse_mask": torch.tensor(ann["coarse_mask"])
        }
        # 应用数据增强（复用 SAM3 现有 transform）
        if self.transform is not None:
            img, mask = self.transform(img, mask)
        return {"img": img, "mask": mask, "prompts": prompts}

    def load_image(self, path):
        # 复用 SAM3 现有图像加载逻辑（参考 sam3/train/data/ 中现有代码）
        from sam3.train.transforms.basic_for_api import load_image
        return load_image(path)

    def load_mask(self, path):
        # 复用 SAM3 现有掩码加载逻辑
        from sam3.train.utils.train_utils import load_mask
        return load_mask(path)

# 构建数据加载器
def build_few_shot_dataloader(cfg, json_path, shot):
    transform = instantiate(cfg.scratch.base_val_transform)  # 复用现有 transform
    dataset = FewShotSegDataset(
        json_path=json_path,
        img_root=cfg.paths.base_annotation_path,  # 从配置中读取图像根目录
        shot=shot,
        transform=transform
    )
    return DataLoader(
        dataset,
        batch_size=cfg.scratch.train_batch_size,
        shuffle=True,
        num_workers=cfg.scratch.num_train_workers
    )
