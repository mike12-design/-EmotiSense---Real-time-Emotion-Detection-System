"""
眼睛模型微调脚本 (Fine-tune Eye Model)

只重新训练 ResNet18 的最后一层 (fc)，
让眼睛模型适配你的数据集分布。

用法：
    python finetune_eye_model.py \
        --dataset-path /Users/asahiyang/Downloads/FacialEmotion/dataset \
        --original-model models/paper_style_eye_region_fold12_best.pth \
        --output-model models/eye_model_finetuned.pth \
        --epochs 10

完成后，把 eye_feature_extractor.py 里的 DEFAULT_EYE_MODEL_PATH
改成新模型路径即可。
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from pathlib import Path
from typing import List, Tuple, Optional
import random
import logging
import argparse
from tqdm import tqdm
from sklearn.metrics import classification_report, accuracy_score

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

# 固定随机种子，确保可重现
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# 第一步：从人脸图片中提取眼睛区域
# ============================================================

class EyeRegionExtractor:
    """复用 eye_feature_extractor.py 中相同的提取逻辑，确保一致性"""

    EYE_SIZE = (224, 224)

    def __init__(self):
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )

    def extract(self, face_img_bgr: np.ndarray) -> Optional[np.ndarray]:
        """
        从 224x224 人脸图中提取眼睛区域
        与 EyeFeatureExtractor.extract_eye_region(face_rect=(0,0,224,224)) 逻辑完全一致
        """
        h, w = face_img_bgr.shape[:2]
        gray = cv2.cvtColor(face_img_bgr, cv2.COLOR_BGR2GRAY)

        # 只在上半脸找眼睛（与推理时逻辑一致）
        upper_roi = gray[0:int(h * 0.6), 0:w]

        eyes = self.eye_cascade.detectMultiScale(
            upper_roi,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(20, 20),
            maxSize=(w // 2, int(h * 0.3))
        )

        if len(eyes) == 0:
            return None

        # 取最大的眼睛
        ex, ey, ew, eh = max(eyes, key=lambda r: r[2] * r[3])
        eye_roi = face_img_bgr[ey:ey + eh, ex:ex + ew]

        if eye_roi.size == 0:
            return None

        return cv2.resize(eye_roi, self.EYE_SIZE)


# ============================================================
# 第二步：构建数据集
# ============================================================

class EyeDataset(Dataset):
    """眼睛区域数据集：只用 sad 和 neutral 两类"""

    # 与原模型标签顺序一致：0=neutral, 1=sad
    LABEL_MAP = {'neutral': 0, 'sad': 1}

    def __init__(self, samples: List[Tuple[np.ndarray, int]], augment: bool = False):
        self.samples = samples
        self.augment = augment

        # 与 eye_feature_extractor.py 推理时相同的预处理
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        # 数据增强（仅训练集）
        self.aug_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            transforms.RandomRotation(degrees=10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        eye_bgr, label = self.samples[idx]
        # BGR → RGB
        eye_rgb = cv2.cvtColor(eye_bgr, cv2.COLOR_BGR2RGB)

        if self.augment:
            tensor = self.aug_transform(eye_rgb)
        else:
            tensor = self.transform(eye_rgb)

        return tensor, label


def build_dataset(
    dataset_path: Path,
    samples_per_class: int = 200
) -> Tuple[List, List]:
    """
    从数据集中提取眼睛区域，分为训练集和验证集

    Returns:
        (train_samples, val_samples) 各为 [(eye_img, label), ...]
    """
    extractor = EyeRegionExtractor()
    all_samples = []
    stats = {'neutral': {'found': 0, 'failed': 0},
             'sad':     {'found': 0, 'failed': 0}}

    for class_name, label in EyeDataset.LABEL_MAP.items():
        # 支持大写/小写文件夹名
        folder = dataset_path / class_name.capitalize()
        if not folder.exists():
            folder = dataset_path / class_name
        if not folder.exists():
            logger.error(f"❌ 找不到文件夹：{class_name}")
            continue

        image_files = sorted(
            list(folder.glob('*.jpg')) +
            list(folder.glob('*.png')) +
            list(folder.glob('*.jpeg'))
        )

        # 固定种子采样
        rng = random.Random(SEED)
        sampled = rng.sample(image_files, min(samples_per_class, len(image_files)))
        logger.info(f"📁 {class_name}：从 {len(image_files)} 张中采样 {len(sampled)} 张")

        for img_path in tqdm(sampled, desc=f"提取 {class_name} 眼睛区域"):
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            img = cv2.resize(img, (224, 224))

            eye_img = extractor.extract(img)
            if eye_img is not None:
                all_samples.append((eye_img, label))
                stats[class_name]['found'] += 1
            else:
                stats[class_name]['failed'] += 1

    # 打印统计
    for cls, s in stats.items():
        total = s['found'] + s['failed']
        rate = s['found'] / total * 100 if total > 0 else 0
        logger.info(f"  {cls}: 成功={s['found']}, 失败={s['failed']}, 检测率={rate:.1f}%")

    # 打乱后按 8:2 分割
    rng = random.Random(SEED)
    rng.shuffle(all_samples)
    split = int(len(all_samples) * 0.8)
    train_samples = all_samples[:split]
    val_samples = all_samples[split:]

    logger.info(f"\n✅ 训练集：{len(train_samples)} 张，验证集：{len(val_samples)} 张")
    return train_samples, val_samples


# ============================================================
# 第三步：加载原始模型，冻结除 fc 以外所有层
# ============================================================

def load_and_freeze_model(model_path: Path, device: str) -> nn.Module:
    """
    加载原始 ResNet18 眼睛模型
    解冻策略：layer3 + layer4 + fc
    - 底层特征（edges, textures）保持不变
    - 高层语义特征（layer3/4）针对新分布重新学习
    - 比只训练 fc 更有表达能力，比全量微调更不容易过拟合
    """
    model = models.resnet18(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(model.fc.in_features, 2)
    )

    # 加载原始权重
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    logger.info(f"✅ 原始模型加载成功：{model_path}")

    # 第一步：冻结所有层
    for param in model.parameters():
        param.requires_grad = False

    # 第二步：解冻 layer3、layer4、fc
    # layer1/layer2 保持冻结（低级特征：边缘、纹理，对新数据集依然有效）
    # layer3/layer4 解冻（高级语义特征：需要适配新的 sad/neutral 分布）
    for param in model.layer3.parameters():
        param.requires_grad = True
    for param in model.layer4.parameters():
        param.requires_grad = True
    for param in model.fc.parameters():
        param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"🔓 可训练参数：{trainable:,} / {total:,}（layer3 + layer4 + fc）")

    return model.to(device)


# ============================================================
# 第四步：训练循环
# ============================================================

def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str,
    epochs: int,
    lr: float,
    output_path: Path
):
    """微调训练循环"""

    # 分层学习率：layer3/4 用小 lr 微调，fc 用大 lr 重新适配
    optimizer = torch.optim.Adam([
        {"params": model.layer3.parameters(), "lr": lr * 0.1},
        {"params": model.layer4.parameters(), "lr": lr * 0.1},
        {"params": model.fc.parameters(),     "lr": lr},
    ], weight_decay=1e-4)

    # 学习率调度：验证 loss 不下降时降低 lr
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, verbose=True
    )

    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_epoch = 0

    logger.info(f"\n🚀 开始微调，共 {epochs} 轮，学习率 {lr}")
    logger.info("=" * 60)

    for epoch in range(1, epochs + 1):
        # ── 训练 ──
        model.train()
        train_loss = 0.0
        train_correct = 0

        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} Train", leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * imgs.size(0)
            train_correct += (outputs.argmax(1) == labels).sum().item()

        train_acc = train_correct / len(train_loader.dataset)
        avg_train_loss = train_loss / len(train_loader.dataset)

        # ── 验证 ──
        model.eval()
        val_preds, val_labels = [], []

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device)
                outputs = model(imgs)
                preds = outputs.argmax(1).cpu().numpy()
                val_preds.extend(preds)
                val_labels.extend(labels.numpy())

        val_acc = accuracy_score(val_labels, val_preds)
        scheduler.step(val_acc)

        logger.info(
            f"Epoch {epoch:3d} | "
            f"Train Loss={avg_train_loss:.4f} Acc={train_acc*100:.1f}% | "
            f"Val Acc={val_acc*100:.1f}%"
        )

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save({'model_state_dict': model.state_dict()}, output_path)
            logger.info(f"  💾 新最佳模型已保存 (Val Acc={val_acc*100:.1f}%)")

    logger.info(f"\n✅ 训练完成！最佳验证准确率：{best_val_acc*100:.1f}% (Epoch {best_epoch})")
    logger.info(f"✅ 最佳模型已保存到：{output_path}")

    # 加载最佳模型做最终分类报告
    checkpoint = torch.load(output_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    final_preds, final_labels = [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            final_preds.extend(outputs.argmax(1).cpu().numpy())
            final_labels.extend(labels.numpy())

    print("\n📋 最终验证集分类报告：")
    print(classification_report(
        final_labels, final_preds,
        target_names=['neutral', 'sad'],
        zero_division=0
    ))


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='眼睛模型微调')
    parser.add_argument('--dataset-path', type=str,
                        default='/Users/asahiyang/Downloads/FacialEmotion/dataset',
                        help='数据集根目录（需含 Sad/ 和 Neutral/ 子文件夹）')
    parser.add_argument('--original-model', type=str,
                        default='models/paper_style_eye_region_fold12_best.pth',
                        help='原始眼睛模型路径')
    parser.add_argument('--output-model', type=str,
                        default='models/eye_model_finetuned.pth',
                        help='微调后模型保存路径')
    parser.add_argument('--samples-per-class', type=int, default=200,
                        help='每类最多使用多少张图（默认200）')
    parser.add_argument('--epochs', type=int, default=15,
                        help='训练轮数（默认15）')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='学习率（默认0.001）')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='批大小（默认32）')
    args = parser.parse_args()

    # 设备检测
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    logger.info(f"使用设备：{device}")

    dataset_path = Path(args.dataset_path)
    original_model_path = Path(args.original_model)
    output_path = Path(args.output_model)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. 提取眼睛数据集
    logger.info("\n📦 第一步：提取眼睛区域数据集")
    train_samples, val_samples = build_dataset(dataset_path, args.samples_per_class)

    if len(train_samples) == 0:
        logger.error("❌ 没有提取到任何眼睛样本，请检查数据集路径")
        return

    # 2. 构建 DataLoader
    train_dataset = EyeDataset(train_samples, augment=True)
    val_dataset = EyeDataset(val_samples, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=0)

    # 3. 加载模型并冻结
    logger.info("\n🔧 第二步：加载原始模型并冻结除 fc 外的所有层")
    model = load_and_freeze_model(original_model_path, device)

    # 4. 训练
    logger.info("\n🏋️ 第三步：开始微调")
    train(model, train_loader, val_loader, device,
          epochs=args.epochs, lr=args.lr, output_path=output_path)

    # 5. 提示下一步
    print("\n" + "=" * 60)
    print("✅ 微调完成！下一步：")
    print(f"   在 core/eye_feature_extractor.py 第33行，把：")
    print(f'   DEFAULT_EYE_MODEL_PATH = Path(__file__).parent.parent / "models" / "paper_style_eye_region_fold12_best.pth"')
    print(f"   改成：")
    print(f'   DEFAULT_EYE_MODEL_PATH = Path(__file__).parent.parent / "models" / "eye_model_finetuned.pth"')
    print("   然后重新运行 benchmark 对比效果。")
    print("=" * 60)


if __name__ == '__main__':
    main()