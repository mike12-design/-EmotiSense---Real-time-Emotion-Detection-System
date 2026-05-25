"""
眼睛区域裁剪脚本 - 仅提取眼部图片

从人脸数据集中提取 Sad 和 Neutral 两类人脸的眼睛区域，
保存为独立图片文件，用于后续分析或训练。

用法：
    python extract_eye_regions.py \
        --dataset-path /Users/asahiyang/Downloads/FacialEmotion/dataset \
        --output-path /Users/asahiyang/Documents/EmotiSense-Real-time-Emotion-Detection-System-main/eye
"""

import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

# 固定随机种子
SEED = 42


class EyeRegionExtractor:
    """眼睛区域提取器 - 与 finetune_eye_model.py 逻辑完全一致"""

    EYE_SIZE = (224, 224)

    def __init__(self):
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )

    def extract(self, face_img_bgr: np.ndarray) -> np.ndarray:
        """
        从 224x224 人脸图中提取眼睛区域
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


def main():
    parser = argparse.ArgumentParser(description='提取眼睛区域图片')
    parser.add_argument('--dataset-path', type=str,
                        default='/Users/asahiyang/Downloads/FacialEmotion/dataset',
                        help='数据集根目录（需含 Sad/ 和 Neutral/ 子文件夹）')
    parser.add_argument('--output-path', type=str,
                        default='/Users/asahiyang/Documents/EmotiSense-Real-time-Emotion-Detection-System-main/eye',
                        help='输出根目录')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='每类最多处理多少张（默认全部）')
    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)
    output_path = Path(args.output_path)
    extractor = EyeRegionExtractor()

    classes = ['Sad', 'Neutral']
    stats = {}

    for cls in classes:
        # 输入文件夹
        src_folder = dataset_path / cls
        if not src_folder.exists():
            src_folder = dataset_path / cls.lower()
        if not src_folder.exists():
            logger.error(f"❌ 找不到源文件夹：{cls}")
            continue

        # 输出文件夹
        dst_folder = output_path / cls
        dst_folder.mkdir(parents=True, exist_ok=True)

        image_files = sorted(
            list(src_folder.glob('*.jpg')) +
            list(src_folder.glob('*.png')) +
            list(src_folder.glob('*.jpeg'))
        )

        if args.max_samples:
            image_files = image_files[:args.max_samples]

        logger.info(f"📁 {cls}：共 {len(image_files)} 张图片")

        success = 0
        failed = 0

        for idx, img_path in enumerate(tqdm(image_files, desc=f"裁剪 {cls}")):
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            img = cv2.resize(img, (224, 224))
            eye_img = extractor.extract(img)

            if eye_img is not None:
                out_name = img_path.stem + f"_{idx:04d}.jpg"
                cv2.imwrite(str(dst_folder / out_name), eye_img)
                success += 1
            else:
                failed += 1

        total = success + failed
        rate = success / total * 100 if total > 0 else 0
        stats[cls] = {'success': success, 'failed': failed, 'rate': rate}

        logger.info(f"  ✅ {cls}：成功={success}, 失败={failed}, 检测率={rate:.1f}%")

    # 总结
    print("\n" + "=" * 60)
    print("✅ 眼睛区域裁剪完成！")
    print("=" * 60)
    for cls, s in stats.items():
        print(f"  {cls}/：{s['success']} 张 (检测率 {s['rate']:.1f}%)")
    print(f"\n输出目录：{output_path}")


if __name__ == '__main__':
    main()
