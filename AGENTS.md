# AGENTS.md - 安全帽 YOLO 检测项目

本指南适用于安全帽 YOLO 目标检测项目的编程代理。

## 项目概述

基于 YOLO 的计算机视觉项目，用于在工作环境中检测安全帽。
- **主要任务**: 使用 YOLO (You Only Look Once) 进行安全帽检测
- **框架**: Ultralytics YOLOv8
- **数据结构**: 标准 YOLO 格式，包含训练/验证/测试分割
- **数据规模**: 5000张图片（训练3500张，验证1000张，测试500张）
- **类别**: 单类别 - "hard_hat" (安全帽)
- **标签格式**: YOLO 文本格式，使用归一化坐标
- **项目文件**: train.py, inference.py, validate_dataset.py, data.yaml

## 构建/环境设置命令

```bash
# 安装 YOLOv8 和依赖
pip install ultralytics

# 安装完整的计算机视觉栈
pip install opencv-python pillow matplotlib seaborn
pip install numpy pandas scikit-learn
pip install tqdm pathlib PyYAML

# 验证 YOLOv8 安装
python -c "from ultralytics import YOLO; print('YOLOv8 安装成功')"

# GPU 训练（推荐）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## 训练命令

```bash
# 使用项目训练脚本（推荐）
python train.py --data data.yaml --model yolov8n.pt --epochs 100 --imgsz 640

# 从检查点恢复训练
python train.py --resume runs/detect/train/weights/last.pt

# 使用项目脚本的高级训练
python train.py --data data.yaml --model yolov8s.pt --epochs 200 --imgsz 640 \
  --batch 16 --device 0 --optimizer AdamW --lr 0.01 --augment

# 直接使用 YOLO 命令
yolo detect train data=data.yaml model=yolov8n.pt epochs=100 imgsz=640

# 在测试集上运行验证
python train.py --data data.yaml --model runs/detect/train/weights/best.pt --epochs 1 --val-only
```

## 推理命令

```bash
# 使用项目推理脚本（推荐）
python inference.py --model best.pt --source test_image.jpg --save-results

# 批量图片推理
python inference.py --model best.pt --source test_images/ --save-results --output-dir results

# 视频推理
python inference.py --model best.pt --source test_video.mp4 --save-results

# 摄像头实时检测
python inference.py --model best.pt --source 0 --webcam

# 直接使用 YOLO 命令
yolo detect predict model=best.pt source=test_image.jpg conf=0.25 save=True
```

## 数据管理命令

```bash
# 使用项目验证脚本（推荐）
python validate_dataset.py --data data.yaml

# 验证并修复标签问题
python validate_dataset.py --data data.yaml --fix-labels

# 验证并可视化样本
python validate_dataset.py --data data.yaml --visualize --num-samples 10

# 统计数据集样本
find train/images -name "*.jpg" -o -name "*.png" | wc -l  # 统计训练图片
find val/images -name "*.jpg" -o -name "*.png" | wc -l      # 统计验证图片
find test/images -name "*.jpg" -o -name "*.png" | wc -l     # 统计测试图片

# 手动验证数据集结构
python -c "
import yaml
with open('data.yaml', 'r') as f:
    data = yaml.safe_load(f)
    print('数据集路径:', data)
    print('类别数量:', data['nc'])
    print('类别名称:', data['names'])
"
```

## 测试和验证

```bash
# 使用项目脚本在验证集上测试模型
python inference.py --model best.pt --source val/images/ --save-results --output-dir val_results

# 使用不同 IoU 阈值测试
yolo detect val model=best.pt data=data.yaml iou=0.6

# 生成详细指标报告
yolo detect val model=best.pt data=data.yaml plots=True save_json=True

# 在验证集上可视化预测结果
python inference.py --model best.pt --source val/images/ --save-results

# 单个测试命令
python validate_dataset.py --data data.yaml --visualize --num-samples 3
```

## 测试单个图片

```bash
# 快速测试单张图片
python inference.py --model best.pt --source test/images/hard_hat_workers10.png --save-results

# 测试带不同置信度阈值
python inference.py --model best.pt --source test/images/hard_hat_workers10.png --conf 0.5

# 验证特定标签文件
python -c "
import yaml
with open('data.yaml', 'r') as f:
    config = yaml.safe_load(f)
print('数据集配置:', config)
"

# 测试数据完整性
python validate_dataset.py --data data.yaml --num-samples 1
```

## 代码风格指南

### 导入约定
```python
# 标准库导入在前，按字母顺序
import argparse
import os
import sys
import time
from pathlib import Path
import json
import yaml

# 第三方库导入，按字母顺序
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch

# YOLO 和深度学习
from ultralytics import YOLO
from ultralytics.utils import LOGGER
```

### 命名约定
- **文件**: snake_case (例如: `train_model.py`, `data_loader.py`)
- **变量**: snake_case (例如: `model_path`, `image_size`, `num_epochs`)
- **类**: PascalCase (例如: `DataLoader`, `ModelTrainer`)
- **常量**: UPPER_SNAKE_CASE (例如: `DEFAULT_IMAGE_SIZE`, `MAX_DETECTIONS`)
- **函数**: snake_case 并使用描述性名称 (例如: `load_dataset()`, `preprocess_image()`)
- **私有方法**: 以下划线开头 (例如: `_validate_label()`, `_merge_issues()`)
- **目录**: 小写字母和下划线 (例如: `train_images/`, `model_outputs/`)

### 数据处理模式
```python
# 标准图片加载
def load_image(image_path):
    """加载并验证图片文件"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"图片未找到: {image_path}")
    
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"加载图片失败: {image_path}")
    
    return image

# 带错误处理的标签加载
def load_yolo_labels(label_path):
    """加载 YOLO 格式标签并进行验证"""
    labels = []
    if not os.path.exists(label_path):
        print(f"警告: 标签文件未找到: {label_path}")
        return labels
    
    with open(label_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            try:
                class_id, x_center, y_center, width, height = map(float, line.strip().split())
                labels.append({
                    'class_id': int(class_id),
                    'x_center': x_center,
                    'y_center': y_center,
                    'width': width,
                    'height': height
                })
            except ValueError as e:
                print(f"解析 {label_path} 第 {line_num} 行时出错: {e}")
                
    return labels
```

### 错误处理最佳实践
```python
# 带错误处理的模型训练
def train_model(data_path, model_config):
    """训练 YOLO 模型并包含全面的错误处理"""
    try:
        # 验证输入
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"数据路径未找到: {data_path}")
        
        # 初始化模型
        model = YOLO(model_config.get('model', 'yolov8n.pt'))
        
        # 带监控的训练
        results = model.train(
            data=data_path,
            epochs=model_config.get('epochs', 100),
            imgsz=model_config.get('image_size', 640),
            batch=model_config.get('batch_size', 16)
        )
        
        return results
        
    except Exception as e:
        print(f"训练失败: {str(e)}")
        # 记录错误详情用于调试
        import traceback
        traceback.print_exc()
        return None
```

### 配置管理
```python
# YAML 配置处理
def load_config(config_path):
    """加载 YAML 配置并进行验证"""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # 验证必填字段
        required_fields = ['train', 'val', 'nc', 'names']
        for field in required_fields:
            if field not in config:
                raise ValueError(f"配置中缺少必填字段: {field}")
                
        return config
        
    except Exception as e:
        print(f"加载配置失败: {e}")
        return None
```

## 项目结构

```
SafetyHelmet_YOLO/
├── train/                    # 训练数据 (3500张)
│   ├── images/              # 训练图片
│   └── labels/              # YOLO 格式标签
├── val/                      # 验证数据 (1000张)
│   ├── images/              # 验证图片
│   └── labels/              # YOLO 格式标签
├── test/                     # 测试数据 (500张)
│   ├── images/              # 测试图片
│   └── labels/              # YOLO 格式标签
├── data.yaml                 # 数据集配置文件
├── train.py                 # 训练脚本
├── inference.py             # 推理脚本
├── validate_dataset.py      # 数据集验证脚本
├── AGENTS.md               # 项目开发指南
├── runs/                   # YOLO 训练输出
│   └── detect/           # 检测结果
└── models/                 # 保存的模型权重
    ├── best.pt           # 最佳模型权重
    └── last.pt          # 最后一次检查点
```

## 常见任务

### 数据增强
```python
# 示例增强管道（train.py 中已实现）
def augment_image(image, labels):
    """对训练图片应用数据增强"""
    # 随机翻转
    if np.random.random() > 0.5:
        image = cv2.flip(image, 1)
        # 为翻转图片更新标签
        for label in labels:
            label['x_center'] = 1.0 - label['x_center']
    
    # 随机亮度
    brightness = np.random.uniform(0.8, 1.2)
    image = cv2.convertScaleAbs(image, alpha=brightness, beta=0)
    
    return image, labels
```

### 标签格式转换
```python
# YOLO 转 COCO 格式示例
def yolo_to_coco(yolo_labels, img_width, img_height):
    """将 YOLO 标签转换为 COCO 格式"""
    coco_annotations = []
    
    for label in yolo_labels:
        # 转换归一化坐标
        x_center = label['x_center'] * img_width
        y_center = label['y_center'] * img_height
        width = label['width'] * img_width
        height = label['height'] * img_height
        
        # 转换为左上角坐标
        x_min = int(x_center - width / 2)
        y_min = int(y_center - height / 2)
        
        coco_annotation = {
            'bbox': [x_min, y_min, width, height],
            'category_id': label['class_id'],
            'category_name': 'hard_hat'
        }
        coco_annotations.append(coco_annotation)
    
    return coco_annotations
```

### 模型评估
```python
# 计算 mAP 和其他指标
def evaluate_model(model_path, test_data_path):
    """全面的模型评估"""
    model = YOLO(model_path)
    
    # 运行验证
    results = model.val(
        data=test_data_path,
        imgsz=640,
        batch=16,
        conf=0.25,
        iou=0.6,
        plots=True,
        save_json=True
    )
    
    print(f"mAP50: {results.box.map50:.4f}")
    print(f"mAP50-95: {results.box.map:.4f}")
    print(f"精确度: {results.box.mp:.4f}")
    print(f"召回率: {results.box.mr:.4f}")
    
    return results
```

## 测试指南

1. **数据验证**: 训练前始终验证数据集完整性
2. **小规模测试**: 首先使用数据子集进行测试
3. **性能监控**: 训练期间跟踪损失和指标
4. **交叉验证**: 如果可能，使用多个验证集
5. **错误分析**: 训练后检查误报/漏报情况

## 性能优化

```python
# 推理的批处理
def batch_inference(model, image_paths, batch_size=16):
    """高效的批处理推理"""
    results = []
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
        batch_results = model(batch_paths)
        results.extend(batch_results)
    return results
```

## 部署考虑

1. **模型优化**: 考虑为边缘部署进行量化
2. **输入预处理**: 确保推理时与训练时的预处理匹配
3. **后处理**: 应用置信度阈值和非极大值抑制(NMS)
4. **硬件兼容性**: 在目标部署硬件上测试

## 重要文件和目录

- `data.yaml` - 数据集配置文件
- `train.py` - 主训练脚本（支持命令行参数）
- `inference.py` - 推理脚本（支持图片/视频/摄像头）
- `validate_dataset.py` - 数据集验证和可视化脚本
- `train/` - 训练数据和标签 (3500张)
- `val/` - 验证数据和标签 (1000张)
- `test/` - 测试数据和标签 (500张)
- `runs/detect/` - 训练输出和结果
- `*.pt` - PyTorch 模型权重

## 常见问题解决方案

1. **CUDA 内存不足**: 减少 `--batch` 参数或降低 `--imgsz`
   ```bash
   python train.py --batch 8 --imgsz 512  # 减小批量大小和图像尺寸
   ```

2. **训练结果差**: 使用验证脚本检查数据质量
   ```bash
   python validate_dataset.py --data data.yaml --visualize
   ```

3. **训练缓慢**: 增加工作进程数或使用 SSD
   ```bash
   python train.py --workers 12  # 增加数据加载进程
   ```

4. **推理错误**: 确保模型和输入路径正确
   ```bash
   python inference.py --model best.pt --source test.jpg --conf 0.3
   ```

5. **标签问题**: 使用自动修复功能
   ```bash
   python validate_dataset.py --data data.yaml --fix-labels
   ```

## 性能优化建议

1. **数据加载优化**: 使用 SSD 存储数据集，增加 `--workers` 参数
2. **模型选择**: 根据精度需求选择合适的模型大小 (n/s/m/l/x)
3. **推理优化**: 使用 TensorRT 或 ONNX 进行模型转换
4. **批处理**: 对大量图片使用批处理推理

## 部署指南

1. **边缘设备部署**: 考虑使用 YOLOv8n 模型并进行量化
2. **实时应用**: 设置合适的置信度阈值 (0.25-0.5)
3. **API 部署**: 使用 Flask/FastAPI 封装推理脚本
4. **容器化**: 创建 Docker 镜像便于部署

## 开发环境检查

```bash
# 检查 CUDA 可用性
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

# 检查 GPU 内存
python -c "import torch; print('GPU memory:', torch.cuda.get_device_properties(0).total_memory // 1024**3, 'GB')"