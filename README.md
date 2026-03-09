# Safety Helmet YOLO

基于 YOLOv8 的安全帽检测项目。

## 项目概述

本项目提供了完整的深度学习安全帽检测解决方案，包括数据集、训练脚本、推理代码和验证工具。

## 数据集规模

- **总图片数**: 5000张图片
- **训练集**: 3500张图片
- **验证集**: 1000张图片  
- **测试集**: 500张图片
- **标注格式**: YOLO 标准格式 (class_id x_center y_center width height)

## 项目结构

```
SafetyHelmet_YOLO/
├── images/           # 图片文件
│   ├── train/          # 训练图片
│   ├── val/            # 验证图片
│   └── test/           # 测试图片
├── AGENTS.md         # 开发指南
├── data.yaml         # 数据集配置
├── train.py          # 训练脚本
├── inference.py      # 推理脚本
├── validate_dataset.py # 数据验证脚本
├── *.pt            # 预训练模型
└── *.onnx           # ONNX 格式模型
└── *.mp4            # 演示视频
```

## 特点

- ✅ **完整的数据集**：包含训练、验证和测试三个部分
- ✅ **模块化代码**：清晰分离的训练、推理和验证模块
- ✅ **详细的文档**：包含 AGENTS.md 开发指南
- ✅ **多种推理方式**：支持图片、视频和摄像头推理
- ✅ **数据验证**：确保数据集质量

## 使用方法

### 快速开始

```bash
# 克隆项目
git clone https://github.com/Doormandd/SafetyHelmet_YOLO.git

# 安装依赖
pip install -r requirements.txt

# 训练模型
python train.py --epochs 50 --data data.yaml --model yolov8n.pt

# 图片推理
python inference.py --model best.pt --source test.jpg

# 验证数据集
python validate_dataset.py --data data.yaml
```

## 技术特点

- **YOLOv8 支持**：使用最新的 YOLOv8 框架
- **GPU 加速**：支持 CUDA 加速训练
- **多种输出格式**：支持图片、视频、JSON 和标注文件
- **数据增强**：丰富的数据增强策略
- **模型优化**：支持模型量化和剪枝

## 适用场景

- 工业安全监控
- 建筑工地管理
- 矿区巡检
- 无人零售分析

## 许可证

MIT 许可证，允许商业和非商业使用。

## 项目文件

- `data.yaml`: 数据集配置文件
- `train.py`: 训练脚本
- `inference.py`: 推理脚本  
- `validate_dataset.py`: 数据验证脚本
- `AGENTS.md`: 开发指南
- `*.pt`: 预训练模型文件

## 联系

- GitHub: https://github.com/Doormandd/SafetyHelmet-YOLO
- 邮箱: Doormandd@github.com