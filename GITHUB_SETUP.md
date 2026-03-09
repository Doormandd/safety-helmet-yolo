# Safety Helmet YOLO 项目

这是 Safety Helmet YOLO 项目的 GitHub 仓库设置文件。

## 项目说明

- **项目名称**: Safety Helmet YOLO
- **语言**: Python
- **目的**: 基于YOLOv8的安全帽检测
- **作者**: Doormandd

## 使用说明

### 快速开始

```bash
# 克隆仓库
git clone git@github.com:Doormandd/SafetyHelmet-YOLO.git

# 安装依赖
pip install -r requirements.txt

# 训练模型
python train.py --epochs 50 --data data.yaml --model yolov8n.pt

# 图片推理
python inference.py --model best.pt --source test.jpg

# 验证数据集
python validate_dataset.py --data data.yaml
```

## 项目结构

```
SafetyHelmet_YOLO/
├── AGENTS.md              # 开发指南
├── data.yaml             # 数据集配置
├── train.py              # 训练脚本
├── inference.py           # 推理脚本
├── validate_dataset.py   # 数据验证脚本
├── README.md             # 项目说明
├── images/              # 项目相关图片
├── train/               # 训练数据和标签
├── val/                 # 验证数据和标签
└── test/                # 测试数据和标签
```

## 功能特点

- ✅ **完整的数据集**: 5000张图片，标注完善
- ✅ **训练脚本**: 支持多种训练配置
- ✅ **推理脚本**: 支持图片、视频、摄像头推理
- ✅ **验证工具**: 数据集质量检查
- ✅ **详细文档**: 开发指南完整

## 部署建议

### GitHub Pages 部署

由于演示页面已经包含在 action-hugo 网站中，可以访问：

- **演示地址**: https://Doormandd.github.io/demo/safety-helmet-yolo/
- **博客文章**: https://Doormandd.github.io/posts/safety-helmet-yolo-detection/

### 独立部署

如果需要独立部署，可以考虑：
- 使用 GitHub Pages (需要单独仓库)
- 使用 Vercel/Netlify 等平台
- 部署到自己的服务器

## 技术支持

- **YOLO版本**: YOLOv8
- **框架**: Ultralytics
- **语言**: Python
- **硬件**: 支持GPU/CPU

## 许可证

本项目使用 MIT 许可证。

---

**最后更新**: 2025-03-09
**版本**: 1.0.0