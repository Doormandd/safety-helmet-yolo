#!/usr/bin/env python3
"""
Safety Helmet YOLO Training Script
训练安全帽检测模型的脚本

Usage:
    python train.py --data data.yaml --model yolov8n.pt --epochs 100 --imgsz 640
    python train.py --resume runs/detect/train/weights/last.pt
"""

import argparse
import os
import sys
from pathlib import Path
import yaml

# YOLOv8 imports
try:
    from ultralytics import YOLO
    from ultralytics.utils import LOGGER
except ImportError:
    print("Error: ultralytics package not found. Please install with: pip install ultralytics")
    sys.exit(1)

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.append(str(PROJECT_ROOT))

def load_config(config_path):
    """加载YAML配置文件"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件未找到: {config_path}")
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 验证必填字段
        required_fields = ['train', 'val', 'nc', 'names']
        for field in required_fields:
            if field not in config:
                raise ValueError(f"配置文件中缺少必填字段: {field}")
        
        LOGGER.info(f"成功加载配置: {config_path}")
        LOGGER.info(f"数据集类别数: {config['nc']}")
        LOGGER.info(f"类别名称: {config['names']}")
        
        return config
        
    except Exception as e:
        LOGGER.error(f"加载配置失败: {e}")
        raise

def validate_paths(config):
    """验证数据路径是否存在"""
    paths_to_check = ['train', 'val', 'test']
    for key in paths_to_check:
        if key in config:
            # 获取图像路径
            img_path = config[key].replace('images', 'labels')
            img_path = config[key]  # 原始图像路径
            
            # 转换为绝对路径
            if not os.path.isabs(img_path):
                img_path = os.path.join(PROJECT_ROOT, img_path)
            
            if not os.path.exists(img_path):
                LOGGER.warning(f"路径不存在: {img_path}")
            else:
                # 统计文件数量
                if os.path.isdir(img_path):
                    file_count = len([f for f in os.listdir(img_path) if f.endswith(('.jpg', '.jpeg', '.png'))])
                    LOGGER.info(f"{key} 图片数量: {file_count}")

def train_model(config, args):
    """训练YOLO模型"""
    try:
        # 初始化模型
        model = YOLO(args.model)
        
        # 设置训练参数
        train_params = {
            'data': args.data,
            'epochs': args.epochs,
            'imgsz': args.imgsz,
            'batch': args.batch,
            'device': args.device,
            'workers': args.workers,
            'name': args.name,
            'exist_ok': args.exist_ok,
            'pretrained': args.pretrained,
            'optimizer': args.optimizer,
            'lr0': args.lr,
            'weight_decay': args.weight_decay,
            'warmup_epochs': args.warmup_epochs,
            'patience': args.patience,
            'save_period': args.save_period,
        }
        
        # 添加数据增强参数
        if args.augment:
            train_params.update({
                'hsv_h': 0.015,  # HSV-Hue augmentation
                'hsv_s': 0.7,    # HSV-Saturation augmentation
                'hsv_v': 0.4,    # HSV-Value augmentation
                'degrees': 0.0,    # rotation (+/- deg)
                'translate': 0.1,   # translation (+/- fraction)
                'scale': 0.5,       # scale (+/- gain)
                'shear': 0.0,       # shear (+/- deg)
                'perspective': 0.0,  # perspective (+/- fraction)
                'flipud': 0.0,      # flip up-down (probability)
                'fliplr': 0.5,      # flip left-right (probability)
                'mosaic': 1.0,      # mosaic augmentation (probability)
                'mixup': 0.0,       # mixup augmentation (probability)
            })
        
        LOGGER.info("开始训练...")
        LOGGER.info(f"训练参数: {train_params}")
        
        # 开始训练
        results = model.train(**train_params)
        
        LOGGER.info("训练完成!")
        LOGGER.info(f"最佳模型保存在: {results.save_dir}/weights/best.pt")
        
        return results
        
    except Exception as e:
        LOGGER.error(f"训练失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def resume_training(checkpoint_path):
    """恢复训练"""
    try:
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"检查点文件未找到: {checkpoint_path}")
        
        LOGGER.info(f"从检查点恢复训练: {checkpoint_path}")
        model = YOLO(checkpoint_path)
        
        # 恢复训练
        results = model.train(resume=True)
        
        LOGGER.info("恢复训练完成!")
        return results
        
    except Exception as e:
        LOGGER.error(f"恢复训练失败: {e}")
        return None

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Safety Helmet YOLO 训练脚本')
    
    # 数据和模型参数
    parser.add_argument('--data', type=str, default='data.yaml', help='数据集配置文件路径')
    parser.add_argument('--model', type=str, default='yolov8n.pt', help='预训练模型路径')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--imgsz', type=int, default=640, help='图像尺寸')
    parser.add_argument('--batch', type=int, default=16, help='批量大小')
    parser.add_argument('--device', type=str, default='0', help='训练设备 (cpu, 0, 1, 2, 3)')
    parser.add_argument('--workers', type=int, default=8, help='数据加载器工作进程数')
    
    # 优化器参数
    parser.add_argument('--optimizer', type=str, default='AdamW', help='优化器 (SGD, Adam, AdamW)')
    parser.add_argument('--lr', type=float, default=0.01, help='初始学习率')
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='权重衰减')
    parser.add_argument('--warmup_epochs', type=int, default=3, help='预热轮数')
    
    # 训练控制
    parser.add_argument('--patience', type=int, default=50, help='早停耐心值')
    parser.add_argument('--save_period', type=int, default=-1, help='保存周期 (每多少轮保存一次)')
    
    # 输出控制
    parser.add_argument('--name', type=str, default='train', help='实验名称')
    parser.add_argument('--exist-ok', action='store_true', help='允许覆盖已存在的实验目录')
    parser.add_argument('--pretrained', action='store_true', help='使用预训练权重')
    
    # 数据增强
    parser.add_argument('--augment', action='store_true', help='启用数据增强')
    
    # 恢复训练
    parser.add_argument('--resume', type=str, help='恢复训练的检查点路径')
    
    args = parser.parse_args()
    
    # 恢复训练模式
    if args.resume:
        return resume_training(args.resume)
    
    # 加载和验证配置
    try:
        config = load_config(args.data)
        validate_paths(config)
    except Exception as e:
        LOGGER.error(f"配置验证失败: {e}")
        return 1
    
    # 开始训练
    results = train_model(config, args)
    
    if results is None:
        return 1
    
    return 0

if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)