"""
YOLOv26x 训练脚本 (GPU版) - 安全帽检测
专为GPU环境优化配置，支持更大batch size和并行数据加载
"""
import time
import json
from pathlib import Path
from datetime import datetime
import torch
from ultralytics import YOLO
import psutil
import GPUtil

class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.metrics = {
            'model': 'YOLOv26x-GPU',
            'training': {},
            'system': {}
        }
    
    def start(self):
        """开始监控"""
        self.start_time = time.time()
        self.metrics['training']['start_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    def end(self):
        """结束监控"""
        self.end_time = time.time()
        self.metrics['training']['end_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.metrics['training']['total_time_seconds'] = self.end_time - self.start_time
        self.metrics['training']['total_time_hours'] = (self.end_time - self.start_time) / 3600
    
    def record_system_info(self):
        """记录系统信息"""
        # CPU信息
        self.metrics['system']['cpu_count'] = psutil.cpu_count()
        self.metrics['system']['cpu_percent'] = psutil.cpu_percent(interval=1)
        
        # 内存信息
        memory = psutil.virtual_memory()
        self.metrics['system']['memory_total_gb'] = memory.total / (1024**3)
        self.metrics['system']['memory_available_gb'] = memory.available / (1024**3)
        self.metrics['system']['memory_percent'] = memory.percent
        
        # GPU信息
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                for idx, gpu in enumerate(gpus):
                    self.metrics['system'][f'gpu_{idx}_name'] = gpu.name
                    self.metrics['system'][f'gpu_{idx}_memory_total_gb'] = gpu.memoryTotal / 1024
                    self.metrics['system'][f'gpu_{idx}_memory_used_gb'] = gpu.memoryUsed / 1024
                    self.metrics['system'][f'gpu_{idx}_load_percent'] = gpu.load * 100
        except:
            self.metrics['system']['gpu_available'] = False
        
        # PyTorch信息
        self.metrics['system']['cuda_available'] = torch.cuda.is_available()
        if torch.cuda.is_available():
            self.metrics['system']['cuda_version'] = torch.version.cuda
            self.metrics['system']['gpu_count'] = torch.cuda.device_count()
            self.metrics['system']['gpu_name_torch'] = torch.cuda.get_device_name(0)
    
    def save(self, output_path):
        """保存监控结果"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.metrics, f, indent=4, ensure_ascii=False)
        print(f"\n性能指标已保存到: {output_path}")

def train_yolo26x_gpu():
    """训练YOLOv26x模型 (GPU模式)"""
    
    print("="*60)
    print("YOLOv26x 安全帽检测模型训练 (GPU模式)")
    print("="*60)
    
    # 初始化性能监控器
    monitor = PerformanceMonitor()
    monitor.record_system_info()
    
    # 配置参数 - GPU优化
    config = {
        'model': 'yolo26x.pt',  # 最新版本YOLO模型
        'data': 'dataset.yaml', # 使用相对路径，便于迁移
        'epochs': 100,
        'imgsz': 640,
        'batch': 16,     # GPU建议16或32
        'device': '0',   # 指定使用第一块GPU
        'workers': 8,    # 增加数据加载线程
        'project': 'runs/train',
        'name': 'yolo26x_helmet_gpu',
        'exist_ok': True,
        'pretrained': True,
        'optimizer': 'auto',
        'lr0': 0.01,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'patience': 50,
        'save': True,
        'save_period': 10,
        'cache': True,
        'amp': True,
        'verbose': True,
        'plots': True,
    }
    
    # 记录配置
    monitor.metrics['training']['config'] = config
    
    print(f"\n训练配置:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # 检查GPU状态
    if not torch.cuda.is_available():
        print("\n警告: 未检测到CUDA环境！将回退到CPU模式，但配置可能不适合CPU。")
        response = input("是否继续? (y/n): ")
        if response.lower() != 'y':
            return None, None
        config['device'] = 'cpu'
        config['batch'] = 4
        config['workers'] = 2
    else:
        print(f"\nGPU就绪: {torch.cuda.get_device_name(0)}")
        print(f"显存可用: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # 加载模型
    print(f"\n加载模型: {config['model']}")
    model = YOLO(config['model'])
    
    # 开始训练
    print("\n开始训练...")
    monitor.start()
    
    try:
        results = model.train(**config)
        
        # 训练完成
        monitor.end()
        
        # 记录训练结果
        print("\n训练完成!")
        print(f"总耗时: {monitor.metrics['training']['total_time_hours']:.2f} 小时")
        
        # 验证模型
        print("\n开始验证...")
        val_results = model.val()
        
        # 记录验证指标
        monitor.metrics['validation'] = {
            'mAP50': float(val_results.box.map50),
            'mAP50-95': float(val_results.box.map),
            'precision': float(val_results.box.mp),
            'recall': float(val_results.box.mr),
        }
        
        print(f"\n验证结果:")
        print(f"  mAP@0.5: {monitor.metrics['validation']['mAP50']:.4f}")
        print(f"  mAP@0.5:0.95: {monitor.metrics['validation']['mAP50-95']:.4f}")
        
    except Exception as e:
        print(f"\n训练过程中出错: {str(e)}")
        monitor.end()
        monitor.metrics['error'] = str(e)
        raise
    
    finally:
        # 保存性能指标
        output_dir = Path(config['project']) / config['name']
        output_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = output_dir / 'performance_metrics.json'
        monitor.save(metrics_path)
    
    return model, monitor.metrics

if __name__ == '__main__':
    train_yolo26x_gpu()
