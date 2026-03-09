#!/usr/bin/env python3
"""
Safety Helmet YOLO Dataset Validation Script
安全帽数据集验证脚本

Usage:
    python validate_dataset.py --data data.yaml
    python validate_dataset.py --data data.yaml --fix-labels
    python validate_dataset.py --data data.yaml --visualize
"""

import argparse
import os
import sys
from pathlib import Path
import yaml
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import random

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.append(str(PROJECT_ROOT))

class DatasetValidator:
    """数据集验证器"""
    
    def __init__(self, data_config_path):
        """
        初始化验证器
        
        Args:
            data_config_path (str): 数据配置文件路径
        """
        self.config = self._load_config(data_config_path)
        self.class_names = self.config['names']
        self.num_classes = self.config['nc']
        
    def _load_config(self, config_path):
        """加载配置文件"""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"配置文件未找到: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        return config
    
    def validate_dataset(self, fix_labels=False, visualize=False, num_samples=5):
        """
        验证数据集
        
        Args:
            fix_labels (bool): 是否修复标签问题
            visualize (bool): 是否可视化样本
            num_samples (int): 可视化样本数量
            
        Returns:
            dict: 验证结果
        """
        results = {
            'total_issues': 0,
            'issues': {
                'missing_images': [],
                'missing_labels': [],
                'invalid_labels': [],
                'empty_labels': [],
                'empty_images': [],
                'size_mismatches': []
            },
            'statistics': {
                'train': {'images': 0, 'labels': 0, 'objects': 0},
                'val': {'images': 0, 'labels': 0, 'objects': 0},
                'test': {'images': 0, 'labels': 0, 'objects': 0}
            }
        }
        
        # 验证各个数据集
        for split in ['train', 'val', 'test']:
            if split in self.config:
                split_results = self._validate_split(split, fix_labels)
                results['statistics'][split] = split_results['statistics']
                results['issues'] = self._merge_issues(results['issues'], split_results['issues'])
                results['total_issues'] += split_results['total_issues']
        
        # 生成报告
        self._print_report(results)
        
        # 可视化样本
        if visualize:
            self._visualize_samples(num_samples)
        
        return results
    
    def _validate_split(self, split, fix_labels):
        """验证数据集分割"""
        results = {
            'total_issues': 0,
            'issues': {
                'missing_images': [],
                'missing_labels': [],
                'invalid_labels': [],
                'empty_labels': [],
                'empty_images': [],
                'size_mismatches': []
            },
            'statistics': {'images': 0, 'labels': 0, 'objects': 0}
        }
        
        # 获取路径
        img_dir = self._get_absolute_path(self.config[split])
        label_dir = img_dir.replace('images', 'labels')
        
        print(f"\n验证 {split} 数据集...")
        print(f"图片目录: {img_dir}")
        print(f"标签目录: {label_dir}")
        
        if not os.path.exists(img_dir):
            print(f"警告: 图片目录不存在: {img_dir}")
            return results
        
        # 获取所有图片文件
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(Path(img_dir).glob(f"*{ext}"))
        
        results['statistics']['images'] = len(image_files)
        
        print(f"找到 {len(image_files)} 张图片")
        
        # 验证每张图片
        for img_file in image_files:
            img_path = str(img_file)
            label_path = str(img_file).replace('images', 'labels').replace(img_file.suffix, '.txt')
            
            # 检查图片
            if not os.path.exists(img_path):
                results['issues']['missing_images'].append(img_path)
                results['total_issues'] += 1
                continue
            
            # 检查标签文件
            if not os.path.exists(label_path):
                results['issues']['missing_labels'].append(label_path)
                results['total_issues'] += 1
                continue
            
            # 验证标签内容
            label_issues = self._validate_label_file(label_path, img_path, fix_labels)
            if label_issues:
                results['issues']['invalid_labels'].extend(label_issues)
                results['total_issues'] += len(label_issues)
            
            # 统计对象数量
            object_count = self._count_objects_in_label(label_path)
            results['statistics']['objects'] += object_count
            
            if object_count > 0:
                results['statistics']['labels'] += 1
            else:
                results['issues']['empty_labels'].append(label_path)
                results['total_issues'] += 1
        
        # 检查标签文件是否有对应的图片
        if os.path.exists(label_dir):
            label_files = list(Path(label_dir).glob('*.txt'))
            for label_file in label_files:
                img_path = str(label_file).replace('labels', 'images').replace('.txt', '.jpg')
                
                # 尝试不同的图片扩展名
                if not os.path.exists(img_path):
                    for ext in ['.jpeg', '.png', '.bmp']:
                        test_path = str(label_file).replace('labels', 'images').replace('.txt', ext)
                        if os.path.exists(test_path):
                            img_path = test_path
                            break
                
                if not os.path.exists(img_path):
                    results['issues']['missing_images'].append(img_path)
                    results['total_issues'] += 1
        
        return results
    
    def _validate_label_file(self, label_path, img_path, fix_labels):
        """验证标签文件"""
        issues = []
        
        try:
            with open(label_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # 检查空文件
            if not lines or not any(line.strip() for line in lines):
                return issues
            
            # 读取图片尺寸
            try:
                img = cv2.imread(img_path)
                if img is None:
                    issues.append(f"无法读取图片: {img_path}")
                    return issues
                
                img_height, img_width = img.shape[:2]
            except Exception as e:
                issues.append(f"读取图片失败 {img_path}: {e}")
                return issues
            
            # 验证每行标签
            fixed_lines = []
            for line_num, line in enumerate(lines, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    parts = line.split()
                    if len(parts) != 5:
                        issues.append(f"{label_path}:{line_num} - 标签格式错误 (应为5个值)")
                        continue
                    
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    
                    # 验证类ID
                    if class_id < 0 or class_id >= self.num_classes:
                        issues.append(f"{label_path}:{line_num} - 无效的类别ID: {class_id}")
                        continue
                    
                    # 验证坐标范围
                    if not (0 <= x_center <= 1 and 0 <= y_center <= 1):
                        issues.append(f"{label_path}:{line_num} - 坐标超出范围 (0-1)")
                        continue
                    
                    if not (0 < width <= 1 and 0 < height <= 1):
                        issues.append(f"{label_path}:{line_num} - 尺寸超出范围 (0-1)")
                        continue
                    
                    # 转换为像素坐标验证
                    pixel_x = int(x_center * img_width)
                    pixel_y = int(y_center * img_height)
                    pixel_w = int(width * img_width)
                    pixel_h = int(height * img_height)
                    
                    x1 = pixel_x - pixel_w // 2
                    y1 = pixel_y - pixel_h // 2
                    x2 = pixel_x + pixel_w // 2
                    y2 = pixel_y + pixel_h // 2
                    
                    if x1 < 0 or y1 < 0 or x2 >= img_width or y2 >= img_height:
                        issues.append(f"{label_path}:{line_num} - 边界框超出图片范围")
                    
                    fixed_lines.append(line)
                    
                except ValueError as e:
                    issues.append(f"{label_path}:{line_num} - 数值转换错误: {e}")
            
            # 修复标签文件
            if fix_labels and not issues and fixed_lines:
                with open(label_path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(fixed_lines))
        
        except Exception as e:
            issues.append(f"读取标签文件失败 {label_path}: {e}")
        
        return issues
    
    def _count_objects_in_label(self, label_path):
        """统计标签文件中的对象数量"""
        try:
            with open(label_path, 'r', encoding='utf-8') as f:
                return len([line for line in f if line.strip()])
        except:
            return 0
    
    def _get_absolute_path(self, path):
        """获取绝对路径"""
        if os.path.isabs(path):
            return path
        return os.path.join(PROJECT_ROOT, path)
    
    def _merge_issues(self, issues1, issues2):
        """合并问题字典"""
        for key in issues2:
            if key in issues1:
                issues1[key].extend(issues2[key])
            else:
                issues1[key] = issues2[key]
        return issues1
    
    def _print_report(self, results):
        """打印验证报告"""
        print("\n" + "="*50)
        print("数据集验证报告")
        print("="*50)
        
        # 统计信息
        print("\n📊 数据集统计:")
        for split, stats in results['statistics'].items():
            print(f"  {split:8}: {stats['images']:4d} 图片, {stats['labels']:4d} 有标签, {stats['objects']:4d} 对象")
        
        # 问题统计
        print(f"\n❌ 发现 {results['total_issues']} 个问题:")
        
        for issue_type, issues in results['issues'].items():
            if issues:
                print(f"  {issue_type}: {len(issues)} 个")
        
        # 详细问题
        if results['total_issues'] > 0:
            print("\n🔍 问题详情:")
            for issue_type, issues in results['issues'].items():
                if issues:
                    print(f"\n  {issue_type}:")
                    for issue in issues[:5]:  # 只显示前5个
                        print(f"    - {issue}")
                    if len(issues) > 5:
                        print(f"    ... 还有 {len(issues) - 5} 个类似问题")
        
        print("\n" + "="*50)
    
    def _visualize_samples(self, num_samples=5):
        """可视化数据集样本"""
        print(f"\n🎨 可视化 {num_samples} 个随机样本...")
        
        for split in ['train', 'val', 'test']:
            if split in self.config:
                img_dir = self._get_absolute_path(self.config[split])
                label_dir = img_dir.replace('images', 'labels')
                
                if not os.path.exists(img_dir):
                    continue
                
                # 获取所有图片
                image_files = []
                for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                    image_files.extend(Path(img_dir).glob(f"*{ext}"))
                
                if not image_files:
                    continue
                
                # 随机选择样本
                sample_files = random.sample(image_files, min(num_samples, len(image_files)))
                
                # 创建可视化
                fig, axes = plt.subplots(1, len(sample_files), figsize=(15, 5))
                if len(sample_files) == 1:
                    axes = [axes]
                
                fig.suptitle(f"{split.upper()} 数据集样本", fontsize=16)
                
                for i, img_file in enumerate(sample_files):
                    img_path = str(img_file)
                    label_path = str(img_file).replace('images', 'labels').replace(img_file.suffix, '.txt')
                    
                    # 读取图片
                    img = cv2.imread(img_path)
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    # 显示图片
                    axes[i].imshow(img_rgb)
                    axes[i].set_title(f"{img_file.name}")
                    axes[i].axis('off')
                    
                    # 绘制边界框
                    if os.path.exists(label_path):
                        with open(label_path, 'r') as f:
                            for line in f:
                                line = line.strip()
                                if not line:
                                    continue
                                
                                try:
                                    parts = line.split()
                                    class_id = int(parts[0])
                                    x_center = float(parts[1])
                                    y_center = float(parts[2])
                                    width = float(parts[3])
                                    height = float(parts[4])
                                    
                                    # 转换为像素坐标
                                    h, w = img.shape[:2]
                                    x = (x_center - width/2) * w
                                    y = (y_center - height/2) * h
                                    rect_width = width * w
                                    rect_height = height * h
                                    
                                    # 绘制矩形
                                    rect = Rectangle((x, y), rect_width, rect_height, 
                                                 linewidth=2, edgecolor='red', facecolor='none')
                                    axes[i].add_patch(rect)
                                    
                                    # 添加标签
                                    if class_id < len(self.class_names):
                                        class_name = self.class_names[class_id]
                                        axes[i].text(x, y-5, class_name, 
                                                    color='red', fontsize=10, 
                                                    bbox=dict(facecolor='white', alpha=0.7))
                                
                                except Exception as e:
                                    print(f"解析标签失败 {label_path}: {e}")
                
                plt.tight_layout()
                
                # 保存可视化
                output_path = os.path.join(PROJECT_ROOT, f"{split}_samples.png")
                plt.savefig(output_path, dpi=150, bbox_inches='tight')
                print(f"  {split} 样本已保存: {output_path}")
                
                plt.close()

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Safety Helmet YOLO 数据集验证脚本')
    
    parser.add_argument('--data', type=str, default='data.yaml', help='数据配置文件路径')
    parser.add_argument('--fix-labels', action='store_true', help='修复标签问题')
    parser.add_argument('--visualize', action='store_true', help='可视化数据集样本')
    parser.add_argument('--num-samples', type=int, default=5, help='可视化样本数量')
    
    args = parser.parse_args()
    
    # 检查配置文件
    if not os.path.exists(args.data):
        print(f"错误: 配置文件不存在: {args.data}")
        return 1
    
    # 执行验证
    try:
        validator = DatasetValidator(args.data)
        results = validator.validate_dataset(
            fix_labels=args.fix_labels,
            visualize=args.visualize,
            num_samples=args.num_samples
        )
        
        # 返回状态码
        return 0 if results['total_issues'] == 0 else 1
        
    except Exception as e:
        print(f"验证失败: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)