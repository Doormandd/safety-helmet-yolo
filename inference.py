#!/usr/bin/env python3
"""
Safety Helmet YOLO Inference Script
安全帽检测推理脚本

Usage:
    python inference.py --model best.pt --source test.jpg
    python inference.py --model best.pt --source images/ --save-results
    python inference.py --model best.pt --source 0 --webcam  # 摄像头实时检测
"""

import argparse
import os
import sys
import time
from pathlib import Path
import cv2
import numpy as np

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

class SafetyHelmetDetector:
    """安全帽检测器类"""
    
    def __init__(self, model_path, conf_threshold=0.25, iou_threshold=0.45):
        """
        初始化检测器
        
        Args:
            model_path (str): 模型文件路径
            conf_threshold (float): 置信度阈值
            iou_threshold (float): IOU阈值
        """
        try:
            self.model = YOLO(model_path)
            self.conf_threshold = conf_threshold
            self.iou_threshold = iou_threshold
            
            LOGGER.info(f"模型加载成功: {model_path}")
            LOGGER.info(f"置信度阈值: {conf_threshold}")
            LOGGER.info(f"IOU阈值: {iou_threshold}")
            
        except Exception as e:
            LOGGER.error(f"模型加载失败: {e}")
            raise
    
    def detect_image(self, image_path, save_path=None):
        """
        检测单张图片
        
        Args:
            image_path (str): 图片路径
            save_path (str, optional): 保存路径
            
        Returns:
            tuple: (image, results) 处理后的图片和检测结果
        """
        try:
            # 读取图片
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"无法读取图片: {image_path}")
            
            # 进行检测
            results = self.model(
                image_path,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                verbose=False
            )
            
            # 处理检测结果
            annotated_image = self._draw_results(image, results[0])
            
            # 保存结果
            if save_path:
                cv2.imwrite(save_path, annotated_image)
                LOGGER.info(f"结果已保存: {save_path}")
            
            return annotated_image, results[0]
            
        except Exception as e:
            LOGGER.error(f"图片检测失败: {e}")
            return None, None
    
    def detect_video(self, video_path, save_path=None):
        """
        检测视频
        
        Args:
            video_path (str): 视频路径
            save_path (str, optional): 保存路径
            
        Returns:
            bool: 是否成功
        """
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"无法打开视频: {video_path}")
            
            # 获取视频信息
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # 设置视频写入器
            writer = None
            if save_path:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(save_path, fourcc, fps, (width, height))
            
            frame_count = 0
            helmet_count = 0
            
            LOGGER.info(f"开始处理视频: {video_path}")
            LOGGER.info(f"视频信息: {width}x{height}, {fps} FPS")
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 检测
                results = self.model(
                    frame,
                    conf=self.conf_threshold,
                    iou=self.iou_threshold,
                    verbose=False
                )
                
                # 绘制结果
                annotated_frame = self._draw_results(frame, results[0])
                
                # 保存帧
                if writer:
                    writer.write(annotated_frame)
                
                frame_count += 1
                helmet_count += len(results[0].boxes)
                
                # 显示进度
                if frame_count % 30 == 0:
                    LOGGER.info(f"已处理 {frame_count} 帧, 检测到 {helmet_count} 个安全帽")
                
                # 显示实时结果
                cv2.imshow('Safety Helmet Detection', annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # 释放资源
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()
            
            LOGGER.info(f"视频处理完成: {frame_count} 帧, 共检测到 {helmet_count} 个安全帽")
            
            return True
            
        except Exception as e:
            LOGGER.error(f"视频检测失败: {e}")
            return False
    
    def detect_webcam(self, camera_index=0):
        """
        摄像头实时检测
        
        Args:
            camera_index (int): 摄像头索引
            
        Returns:
            bool: 是否成功
        """
        try:
            cap = cv2.VideoCapture(camera_index)
            if not cap.isOpened():
                raise ValueError(f"无法打开摄像头: {camera_index}")
            
            LOGGER.info("开始摄像头实时检测 (按 'q' 退出)")
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 检测
                results = self.model(
                    frame,
                    conf=self.conf_threshold,
                    iou=self.iou_threshold,
                    verbose=False
                )
                
                # 绘制结果
                annotated_frame = self._draw_results(frame, results[0])
                
                # 显示结果
                cv2.imshow('Safety Helmet Detection', annotated_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # 释放资源
            cap.release()
            cv2.destroyAllWindows()
            
            return True
            
        except Exception as e:
            LOGGER.error(f"摄像头检测失败: {e}")
            return False
    
    def _draw_results(self, image, results):
        """
        在图片上绘制检测结果
        
        Args:
            image: 原始图片
            results: YOLO检测结果
            
        Returns:
            image: 标注后的图片
        """
        annotated_image = image.copy()
        
        if results is None or len(results.boxes) == 0:
            return annotated_image
        
        # 获取检测框
        boxes = results.boxes
        
        for i, box in enumerate(boxes):
            # 获取坐标和置信度
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            confidence = box.conf[0].cpu().numpy()
            class_id = int(box.cls[0].cpu().numpy())
            
            # 转换为整数
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # 绘制边界框
            color = (0, 255, 0) if class_id == 0 else (0, 0, 255)  # 安全帽用绿色
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
            
            # 绘制标签
            label = f"Safety Hat: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(
                annotated_image,
                (x1, y1 - label_size[1] - 10),
                (x1 + label_size[0], y1),
                color,
                -1
            )
            cv2.putText(
                annotated_image,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2
            )
        
        return annotated_image

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Safety Helmet YOLO 推理脚本')
    
    # 模型和输入参数
    parser.add_argument('--model', type=str, required=True, help='模型文件路径 (.pt)')
    parser.add_argument('--source', type=str, required=True, help='输入源 (图片/视频/摄像头索引)')
    
    # 检测参数
    parser.add_argument('--conf', type=float, default=0.25, help='置信度阈值')
    parser.add_argument('--iou', type=float, default=0.45, help='IOU阈值')
    
    # 输出参数
    parser.add_argument('--save-results', action='store_true', help='保存检测结果')
    parser.add_argument('--output-dir', type=str, default='results', help='输出目录')
    
    # 特殊模式
    parser.add_argument('--webcam', action='store_true', help='摄像头模式')
    
    args = parser.parse_args()
    
    # 检查模型文件
    if not os.path.exists(args.model):
        LOGGER.error(f"模型文件不存在: {args.model}")
        return 1
    
    # 创建输出目录
    if args.save_results and not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # 初始化检测器
    try:
        detector = SafetyHelmetDetector(
            model_path=args.model,
            conf_threshold=args.conf,
            iou_threshold=args.iou
        )
    except Exception as e:
        LOGGER.error(f"检测器初始化失败: {e}")
        return 1
    
    # 执行检测
    success = False
    
    if args.webcam or args.source.isdigit():
        # 摄像头检测
        camera_index = int(args.source) if args.source.isdigit() else 0
        success = detector.detect_webcam(camera_index)
        
    elif os.path.isfile(args.source):
        # 单个文件检测
        file_ext = os.path.splitext(args.source)[1].lower()
        
        if file_ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            # 图片检测
            save_path = None
            if args.save_results:
                filename = os.path.basename(args.source)
                name, ext = os.path.splitext(filename)
                save_path = os.path.join(args.output_dir, f"{name}_result{ext}")
            
            _, results = detector.detect_image(args.source, save_path)
            
            if results is not None:
                helmet_count = len(results.boxes)
                LOGGER.info(f"检测完成，发现 {helmet_count} 个安全帽")
                success = True
                
        elif file_ext in ['.mp4', '.avi', '.mov', '.mkv']:
            # 视频检测
            save_path = None
            if args.save_results:
                filename = os.path.basename(args.source)
                name, ext = os.path.splitext(filename)
                save_path = os.path.join(args.output_dir, f"{name}_result{ext}")
            
            success = detector.detect_video(args.source, save_path)
            
        else:
            LOGGER.error(f"不支持的文件格式: {file_ext}")
            
    elif os.path.isdir(args.source):
        # 目录批量检测
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            image_files.extend(Path(args.source).glob(f"*{ext}"))
        
        if not image_files:
            LOGGER.error(f"目录中未找到图片文件: {args.source}")
            return 1
        
        LOGGER.info(f"开始批量检测 {len(image_files)} 张图片")
        
        total_helmets = 0
        for i, image_file in enumerate(image_files):
            save_path = None
            if args.save_results:
                save_path = os.path.join(args.output_dir, f"{image_file.stem}_result{image_file.suffix}")
            
            _, results = detector.detect_image(str(image_file), save_path)
            
            if results is not None:
                helmet_count = len(results.boxes)
                total_helmets += helmet_count
                LOGGER.info(f"[{i+1}/{len(image_files)}] {image_file.name}: {helmet_count} 个安全帽")
        
        LOGGER.info(f"批量检测完成，总共发现 {total_helmets} 个安全帽")
        success = True
        
    else:
        LOGGER.error(f"无效的输入源: {args.source}")
        return 1
    
    return 0 if success else 1

if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)