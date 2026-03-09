"""
YOLOv8 安全帽检测图片推理脚本
加载训练好的模型对images文件夹中的图片进行推理并保存结果
"""
import cv2
from pathlib import Path
from ultralytics import YOLO
import time

def inference_images(model_path, images_dir, output_dir, conf=0.25, save_txt=False):
    """
    对图片文件夹进行批量推理
    
    Args:
        model_path: 模型文件路径
        images_dir: 输入图片文件夹路径
        output_dir: 输出结果文件夹路径
        conf: 置信度阈值
        save_txt: 是否保存YOLO格式标注文件
    """
    print("="*60)
    print("YOLOv26 安全帽检测 - 图片批量推理")
    print("="*60)
    
    # 路径处理
    model_path = Path(model_path)
    images_dir = Path(images_dir)
    output_dir = Path(output_dir)
    
    # 检查模型和图片文件夹
    if not model_path.exists():
        print(f"错误: 模型文件不存在: {model_path}")
        return
    
    if not images_dir.exists():
        print(f"错误: 图片文件夹不存在: {images_dir}")
        return
    
    # 创建输出文件夹
    output_dir.mkdir(parents=True, exist_ok=True)
    if save_txt:
        labels_dir = output_dir / 'labels'
        labels_dir.mkdir(exist_ok=True)
    
    print(f"\n加载模型: {model_path}")
    model = YOLO(str(model_path))
    
    # 支持的图片格式
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}
    
    # 获取所有图片文件
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(images_dir.glob(f'*{ext}')))
        image_files.extend(list(images_dir.glob(f'*{ext.upper()}')))
    
    if not image_files:
        print(f"错误: 在 {images_dir} 中未找到图片文件")
        return
    
    print(f"\n找到 {len(image_files)} 张图片")
    print(f"输出目录: {output_dir}")
    print(f"置信度阈值: {conf}")
    print(f"\n开始推理...\n")
    
    start_time = time.time()
    results_summary = {
        'total': len(image_files),
        'processed': 0,
        'detections': 0,
        'helmet_count': 0,
        'head_count': 0,
        'person_count': 0
    }
    
    # 逐张处理图片
    for idx, image_path in enumerate(image_files, 1):
        try:
            # 记录单张图片开始时间
            img_start_time = time.time()
            
            # 读取图片
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"警告: 无法读取图片 {image_path.name}，跳过")
                continue
            
            # 推理
            inference_start = time.time()
            results = model(image, conf=conf, verbose=False)
            inference_time = time.time() - inference_start
            
            # 统计检测结果
            for result in results:
                boxes = result.boxes
                if boxes is not None and len(boxes) > 0:
                    results_summary['detections'] += len(boxes)
                    
                    # 统计各类别
                    helmet_count = sum(1 for cls in boxes.cls if int(cls) == 0)
                    head_count = sum(1 for cls in boxes.cls if int(cls) == 1)
                    person_count = sum(1 for cls in boxes.cls if int(cls) == 2)
                    
                    results_summary['helmet_count'] += helmet_count
                    results_summary['head_count'] += head_count
                    results_summary['person_count'] += person_count
                    
            # 绘制结果
            annotated_image = results[0].plot()
            
            # 保存标注后的图片
            output_path = output_dir / f"detected_{image_path.name}"
            cv2.imwrite(str(output_path), annotated_image)
            
            # 计算单张图片总耗时
            img_total_time = time.time() - img_start_time
            
            # 打印检测信息（包含耗时）
            print(f"[{idx}/{len(image_files)}] {image_path.name}: "
                  f"Helmet={helmet_count}, Head={head_count}, Person={person_count}, "
                  f"Total={len(boxes) if boxes is not None and len(boxes) > 0 else 0} | "
                  f"推理耗时: {inference_time*1000:.1f}ms, 总耗时: {img_total_time*1000:.1f}ms")
            
            # 保存YOLO格式标注（可选）
            if save_txt and boxes is not None and len(boxes) > 0:
                txt_path = labels_dir / f"{image_path.stem}.txt"
                with open(txt_path, 'w') as f:
                    for box in boxes:
                        # YOLO格式: class x_center y_center width height
                        cls_id = int(box.cls[0])
                        x_center, y_center, w, h = box.xywhn[0].tolist()
                        conf_score = float(box.conf[0])
                        f.write(f"{cls_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f} {conf_score:.4f}\n")
            
            results_summary['processed'] += 1
            
        except Exception as e:
            print(f"错误: 处理图片 {image_path.name} 时出错: {str(e)}")
            continue
    
    # 统计信息
    elapsed = time.time() - start_time
    
    print("\n" + "="*60)
    print("推理完成!")
    print("="*60)
    print(f"\n处理统计:")
    print(f"  总图片数: {results_summary['total']}")
    print(f"  成功处理: {results_summary['processed']}")
    print(f"  总检测数: {results_summary['detections']}")
    print(f"  - 安全帽 (Helmet): {results_summary['helmet_count']}")
    print(f"  - 未戴帽头部 (Head): {results_summary['head_count']}")
    print(f"  - 人 (Person): {results_summary['person_count']}")
    print(f"\n性能统计:")
    print(f"  总耗时: {elapsed:.2f} 秒")
    print(f"  平均速度: {results_summary['processed']/elapsed:.2f} 张/秒")
    print(f"\n结果保存位置:")
    print(f"  标注图片: {output_dir}")
    if save_txt:
        print(f"  标注文件: {labels_dir}")
    
    return results_summary

if __name__ == '__main__':
    # 配置参数
    MODEL_PATH = 'yolov26n_helmet_best.pt'  # YOLO26n模型路径
    IMAGES_DIR = 'images'                    # 输入图片文件夹
    OUTPUT_DIR = 'output_images_yolo26n'     # 输出文件夹
    CONFIDENCE = 0.25                         # 置信度阈值
    SAVE_TXT = True                           # 是否保存YOLO格式标注文件
    
    inference_images(
        model_path=MODEL_PATH,
        images_dir=IMAGES_DIR,
        output_dir=OUTPUT_DIR,
        conf=CONFIDENCE,
        save_txt=SAVE_TXT
    )
