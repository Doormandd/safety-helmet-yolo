"""
YOLOv8 安全帽检测推理脚本
加载训练好的模型对视频进行推理并播放结果
"""
import cv2
from pathlib import Path
from ultralytics import YOLO
import time

def inference_video(model_path, video_path, output_path=None, show=True, conf=0.25):
    """
    对视频进行推理
    
    Args:
        model_path: 模型文件路径
        video_path: 输入视频路径
        output_path: 输出视频路径（可选）
        show: 是否实时显示结果
        conf: 置信度阈值
    """
    print("="*60)
    print("YOLOv8 安全帽检测推理")
    print("="*60)
    
    # 检查文件是否存在
    model_path = Path(model_path)
    video_path = Path(video_path)
    
    if not model_path.exists():
        print(f"错误: 模型文件不存在: {model_path}")
        return
    
    if not video_path.exists():
        print(f"错误: 视频文件不存在: {video_path}")
        return
    
    print(f"\n加载模型: {model_path}")
    model = YOLO(str(model_path))
    
    print(f"打开视频: {video_path}")
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        print("错误: 无法打开视频文件")
        return
    
    # 读取第一帧以获取准确的尺寸（解决元数据尺寸不一致问题）
    ret, first_frame = cap.read()
    if not ret:
        print("错误: 无法读取视频内容")
        return
    
    # 重置视频到开始位置
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # 获取视频信息
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    # 使用实际帧的尺寸，确保输出比例正确
    height, width = first_frame.shape[:2]
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"\n视频信息:")
    print(f"  分辨率: {width}x{height}")
    print(f"  帧率: {fps} FPS")
    print(f"  总帧数: {total_frames}")
    print(f"  时长: {total_frames/fps:.2f} 秒")
    
    # 设置输出视频
    writer = None
    if output_path:
        output_path = Path(output_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        print(f"\n输出视频: {output_path}")
    
    print(f"\n开始推理 (置信度阈值: {conf})...")
    print("按 'q' 键退出，'p' 键暂停/继续\n")
    
    frame_count = 0
    start_time = time.time()
    paused = False
    
    # 类别颜色（BGR格式）
    colors = {
        'helmet': (0, 255, 0),    # 绿色 - 戴安全帽
        'head': (0, 0, 255),      # 红色 - 未戴安全帽
        'person': (255, 0, 0)     # 蓝色 - 人
    }
    
    try:
        while cap.isOpened():
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # 使用YOLO进行推理
                results = model(frame, conf=conf, verbose=False)
                
                # 绘制结果
                annotated_frame = results[0].plot()
                
                # 添加统计信息
                stats_text = []
                for result in results:
                    boxes = result.boxes
                    if boxes is not None and len(boxes) > 0:
                        # 统计各类别数量
                        helmet_count = sum(1 for cls in boxes.cls if int(cls) == 0)
                        head_count = sum(1 for cls in boxes.cls if int(cls) == 1)
                        person_count = sum(1 for cls in boxes.cls if int(cls) == 2)
                        
                        stats_text.append(f"Helmet: {helmet_count}")
                        stats_text.append(f"Head: {head_count}")
                        stats_text.append(f"Person: {person_count}")
                
                # 在画面上显示统计信息
                y_offset = 30
                for i, text in enumerate(stats_text):
                    cv2.putText(annotated_frame, text, (10, y_offset + i*30),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                # 显示帧数和进度
                progress_text = f"Frame: {frame_count}/{total_frames} ({frame_count/total_frames*100:.1f}%)"
                cv2.putText(annotated_frame, progress_text, (10, height-20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # 写入输出视频
                if writer is not None:
                    writer.write(annotated_frame)
                
                # 显示结果
                if show:
                    cv2.imshow('YOLOv8 Safety Helmet Detection', annotated_frame)
                
                # 打印进度
                if frame_count % 30 == 0:
                    elapsed = time.time() - start_time
                    fps_actual = frame_count / elapsed
                    eta = (total_frames - frame_count) / fps_actual
                    print(f"进度: {frame_count}/{total_frames} ({frame_count/total_frames*100:.1f}%) | "
                          f"FPS: {fps_actual:.1f} | ETA: {eta:.1f}s")
            
            # 处理键盘输入
            if show:
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n用户中断")
                    break
                elif key == ord('p'):
                    paused = not paused
                    print("\n已暂停" if paused else "\n继续播放")
    
    except KeyboardInterrupt:
        print("\n\n用户中断")
    
    finally:
        # 释放资源
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
        
        # 统计信息
        elapsed = time.time() - start_time
        print(f"\n推理完成!")
        print(f"  处理帧数: {frame_count}/{total_frames}")
        print(f"  总耗时: {elapsed:.2f} 秒")
        print(f"  平均FPS: {frame_count/elapsed:.1f}")
        
        if output_path and output_path.exists():
            print(f"  输出已保存: {output_path}")

if __name__ == '__main__':
    # 配置参数
    MODEL_PATH = 'yolov26n_helmet_best.pt'  # YOLOv26n模型路径
    VIDEO_PATH = 'v.mp4'                    # 输入视频
    OUTPUT_PATH = 'output_yolov26n.mp4'     # 输出视频（可选，设为None则不保存）
    CONFIDENCE = 0.25                        # 置信度阈值
    SHOW = True                              # 是否实时显示
    
    inference_video(
        model_path=MODEL_PATH,
        video_path=VIDEO_PATH,
        output_path=OUTPUT_PATH,
        show=SHOW,
        conf=CONFIDENCE
    )
