import cv2
from ultralytics import YOLO
import argparse
import numpy as np
import matplotlib.pyplot as plt

def main(args):
    # 1. 加载模型
    print(f"正在加载模型: {args.weights}")
    model = YOLO(args.weights)

    # 2. 读取视频
    print(f"正在读取视频: {args.source}")
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        print("错误：无法打开视频文件，请检查路径。")
        return

    # 获取视频参数
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 3. 设置输出视频写入器 (mp4v 编码)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    print("开始推理，请稍候...")

    all_confidences = []

    # 4. 逐帧推理
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 推理当前帧
        results = model.predict(source=frame, conf=0.25, verbose=False)
        result = results[0]

        if result.boxes is not None and len(result.boxes) > 0:
            all_confidences.extend(result.boxes.conf.cpu().numpy().tolist())

        # 5. 绘制检测框、类别、置信度和关键点
        # result.plot() 会自动将模型输出的关键点画在图上
        annotated_frame = result.plot()

        # 写入输出视频
        out.write(annotated_frame)

    # 释放资源
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"推理完成！结果已保存至: {args.output}")

    if len(all_confidences) > 0:
        avg_conf = np.mean(all_confidences)
        print(f"【统计结果】共检测到{len(all_confidences)}个目标。")
        print(f"【统计结果】平均置信度(Average Confidence):{avg_conf:.4f}")

        plt.figure(figsize=(10, 6))
        plt.hist(all_confidences, bins=20, range=(0.0, 1.0), color='skyblue', edgecolor='black')
        plt.title('Confidence Distribution of Detections')
        plt.xlabel('Confidence Score')
        plt.ylabel('Frequency')
        plt.grid(axis='y', alpha=0.75)

        # 保存图表为图片文件
        plot_filename = 'confidence_distribution.png'
        plt.savefig(plot_filename)
        print(f"【统计图表】已生成置信度分布图: {plot_filename}")
        plt.close()
    else:
        print("推理完成，但未检测到任何目标。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO26n 装甲板视频推理脚本")
    parser.add_argument('--weights', type=str, default='best.pt', help='模型权重文件路径')
    parser.add_argument('--source', type=str, required=True, help='输入视频路径')
    parser.add_argument('--output', type=str, default='output_video.mp4', help='输出视频保存路径')
    
    args = parser.parse_args()
    main(args)