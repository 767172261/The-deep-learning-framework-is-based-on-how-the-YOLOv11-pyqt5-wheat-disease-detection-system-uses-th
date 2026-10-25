基于YOLOv11+pyqt5的小麦病害检测系统
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/c952e73b4fae4ab3ab45476dee368961.png)

小麦病虫害数据集。数据集已经标注好，txt标注，yolo格式，目标检测数据集。适用于训练yolo5 7 8 9 10等系列目标检测模型。数据集已经划分好训练集（1164张图片）验证集（218张图片）测试集（165张图片），含类别标签说明文件，分为5个类别：叶锈病、健康、散黑穗病、黄锈病、秆锈病。注意该数据集含有数据增强的图片（原图随机角度旋转）

训练集：1164
验证集：218
测试集：165
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/5905149baa564a99948352e1512b921f.png)


**yolov11 500epoch训练结果
也可自行替换模型，使用该界面做其他，实现检测目标自定义
Python + PyQt5可视化界面
可对图片，视频，摄像头进行识别**

以下是基于 **YOLOv11 + PyQt5** 实现小麦病害检测系统的完整代码，包含界面、模型加载、图片/视频/摄像头检测等功能：


### 一、环境依赖
首先安装所需库：
```bash
pip install torch torchvision opencv-python pyqt5 numpy pillow
```

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/388e8bfae9f74246b338ca9565508249.png)

### 二、完整代码
```python
import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel, QPushButton, 
                             QSpinBox, QDoubleSpinBox, QComboBox, QTableWidget, QTableWidgetItem,
                             QVBoxLayout, QHBoxLayout, QFileDialog, QTextEdit, QGroupBox)
from PyQt5.QtGui import QPixmap, QImage, QFont
from PyQt5.QtCore import Qt, QTimer
from ultralytics import YOLO


class WheatDiseaseDetector(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("基于YOLOv11的小麦病害检测系统")
        self.setGeometry(100, 100, 1200, 800)
        
        # 加载YOLOv11模型（替换为你的模型路径）
        self.model = YOLO("yolov11_weights.pt")  # 需替换为训练好的权重文件
        self.classes = ["叶锈病", "健康", "散黑穗病", "黄锈病", "秆锈病"]  # 对应数据集类别
        
        # 初始化变量
        self.current_image = None
        self.video_cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_video_frame)
        
        # 构建界面
        self.init_ui()

    def init_ui(self):
        # 主布局
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        # ========== 左侧：图像显示区 ==========
        left_group = QGroupBox("检测结果展示")
        left_layout = QVBoxLayout(left_group)
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("border: 1px solid #ccc;")
        left_layout.addWidget(self.image_label)
        main_layout.addWidget(left_group, stretch=2)

        # ========== 右侧：参数+结果+操作区 ==========
        right_layout = QVBoxLayout()

        # 1. 检测参数设置
        param_group = QGroupBox("检测参数设置")
        param_layout = QHBoxLayout(param_group)
        param_layout.addWidget(QLabel("置信度阈值:"))
        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.01, 1.0)
        self.conf_spin.setValue(0.25)
        param_layout.addWidget(self.conf_spin)
        param_layout.addWidget(QLabel("交并比阈值:"))
        self.iou_spin = QDoubleSpinBox()
        self.iou_spin.setRange(0.01, 1.0)
        self.iou_spin.setValue(0.70)
        param_layout.addWidget(self.iou_spin)
        right_layout.addWidget(param_group)

        # 2. 检测结果展示
        result_group = QGroupBox("检测结果")
        result_layout = QVBoxLayout(result_group)
        # 结果标签
        self.result_label = QLabel("显示标签名称与置信度\n总目标数：0  用时：0 s")
        result_layout.addWidget(self.result_label)
        # 类型+置信度
        type_layout = QHBoxLayout()
        type_layout.addWidget(QLabel("类型:"))
        self.type_label = QLabel("无")
        type_layout.addWidget(self.type_label)
        type_layout.addWidget(QLabel("置信度:"))
        self.conf_label = QLabel("0%")
        result_layout.addLayout(type_layout)
        # 目标位置
        pos_layout = QHBoxLayout()
        pos_layout.addWidget(QLabel("xmin:"))
        self.xmin_label = QLabel("0")
        pos_layout.addWidget(QLabel("ymin:"))
        self.ymin_label = QLabel("0")
        pos_layout.addWidget(QLabel("xmax:"))
        self.xmax_label = QLabel("0")
        pos_layout.addWidget(QLabel("ymax:"))
        self.ymax_label = QLabel("0")
        result_layout.addLayout(pos_layout)
        right_layout.addWidget(result_group)

        # 3. 检测结果表格
        table_group = QGroupBox("检测结果与位置信息")
        self.result_table = QTableWidget()
        self.result_table.setColumnCount(6)
        self.result_table.setHorizontalHeaderLabels(["序号", "文件路径", "目标编号", "类别", "置信度", "坐标"])
        table_layout = QVBoxLayout(table_group)
        table_layout.addWidget(self.result_table)
        right_layout.addWidget(table_group)

        # 4. 操作按钮
        btn_layout = QVBoxLayout()
        self.open_img_btn = QPushButton("打开图片")
        self.open_img_btn.clicked.connect(self.open_image)
        btn_layout.addWidget(self.open_img_btn)
        self.open_video_btn = QPushButton("打开视频")
        self.open_video_btn.clicked.connect(self.open_video)
        btn_layout.addWidget(self.open_video_btn)
        self.open_cam_btn = QPushButton("打开摄像头")
        self.open_cam_btn.clicked.connect(self.open_camera)
        btn_layout.addWidget(self.open_cam_btn)
        self.save_btn = QPushButton("保存")
        self.save_btn.clicked.connect(self.save_result)
        btn_layout.addWidget(self.save_btn)
        self.exit_btn = QPushButton("退出")
        self.exit_btn.clicked.connect(self.close)
        btn_layout.addWidget(self.exit_btn)
        right_layout.addLayout(btn_layout)

        main_layout.addLayout(right_layout, stretch=1)

    def detect(self, img):
        """执行YOLOv11检测"""
        results = self.model.predict(
            img,
            conf=self.conf_spin.value(),
            iou=self.iou_spin.value(),
            classes=list(range(len(self.classes)))
        )
        # 绘制检测框
        result_img = results[0].plot()  # 自动绘制框和标签
        # 获取检测信息
        det_info = []
        for i, box in enumerate(results[0].boxes):
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            xmin, ymin, xmax, ymax = map(int, box.xyxy[0])
            det_info.append({
                "序号": len(det_info)+1,
                "文件路径": "当前文件",
                "目标编号": i+1,
                "类别": self.classes[cls],
                "置信度": f"{conf*100:.2f}%",
                "坐标": f"[{xmin}, {ymin}, {xmax}, {ymax}]"
            })
        return result_img, det_info, results[0].speed["inference"]  # 推理时间

    def update_ui(self, result_img, det_info, infer_time):
        """更新界面显示"""
        # 显示检测后的图像
        h, w, ch = result_img.shape
        bytes_per_line = ch * w
        q_img = QImage(result_img.data, w, h, bytes_per_line, QImage.Format_BGR888)
        self.image_label.setPixmap(QPixmap.fromImage(q_img).scaled(self.image_label.size(), Qt.KeepAspectRatio))

        # 更新结果标签
        self.result_label.setText(f"显示标签名称与置信度\n总目标数：{len(det_info)}  用时：{infer_time/1000:.3f} s")
        
        # 更新表格
        self.result_table.setRowCount(len(det_info))
        for row, info in enumerate(det_info):
            self.result_table.setItem(row, 0, QTableWidgetItem(str(info["序号"])))
            self.result_table.setItem(row, 1, QTableWidgetItem(info["文件路径"]))
            self.result_table.setItem(row, 2, QTableWidgetItem(str(info["目标编号"])))
            self.result_table.setItem(row, 3, QTableWidgetItem(info["类别"]))
            self.result_table.setItem(row, 4, QTableWidgetItem(info["置信度"]))
            self.result_table.setItem(row, 5, QTableWidgetItem(info["坐标"]))

        # 选中第一个目标的信息（示例）
        if det_info:
            first = det_info[0]
            self.type_label.setText(first["类别"])
            self.conf_label.setText(first["置信度"])
            coords = eval(first["坐标"])
            self.xmin_label.setText(str(coords[0]))
            self.ymin_label.setText(str(coords[1]))
            self.xmax_label.setText(str(coords[2]))
            self.ymax_label.setText(str(coords[3]))

    def open_image(self):
        """打开图片并检测"""
        file_path, _ = QFileDialog.getOpenFileName()
        if not file_path:
            return
        img = cv2.imread(file_path)
        self.current_image = img
        result_img, det_info, infer_time = self.detect(img)
        self.update_ui(result_img, det_info, infer_time)

    def open_video(self):
        """打开视频并检测"""
        file_path, _ = QFileDialog.getOpenFileName()
        if not file_path:
            return
        self.video_cap = cv2.VideoCapture(file_path)
        self.timer.start(30)  # 30ms/帧（约33FPS）

    def open_camera(self):
        """打开摄像头并检测"""
        self.video_cap = cv2.VideoCapture(0)
        self.timer.start(30)

    def update_video_frame(self):
        """更新视频/摄像头帧"""
        ret, frame = self.video_cap.read()
        if not ret:
            self.timer.stop()
            self.video_cap.release()
            return
        result_img, det_info, infer_time = self.detect(frame)
        self.update_ui(result_img, det_info, infer_time)

    def save_result(self):
        """保存当前检测结果（示例：保存图像）"""
        if self.current_image is None:
            return
        save_path, _ = QFileDialog.getSaveFileName(filter="PNG(*.png);;JPG(*.jpg)")
        if save_path:
            cv2.imwrite(save_path, self.current_image)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = WheatDiseaseDetector()
    window.show()
    sys.exit(app.exec_())
```


### 三、使用说明
1. **模型替换**：将代码中 `self.model = YOLO("yolov11_weights.pt")` 替换为你训练好的YOLOv11权重文件路径。
2. **数据集适配**：确保 `self.classes` 与你的数据集类别顺序一致（需和训练时的 `names.yaml` 对应）。
3. **功能扩展**：可根据需求增加“批量检测”“结果导出为Excel”等功能。


![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/52f9dba73a734a229a2c88320dca27fc.png)
以上文字及代码仅供参考
