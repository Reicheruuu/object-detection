import cv2
import numpy as np
from PIL import Image
import torch
from torchvision.transforms.functional import resize
import sys
from PyQt5 import QtGui
from PyQt5.uic import loadUi
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QDialog, QApplication, QTabWidget, QWidget

# Define the class names for your objects
classes = ['ID Lace', 'Pants', 'Polo', 'Skirt']

# Load the pre-trained model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/runs/train/yolov5s_results8/weights/best.pt')

# Move the model to the GPU (if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Set the model to evaluation mode
model.eval()

# Start the video stream
cap = cv2.VideoCapture(0)

class MainWindow(QDialog):
    def __init__(self):
        super(MainWindow, self).__init__()
        loadUi("Tab.ui", self)

        # Set the window title
        self.setWindowTitle("Object Detection")

        self.tabWidget.setCurrentIndex(0)

        self.Run.clicked.connect(self.camera)

        self.cap = cv2.VideoCapture(0)

    def camera(self):
        self.tabWidget.setCurrentIndex(1)
        while True:
            # Initialize counters for each class
            counters = {cls: 0 for cls in classes}

            # Read a frame from the camera
            ret, frame = cap.read()

            # Resize the frame
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height, width, _ = frame.shape
            max_size = 900
            if width > height:
                new_width = max_size
                new_height = int(height / (width / max_size))
            else:
                new_height = max_size
                new_width = int(width / (height / max_size))
            frame = resize(Image.fromarray(frame), (new_height, new_width))

            # Convert frame to numpy array
            frame = np.array(frame)

            # Perform inference with YOLOv5 model
            results = model(frame)

            # Get detected objects and their positions
            boxes = results.xyxy[0].cpu().numpy()
            labels = results.xyxyn[0].cpu().numpy()[:, -1].astype(int)

            # Initialize the counters for each class and the total number of detections
            counters = {c: 0 for c in classes}
            total_detections = 0
            correct_detections = 0

            # Initialize the counters for each class and the total number of detections
            counters = {c: 0 for c in classes}
            total_detections = 0
            correct_detections = 0

            # Loop through all detected objects
            for i, box in enumerate(boxes):
                # Get object class and confidence
                cls = int(box[5])
                conf = box[4]

                # Check if the class index is valid
                if cls < len(classes):
                    class_name = classes[cls]

                    # Increment the counter for this class
                    counters[class_name] += 1

                    # Only show results above a certain confidence threshold
                    if conf > 0.5:
                        # Set the default bounding box color to red
                        color = (255, 0, 0)  # red

                        # Check if the Polo, ID Lace, and Pants or Polo, ID Lace, and Skirt are detected
                        if "Polo" in class_name and "ID Lace" in class_name and (
                                "Pants" in class_name or "Skirt" in class_name):
                            # All three items are detected, set bounding box color to green
                            color = (0, 255, 0)  # green
                            correct_detections += 1

                        # Draw the bounding box and label on the frame
                        xmin, ymin, xmax, ymax = map(int, box[:4])
                        if correct_detections < 2:
                            color = (255, 0, 0)  # red
                        frame = cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
                        frame = cv2.putText(frame, f"{class_name} {conf:.2f}", (xmin, ymin - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # Check if the uniform is complete and add to the UI
            if correct_detections == 3:
                cv2.putText(frame, 'Complete Uniform', (50, 500),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(frame, 'Incomplete Uniform', (50, 500),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)


                # Increment the total number of detections
                total_detections += 1

                # Calculate the accuracy
                if total_detections > 0:
                    accuracy = correct_detections / total_detections
                else:
                    accuracy = 0.0

                # Check for complete and incomplete uniform based on accuracy and detection counts
                id_lace_detected = counters['ID Lace'] > 0
                pants_detected = counters['Pants'] > 0
                skirt_detected = counters['Skirt'] > 0
                polo_detected = counters['Polo'] > 0

                if id_lace_detected and pants_detected and polo_detected:
                    print('Complete uniform')
                else:
                    if skirt_detected and (polo_detected or id_lace_detected) or pants_detected and (polo_detected or id_lace_detected):
                        print('Incomplete uniform')


            # Convert the frame to QPixmap
            qImg = QtGui.QImage(frame.data, frame.shape[1], frame.shape[0], QtGui.QImage.Format_RGB888)
            pixmap = QtGui.QPixmap.fromImage(qImg)

            # Display the frame on self.Camera_2
            self.Camera_2.setPixmap(pixmap)

            # Update the GUI
            QtWidgets.QApplication.processEvents()

        # Release the camera
        cap.release()

#main
app = QApplication(sys.argv)
mainwindow = MainWindow()
widget = QtWidgets.QStackedWidget()
widget.addWidget(mainwindow)
widget.setFixedHeight(600)
widget.setFixedWidth(900)
widget.show()

try:
    sys.exit(app.exec_())
except:
    print("Exiting")