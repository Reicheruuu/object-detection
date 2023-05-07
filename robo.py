
from roboflow import Roboflow
rf = Roboflow(api_key="3GSc6HlJbhrcP1cN3Mz4")
project = rf.workspace("yolov5-6agzx").project("school-uniform")
dataset = project.version(1).download("yolov5")
