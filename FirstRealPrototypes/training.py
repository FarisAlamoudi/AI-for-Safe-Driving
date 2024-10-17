from roboflow import Roboflow
from ultralytics import YOLO
import sys

model_name = sys.argv[1]
model_name_ext = model_name + '.pt'

rf = Roboflow(api_key="MRUSuDWuLjlkWAF0se8m")
project = rf.workspace("inside-ai-for-safe-driving").project("training-data-crusx")
version = project.version(7)
dataset = version.download("yolov8")

path = dataset.location + "/data.yaml"
model = YOLO(model_name_ext)

model.train(data=path, epochs=100, imgsz=640, batch=16, device=0)

metrics = model.val()
with open(f'./{model_name}/{model_name}_metrics.txt', 'w') as f:
    f.write(f'Validation Metrics:\n{metrics}')

model_path = model.export(format='onnx')
with open(f'./{model_name}/trained_{model_name}_path.txt', 'w') as f:
    f.write(f'Exported Model Location: {model_path}')
