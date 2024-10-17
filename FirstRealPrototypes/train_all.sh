mkdir yolov8n
mkdir yolov8s
mkdir yolov8m
mkdir yolov8l
mkdir yolov8x

echo "training yolov8n model..."
python3 training.py yolov8n > ./yolov8n/yolov8n_output.txt

echo "training yolov8s model..."
python3 training.py yolov8s > ./yolov8s/yolov8s_output.txt

echo "training yolov8m model..."
python3 training.py yolov8m > ./yolov8m/yolov8m_output.txt

echo "training yolov8l model..."
python3 training.py yolov8l > ./yolov8l/yolov8l_output.txt

echo "training yolov8x model"
python3 training.py yolov8x > ./yolov8x/yolov8x_output.txt


echo "done"