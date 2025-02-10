from model_testing import InsideModel

model = InsideModel(0,"C:/Users/ethan/OneDrive/Desktop/yolo11s.pt")

model.run_inferences(10, False, True)