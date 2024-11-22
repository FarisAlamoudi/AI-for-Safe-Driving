from ultralytics import YOLO
import cv2
import time
import math
from datetime import datetime

class InsideModel():
    def __init__(self, cap_val: int, model_path: str, db=None):
        self.model = YOLO(model_path)

        self.cap = cv2.VideoCapture(cap_val)
        self.cap.set(3, 736)
        self.cap.set(4, 736)

        self.classNames = [
            'Drinking', 'Eating', 'Hands Off Wheel', 'Hands On Wheel', 'Phone', 'Seatbelt-On', 'Sleeping', 'Smoking'
        ]

        if db != None:
            self.db = db
        else:
            self.db=None
    
    def run_inferences(self, runtime: int, export: bool, display_feed: bool):
        """
        Description:
            Runs model for a certain amount of time to make inferrences
        Parameters:
            runtime: int
                - Amount of time to make inferences for. If value passed is -1 then model will run indefinitely
            export: bool
                - Determines if output is exported to a db. Requires db object to be instatiated.
            display_feed: bool
                - Determines if display_feed is created.
        """
        if export and self.db == None:
            raise Exception("Cannot export data to a database if it is not instantiated.")
        
        start = time.time()
        while time.time() - start < runtime:
            ret, frame = self.cap.read()
            if not ret:
                raise Exception("Unable to grab frame.")

            results = self.model(frame, stream=True)
            detections = []
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    detections.append(int(box.cls[0]))
                    if display_feed:
                        x1, y1, x2, y2 = box.xyxy[0]
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)

                        # confidence
                        confidence = math.ceil((box.conf[0]*100))/100
                        print("Confidence --->",confidence)

                        # class name
                        cls = int(box.cls[0])
                        print("Class name -->", self.classNames[cls])

                        # object details
                        org = [x1, y1]
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        fontScale = 1
                        color = (255, 0, 0)
                        thickness = 2

                        cv2.putText(frame, self.classNames[cls], org, font, fontScale, color, thickness)

            if display_feed:
                cv2.imshow('YOLO Video Feed', frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            output_dict = {
                "time": str(datetime.now().strftime("%H:%M:%S")),
                "Drinking": 0,
                "Eating": 0,
                "Hands Off Wheel": 0,
                "Hands on Wheel": 0,
                "Phone": 0,
                "Seatbelt-On": 0,
                "Sleeping": 0,
                "Smoking": 0
            }

            for cls in detections:
                if self.classNames[cls] == "Drinking":
                    output_dict['Drinking'] = 1
                if self.classNames[cls] == "Eating":
                    output_dict["Eating"] = 1
                if self.classNames[cls] == "Hands Off Wheel":
                    output_dict["Hands Off Wheel"] += 1 if (output_dict["Hands Off Wheel"] + output_dict['Hands on Wheel']) < 2 else 0
                if self.classNames[cls] == "Hands on Wheel":
                    output_dict["Hands on Wheel"] += 1 if (output_dict["Hands Off Wheel"] + output_dict['Hands on Wheel']) < 2 else 0
                if self.classNames[cls] == "Phone":
                    output_dict['Phone'] = 1
                if self.classNames[cls] == "Seatbelt-On":
                    output_dict['Seatbelt-On'] = 1
                if self.classNames[cls] == "Sleeping":
                    output_dict['Sleeping'] = 1
                if self.classNames[cls] == "Smoking":
                    output_dict['Smoking'] = 1

            if export:
                self.db.execute(f'INSERT INTO InsideValues (Time, Drinking, Eating, HandsOffWheel, HandsOnWheel, Phone, SeatbeltOn, Sleeping, Smoking) VALUES ("{output_dict['time']}", {output_dict['Drinking']}, {output_dict['Eating']}, {output_dict['Hands Off Wheel']}, {output_dict["Hands on Wheel"]}, {output_dict['Phone']}, {output_dict['Seatbelt-On']}, {output_dict['Sleeping']}, {output_dict['Smoking']})')

        self.cap.release()
        cv2.destroyAllWindows()