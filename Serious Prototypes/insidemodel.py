import math
from datetime import datetime
from ultralytics import YOLO
import cv2
import time
import sqlite3


class InsideModel():
    def __init__(self, cap_val: int, model_path: str, db:sqlite3.Cursor=None):
        """
        Description:
            Constructor for inside model inference class
        Arguments:
            cap_val: int
                - Desired camera index value in system
            model_path: str
                - Yolo model path or name. If name is provided then an internet connection is required
            db: sqlite3.Cursor
                - Cursor for desired database connection to be used for exporting data
        """
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
            self.db = None
    
    def run_inferences(self, runtime: int, export: bool, display_feed: bool, confidence_interval: int=50) -> list[dict[str, int|str]]:
        """
        Description:
            Runs model for a predetermined amount of time to make inferences on a live camera feed.
        Arguments:
            runtime: int
                - Amount of seconds to run model for. If -1 is passed, then model will run indefinitely
            export: bool
                - Determines if output is exported to the database instance. Requires a sqlite.Cursor to be provided to the InsideModel Class
            display_feed: bool
                - Determined if a display_feed for live playback is created.
            confidence_interval: int
                - Lowest possible confidence for model inference
        """

        assert not (export and self.db == None)

        start = time.time()
        inferences = []
        while time.time() - start < runtime or runtime == -1:
            ret, frame = self.cap.read()
            assert ret

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
                        print("Confidence --->", confidence)

                        # class name
                        cls = int(box.cls[0])
                        print("Class name --->", self.classNames[cls])

                        # object details
                        org = [x1, y1]
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        fontScale = 1
                        color = (255, 0, 0)
                        thickness = 2

                        cv2.putText(frame, self.classNames[cls], org, font, fontScale, color, thickness)

            if display_feed:
                cv2.imshow('Insade Camera Video Feed', frame)
            
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

            inferences.append(output_dict.copy())
            if export:
                self.db.execute(f'INSERT INTO InsideValues (Time, Drinking, Eating, HandsOffWheel, HandsOnWheel, Phone, SeatbeltOn, Sleeping, Smoking) VALUES ("{output_dict['time']}", {output_dict['Drinking']}, {output_dict['Eating']}, {output_dict['Hands Off Wheel']}, {output_dict["Hands on Wheel"]}, {output_dict['Phone']}, {output_dict['Seatbelt-On']}, {output_dict['Sleeping']}, {output_dict['Smoking']})')

        self.cap.release()
        cv2.destroyAllWindows()

        return inferences

    def run_inferences_on_video(self, runtime: int, export: bool, display_feed: bool, pic_or_video_path: str = None, confidence_interval: int=50) -> list[dict[str, int|str]]:
        """
        Description:
            Runs model for a predetermined amount of time to make inferences on a live camera feed.
        Arguments:
            runtime: int
                - Amount of seconds to run model for. If -1 is passed, then model will run indefinitely
            export: bool
                - Determines if output is exported to the database instance. Requires a sqlite.Cursor to be provided to the InsideModel Class
            display_feed: bool
                - Determined if a display_feed for live playback is created.
            confidence_interval: int
                - Lowest possible confidence for model inference
            pic_or_video_path: str
                - Path to media to make inferences on
        """

        assert not (export and self.db == None)
        assert pic_or_video_path != None

        cap = cv2.VideoCapture(pic_or_video_path)
        start = time.time()
        inferences = []
        while time.time() - start < runtime or runtime == -1:
            ret, frame = cap.read()
            assert ret

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
                        print("Confidence --->", confidence)

                        # class name
                        cls = int(box.cls[0])
                        print("Class name --->", self.classNames[cls])

                        # object details
                        org = [x1, y1]
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        fontScale = 1
                        color = (255, 0, 0)
                        thickness = 2

                        cv2.putText(frame, self.classNames[cls], org, font, fontScale, color, thickness)

            if display_feed:
                cv2.imshow('Insade Camera Video Feed', frame)
            
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

            inferences.append(output_dict.copy())
            if export:
                self.db.execute(f'INSERT INTO InsideValues (Time, Drinking, Eating, HandsOffWheel, HandsOnWheel, Phone, SeatbeltOn, Sleeping, Smoking) VALUES ("{output_dict['time']}", {output_dict['Drinking']}, {output_dict['Eating']}, {output_dict['Hands Off Wheel']}, {output_dict["Hands on Wheel"]}, {output_dict['Phone']}, {output_dict['Seatbelt-On']}, {output_dict['Sleeping']}, {output_dict['Smoking']})')

        cap.release()
        cv2.destroyAllWindows()

        return inferences