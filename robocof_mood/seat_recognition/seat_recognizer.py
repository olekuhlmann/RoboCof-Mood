import math
import torch
import cv2
import json
import warnings
warnings.filterwarnings("ignore")
from collections import Counter


import asyncio
from time import sleep
from enum import Enum
from robocof_mood.input_stream.input_stream import InputStream
from robocof_mood.input_stream.webcam_input_stream import WebcamInputStream


#TODO: store model locally,
#TODO: warm up model once at beginning so detection can be instant

class SeatStatus(Enum):
    NO_CHAIRS_NO_PEOPLE = 0
    UNSURE = 1
    """
    Unsure triggers when at least one person is detected but no suitable chair is detected.
    Most common causes are either:
        the chair is empty and there are people in the backround
        the chair is blocked by an item or the person sitting on it, stopping the model from identifying the chair
        the chair is not where it should be (out of frame or too far back)

    """
    SEAT_EMPTY = 2
    SEAT_OCCUPIED = 3


class SeatRecognizer:
    def __init__(self, input_stream: InputStream):
        self.__input_stream = input_stream
        #Save and load the initialized model: If you're calling it repeatedly, warm it once and serialize with torch.save(model.state_dict()) or TorchScript.
        #debug mode?
               # Loading Model
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s')


        # Configuring Model
        self.model.cpu()  # .cpu() ,or .cuda()
        self.model.conf = 0.25  # NMS confidence threshold
        self.model.iou = 0.45  # NMS IoU threshold
        self.model.agnostic = False  # NMS class-agnostic
        self.model.multi_label = False  # NMS multiple labels per box
        # (optional list) filter by class, i.e. = [0, 15, 16] for COCO persons, cats and dogs
        self.model.classes = [0, 56]
        self.model.max_det = 20  # maximum number of detections per image
        self.model.amp = False  # Automatic Mixed Precision (AMP) inference

        #counter
        self.seatStatus_counter = Counter()
            


    def recognize(self, frame, model):
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Inference
        results = model(image, size=720)  # includes NMS

        # Results
        # results.print()  # .print() , .show(), .save(), .crop(), .pandas(), etc.
        # results.show()

        results.xyxy[0]  # im predictions (tensor)
        df = results.pandas().xyxy[0]  # im predictions (pandas)
        #      xmin    ymin    xmax   ymax  confidence  class    name
        # 0  749.50   43.50  1148.0  704.5    0.874023      0  person
        # 2  114.75  195.75  1095.0  708.0    0.624512      0  person
        # 3  986.00  304.00  1028.0  420.0    0.286865     27     tie
        df_chair = df[df['class'] == 56]
        df_person = df[df['class'] == 0]

        # TODO: fix this hack and do it properly
        json_person = df_person.to_json(orient="records")
        json_chair = df_chair.to_json(orient="records")
        df_person = json.loads(json_person)
        df_chair = json.loads(json_chair)
        
        # Accessing each individual object and then getting its xmin, ymin, xmax and ymax to calculate its centroid
        
        min_size = 10000 #minimum size of bounding box for chair (to avoid background chairs). currently chosen arbitrarily
        big_chair = [min_size, 0, 0] #min_size, bottom left corner, top right corner
        flag = False
        for objects in df_chair:
            xmin_chair = objects["xmin"]
            ymin_chair = objects["ymin"]
            xmax_chair = objects["xmax"]
            ymax_chair = objects["ymax"]
            size_chair = (xmax_chair - xmin_chair) * (ymax_chair - ymin_chair)
            
            if size_chair > big_chair[0]:
                big_chair[1] = (xmin_chair, ymin_chair) #bottom left
                big_chair[2] = (xmax_chair, ymax_chair) #top right
                big_chair[0] = size_chair
                #print(big_chair)
                flag = True


        if flag: #chair with bounding box > min_size found
            xmin_chair = big_chair[1][0]
            ymin_chair = big_chair[1][1]
            xmax_chair = big_chair[2][0]
            ymax_chair = big_chair[2][1]
            cx_chair = int((xmin_chair+xmax_chair)/2.0)
            cy_chair = int((ymin_chair+ymax_chair)/2.0)
            
            
            for object_person in df_person:
                xmin_person = object_person["xmin"]
                ymin_person = object_person["ymin"]
                xmax_person = object_person["xmax"]
                ymax_person = object_person["ymax"]
                cx_person = int((xmin_person+xmax_person)/2.0)
                cy_person = int((ymin_person+ymax_person)/2.0)
                chair_list = [cx_chair, cy_chair]
                person_list = [cx_person, cy_person]
                #c1 = (xmin_chair, ymin_chair)
                #c2 = (xmax_chair, ymax_chair)
                centroid_dist = math.dist(chair_list, person_list)
                
                #print(centroid_dist)
                
                #print(c1 , c2)
                #print("......")
                
                if (centroid_dist >= 190): 
                    #cv2.rectangle(image, (int(c1[0]), int(c1[1])), (int(c2[0]), int(c2[1])), (0, 150, 0), 2)
                    #cv2.putText(image, 'Empty', (int(c1[0]), int(c1[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 150, 0), 2)
                    return self.parse_result([big_chair, df_person, True])
                
        return self.parse_result([big_chair if big_chair[1] else None, df_person, False])
                

    def parse_result(self, result):
        """
        result in the form [bounding box of closest chair if exists, list of people, is seat empty boolesn]
        """
        #print(result)
        if result[2] == True: #seat empty
            return SeatStatus.SEAT_EMPTY
        
        elif result[0]: #chair detected
            if result[1]: #if people detected
                return SeatStatus.SEAT_EMPTY
            
            return SeatStatus.SEAT_OCCUPIED
        
        
        elif result[1]: #chair not detected, people detected
            return SeatStatus.UNSURE
    
        else:
            return SeatStatus.NO_CHAIRS_NO_PEOPLE

    async def start(self):
        """
        Starts check for seats and people

        Returns:
            SeatStatus code corresponding to the specific scenario
        """
 
        
        while True:
            frame = self.__input_stream.capture_frame()
            if frame is None:
                print("[Seat-detection]: Failed to capture image.")
                continue
        
            status = self.recognize(frame, self.model)
            #print(status)
            self.seatStatus_counter[status] += 1
            # yield control to allow other tasks to run
            await asyncio.sleep(0.01)
            
                    
    def output(self):
        return self.seatStatus_counter.most_common(1)[0][0]

if __name__ == "__main__":
    async def main():
        # Example usage
        inputstream = WebcamInputStream()
        inputstream.start()
        await SeatRecognizer(inputstream).start()
    asyncio.run(main())

