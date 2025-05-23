import asyncio
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from enum import Enum
from typing import Optional
from input_stream.input_stream import InputStream 


MODEL_PATH = 'https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/latest/gesture_recognizer.task'

class Gesture(Enum):
    UNKNOWN = 0
    CLOSED_FIST = 1
    OPEN_PALM = 2
    POINTING_UP = 3
    THUMB_DOWN = 4
    THUMB_UP = 5
    VICTORY = 6
    LOVE = 7


class GestureRecognizer:
    def __init__(self, gestures: list[Gesture], input_stream: InputStream):
        """Constructor

        Args:
            gestures (list[Gesture]): List of gestures to recognize. Will stop active recognition if one of these gestures is detected.
            input_stream (InputStream): The input stream to capture frames from.
        """
        self.__gestures = gestures
        base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
        options = vision.GestureRecognizerOptions(base_options=base_options)
        self.__recognizer = vision.GestureRecognizer.create_from_options(options)
        self.__input_stream = input_stream
        
    def start(self, ) -> list[Gesture]:
        """
        Starts the gesture recognition process. Returns the recognized gesture once a gesture from `gestures` is recognized.
        
        Returns:
            Gesture: The recognized gesture.
        """        
        while True:
            frame = self.__input_stream.capture_frame() 
            if frame is None:
                print("Error: Failed to capture image.")
                break

            # Convert the frame to a MediaPipe Image object
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

            # Recognize the gestures in the current frame
            gestures = self.recognize(mp_image)

            if gestures:
                # filter the recognized gestures to check if any of them are in the list of gestures
                recognized_gestures = [gesture for gesture in gestures if gesture in self.__gestures]
                if recognized_gestures:
                    return recognized_gestures
            
            # yield control to allow other tasks to run TODO enable again and make def async
            #await asyncio.sleep(0.01)
        
        
    
    
    
    def recognize(self, image: mp.Image) -> list[Gesture]:
        """
        Recognizes the gesture in the given image.

        Args:
            image (mp.Image): The image to recognize the gesture in.

        Returns:
            Gesture: The recognized gesture.
        """
        result = self.__recognizer.recognize(image)
        gestures = self.__parse_result(result)
        return gestures
    
    def __parse_result(self, result) -> list[Gesture]:
        """Parses the result of the gesture recognition.

        Args:
            result (vision.GestureRecognizerResult): The result of the gesture recognition.

        Returns:
            list[Gesture]: All gestures recognized.
        """
        gestures = result.gestures
        ret = [self.__parse_gesture(gesture.category_name) for gesture in gestures]
        return ret
    
    def __parse_gesture(self, gesture: Optional[str]) -> Gesture:
        """Parses the gesture label from GestureRecognizerResult to Gesture enum.

        Args:
            gesture (Optional[str]): category_name of the gesture.
            If None, returns Gesture.UNKNOWN.

        Returns:
            Gesture: The parsed gesture.
        """
        if gesture is None:
            return Gesture.UNKNOWN
        try:
            return Gesture[gesture.upper()]
        except KeyError:
            return Gesture.UNKNOWN
        
    
    def stop(self):
        """
        Stops the gesture recognition process.
        """
        pass
        
    def __get_gestures(self):
        return self.__gestures
        
    gestures = property(__get_gestures)