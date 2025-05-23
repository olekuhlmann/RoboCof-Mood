import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from enum import Enum


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
    def __init__(self, gestures: list[Gesture]):
        """Constructor

        Args:
            gestures (list[Gesture]): List of gestures to recognize. Will stop active recognition if one of these gestures is detected.
        """
        self.__gestures = gestures
        base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
        options = vision.GestureRecognizerOptions(base_options=base_options)
        self.__recognizer = vision.GestureRecognizer.create_from_options(options)
        
    def start(self) -> Gesture:
        """
        Starts the gesture recognition process. Returns the recognized gesture once a gesture from `gestures` is recognized.
        
        Returns:
            Gesture: The recognized gesture.
        """        
        pass
    
    def recognize(self, image: mp.Image) -> Gesture:
        """
        Recognizes the gesture in the given image.

        Args:
            image (mp.Image): The image to recognize the gesture in.

        Returns:
            Gesture: The recognized gesture.
        """
        result = self.__recognizer.recognize(image)
        gesture = self.__parse_result(result)
        return gesture
    
    def __parse_result(self, result) -> Gesture:
        """Parses the result of the gesture recognition.
        This method is called when a gesture is recognized.

        Args:
            result (vision.GestureRecognizerResult): The result of the gesture recognition.

        Returns:
            Gesture: Gesture recognized.
        """
        pass
        
    
    def stop(self):
        """
        Stops the gesture recognition process.
        """
        pass
        
    def __get_gestures(self):
        return self.__gestures
        
    gestures = property(__get_gestures)