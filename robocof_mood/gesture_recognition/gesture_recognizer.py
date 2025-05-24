import cv2
from mediapipe.python.solutions import drawing_utils as mp_drawing
from mediapipe.python.solutions import hands as mp_hands
from mediapipe.framework.formats import landmark_pb2





import asyncio
from time import sleep
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from enum import Enum
from typing import Optional
from robocof_mood.input_stream.input_stream import InputStream


MODEL_PATH = "models/gesture_recognizer.task"


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
    def __init__(self, gestures: list[Gesture], input_stream: InputStream, debug_mode: bool = True):
        """Constructor

        Args:
            gestures (list[Gesture]): List of gestures to recognize. Will stop active recognition if one of these gestures is detected.
            input_stream (InputStream): The input stream to capture frames from.
            debug_mode (bool, optional): If True, will not return any gesture recognized and will only print debug information. Defaults to False.
        """
        self.__gestures = gestures
        base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
        options = vision.GestureRecognizerOptions(
            base_options=base_options, num_hands=2,
            min_hand_detection_confidence=0.2,
            min_hand_presence_confidence=0.2,
            min_tracking_confidence=0.2
        )
        self.__recognizer = vision.GestureRecognizer.create_from_options(options)
        self.__input_stream = input_stream
        self.__debug_mode = debug_mode

    async def start(
        self,
    ) -> list[Gesture]:
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

            # In capture_frame() before returning
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            equalized = cv2.equalizeHist(gray)
            frame = cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)

            alpha = 1.2  # contrast
            beta = 20  # brightness
            frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)

            # Convert the frame to a MediaPipe Image object
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

            # Recognize the gestures in the current frame
            gestures = self.recognize(mp_image)

            if gestures:
                # filter the recognized gestures to check if any of them are in the list of gestures
                recognized_gestures = [
                    gesture for gesture in gestures if gesture in self.__gestures
                ]
                if recognized_gestures:
                    if not self.__debug_mode:
                        return recognized_gestures

            # yield control to allow other tasks to run
            await asyncio.sleep(0.01)

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
        print(f"[DEBUG] Recognized gestures: {gestures}")

        if self.__debug_mode:
            self.__visualize_debug(image, result)

        return gestures

    def __parse_result(self, result) -> list[Gesture]:
        """Parses the result of the gesture recognition.

        Args:
            result (vision.GestureRecognizerResult): The result of the gesture recognition.

        Returns:
            list[Gesture]: All gestures recognized.
        """
        gestures = result.gestures
        ret = [
            self.__parse_gesture(gesture.category_name)
            for gesture_list in gestures
            for gesture in gesture_list
        ]
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

    def __visualize_debug(self, mp_image: mp.Image, result):
        frame_rgb = mp_image.numpy_view()
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        # ✅ Convert task landmarks to protobuf landmarks
        if hasattr(result, "hand_landmarks") and result.hand_landmarks:
            for task_landmarks in result.hand_landmarks:
                proto_landmarks = landmark_pb2.NormalizedLandmarkList(
                    landmark=[
                        landmark_pb2.NormalizedLandmark(
                            x=l.x,
                            y=l.y,
                            z=l.z
                        ) for l in task_landmarks
                    ]
                )
                mp_drawing.draw_landmarks(
                    frame_bgr,
                    proto_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2),
                )

        # Draw gesture labels if available
        if result.gestures:
            for gesture_list in result.gestures:
                for gesture in gesture_list:
                    category = gesture.category_name
                    score = gesture.score
                    cv2.putText(
                        frame_bgr,
                        f"{category} ({score:.2f})",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 255, 255),
                        2,
                    )

        cv2.imshow("Gesture Debug", frame_bgr)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            exit()