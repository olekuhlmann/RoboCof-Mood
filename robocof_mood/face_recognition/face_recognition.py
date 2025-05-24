import asyncio
import numpy as np
import face_recognition
from typing import Optional
from robocof_mood.input_stream.input_stream import InputStream
import cv2


class FaceRecognizer:
    def __init__(
            self,
            input_stream: InputStream,
            face_names=None,
            known_face_encodings=None,
            debug_mode: bool = False,
    ):
        """Constructor

        Args:
            gestures (list[Gesture]): List of gestures to recognize. Will stop active recognition if one of these gestures is detected.
            input_stream (InputStream): The input stream to capture frames from.
            debug_mode (bool, optional): If True, will not return any gesture recognized and will only print debug information. Defaults to False.
        """

        if known_face_encodings is None:
            known_face_encodings = []
        if face_names is None:
            face_names = []
        self.__input_stream = input_stream
        self.__debug_mode = debug_mode
        self.__known_face_encodings = known_face_encodings
        self.__known_face_names = face_names

    def add_face_image(self, name: str, image_path: str):
        """Adds a new face to the recognizer from an image file.

        Args:
            name (str): The name of the person.
            image_path (str): The path to the image file.
        """
        image = face_recognition.load_image_file(image_path)
        encoding = face_recognition.face_encodings(image)[0]
        self.add_face_encoding(name, encoding)

    def find_face_image(self, name: str, image_path: str) -> Optional[np.ndarray]:
        """Finds a face in an image file and returns the face image.

        Args:
            name (str): The name of the person.
            image_path (str): The path to the image file.

        Returns:
            Optional[np.ndarray]: The face image if found, None otherwise.
        """
        image = face_recognition.load_image_file(image_path)
        face_locations = face_recognition.face_locations(image)
        if not face_locations:
            print(f"[Face Recognizer]: No face found for {name} in {image_path}.")
            return None
        top, right, bottom, left = face_locations[0]
        return image[top:bottom, left:right]
    
    def find_face_image_from_array(self, name: str, image: np.ndarray) -> Optional[np.ndarray]:
        """Finds a face in an image array and returns the face image.

        Args:
            name (str): The name of the person.
            image (np.ndarray): The image array.

        Returns:
            Optional[np.ndarray]: The face image if found, None otherwise.
        """
        face_locations = face_recognition.face_locations(image)
        if not face_locations:
            print(f"[Face Recognizer]: No face found for {name}.")
            return None
        top, right, bottom, left = face_locations[0]
        return image[top:bottom, left:right]
    


    def add_face_image_from_array(self, name: str, image: np.ndarray):
        """Adds a new face to the recognizer from an image array.

        Args:
            name (str): The name of the person.
            image (np.ndarray): The image array.
        """
        encoding = face_recognition.face_encodings(image)[0]
        self.add_face_encoding(name, encoding)

    def add_face_encoding(self, name: str, encoding: np.ndarray):
        """Adds a new face to the recognizer.

        Args:
            name (str): The name of the person.
            encoding (np.ndarray): The face encoding of the person.
        """
        self.__known_face_names.append(name)
        self.__known_face_encodings.append(encoding)

    def remove_face_encoding(self, name: str):
        """Removes a face from the recognizer.

        Args:
            name (str): The name of the person to remove.
        """
        if name in self.__known_face_names:
            index = self.__known_face_names.index(name)
            del self.__known_face_names[index]
            del self.__known_face_encodings[index]
        else:
            print(f"[Face Recognizer]: Face with name {name} not found.")

    def get_known_faces(self) -> list[str]:
        """Returns the list of known faces.

        Returns:
            list[str]: The list of known faces.
        """
        return self.__known_face_names

    def get_known_face_encodings(self) -> list[np.ndarray]:
        """Returns the list of known face encodings.

        Returns:
            list[np.ndarray]: The list of known face encodings.
        """
        return self.__known_face_encodings
    
    def is_person_in_image(self, image: np.ndarray, name: str) -> bool:
        """
        Checks if a person is in the given image.

        Args:
            image (np.ndarray): The image to check.
            name (str): The name of the person to check for.

        Returns:
            bool: True if the person is in the image, False otherwise.
        """
        if name not in self.__known_face_names:
            print(f"[Face Recognizer]: Person {name} not found in known faces.")
            return False

        encoding = self.__known_face_encodings[self.__known_face_names.index(name)]
        face_locations = face_recognition.face_locations(image)
        face_encodings = face_recognition.face_encodings(image, face_locations)

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces([encoding], face_encoding)
            if matches[0]:
                return True

        return False

    def recognize(self, image: np.ndarray) -> list[str]:
        """
        Recognizes faces in the given image.

        Args:
            image (np.ndarray): The image to recognize faces in.

        Returns:
            list[str]: The names of the recognized faces.
        """
        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_image)
        face_encodings = face_recognition.face_encodings(rgb_image, face_locations)

        recognized_faces = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(self.__known_face_encodings, face_encoding)
            name = "Unknown"

            # Or instead, use the known face with the smallest distance to the new face
            if True in matches:
                first_match_index = matches.index(True)
                name = self.__known_face_names[first_match_index]

            recognized_faces.append(name)

        return recognized_faces

    async def recognize_async(self, image: np.ndarray) -> list[str]:
        """
        Asynchronously recognizes faces in the given image.

        Args:
            image (np.ndarray): The image to recognize faces in.

        Returns:
            list[str]: The names of the recognized faces.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.recognize, image)

    async def recognize_from_stream(self) -> Optional[list[str]]:
        """
        Asynchronously recognizes faces from the input stream.
        Returns:
            Optional[list[str]]: The names of the recognized faces or None if no faces are recognized.
        """
        frame = await self.__input_stream.get_frame()
        if frame is None:
            return None

        recognized_faces = await self.recognize_async(frame)

        if self.__debug_mode:
            print(f"[Face Recognizer]: Recognized faces: {recognized_faces}")

        return recognized_faces

    async def recognize_from_stream_loop(self):
        """
        Asynchronously recognizes faces from the input stream in a loop.
        This method will continuously capture frames from the input stream and recognize faces in each frame.
        """
        while True:
            recognized_faces = await self.recognize_from_stream()
            if recognized_faces is not None:
                if self.__debug_mode:
                    print(f"[Face Recognizer]: Recognized faces: {recognized_faces}")
            await asyncio.sleep(0.1)

    def start_recognition_loop(self):
        """
        Starts the recognition loop.
        This method will run the recognition loop in a separate thread.
        """
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.recognize_from_stream_loop())

    @staticmethod
    def stop_recognition_loop():
        """
        Stops the recognition loop.
        This method will stop the recognition loop.
        """
        loop = asyncio.get_event_loop()
        loop.stop()
        loop.close()
        print("[Face Recognizer]: Recognition loop stopped.")

    def get_input_stream(self) -> InputStream:
        """
        Returns the input stream used by the recognizer.

        Returns:
            InputStream: The input stream used by the recognizer.
        """
        return self.__input_stream

    def set_input_stream(self, input_stream: InputStream):
        """
        Sets the input stream used by the recognizer.
        Args:
            input_stream (InputStream): The input stream to set.
        """
        self.__input_stream = input_stream
        if self.__debug_mode:
            print(f"[Face Recognizer]: Input stream set to {self.__input_stream}")

    def is_debug_mode(self) -> bool:
        """
        Returns whether the recognizer is in debug mode.

        Returns:
            bool: True if the recognizer is in debug mode, False otherwise.
        """
        return self.__debug_mode

    def set_debug_mode(self, debug_mode: bool):
        """
        Sets whether the recognizer is in debug mode.
        Args:
            debug_mode (bool): True to enable debug mode, False to disable it.
        """
        self.__debug_mode = debug_mode
        if self.__debug_mode:
            print("[Face Recognizer]: Debug mode enabled.")
        else:
            print("[Face Recognizer]: Debug mode disabled.")
