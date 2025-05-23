from .input_stream import InputStream
import cv2

class WebcamInputStream(InputStream):
    def __init__(self):
        self.cap = None

    def start(self):
        """Start capturing frames from the webcam."""
        self.cap = cv2.VideoCapture(0)  # 0 is the default camera

        if not self.cap.isOpened():
            print("Error: Could not access the webcam.")
            exit()

    def capture_frame(self):
        """Capture and return a frame from the webcam."""
        ret, frame = self.cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            return None
        
        # 🖼️ Show the frame for debugging
        cv2.imshow("Debug Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            exit()  # Press 'q' to exit the window
        
        return frame

    def stop(self):
        """Release the webcam and close OpenCV windows."""
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()