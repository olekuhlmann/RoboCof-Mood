from abc import ABC, abstractmethod
import cv2
import numpy as np


class InputStream(ABC):
    @abstractmethod
    def start(self):
        """Start the input stream."""
        pass

    @abstractmethod
    def capture_frame(
        self, square_crop: bool = False, transform: bool = False
    ) -> np.ndarray | None:
        """Capture a frame from the input stream.

        Args:
            square_crop : bool, optional
                If True, center-crop the frame to a square. Defaults to False.
            transform : bool, optional
                If True, apply transformations to the frame (e.g., greyscale, contrast, brightness). Defaults to False.

        Returns:
            np.ndarray | None
                The captured frame as a NumPy array, or None if no frame is available.
        """
        pass

    @abstractmethod
    def stop(self):
        """Stop the input stream."""
        pass

    def center_crop_square(self, frame: np.ndarray) -> np.ndarray:
        """
        Center-crop an HxWxC BGR/RGB frame to the largest possible square.

        Args:
            frame : np.ndarray
                Original image with shape (height, width, channels).

        Returns:
            np.ndarray
                Square crop (zero-copy).
        """
        if frame.ndim != 3:
            raise ValueError("Expected an image with shape (H, W, C)")

        h, w = frame.shape[:2]

        if h == w:
            return frame  # already square

        if w > h:  # crop left/right
            offset = (w - h) // 2
            return frame[:, offset : offset + h]
        else:  # crop top/bottom
            offset = (h - w) // 2
            return frame[offset : offset + w, :]

    def transform_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Convert a BGR/RGB frame to greyscale and apply contrast and brightness adjustments.

        Args:
            frame : np.ndarray
                Original image with shape (height, width, channels).

        Returns:
            np.ndarray
                Greyscale image with adjusted contrast and brightness.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        equalized = cv2.equalizeHist(gray)
        frame = cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)

        alpha = 1.2  # contrast
        beta = 20  # brightness
        frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)

        return frame
