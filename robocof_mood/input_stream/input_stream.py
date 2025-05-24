from abc import ABC, abstractmethod
import numpy as np


class InputStream(ABC):
    @abstractmethod
    def start(self):
        """Start the input stream."""
        pass

    @abstractmethod
    def capture_frame(self, square_crop: bool = False) -> np.ndarray | None:
        """Capture a frame from the input stream.
        
        Args:
            square_crop : bool, optional
                If True, center-crop the frame to a square. Defaults to False.
                
        Returns:
            np.ndarray | None
                The captured frame as a NumPy array, or None if no frame is available.
        """
        pass

    @abstractmethod
    def stop(self):
        """Stop the input stream."""
        pass

    def center_crop_square(frame: np.ndarray) -> np.ndarray:
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
            return frame                     # already square

        if w > h:                            # crop left/right
            offset = (w - h) // 2
            return frame[:, offset:offset + h]
        else:                                # crop top/bottom
            offset = (h - w) // 2
            return frame[offset:offset + w, :]