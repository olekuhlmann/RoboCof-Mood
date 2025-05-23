from abc import ABC, abstractmethod


class InputStream(ABC):
    @abstractmethod
    def start(self):
        """Start the input stream."""
        pass

    @abstractmethod
    def capture_frame(self):
        """Capture a frame from the input stream."""
        pass

    @abstractmethod
    def stop(self):
        """Stop the input stream."""
        pass