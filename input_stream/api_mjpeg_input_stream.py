from __future__ import annotations
from input_stream.input_stream import InputStream

import cv2
import numpy as np
import requests, threading, time
from collections import deque


class MJPEGAPIInputStream(InputStream):
    """
    Reads a multipart/x-mixed-replace MJPEG stream in a background thread
    and always exposes the **latest decoded frame** via `capture_frame()`.
    Designed for >30 fps, limited only by network + decode time.
    """

    def __init__(
        self,
        url: str,
        *,
        boundary: bytes = b"--frame",
        chunk_size: int = 8192,
        max_queue: int = 3,  # how many JPEGs to hold undecoded
        timeout: float = 5.0,
        headers: dict | None = None,
    ):
        self.url = url
        self.boundary = boundary + b"\r\n"  # match server delimiter
        self.chunk_size = chunk_size
        self.max_queue = max_queue
        self.timeout = timeout
        self.headers = headers or {}

        self._session: requests.Session | None = None
        self._worker: threading.Thread | None = None
        self._stop_flag = threading.Event()
        self._latest_frame: np.ndarray | None = None
        self._frame_lock = threading.Lock()
        self._jpeg_q = deque(maxlen=max_queue)

    # ------------------------------------------------------------------ #
    # public API required by InputStream
    # ------------------------------------------------------------------ #
    def start(self):
        self._session = requests.Session()
        self._worker = threading.Thread(target=self._reader, daemon=True)
        self._worker.start()

    def capture_frame(
        self, square_crop: bool = False, transform: bool = False
    ) -> np.ndarray | None:
        """
        Returns a *copy* of the most recent frame so the caller can mutate it
        without racing the reader thread. Returns None until first frame lands.
        """
        with self._frame_lock:
            if self._latest_frame is None:
                return None

            if square_crop:
                frame = self.center_crop_square(self._latest_frame.copy())
            else:
                frame = self._latest_frame.copy()

            if transform:
                frame = self.transform_frame(frame)

            # 🖼️ Show the frame for debugging
            cv2.imshow("Debug Frame", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                exit()  # Press 'q' to exit the window
            return frame

    def stop(self):
        self._stop_flag.set()
        if self._worker:
            self._worker.join(timeout=3)
        if self._session:
            self._session.close()

    # ------------------------------------------------------------------ #
    # internal reader
    # ------------------------------------------------------------------ #
    def _reader(self):
        try:
            with self._session.get(
                self.url, headers=self.headers, stream=True, timeout=self.timeout
            ) as resp:
                resp.raise_for_status()

                # consume the HTTP stream
                buf = bytearray()
                boundary = self.boundary

                for chunk in resp.iter_content(self.chunk_size):
                    if self._stop_flag.is_set():
                        break
                    buf.extend(chunk)

                    # split whenever we already have *two* boundaries in buffer;
                    # that guarantees a full JPEG part in between.
                    while True:
                        first = buf.find(boundary)
                        if first == -1:
                            break
                        second = buf.find(boundary, first + len(boundary))
                        if second == -1:
                            break  # need more bytes

                        part = bytes(
                            buf[first + len(boundary) : second]
                        )  # includes headers+JPEG
                        del buf[:second]  # consume up to the second boundary

                        # Strip headers — everything up to \r\n\r\n
                        header_end = part.find(b"\r\n\r\n")
                        if header_end == -1:
                            continue  # malformed
                        jpeg_bytes = part[header_end + 4 :]  # skip the empty line

                        self._jpeg_q.append(jpeg_bytes)

                        # Keep only the *last* JPEG if decoding lags behind
                        while self._jpeg_q:
                            jpg = self._jpeg_q.pop()
                            img = cv2.imdecode(
                                np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR
                            )
                            if img is not None:
                                with self._frame_lock:
                                    self._latest_frame = img
                                break  # decoded newest; drop older ones

        except Exception as exc:
            print(f"[MJPEG reader] stopped because: {exc}")


def smoke_test():
    stream = MJPEGAPIInputStream("http://10.143.186.203:5000/video_feed")
    stream.start()
    t0 = time.time()
    frames = 0
    while True:
        frame = stream.capture_frame()
        if frame is not None:
            frames += 1
            cv2.imshow("MJPEG Feed", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    stream.stop()
    cv2.destroyAllWindows()
    print(f"Captured {frames} frames in {time.time() - t0:.1f} s")
