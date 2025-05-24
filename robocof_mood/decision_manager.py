import asyncio
from typing import Optional
from robocof_mood.input_stream.input_stream import InputStream
from robocof_mood.input_stream.webcam_input_stream import WebcamInputStream
from robocof_mood.input_stream.api_mjpeg_input_stream import MJPEGAPIInputStream, smoke_test
from robocof_mood.gesture_recognition.gesture_recognizer import GestureRecognizer, Gesture
from enum import Enum
from robocof_mood.face_recognition.face_recognition import FaceRecognizer


class Decision(Enum):
    USER_ABORT = 0
    CARRY_OUT_ACTION = 1
    TIMEOUT_NO_USER_PRESENT = 2
    TIMEOUT_WRONG_USER_PRESENT = 3
    TIMEOUT_CORRECT_USER_PRESENT = 4
    TIMEOUT = 5
    ERROR = 6


GESTURES_POSITIVE = [Gesture.THUMB_UP, Gesture.CLOSED_FIST]
GESTURES_NEGATIVE = [Gesture.OPEN_PALM]


class DecisionManager:
    """A class to manage the decision-making process for the robot of whether or not to carry out an action."""

    def __init__(
        self, input_stream: InputStream, timeout: int = 15, debug_mode: bool = False
    ):
        """Constructor

        Args:
            live_feed (LiveFeed): The live feed object to get the current image from.
            timeout (int, optional): The timeout in seconds to wait for a decision. Defaults to 15.
            debug_mode (bool, optional): Does not return any decision and only prints debug information. Defaults to False.
        """
        self.input_stream = input_stream
        self.__gesture_recognizer = GestureRecognizer(
            GESTURES_POSITIVE + GESTURES_NEGATIVE, input_stream, debug_mode=debug_mode
        )
        self.__face_recognizer = FaceRecognizer(
            input_stream, debug_mode=debug_mode
        )
        self.__debug_mode = debug_mode
        self.__timeout = timeout if not debug_mode else float("inf")

    async def make_decision(self) -> Decision:
        """
        Makes a decision bpased on the live feed.

        Returns:
            Decision.ABORT if the decision is to abort the action,
            Decision.CARRY_OUT_ACTION if the decision is to carry out the action,
            Decision.TIMEOUT if no other decision was taken within the timeout.
        """

        async def gesture_recognition_task():
            """A task to run the gesture recognition in the background."""
            return await self.__gesture_recognizer.start()

        async def seat_recognition_task():
            """A task to run the seat recognition in the background."""
            # Placeholder for seat recognition logic
            return -1

        async def face_recognition_task():
            """A task to run the face recognition in the background."""
            # Placeholder for face recognition logic
            return await self.__face_recognizer.is_person_in_stream("");

        async def timeout_task(timeout: int):
            """A task to wait for a timeout (in seconds)"""
            await asyncio.sleep(timeout)
            return None

        self.input_stream.start()

        tasks = {
            asyncio.create_task(gesture_recognition_task()): "gesture",
            asyncio.create_task(seat_recognition_task()): "seat",
            asyncio.create_task(face_recognition_task()): "face",
            asyncio.create_task(timeout_task(self.timeout)): "timeout",
        }

        try:
            while tasks:
                done, pending = await asyncio.wait(
                    tasks, return_when=asyncio.FIRST_COMPLETED
                )

                decision = None

                for task in done:
                    result = task.result()
                    task_name = tasks[task]
                    print(f"Task {task_name} completed with result: {result}")

                    if task_name == "gesture":
                        if any(gesture in result for gesture in GESTURES_POSITIVE):
                            decision = Decision.CARRY_OUT_ACTION
                            break
                        if any(gesture in result for gesture in GESTURES_NEGATIVE):
                            decision = Decision.USER_ABORT
                            break

                    elif task_name == "seat":
                        pass
                    elif task_name == "face":
                        result:Optional[bool] = result
                        if result is None:
                            decision = Decision.TIMEOUT_NO_USER_PRESENT
                        elif result == False:
                            decision = Decision.TIMEOUT_WRONG_USER_PRESENT
                        else:
                            decision = Decision.TIMEOUT_CORRECT_USER_PRESENT
                        break
                            

                    elif task_name == "timeout":
                        decision = Decision.TIMEOUT
                        break

                    # Remove the completed task from the dictionary
                    del tasks[task]

                if decision is not None:
                    print(f"Decision made: {decision}")

                    # Cancel pending tasks
                    for task in pending:
                        task.cancel()

                    return decision

        except asyncio.CancelledError as e:
            print(f"Decision-making process was cancelled: {e}")
        finally:
            self.input_stream.stop()

        return Decision.ERROR

    def __get_timeout(self) -> int:
        """Get the timeout for the decision-making process."""
        return self.__timeout

    def __set_timeout(self, timeout: int):
        """Set the timeout for the decision-making process."""
        if not self.__debug_mode:
            self.__timeout = timeout

    timeout = property(__get_timeout, __set_timeout)


# start using command python -m robocof_mood.decision_manager from the root dir
if __name__ == "__main__":
    # Example usage
    input_stream = WebcamInputStream()
    # input_stream = stream = MJPEGAPIInputStream("http://192.168.137.203:8000/video_feed")

    async def main():
        decision_manager = DecisionManager(input_stream, debug_mode=True)
        decision = await decision_manager.make_decision()
        print(f"Final decision: {decision}")

    asyncio.run(main())
