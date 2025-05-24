import asyncio
from input_stream.input_stream import InputStream
from input_stream.webcam_input_stream import WebcamInputStream
from gesture_recognition.gesture_recognizer import GestureRecognizer, Gesture
from enum import Enum


class Decision(Enum):
    USER_ABORT = 0
    TIMEOUT = 1
    CARRY_OUT_ACTION = 2
    NO_USER_PRESENT = 3
    ERROR = 4


class DecisionManager:
    """A class to manage the decision-making process for the robot of whether or not to carry out an action."""

    def __init__(self, input_stream: InputStream, timeout: int = 15):
        """Constructor

        Args:
            live_feed (LiveFeed): The live feed object to get the current image from.
        """
        self.input_stream = input_stream
        self.__gesture_recognizer = GestureRecognizer(
            [Gesture.THUMB_UP, Gesture.OPEN_PALM], input_stream
        )
        self.__timeout = timeout

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
            return -1

        async def timeout_task(timeout: int):
            """A task to wait for a timeout (in seconds)"""
            await asyncio.sleep(timeout)
            return None

        tasks = {
            asyncio.create_task(gesture_recognition_task()): "gesture",
            asyncio.create_task(seat_recognition_task()): "seat",
            asyncio.create_task(face_recognition_task()): "face",
            asyncio.create_task(timeout_task(self.__timeout)): "timeout",
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
                        if Gesture.THUMB_UP in result:
                            decision = Decision.CARRY_OUT_ACTION
                            break
                        if Gesture.OPEN_PALM in result:
                            decision = Decision.USER_ABORT
                            break

                    elif task_name == "seat":
                        pass
                    elif task_name == "face":
                        pass

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

        return Decision.ERROR


if __name__ == "__main__":
    # Example usage
    input_stream = WebcamInputStream()
    input_stream.start()
    decision_manager = DecisionManager(input_stream)

    async def main():
        decision = await decision_manager.make_decision()
        input_stream.stop()
        print(f"Final decision: {decision}")

    asyncio.run(main())
