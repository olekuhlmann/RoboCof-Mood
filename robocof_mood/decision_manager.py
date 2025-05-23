from input_stream.input_stream import InputStream
from input_stream.webcam_input_stream import WebcamInputStream
from gesture_recognition.gesture_recognizer import GestureRecognizer, Gesture
from enum import Enum



class Decision(Enum):
    ABORT = 0
    TIMEOUT = 1
    CARRY_OUT_ACTION = 2

class DecisionManager:
    """A class to manage the decision-making process for the robot of whether or not to carry out an action.
    """
    
    def __init__(self, input_stream: InputStream):
        """Constructor

        Args:
            live_feed (LiveFeed): The live feed object to get the current image from.
        """
        self.input_stream = input_stream
        self.__gesture_recognizer = GestureRecognizer([Gesture.THUMB_UP, Gesture.OPEN_PALM], input_stream)
        
    def make_decision(self) -> Decision: # TODO make it async again
        """
        Makes a decision bpased on the live feed.
        
        Returns:
            Decision.ABORT if the decision is to abort the action,
            Decision.CARRY_OUT_ACTION if the decision is to carry out the action,
            Decision.TIMEOUT if no other decision was taken within the timeout.
        """
        gestures = self.__gesture_recognizer.start()
        print(f"Recognized gestures: {gestures}")
        
        return Decision.CARRY_OUT_ACTION if gestures else Decision.ABORT # dummy TODO change to real decision-making process
        
        

if __name__ == "__main__":
    # Example usage
    input_stream = WebcamInputStream()
    input_stream.start()
    decision_manager = DecisionManager(input_stream)
    decision = decision_manager.make_decision()
    print(f"Decision: {decision}")
    input_stream.stop()