from input_stream.input_stream import InputStream
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
        self.__live_feed = input_stream
        
    async def make_decision(self) -> Decision:
        """
        Makes a decision based on the live feed.
        
        Returns:
            Decision.ABORT if the decision is to abort the action,
            Decision.CARRY_OUT_ACTION if the decision is to carry out the action,
            Decision.TIMEOUT if no other decision was taken within the timeout.
        """
        pass
        
        