from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from robocof_mood.input_stream.webcam_input_stream import WebcamInputStream
from robocof_mood.decision_manager import DecisionManager

app = FastAPI()


# Define a Pydantic model for the request body
class DecisionRequest(BaseModel):
    timeout: int = 15  # Default timeout if not provided


@app.get("/make_decision/")
async def make_decision(timeout: int = 15):
    """
    Make a decision based on the webcam input.
    The `timeout` parameter is optional and defaults to 15 seconds.
    """

    input_stream = WebcamInputStream()
    input_stream.start()

    # Initialize DecisionManager with the requested timeout
    decision_manager = DecisionManager(input_stream, timeout=timeout)

    try:
        # Make decision asynchronously
        decision = await decision_manager.make_decision()
        input_stream.stop()
        return {"decision": decision.name}
    except Exception as e:
        input_stream.stop()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
def read_root():
    return {"message": "Welcome to RoboCof Decision API!"}
