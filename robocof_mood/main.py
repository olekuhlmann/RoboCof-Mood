import httpx
from fastapi import FastAPI, Request, BackgroundTasks, Depends
from pydantic import HttpUrl, BaseModel, Field
from contextlib import asynccontextmanager
from robocof_mood.input_stream.api_mjpeg_input_stream import MJPEGAPIInputStream
from robocof_mood.input_stream.webcam_input_stream import WebcamInputStream
from robocof_mood.decision_manager import DecisionManager

LIVESTREAM_URL = "http://192.168.137.204:8000/video_feed"
# Default timeout in seconds
DEFAULT_TIMEOUT = 15
MAX_TIMEOUT = 60 * 2  # Maximum timeout allowed for a decision request


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for the FastAPI application.
    This function is called when the application starts up and shuts down.

    Exposes the `decision_manager` instance to the application context.
    """
    # ------------- STARTUP ------------- #
    input_stream = MJPEGAPIInputStream(LIVESTREAM_URL)
    #input_stream = WebcamInputStream() # for debugging
    decision_manager = DecisionManager(input_stream, timeout=DEFAULT_TIMEOUT)

    app.state.decision_manager = decision_manager

    try:
        yield  # Application is running
    finally:
        # ------------- SHUTDOWN ------------- #
        print("Application shutdown complete.")


app = FastAPI(lifespan=lifespan)


def get_dm(request: Request) -> DecisionManager:
    return request.app.state.decision_manager


# Default route
@app.get("/")
async def root():
    return {"message": "Welcome to the RoboCof decision-making API!"}


class DecisionRequest(BaseModel):
    callback_url: HttpUrl
    robot_run_id: int = Field(ge=1, description="Unique identifier for the robot run")
    timeout: int = Field(DEFAULT_TIMEOUT, ge=1, le=MAX_TIMEOUT)


async def _decide_and_callback(
    dm: DecisionManager, callback: HttpUrl, robot_run_id: int
):
    """
    1. Wait for the DecisionManager to finish.
    2. POST the result to the client-supplied callback URL.
    """
    try:
        decision = await dm.make_decision()
    except Exception as exc:  # still log or alert
        print(f"[decision] failed: {exc}")
        return

    payload = {"decision": str(decision), "robot_run_id": robot_run_id}

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.post(str(callback), json=payload)
            r.raise_for_status()
    except Exception as exc:
        print(f"[callback] POST {callback} failed: {exc}")


# Get a decision request
@app.post("/decision", status_code=202)
async def decision_entrypoint(
    body: DecisionRequest,
    background_tasks: BackgroundTasks,
    dm: DecisionManager = Depends(get_dm),
):
    """
    1. Validate timeout bounds (handled by Pydantic).
    2. Adjust DecisionManager timeout.
    3. Kick off background task.
    4. Respond 202 Accepted immediately.
    """
    dm.timeout = body.timeout

    # Fire-and-forget
    background_tasks.add_task(
        _decide_and_callback, dm, body.callback_url, body.robot_run_id
    )

    return {"detail": "Decision accepted, result will be sent to callback"}



# start using uvicorn robocof_mood.main:app --reload   