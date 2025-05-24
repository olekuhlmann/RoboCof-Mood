import httpx
from fastapi import FastAPI, Request, BackgroundTasks, Depends, Form, File, UploadFile, HTTPException
from pydantic import HttpUrl, BaseModel, Field
from contextlib import asynccontextmanager
from robocof_mood.input_stream.api_mjpeg_input_stream import MJPEGAPIInputStream
from robocof_mood.input_stream.webcam_input_stream import WebcamInputStream
from robocof_mood.decision_manager import DecisionManager
import numpy as np
import cv2

LIVESTREAM_URL = "http://192.168.137.204:8000/video_feed"
# Default timeout in seconds
DEFAULT_TIMEOUT = 15
MAX_TIMEOUT = 120  # 60 * 2


@asynccontextmanager
async def lifespan(app: FastAPI):
    input_stream = MJPEGAPIInputStream(LIVESTREAM_URL)
    #input_stream = WebcamInputStream() # for debugging
    decision_manager = DecisionManager(input_stream, timeout=DEFAULT_TIMEOUT)
    app.state.decision_manager = decision_manager
    try:
        yield
    finally:
        print("Application shutdown complete.")


app = FastAPI(lifespan=lifespan)


def get_dm(request: Request) -> DecisionManager:
    return request.app.state.decision_manager


@app.get("/")
async def root():
    return {"message": "Welcome to the RoboCof decision-making API!"}


async def _decide_and_callback(dm: DecisionManager, callback: HttpUrl, robot_run_id: int, image_bytes: bytes | None = None, name: str = "user"):
    if image_bytes:
        # Convert bytes to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        # Decode image (assuming JPEG)
        img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_np is None:
            print("[image] Failed to decode image bytes")
        dm.set_image_from_user(name, img_np)

    try:
        decision = await dm.make_decision(name=name)
    except Exception as exc:
        print(f"[decision] failed: {exc}")
        return

    payload = {"decision": str(decision), "robot_run_id": robot_run_id}

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.post(str(callback), json=payload)
            r.raise_for_status()
    except Exception as exc:
        print(f"[callback] POST {callback} failed: {exc}")


@app.post("/decision", status_code=202)
async def decision_entrypoint(
    background_tasks: BackgroundTasks,
    image: UploadFile | None = File(default=None),
    callback_url: HttpUrl = Form(...),
    robot_run_id: int = Form(...),
    timeout: int = Form(DEFAULT_TIMEOUT),
    dm: DecisionManager = Depends(get_dm),
):
    if timeout < 1 or timeout > MAX_TIMEOUT:
        raise HTTPException(status_code=400, detail=f"Timeout must be between 1 and {MAX_TIMEOUT} seconds.")

    image_bytes = None if image is None else image.file.read()

    dm.timeout = timeout

    background_tasks.add_task(
        _decide_and_callback, dm, callback_url, robot_run_id, image_bytes
    )

    return {"detail": "Decision accepted, result will be sent to callback"}



# start using uvicorn robocof_mood.main:app --reload   