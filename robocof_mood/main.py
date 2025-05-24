import httpx
import uvicorn
from fastapi import FastAPI, Request, HTTPException, BackgroundTasks, Depends, Form, File, UploadFile
from pydantic import HttpUrl
from contextlib import asynccontextmanager
from robocof_mood.input_stream.api_mjpeg_input_stream import MJPEGAPIInputStream
from robocof_mood.decision_manager import DecisionManager

LIVESTREAM_URL = "http://10.143.186.203:5000/video_feed"
DEFAULT_TIMEOUT = 15
MAX_TIMEOUT = 120  # 60 * 2


@asynccontextmanager
async def lifespan(app: FastAPI):
    input_stream = MJPEGAPIInputStream(LIVESTREAM_URL)
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


async def _decide_and_callback(dm: DecisionManager, callback: HttpUrl, robot_run_id: int, image_bytes: bytes | None = None):
    if image_bytes:
        # TODO use face recognition
        pass

    try:
        decision = await dm.make_decision()
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


if __name__ == "__main__":
    uvicorn.run("robocof_mood.main:app", host="0.0.0.0", port=8000, reload=True)
