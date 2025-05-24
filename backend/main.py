from fastapi import FastAPI, HTTPException, Request
from pydantic import PositiveInt
from time import time
from robocof_mood.decision_manager import DecisionManager
from robocof_mood.input_stream.webcam_input_stream import WebcamInputStream

app = FastAPI()

# Default timeout in seconds
DEFAULT_TIMEOUT = 15
MAX_TIMEOUT = 60  # Maximum timeout allowed for a decision request

# Rate limit configuration (simple in-memory storage of timestamps)
# Set a maximum of 5 requests per 30 seconds
RATE_LIMIT_WINDOW = 30  # in seconds
MAX_REQUESTS = 5
request_timestamps = {}


# Helper function to check rate limit
def is_rate_limited(ip: str) -> bool:
    current_time = time()
    if ip not in request_timestamps:
        request_timestamps[ip] = []

    # Remove outdated timestamps (older than RATE_LIMIT_WINDOW seconds)
    request_timestamps[ip] = [
        timestamp for timestamp in request_timestamps[ip] if current_time - timestamp < RATE_LIMIT_WINDOW
    ]

    # Check if the rate limit is exceeded
    if len(request_timestamps[ip]) >= MAX_REQUESTS:
        return True

    # Otherwise, log the current timestamp and allow the request
    request_timestamps[ip].append(current_time)
    return False



# Helper function for decision_making
async def make_decision(request: Request, timeout: int = DEFAULT_TIMEOUT):
    # Get client IP address for rate-limiting
    client_ip = request.client.host #"127.0.0.1"  # TODO get this from the request headers

    # Check if the request exceeds the rate limit
    if is_rate_limited(client_ip):
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Try again later.")

    input_stream = WebcamInputStream()
    input_stream.start()
    decision_manager = DecisionManager(input_stream=input_stream, timeout=timeout)

    try:
        decision = await decision_manager.make_decision()
        input_stream.stop()
        return {"decision": decision.name}
    except Exception as e:
        input_stream.stop()
        raise HTTPException(status_code=500, detail=f"Error making decision: {str(e)}")


# Default route
@app.get("/")
async def root():
    return {"message": "Welcome to the RoboCof decision-making API!"}


@app.get("/make_decision")
async def make_decision_with_default_timeout(request: Request):
    """
    Endpoint to trigger decision-making with the default timeout.
    """
    return await make_decision(request=request)


@app.get("/make_decision/{timeout}")
async def make_decision_with_custom_timeout(request: Request, timeout: PositiveInt):
    """
    Endpoint to trigger decision-making with a custom timeout.
    """
    if timeout > MAX_TIMEOUT:
        raise HTTPException(status_code=400, detail=f"Timeout must be less than {MAX_TIMEOUT} seconds")

    return await make_decision(request=request, timeout=timeout)

