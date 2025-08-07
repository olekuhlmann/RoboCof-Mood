# RoboCof-Mood 🤖☕

**RoboCof-Mood** acts as one of three components of the **RoboCof** application, developed as part of the hackathon _Code Your Way Stockholm!_ held by Itestra in May 2025.
Our module is responsible for the final, crucial step of the RoboCof application: After an employee has sent the robot to autonomously find their favourite colleague, we analyse the recipient colleague's "mood" to decide whether they are available and willing to be approached by the robot. Using computer vision, we check for explicit consent via gestures while also determining if the person is present at their desk.

## The RoboCof Project

In the vast Itestra Munich office, a clean desk policy can make it a real challenge to find your favorite colleagues for a quick chat or a coffee break. The RoboCof project is a fun, innovative solution to this problem.

An employee can use the **RoboCof-App** to send a message or a fun greeting to a colleague. A small, autonomous robot is then dispatched, using the **RoboCof-Pathfinding** module to navigate the office and find the recipient's desk. Once the robot arrives, our **RoboCof-Mood** module takes over to ensure the interaction is appropriate and non-disruptive.

The project is a collaboration between three teams, with each repository representing a core component:

  * **[RoboCof-App](https://github.com/amanuelsen/Fullstack-Itestra)**: The main web application for sending requests.
  * **[RoboCof-Pathfinding](https://github.com/djolodjolo23/Robot-Team-2)**: The robot's navigation and pathfinding system.
  * **RoboCof-Mood** (This repository): The final decision-making engine at the destination.


## How RoboCof-Mood Works

Our service is initiated via a single API call from the main app once the robot arrives at the target colleague's desk. The system analyzes the robot's camera feed to make a decision and then sends the outcome to a provided callback URL.

The core principle is **strict opt-in**: the robot will only perform its action if the colleague gives explicit, positive consent. In all other cases, the interaction is gracefully aborted.

### API & Decision Manager

  * **API Endpoint**: The process begins when a `POST` request is sent to the `/decision` endpoint in `main.py`. This request includes a callback URL and a timeout for the decision process.
  * **Decision Manager**: The `decision_manager.py` orchestrates the different recognition modules concurrently. It immediately terminates and makes a decision upon detecting an opt-in or opt-out gesture. If no gesture is detected before the timeout, it uses data from the other modules to provide a reason for aborting.

### Recognition Modules

#### 👍 Gesture Recognition

This is the most critical module for our opt-in system. It uses **Google's MediaPipe Gesture Recognizer** to analyze the video stream from the robot's camera.

  * **Opt-In (`THUMBS_UP`)**: If the colleague gives a thumbs-up, the decision is `CARRY_OUT_ACTION`, and the robot proceeds with its action (e.g., delivering a message).
  * **Opt-Out (`OPEN_PALM`)**: If the colleague shows an open palm (stop gesture), the decision is `USER_ABORT`, and the robot leaves without performing the action.

#### 🪑 Seat Recognition

Running in parallel, this module determines the status of the colleague's chair using the **YOLOv5 object detection model**. It checks if the seat is occupied or empty. This is primarily used to provide context if the gesture recognition times out. For example, if no gesture is seen and the seat is empty, the robot can report that the colleague wasn't present.

#### 👤 Face Recognition

This module also runs concurrently to identify who is at the desk, using the Python `face_recognition` library. This helps differentiate between several timeout scenarios:

  * The correct colleague is present but didn't respond.
  * A different, unknown colleague is at the desk.
  * No person is visible at all.

### Final Decision Outcomes

The `DecisionManager` returns one of the following outcomes to the main app:

| Decision                       | Reason                                                                      |
| ------------------------------ | --------------------------------------------------------------------------- |
| `CARRY_OUT_ACTION`             | The user gave a **THUMBS\_UP** gesture (Opt-In).                             |
| `USER_ABORT`                   | The user gave an **OPEN\_PALM** gesture (Opt-Out).                           |
| `TIMEOUT_NO_USER_PRESENT`      | Timeout reached. The seat recognizer determined the desk is empty.          |
| `TIMEOUT_WRONG_USER_PRESENT`   | Timeout reached. A person is present, but it's not the target colleague.    |
| `TIMEOUT_CORRECT_USER_PRESENT` | Timeout reached. The correct colleague is present but gave no clear signal. |

## Installation and Usage

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/olekuhlmann/RoboCof-Mood.git
    cd RoboCof-Mood
    ```

2.  **Create a virtual environment and install dependencies:**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```

3.  **Run the application:**
    The application is built with FastAPI. To start the server, run:

    ```bash
    uvicorn robocof_mood.main:app --reload
    ```

    The API will be available at `http://127.0.0.1:8000`.

## Contributors

This project was brought to life by:

  * **Ole Kuhlmann** ([@olekuhlmann](https://github.com/olekuhlmann)) - Team Lead & Gesture Recognition
  * **Jonas Jostan** ([@halo34](https://github.com/halo34)) - Face Recognition
  * **Uku** ([@uuuuuu-k](https://github.com/uuuuuu-k)) - Seat Recognition
  * **Lauri** - Gesture Recognition

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
