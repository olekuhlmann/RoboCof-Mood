# Use official Python 3.12 slim image
FROM python:3.12-slim

# Install system dependencies required by OpenCV and libGL
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Create and activate a virtual environment
RUN python -m venv /env

# Use the virtual environment
ENV PATH="/env/bin:$PATH"

# Copy requirements and install dependencies inside the virtual environment
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy the rest of your code
COPY . .

# Expose port for FastAPI (documentation only)
EXPOSE 8000

# Run the app with uvicorn
CMD ["uvicorn", "robocof_mood.main:app", "--host", "0.0.0.0", "--port", "8000"]
