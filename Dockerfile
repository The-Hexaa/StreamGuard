# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt
RUN apt-get update && apt-get install -y libgl1-mesa-glx
RUN apt-get install -y libglib2.0-0 libsm6 libxext6 libxrender-dev
run apt-get install -y ffmpeg  libavcodec-extra libx264-dev


# Copy the current directory contents into the container at /app
COPY . .

# Make port 8000 available to the world outside this container
EXPOSE 8002

# Define environment variable for running in production
ENV ENVIRONMENT production

# Run FastAPI with uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8002"]
