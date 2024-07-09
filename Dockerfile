# Use the official Python image from the Docker Hub
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# Copy the dependencies file to the working directory
COPY requirements.txt .

# Install any dependencies
RUN pip install -r requirements.txt

# Set environment variable for protobuf if choosing the workaround
# ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

# Copy the content of the local src directory to the working directory
COPY . .

# Specify the command to run on container start
ENTRYPOINT ["streamlit", "run", "app.py", "--server.enableCORS=false", "--server.address=0.0.0.0", "--server.port=8080", "--browser.gatherUsageStats=false"]
