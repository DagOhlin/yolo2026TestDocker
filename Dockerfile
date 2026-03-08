# Start from the Ultralytics base image
FROM ultralytics/ultralytics:latest

# Permanently install Flask into the image
RUN pip install flask