FROM waggle/plugin-base:1.1.0-ml-cuda11.0-amd64

# install python3 dev environment
# (some dependencies use use PEP 517,
#  and may need to be built from source):
RUN apt-get update && apt-get install python3-dev build-essential -y

# install pip dependencies:
COPY requirements.txt /app/
RUN pip3 install --upgrade pip setuptools wheel
RUN pip3 install --no-cache-dir -r /app/requirements.txt

# Copy plugin app content:
COPY main.py detector.py capture.py object_tracker.py test.py /app/
COPY test /app/test/

# Establish entrypoint:
# ----- Arguments -----
#   input:      video source device
#   fps:        captured frames per second
#   interval:   minimum interval between data publishes (in seconds)
#   detector:   detector method to be used (one of the following):
#                   1. "bg_subtraction" (simple KNN background subtraction)
#                   2. "dense_optflow"  (dense optical flow; recommended)
#                   3. "yolo"           (small pre-trained YOLO model)
WORKDIR /app
ENTRYPOINT ["python3", "/app/main.py", \
    "--fps",      "20", \
    "--interval", "10", \
    "--detector", "dense_optflow"]
