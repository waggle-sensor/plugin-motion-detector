FROM waggle/plugin-base:1.1.1-ml
COPY requirements.txt /app/

RUN pip3 install --no-cache-dir -r /app/requirements.txt
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

COPY . /app/
WORKDIR /app
ENTRYPOINT ["python3", "/app/main.py"]
