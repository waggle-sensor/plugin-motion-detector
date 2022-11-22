FROM waggle/plugin-base:1.1.1-base

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

COPY requirements.txt /app/
RUN pip3 install --upgrade pip \
  && pip3 install --no-cache-dir -r /app/requirements.txt

COPY . /app/
WORKDIR /app
ENTRYPOINT ["python3", "/app/main.py"]
