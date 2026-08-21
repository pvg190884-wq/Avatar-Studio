FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

WORKDIR /app
RUN apt-get update && apt-get install -y ffmpeg libgl1 libglib2.0-0 curl && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/OpenTalker/SadTalker.git
WORKDIR /app/SadTalker

RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir "imageio>=2.31,<2.34" runpod

RUN bash scripts/download_models.sh

RUN mkdir -p gfpgan/weights && \
    curl -L -o gfpgan/weights/alignment_WFLW_4HG.pth \
    https://github.com/xinntao/facexlib/releases/download/v0.1.0/alignment_WFLW_4HG.pth

COPY handler.py /app/handler.py

CMD ["python", "-u", "/app/handler.py"]
