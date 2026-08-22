FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

WORKDIR /app
RUN apt-get update && apt-get install -y ffmpeg libgl1 libglib2.0-0 curl espeak-ng && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/OpenTalker/SadTalker.git
WORKDIR /app/SadTalker

RUN python3 -c "\
import re, glob; \
[open(f, 'w').write(re.sub(r'\\bnp\\.bool\\b', 'bool', re.sub(r'\\bnp\\.int\\b', 'int', re.sub(r'\\bnp\\.float\\b', 'float', open(f).read())))) for f in glob.glob('/app/SadTalker/**/*.py', recursive=True)]"

RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir "imageio>=2.31,<2.34" runpod

RUN bash scripts/download_models.sh

RUN mkdir -p gfpgan/weights && \
    curl -L -o gfpgan/weights/alignment_WFLW_4HG.pth \
    https://github.com/xinntao/facexlib/releases/download/v0.1.0/alignment_WFLW_4HG.pth

RUN pip install --no-cache-dir "coqui-tts==0.25.2"
ENV COQUI_TOS_AGREED=1
RUN python -c "from TTS.api import TTS; TTS('tts_models/multilingual/multi-dataset/xtts_v2')"

COPY handler.py /app/handler.py

CMD ["python", "-u", "/app/handler.py"]
