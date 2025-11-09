# Pix2Pix (seg->photo) Dockerfile
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn8-runtime

WORKDIR /workspace

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

ENV PYTHONUNBUFFERED=1

# Copy only code; data and outputs are mounted
COPY src/ src/

RUN mkdir -p /workspace/outputs

CMD ["python", "src/train_pix2pix.py"]
