FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Set up environment
ENV DEBIAN_FRONTEND=noninteractive

RUN echo 'APT::Sandbox::User "root";' > /etc/apt/apt.conf.d/sandbox-disable

RUN apt-get update && \
    apt-get install -y python3 python3-pip python3-venv git ffmpeg libsm6 libxext6 && \
    rm -rf /var/lib/apt/lists/*

# Set python3 as default
RUN ln -sf /usr/bin/python3 /usr/bin/python

# Create working directory
WORKDIR /workspace

# Copy project files
COPY . /workspace

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu121

# (Optional) Install other dependencies from requirements.txt if present
COPY requirements.txt /workspace/requirements.txt
RUN pip install -r requirements.txt || true

# Set entrypoint
CMD ["python", "train.py", "--help"]
