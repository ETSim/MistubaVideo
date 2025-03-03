# Use an official CUDA base image
FROM nvidia/cuda:12.0.1-cudnn8-devel-ubuntu22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/root/.local/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Install system dependencies for Mitsuba and Python
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    python3.11 \
    python3.11-dev \
    python3-pip \
    python3-setuptools \
    ffmpeg \
    libboost-all-dev \
    libopenexr-dev \
    libglewmx-dev \
    libxrandr-dev \
    libxinerama-dev \
    libxcursor-dev \
    libxi-dev \
    libx11-dev \
    libcolorio-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libfreetype6-dev \
    libfftw3-dev \
    libopenimageio-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create a working directory
WORKDIR /app

# Install mitsuba3
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir mitsuba

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the project files
COPY . .

# Set Python path to include src directory
ENV PYTHONPATH="${PYTHONPATH}:/app:/app/src"

# Create output directory structure
RUN mkdir -p output/exr output/png output/video output/gif output/scenes output/meshes

# Expose the Gradio UI port
EXPOSE 7860

# Set entry point to run our main script
ENTRYPOINT ["python3", "src/main.py"]

# Default command if none is specified at runtime
CMD ["--help"]
