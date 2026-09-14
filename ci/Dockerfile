FROM nvidia/cuda:12.8.1-devel-ubuntu24.04

# The CUDA toolkit comes from the base image; this is the host toolchain DMK builds against.
# openmpi-bin is named explicitly so mpirun survives --no-install-recommends.
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
      clang \
      cmake \
      git \
      libomp-dev \
      libopenblas-dev \
      libopenmpi-dev \
      openmpi-bin \
      pkg-config \
 && rm -rf /var/lib/apt/lists/*
