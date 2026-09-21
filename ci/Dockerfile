FROM nvidia/cuda:12.8.1-devel-ubuntu24.04

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
      ca-certificates \
      clang \
      cmake \
      git \
      libnuma-dev \
      libomp-dev \
      libopenblas-dev \
      pkg-config \
      wget \
 && rm -rf /var/lib/apt/lists/*

ARG UCX_VERSION=1.20.1
RUN wget -qO- https://github.com/openucx/ucx/releases/download/v${UCX_VERSION}/ucx-${UCX_VERSION}.tar.gz | tar xz \
 && cd ucx-${UCX_VERSION} \
 && ./configure --prefix=/opt/ucx --with-cuda=/usr/local/cuda \
      --enable-mt --disable-static --disable-doxygen-doc \
 && make -j"$(nproc)" && make install \
 && cd .. && rm -rf ucx-${UCX_VERSION}

ARG OPENMPI_VERSION=4.1.8
RUN wget -qO- https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-${OPENMPI_VERSION}.tar.bz2 | tar xj \
 && cd openmpi-${OPENMPI_VERSION} \
 && ./configure --prefix=/opt/ompi --with-ucx=/opt/ucx \
      --with-cuda=/usr/local/cuda --with-cuda-libdir=/usr/local/cuda/lib64/stubs \
      --disable-static \
 && make -j"$(nproc)" && make install \
 && cd .. && rm -rf openmpi-${OPENMPI_VERSION}

ENV PATH=/opt/ompi/bin:$PATH
ENV LD_LIBRARY_PATH=/opt/ompi/lib:/opt/ucx/lib:$LD_LIBRARY_PATH
# Avoids ssh/rsh dependency by asserting MPI is local only
ENV OMPI_MCA_plm=isolated

RUN ompi_info --parsable --all | grep -q '^mca:opal:base:param:opal_built_with_cuda_support:value:true$' \
 && ls /opt/ucx/lib/ucx | grep -q libuct_cuda
