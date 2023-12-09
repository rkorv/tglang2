FROM debian:10

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y g++ wget make git libssl-dev libomp-dev python3 python3-pip && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Build CMake (required for tensorflow)
RUN cd /build/ && wget https://github.com/Kitware/CMake/releases/download/v3.27.7/cmake-3.27.7.tar.gz && \
    tar -xvf cmake-3.27.7.tar.gz && \
    cd cmake-3.27.7 && \
    ./bootstrap && \
    make -j && \
    make install

# Build TensorFlow Lite from source
RUN git clone https://github.com/tensorflow/tensorflow.git tensorflow_src && \
    cd tensorflow_src && \
    git checkout v2.14.0

RUN mkdir tflite_build && \
    cd tflite_build && \
    cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-O3" -DTFLITE_ENABLE_XNNPACK=ON ../tensorflow_src/tensorflow/lite/c && \
    cmake --build . -j

# Build ONNX Runtime from source
RUN git clone --recursive https://github.com/microsoft/onnxruntime.git && \
    cd onnxruntime && \
    git checkout v1.15.1 && \
    ./build.sh --config RelWithDebInfo --parallel --minimal_build --compile_no_warning_as_error --allow_running_as_root --skip_submodule_sync --cmake_extra_defines CMAKE_OSX_ARCHITECTURES=x86_64
