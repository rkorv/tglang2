#!/bin/bash
cd "$(dirname "$0")"/..

# Build environment
docker build -f ./build.Dockerfile -t tglang_build .

# Build libtglang
docker run --rm -it -v `pwd`/data:/data -v `pwd`:/app tglang_build bash -c "cd /app && mkdir -p build/libtglang && cd build/libtglang && cmake ../../libtglang && make"

cp ./build/libtglang/libtglang.so ../