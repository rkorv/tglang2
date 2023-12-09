cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

project(tglang VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-O3 -Wno-trigraphs")

include_directories(/data/model)

set(LIB_SOURCES
    tglang.cpp
)

add_library(tglang SHARED ${LIB_SOURCES})

target_compile_definitions(tglang PRIVATE -DINFERENCE_TFLITE)

### ONNX_RUNTIME ###
# set(ONNX_ROOT /build/onnxruntime)
# include_directories(${ONNX_ROOT}/include)
# set(DEPS ${ONNX_ROOT}/build/Linux/RelWithDebInfo/_deps)

# include_directories(/build/tflite_build/flatbuffers/include/)
# target_link_libraries(tglang
#     dl
#     ${ONNX_ROOT}/build/Linux/RelWithDebInfo/libonnxruntime_session.a
#     ${ONNX_ROOT}/build/Linux/RelWithDebInfo/libonnxruntime_optimizer.a
#     ${ONNX_ROOT}/build/Linux/RelWithDebInfo/libonnxruntime_providers.a
#     ${ONNX_ROOT}/build/Linux/RelWithDebInfo/libonnxruntime_util.a
#     ${ONNX_ROOT}/build/Linux/RelWithDebInfo/libonnxruntime_framework.a
#     ${ONNX_ROOT}/build/Linux/RelWithDebInfo/libonnxruntime_graph.a
#     ${ONNX_ROOT}/build/Linux/RelWithDebInfo/libonnxruntime_mlas.a
#     ${ONNX_ROOT}/build/Linux/RelWithDebInfo/libonnxruntime_common.a
#     ${ONNX_ROOT}/build/Linux/RelWithDebInfo/libonnxruntime_flatbuffers.a
#     ${ONNX_ROOT}/build/Linux/RelWithDebInfo/libonnx_test_data_proto.a

#     # version 1.14.1: git checkout c57cf374b67f72575546d7b4c69a1af4972e2b54
#     ${DEPS}/abseil_cpp-build/absl/base/libabsl_base.a
#     ${DEPS}/abseil_cpp-build/absl/base/libabsl_log_severity.a
#     ${DEPS}/abseil_cpp-build/absl/base/libabsl_malloc_internal.a
#     ${DEPS}/abseil_cpp-build/absl/base/libabsl_raw_logging_internal.a
#     ${DEPS}/abseil_cpp-build/absl/base/libabsl_spinlock_wait.a
#     ${DEPS}/abseil_cpp-build/absl/base/libabsl_throw_delegate.a
#     ${DEPS}/abseil_cpp-build/absl/container/libabsl_hashtablez_sampler.a
#     ${DEPS}/abseil_cpp-build/absl/container/libabsl_raw_hash_set.a
#     ${DEPS}/abseil_cpp-build/absl/hash/libabsl_hash.a
#     ${DEPS}/abseil_cpp-build/absl/hash/libabsl_city.a
#     ${DEPS}/abseil_cpp-build/absl/hash/libabsl_low_level_hash.a
#     ${DEPS}/abseil_cpp-build/absl/numeric/libabsl_int128.a
#     ${DEPS}/abseil_cpp-build/absl/profiling/libabsl_exponential_biased.a
#     ${DEPS}/abseil_cpp-build/absl/strings/libabsl_cord.a
#     ${DEPS}/abseil_cpp-build/absl/strings/libabsl_cord_internal.a
#     ${DEPS}/abseil_cpp-build/absl/strings/libabsl_cordz_functions.a
#     ${DEPS}/abseil_cpp-build/absl/strings/libabsl_cordz_handle.a
#     ${DEPS}/abseil_cpp-build/absl/strings/libabsl_cordz_info.a
#     ${DEPS}/abseil_cpp-build/absl/strings/libabsl_strings.a
#     ${DEPS}/abseil_cpp-build/absl/strings/libabsl_strings_internal.a
#     ${DEPS}/abseil_cpp-build/absl/synchronization/libabsl_graphcycles_internal.a
#     ${DEPS}/abseil_cpp-build/absl/synchronization/libabsl_synchronization.a
#     ${DEPS}/abseil_cpp-build/absl/time/libabsl_civil_time.a
#     ${DEPS}/abseil_cpp-build/absl/time/libabsl_time.a
#     ${DEPS}/abseil_cpp-build/absl/time/libabsl_time_zone.a
#     ${DEPS}/abseil_cpp-build/absl/types/libabsl_bad_optional_access.a
#     ${DEPS}/abseil_cpp-build/absl/types/libabsl_bad_variant_access.a
#     ${DEPS}/flatbuffers-build/libflatbuffers.a
#     ${DEPS}/google_nsync-build/libnsync_cpp.a
#     ${DEPS}/onnx-build/libonnx.a
#     ${DEPS}/onnx-build/libonnx_proto.a
#     ${DEPS}/protobuf-build/libprotobuf-lite.a
#     ${DEPS}/protobuf-build/libprotobuf.a
#     ${DEPS}/protobuf-build/libprotoc.a
#     ${DEPS}/pytorch_cpuinfo-build/libcpuinfo.a
#     ${DEPS}/pytorch_cpuinfo-build/deps/clog/libclog.a
#     ${DEPS}/re2-build/libre2.a

#     pthread
#     atomic
# )


### TFLITE ###
include_directories(/build/tensorflow_src)
include_directories(/build/tflite_build/flatbuffers/include/)

set(TFLITE_ROOT /build/tflite_build)

target_link_libraries(tglang
    dl
    ${TFLITE_ROOT}/tensorflow-lite/libtensorflow-lite.a
    ${TFLITE_ROOT}/_deps/fft2d-build/libfft2d_fftsg.a
    ${TFLITE_ROOT}/_deps/fft2d-build/libfft2d_fftsg2d.a
    ${TFLITE_ROOT}/_deps/flatbuffers-build/libflatbuffers.a
    ${TFLITE_ROOT}/_deps/ruy-build/ruy/libruy_ctx.a
    ${TFLITE_ROOT}/_deps/ruy-build/ruy/libruy_allocator.a
    ${TFLITE_ROOT}/_deps/ruy-build/ruy/libruy_frontend.a
    ${TFLITE_ROOT}/_deps/ruy-build/ruy/libruy_context_get_ctx.a
    ${TFLITE_ROOT}/_deps/ruy-build/ruy/libruy_context.a
    ${TFLITE_ROOT}/_deps/ruy-build/ruy/libruy_apply_multiplier.a
    ${TFLITE_ROOT}/_deps/ruy-build/ruy/libruy_prepacked_cache.a
    ${TFLITE_ROOT}/_deps/ruy-build/ruy/libruy_tune.a
    ${TFLITE_ROOT}/_deps/ruy-build/ruy/libruy_cpuinfo.a
    ${TFLITE_ROOT}/_deps/ruy-build/ruy/libruy_system_aligned_alloc.a
    ${TFLITE_ROOT}/_deps/ruy-build/ruy/libruy_prepare_packed_matrices.a
    ${TFLITE_ROOT}/_deps/ruy-build/ruy/libruy_trmul.a
    ${TFLITE_ROOT}/_deps/ruy-build/ruy/libruy_block_map.a
    ${TFLITE_ROOT}/_deps/ruy-build/ruy/libruy_denormal.a
    ${TFLITE_ROOT}/_deps/ruy-build/ruy/libruy_thread_pool.a
    ${TFLITE_ROOT}/_deps/ruy-build/ruy/libruy_blocking_counter.a
    ${TFLITE_ROOT}/_deps/ruy-build/ruy/libruy_wait.a
    ${TFLITE_ROOT}/_deps/ruy-build/ruy/libruy_kernel_avx.a
    ${TFLITE_ROOT}/_deps/ruy-build/ruy/libruy_kernel_avx2_fma.a
    ${TFLITE_ROOT}/_deps/ruy-build/ruy/libruy_kernel_avx512.a
    ${TFLITE_ROOT}/_deps/ruy-build/ruy/libruy_pack_avx.a
    ${TFLITE_ROOT}/_deps/ruy-build/ruy/libruy_pack_avx2_fma.a
    ${TFLITE_ROOT}/_deps/ruy-build/ruy/libruy_pack_avx512.a
    ${TFLITE_ROOT}/_deps/ruy-build/ruy/libruy_have_built_path_for_avx.a
    ${TFLITE_ROOT}/_deps/ruy-build/ruy/libruy_have_built_path_for_avx2_fma.a
    ${TFLITE_ROOT}/_deps/ruy-build/ruy/libruy_have_built_path_for_avx512.a
    ${TFLITE_ROOT}/_deps/farmhash-build/libfarmhash.a
    ${TFLITE_ROOT}/_deps/xnnpack-build/libXNNPACK.a
    ${TFLITE_ROOT}/_deps/cpuinfo-build/libcpuinfo.a
    ${TFLITE_ROOT}/pthreadpool/libpthreadpool.a
)
