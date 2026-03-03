# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.15)
project(EmbeddingGemmaBenchmark LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenVINO GenAI (setupvars.ps1 sets OpenVINOGenAI_DIR / OpenVINO_DIR)
find_package(OpenVINOGenAI REQUIRED
    PATHS
        "${CMAKE_BINARY_DIR}"
        ${OpenVINO_DIR}
    NO_CMAKE_FIND_ROOT_PATH
)

# 1. Raw OV API benchmark (handles position_ids manually)
add_executable(embedding_gemma_benchmark embedding_gemma_benchmark.cpp)
target_link_libraries(embedding_gemma_benchmark PRIVATE openvino::genai)

# 2. GenAI TextEmbeddingPipeline test (high-level SDK API)
add_executable(genai_embedding_test genai_embedding_test.cpp)
target_link_libraries(genai_embedding_test PRIVATE openvino::genai)

foreach(target_name embedding_gemma_benchmark genai_embedding_test)
    set_target_properties(${target_name} PROPERTIES
        COMPILE_PDB_NAME ${target_name}
        INSTALL_RPATH_USE_LINK_PATH ON
    )
    if(MSVC)
        target_compile_options(${target_name} PRIVATE /EHsc /W3)
        target_compile_definitions(${target_name} PRIVATE
            _CRT_SECURE_NO_WARNINGS _SCL_SECURE_NO_WARNINGS)
    endif()
endforeach()

install(TARGETS embedding_gemma_benchmark genai_embedding_test
        RUNTIME DESTINATION bin/
        COMPONENT samples_bin
        EXCLUDE_FROM_ALL)
