cmake_minimum_required(VERSION 3.20)
project(rf-detr-cpp CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(third_party/ggml)

add_library(rfdetr STATIC
    src/ops.cpp
    src/backbone.cpp
    src/projector.cpp
    src/deform_attn.cpp
    src/decoder.cpp
    src/segmentation.cpp
    src/keypoints.cpp
)
target_link_libraries(rfdetr PUBLIC ggml)
target_include_directories(rfdetr PUBLIC src)

foreach(t test_backbone_smoke test_backbone test_backbone_base test_projector test_projector_multiscale test_deform_attn test_deform_attn_multilevel test_decoder test_decoder_base test_decoder_small test_decoder_medium test_decoder_large test_segmentation test_keypoints test_norm_backward)
    add_executable(${t} tests/${t}.cpp)
    target_link_libraries(${t} PRIVATE rfdetr)
endforeach()
