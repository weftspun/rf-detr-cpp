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

foreach(t test_backbone_smoke test_backbone test_projector test_deform_attn test_decoder test_segmentation test_keypoints)
    add_executable(${t} tests/${t}.cpp)
    target_link_libraries(${t} PRIVATE rfdetr)
endforeach()
