add_executable(tile_example_topk_softmax EXCLUDE_FROM_ALL topk_softmax.cpp topk_softmax_api.cpp)
target_include_directories(tile_example_topk_softmax PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/)
target_link_libraries(tile_example_topk_softmax PRIVATE utility)

target_compile_options(tile_example_topk_softmax PRIVATE
    -Wno-undefined-func-template
    -Wno-float-equal)
