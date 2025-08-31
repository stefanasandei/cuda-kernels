project(lib)

add_library(common STATIC common/common.cu)
set_target_properties(common PROPERTIES LINKER_LANGUAGE CUDA)
target_include_directories(common PUBLIC ./common)
