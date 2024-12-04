cmake_minimum_required(VERSION 3.27)

project(${SKBUILD_PROJECT_NAME} LANGUAGES HIP)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 REQUIRED CONFIG)

pybind11_add_module(_C flash_attn.hip)

set_property(TARGET _C PROPERTY BUILD_WITH_INSTALL_RPATH TRUE)
set_property(TARGET _C PROPERTY HIP_ARCHITECTURES gfx1100 gfx1101)
set_property(TARGET _C PROPERTY HIP_STANDARD 20)

target_compile_options(_C PRIVATE "-D_GLIBCXX_USE_CXX11_ABI=0")
target_compile_options(_C PRIVATE "-fPIC")
target_compile_options(_C PRIVATE "-O3")

target_include_directories(_C PRIVATE /opt/rocm/include)
target_include_directories(_C PRIVATE ${Python_SITEARCH}/torch/include)
target_include_directories(_C PRIVATE ${Python_SITEARCH}/torch/include/torch/csrc/api/include)

target_link_directories(_C PRIVATE ${Python_SITEARCH}/torch/lib)

target_link_libraries(_C PRIVATE c10)
target_link_libraries(_C PRIVATE torch)
target_link_libraries(_C PRIVATE torch_python)

install(TARGETS _C LIBRARY DESTINATION ${SKBUILD_PROJECT_NAME})
