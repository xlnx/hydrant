cmake_minimum_required(VERSION 3.0)
project(hydrant)
set(CMAKE_CXX_STANDARD 14)

option(HYDRA_BUILD_ENGINE "build cuda renderer library" ON)
option(HYDRA_BUILD_GUI "build glfw & imgui library" ON)
option(HYDRA_BUILD_LOCAL "build local intereactive renderer" ON)
option(HYDRA_BUILD_SERVER "build intereactive render server" ON)

find_package(Git)
execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive)

# install external modules
include(scripts/external.cmake)

find_package(CUDA REQUIRED)
set(CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE ON)
set(CUDA_SEPARABLE_COMPILATION ON)
set(CUDA_NVCC_FLAGS ${CMAKE_NVCC_FLAGS}
  -std=c++14
  #	-v
  --expt-extended-lambda
  --use_fast_math
  --keep-device-functions
  -keep
  -maxrregcount
  32
  #	-w  
  -lineinfo
  -Xcudafe
  --display_error_number
  --relocatable-device-code=true
  -Xcudafe
  --diag_suppress="2913" 
)

include_directories(${PROJECT_SOURCE_DIR}/include)

add_subdirectory(src)

install(FILES README.md 
  DESTINATION .
  COMPONENT runtime
)

install(DIRECTORY configs
  DESTINATION .
  COMPONENT runtime
)
