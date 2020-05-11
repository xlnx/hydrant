cmake_minimum_required(VERSION 3.0)
project(hydrant)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(HYDRA_BUILD_SERVER "build cuda renderer server" ON)
option(HYDRA_BUILD_CLIENT "build glfw & imgui client" ON)

find_package(Git)
execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive)

# install external modules
include(scripts/external.cmake)

#find_package(OpenMP REQUIRED)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
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
  -Xcompiler
  -fPIC
  -Xcudafe
  --display_error_number
  --relocatable-device-code=true
  -Xcudafe
  --diag_suppress="2913" 
  -I${PROJECT_SOURCE_DIR}/build/external/glm-cuda/include
  -I${PROJECT_SOURCE_DIR}/build/external/VMUtils/include
  -I${PROJECT_SOURCE_DIR}/build/external/cuda-fx/include
  -I${PROJECT_SOURCE_DIR}/build/external/stbi/include
  -L/usr/local/cuda/lib64/stubs
)
set(CUDA_CUDA_LIBRARY libcuda.so)

link_directories(/usr/local/cuda/lib64/stubs)
include_directories(
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/ui
)

add_subdirectory(src)

install(FILES README.md 
  DESTINATION .
  COMPONENT runtime
)

install(DIRECTORY configs
  DESTINATION .
  COMPONENT runtime
)
install(FILES build_dataset.sh infer_dimension.pl
  DESTINATION .
)
