cmake_minimum_required(VERSION 2.8.3)
project(plan_manage)

set(CMAKE_VERBOSE_MAKEFILE "true")
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++17 -g ${TORCH_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -fPIC")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall" )
list(APPEND CMAKE_PREFIX_PATH
    "${CMAKE_SOURCE_DIR}/../libtorch"
    /usr/local/cuda
    )
set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda")
set(Torch_DIR "${CMAKE_SOURCE_DIR}/../libtorch/share/cmake/Torch")
set(ENABLE_CUDA true)

find_package(Torch REQUIRED)

if(ENABLE_CUDA)
  find_package(CUDA REQUIRED)
  SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3 -use_fast_math)
  set(CUDA_NVCC_FLAGS 
    -gencode arch=compute_74,code=sm_74
  )
endif()


find_package(Eigen3 REQUIRED)

find_package(PCL REQUIRED)
find_package(catkin REQUIRED COMPONENTS
    roscpp
    visualization_msgs
    std_msgs
    tools
    path_search
    cv_bridge
    decomp_ros_utils
)

include_directories(
    include
    ${catkin_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}
)


#if this catkin packge's header is used by other packages, use catkin_package to
#declare the include directories of this package.
catkin_package(
    INCLUDE_DIRS include 
)
add_library(plan_manage
    src/plan_manage.cpp
)
target_link_libraries(plan_manage 
    kino_astar
    nn_pathsearch
    ${TORCH_LIBRARIES}
    ${catkin_LIBRARIES}
)
add_executable(plan_node
    src/plan_node.cpp
)
target_link_libraries(plan_node 
    plan_manage
    ${catkin_LIBRARIES}
    ${PCL_LIBRARIES}
)
