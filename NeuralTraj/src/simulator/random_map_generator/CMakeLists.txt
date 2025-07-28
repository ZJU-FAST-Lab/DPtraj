cmake_minimum_required(VERSION 2.8.3)
project(random_map_generator)

set(CMAKE_BUILD_TYPE "Release")
ADD_COMPILE_OPTIONS(-std=c++11 )
ADD_COMPILE_OPTIONS(-std=c++14 )
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  sensor_msgs
  std_msgs
  visualization_msgs
  pcl_conversions
  tools
  cv_bridge
  roslib
)
find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED)

catkin_package(
 CATKIN_DEPENDS roscpp std_msgs sensor_msgs visualization_msgs roslib
)

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${PCL_INCLUDE_DIRS}
)

add_executable (random_map src/random_map_node.cpp )
target_link_libraries(random_map
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
)

add_executable (egeneration_node src/egeneration_node.cpp )
target_link_libraries(egeneration_node
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  
)
add_executable (enet_test_node src/enet_test.cpp )
target_link_libraries(enet_test_node
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
)
add_executable (image_node src/image_pro.cpp )
target_link_libraries(image_node
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  
)
add_executable (corridor_pro_node src/corridor_pro.cpp )
target_link_libraries(corridor_pro_node
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  
)