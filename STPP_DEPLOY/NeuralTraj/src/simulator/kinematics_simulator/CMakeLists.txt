cmake_minimum_required(VERSION 2.8.3)
project(kinematics_simulator)

set(CMAKE_BUILD_TYPE "Release")
ADD_COMPILE_OPTIONS(-std=c++11 )
ADD_COMPILE_OPTIONS(-std=c++14 )
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  nav_msgs
  std_msgs
  visualization_msgs
)
find_package(Eigen3 REQUIRED)

add_message_files(FILES
  ControlCmd.msg
)

generate_messages(DEPENDENCIES
	std_msgs
)

catkin_package(
 CATKIN_DEPENDS roscpp std_msgs message_runtime
)

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
)

add_executable (kinematics_simulator_node src/kinematics_sim.cpp )
target_link_libraries(kinematics_simulator_node
  ${catkin_LIBRARIES})
