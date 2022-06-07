cmake_minimum_required(VERSION 3.16)
project(recon)

list(PREPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# Libigl
option(LIBIGL_GLFW "Build target igl::glfw" ON)
#option(LIBIGL_WITH_OPENGL_GLFW_IMGUI "Use ImGui" ON)
include(libigl)

# Add your project files
file(GLOB SRC_FILES *.cpp)
#add_executable(${PROJECT_NAME} ${SRC_FILES})
add_executable(${PROJECT_NAME} main.cpp)
target_link_libraries(${PROJECT_NAME} PUBLIC igl::glfw)
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS "-O3")

