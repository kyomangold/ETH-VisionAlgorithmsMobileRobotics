cmake_minimum_required(VERSION 3.10)

project(VAMR 
    VERSION 1.0.0 
    )

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(OpenCV REQUIRED)
add_definitions(-DOPENCV_VERSION=100*${OpenCV_VERSION_MAJOR}+${OpenCV_VERSION_MINOR})
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)

find_package(Threads REQUIRED)
find_package(OpenMP)

include_directories(
    include
    ${OpenCV_INCLUDE_DIRS}
    )

aux_source_directory(src SRC_FILES)

add_executable(main ${SRC_FILES})

add_executable(example example.cpp)

target_link_libraries(main
    ${OpenCV_LIBS}
    Threads::Threads
    )

target_link_libraries(example
    ${OpenCV_LIBS}
    )