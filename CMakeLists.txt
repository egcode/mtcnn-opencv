cmake_minimum_required(VERSION 3.2)

project (MTCNN_APP)

set (CMAKE_CXX_STANDARD 11)
set (CMAKE_CXX_STANDARD_REQUIRED ON)

find_package( OpenCV REQUIRED )
find_package( Boost REQUIRED COMPONENTS system filesystem timer )
include_directories( ${Boost_INCLUDE_DIRS} )

set(sources
./mtcnn/detector.cpp
./mtcnn/face.h 
./mtcnn/onet.cpp 
./mtcnn/pnet.cpp ./mtcnn/rnet.cpp
./mtcnn/detector.h ./mtcnn/helpers.h ./mtcnn/onet.h ./mtcnn/pnet.h ./mtcnn/rnet.h
)
add_executable (mtcnn_app main.cpp ${sources})

target_link_libraries (mtcnn_app LINK_PUBLIC ${OpenCV_LIBS} ${Boost_LIBRARIES} )
