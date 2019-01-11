CMAKE_MINIMUM_REQUIRED(VERSION 3.0)

PROJECT(Coherent-Line-Drawing)

FIND_PACKAGE(OpenCV REQUIRED)
FIND_PACKAGE(wxWidgets REQUIRED)
# cmake寻找cuda，这个要现在系统里面装好cuda，设置好cuda的环境参数啥的
FIND_PACKAGE(CUDA REQUIRED)

INCLUDE(${wxWidgets_USE_FILE})
INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/include)

CUDA_ADD_EXECUTABLE(Coherent-Line-Drawing
"${PROJECT_SOURCE_DIR}/src/main.cpp" 
"${PROJECT_SOURCE_DIR}/src/ETF.cu" 
"${PROJECT_SOURCE_DIR}/src/CLD.cpp" 
"${PROJECT_SOURCE_DIR}/src/postProcessing.cpp" 
"${PROJECT_SOURCE_DIR}/src/gui.cpp"  
"${PROJECT_SOURCE_DIR}/include/ETF.h" 
"${PROJECT_SOURCE_DIR}/include/CLD.h" 
"${PROJECT_SOURCE_DIR}/include/postProcessing.h" 
"${PROJECT_SOURCE_DIR}/include/gui.h" 
)

TARGET_LINK_LIBRARIES(Coherent-Line-Drawing ${OpenCV_LIBS} ${wxWidgets_LIBRARIES})
