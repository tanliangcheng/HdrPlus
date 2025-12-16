set(mat_srcs
    MemPool.cpp
    MultiUshortImage.cpp
    YUVToJpeg.cpp
    Common.cpp
    MultiIntImage.cpp
    SingleUcharImage.cpp
    MathFunction.cpp
    MultiShortImage.cpp
    TimeStatistic.cpp
    Matrix.cpp
    MultiUcharImage.cpp
    YUV420Image.cpp
)

add_library(mat SHARED ${mat_srcs})

target_compile_features(mat PUBLIC cxx_std_11)
set_target_properties(mat PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(mat
    PUBLIC
        ${PROJECT_SOURCE_DIR}/Mat
)