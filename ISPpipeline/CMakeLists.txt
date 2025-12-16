add_executable(testrawisp testrawisp.cpp)

# # 如果 testrawisp.cpp 也需要包含项目头文件，则：
# target_include_directories(testrawisp
#     PUBLIC
#         ${PROJECT_SOURCE_DIR}/Layer
#         ${PROJECT_SOURCE_DIR}/Mat
# )

# 链接主工程的库
target_link_libraries(testrawisp
    PRIVATE
        layer
        mat
)

add_executable(DecodeCR2 DecodeCR2.cpp)
target_link_libraries(DecodeCR2 PRIVATE libraw::libraw)