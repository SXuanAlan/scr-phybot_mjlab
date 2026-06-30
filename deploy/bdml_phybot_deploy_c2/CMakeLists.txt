cmake_minimum_required(VERSION 3.12)
project(bdml_phybot_deploy_c2 CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

enable_testing()

add_executable(test_task_manager core/test/test_task_manager.cpp)
target_include_directories(test_task_manager PRIVATE core/include core/test)

add_test(NAME core_task_manager COMMAND test_task_manager)
