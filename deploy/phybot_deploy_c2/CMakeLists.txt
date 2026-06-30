cmake_minimum_required(VERSION 3.12)  # 更新最小 CMake 版本
# cmake_policy(SET CMP0148 NEW)

set(CMAKE_WARN_DEVELOPER OFF)
set(CMAKE_BUILD_TYPE "Release")

add_compile_options(-O0  -g )
#add_compile_options(-O3  -ffast-math )

# Set the C standard to C99
set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 17)
# add_compile_options(-std=c++17)

project(phybot_software)

# LCM
set(LCM_INSTALL_PATH ${CMAKE_SOURCE_DIR}/lcm_server/lcm/release)
set(LCM_LIBRARY ${LCM_INSTALL_PATH}/lib/liblcm.so)
set(GLIB_LIBRARY ${LCM_INSTALL_PATH}/lib/libglib-2.0.so.0)

# LCM 头文件路径
include_directories(${LCM_INSTALL_PATH}/include)
# LCM 库路径
link_directories(${LCM_INSTALL_PATH}/lib)
message(STATUS "Project source directory: ${PROJECT_SOURCE_DIR}/..")
add_compile_options(-Wno-return-type)

# # catkin 包信息
# catkin_package(
#   INCLUDE_DIRS include
#   LIBRARIES ${PROJECT_NAME}
#   CATKIN_DEPENDS roscpp std_msgs tf geometry_msgs
# )

# set(THREADS_PREFER_PTHREAD_FLAG ON)
# find_package(Threads REQUIRED)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")

# link_directories("/usr/local/lib")
# INCLUDE_DIRECTORIES("/usr/local/include")
# link_directories("/usr/lib")
# INCLUDE_DIRECTORIES("/usr/include")
# include_directories("/usr/include/GL")
# find_package(OpenGL REQUIRED)
# include_directories(${OPENGL_INCLUDE_DIR})

set(CMAKE_PREFIX_PATH "/opt/ros/noetic" ${PROJECT_SOURCE_DIR}/ThirdParty/libtorch/share/cmake/Torch)
find_package(Torch REQUIRED)
# find_package(yaml-cpp REQUIRED)

include_directories(${PROJECT_SOURCE_DIR})
include_directories(${PROJECT_SOURCE_DIR}/MujocoInterface/include/simulate)
include_directories(${PROJECT_SOURCE_DIR}/ThirdParty)
include_directories(${PROJECT_SOURCE_DIR}/ThirdParty/eigen3)
include_directories(${PROJECT_SOURCE_DIR}/ThirdParty/LCM/include)
include_directories(${PROJECT_SOURCE_DIR}/ThirdParty/libtorch/include)
include_directories(${PROJECT_SOURCE_DIR}/ThirdParty/boost/include)
include_directories(${PROJECT_SOURCE_DIR}/ThirdParty/yaml/include)
include_directories(${PROJECT_SOURCE_DIR}/ThirdParty/spdlog/include)
include_directories(${PROJECT_SOURCE_DIR}/lcm_server/include)
include_directories(${PROJECT_SOURCE_DIR}/MotorList/include)

link_directories(${PROJECT_SOURCE_DIR}/ThirdParty/LCM/lib)
link_directories(${PROJECT_SOURCE_DIR}/ThirdParty/LCM/lib/x86_64-linux-gnu)
link_directories(${PROJECT_SOURCE_DIR}/ThirdParty/boost/lib)
link_directories(${PROJECT_SOURCE_DIR}/ThirdParty/boost/lib)
link_directories(${PROJECT_SOURCE_DIR}/ThirdParty/yaml/lib)
link_directories(${PROJECT_SOURCE_DIR}/MotorList/lib)
link_directories(${PROJECT_SOURCE_DIR}/lcm_server/lib)

aux_source_directory(${PROJECT_SOURCE_DIR}/DataPackage/src dir_datapackage)
# add_library(DataPackage SHARED ${dir_datapackage})

# target_link_libraries(DataPackage pinocchio yaml-cpp)



# =========================
# Common Include Collection
# =========================

include_directories(
    ${PROJECT_SOURCE_DIR}/ThirdParty/yaml/include
    ${PROJECT_SOURCE_DIR}/DataPackage/include
    ${PROJECT_SOURCE_DIR}/StateMachine/include
    ${PROJECT_SOURCE_DIR}/LowPassFilter/include
    ${PROJECT_SOURCE_DIR}/ZeroState/include
    ${PROJECT_SOURCE_DIR}/RL_deploy_long_motion/include
    ${PROJECT_SOURCE_DIR}/RL_deploy_kongfu/include
    ${PROJECT_SOURCE_DIR}/RL_deploy_taichi/include
    ${PROJECT_SOURCE_DIR}/RL_deploy_forward_punch/include
    ${PROJECT_SOURCE_DIR}/RL_deploy_forward_kick/include
    ${PROJECT_SOURCE_DIR}/RL_deploy_mimic/include
    ${PROJECT_SOURCE_DIR}/RL_deploy_cpg/include
    ${PROJECT_SOURCE_DIR}/RL_deploy_walk_amp/include

    ${PROJECT_SOURCE_DIR}/RL_deploy_climb_up/include

    ${PROJECT_SOURCE_DIR}/device/Imu_hipnuc/linux
    ${PROJECT_SOURCE_DIR}/device/Imu_hipnuc/linux/fw_downloader
    ${PROJECT_SOURCE_DIR}/device/Imu_hipnuc/lib
)


# =========================
# Common Source Collection
# =========================

aux_source_directory(${PROJECT_SOURCE_DIR}/StateMachine/src dir_FSM)
aux_source_directory(${PROJECT_SOURCE_DIR}/LowPassFilter/src dir_LPF)
aux_source_directory(${PROJECT_SOURCE_DIR}/ZeroState/src dir_ZSM)

aux_source_directory(${PROJECT_SOURCE_DIR}/RL_deploy_long_motion/src dir_rl_long_motion)
aux_source_directory(${PROJECT_SOURCE_DIR}/RL_deploy_kongfu/src dir_rl_kongfu)
aux_source_directory(${PROJECT_SOURCE_DIR}/RL_deploy_taichi/src dir_rl_taichi)
aux_source_directory(${PROJECT_SOURCE_DIR}/RL_deploy_forward_punch/src dir_rl_forward_punch)
aux_source_directory(${PROJECT_SOURCE_DIR}/RL_deploy_forward_kick/src dir_rl_forward_kick)
aux_source_directory(${PROJECT_SOURCE_DIR}/RL_deploy_mimic/src dir_rl_mimic)
aux_source_directory(${PROJECT_SOURCE_DIR}/RL_deploy_cpg/src dir_rl_cpg)
aux_source_directory(${PROJECT_SOURCE_DIR}/RL_deploy_walk_amp/src dir_rl_walk_amp)

aux_source_directory(${PROJECT_SOURCE_DIR}/RL_deploy_climb_up/src dir_rl_climb_up)
aux_source_directory(${PROJECT_SOURCE_DIR}/RL_deploy_crane_down/src dir_rl_crane_down)
aux_source_directory(${PROJECT_SOURCE_DIR}/lcm_server/src lcm_server)
# =========================
# IMU Sources
# =========================

set(IMU_SOURCES
    ${PROJECT_SOURCE_DIR}/device/Imu_hipnuc/linux/HipnucReader.cpp
    ${PROJECT_SOURCE_DIR}/device/Imu_hipnuc/linux/serial_port.c
    ${PROJECT_SOURCE_DIR}/device/Imu_hipnuc/linux/log.c
    ${PROJECT_SOURCE_DIR}/device/Imu_hipnuc/linux/fw_downloader/hex2bin.c
    ${PROJECT_SOURCE_DIR}/device/Imu_hipnuc/linux/fw_downloader/kboot.c
    ${PROJECT_SOURCE_DIR}/device/Imu_hipnuc/lib/example_data.c
    ${PROJECT_SOURCE_DIR}/device/Imu_hipnuc/lib/hipnuc_dec.c
    ${PROJECT_SOURCE_DIR}/device/Imu_hipnuc/lib/nmea_dec.c
)

# =========================
# Common Sources
# =========================

set(COMMON_SOURCES
    ${IMU_SOURCES}
    ${dir_FSM}
    ${dir_LPF}
    ${dir_ZSM}
    ${dir_rl_long_motion}
    ${dir_rl_kongfu}
    ${dir_rl_taichi}
    ${dir_rl_forward_punch}
    ${dir_rl_forward_kick}
    ${dir_rl_mimic}
    ${dir_rl_cpg}
    ${dir_rl_walk_amp}
    ${dir_rl_climb_up}
    ${dir_rl_crane_down}
    ${dir_datapackage}
     ${lcm_server}

)


add_executable(walk_amp_dry_run_check
    ${PROJECT_SOURCE_DIR}/RL_deploy_walk_amp/test/dry_run_check.cpp
    ${PROJECT_SOURCE_DIR}/RL_deploy_walk_amp/src/rl_deploy.cpp
    ${PROJECT_SOURCE_DIR}/DataPackage/src/DataPackage.cpp
)
target_link_libraries(walk_amp_dry_run_check PUBLIC
    ${TORCH_LIBRARIES}
    yaml-cpp
)


# Golden-fixture generator for the official CPG walk task (additive, non-behavioral).
# Mirrors the walk_amp_dry_run_check pattern; standalone torch executable.
add_executable(cpg_golden_gen
    ${PROJECT_SOURCE_DIR}/RL_deploy_cpg/test/cpg_golden_gen.cpp
    ${PROJECT_SOURCE_DIR}/RL_deploy_cpg/src/rl_deploy.cpp
    ${PROJECT_SOURCE_DIR}/DataPackage/src/DataPackage.cpp
)
target_link_libraries(cpg_golden_gen PUBLIC
    ${TORCH_LIBRARIES}
    yaml-cpp
)


if(WHICH_ENV AND(WHICH_ENV STREQUAL "realrobot_mini"))



  

    aux_source_directory(${PROJECT_SOURCE_DIR}/MotorList/src dir_ML)



    link_directories(${PROJECT_SOURCE_DIR}/ThirdParty/yaml/lib)
    link_directories(${PROJECT_SOURCE_DIR}/MotorList/lib)




    add_executable(main ${PROJECT_SOURCE_DIR}/RobotStart/realrobot/run_mini.cpp

                        ${dir_ML}
                        ${COMMON_SOURCES}

                        )
    target_link_libraries(main  PUBLIC 
                            ${TORCH_LIBRARIES}
                            pthread
                            yaml-cpp
                            MotorDrive
                            ${LCM_LIBRARY}
                            ${GLIB_LIBRARY}
                            # HARDWARE_TOP
                            # Main_Control_Board_Device
                            )

    add_executable(set_zero ${PROJECT_SOURCE_DIR}/RobotStart/realrobot/set_zero_mini.cpp

                            # ${IMU_SOURCES}
                            ${dir_LPF}

                            ${dir_datapackage}
                            ${dir_ML}
                            ${dir_ZSM}
                            )
    target_link_libraries(set_zero  PUBLIC 
                                ${TORCH_LIBRARIES}
                                pthread
                                yaml-cpp
                                MotorDrive
                                # HARDWARE_TOP
                                # Main_Control_Board_Device
                                )

    add_executable(set_one_zero ${PROJECT_SOURCE_DIR}/RobotStart/test/set_one_zero.cpp

                            # ${IMU_SOURCES}
                            ${dir_LPF}

                            ${dir_datapackage}
                            ${dir_ML}
                            ${dir_ZSM}
                            )
    target_link_libraries(set_one_zero  PUBLIC 
                                ${TORCH_LIBRARIES}
                                pthread
                                yaml-cpp
                                MotorDrive
                                # HARDWARE_TOP
                                # Main_Control_Board_Device
                                )
elseif(WHICH_ENV AND(WHICH_ENV STREQUAL "mujoco_sim_mini"))

    include_directories(${PROJECT_SOURCE_DIR}/ThirdParty/mujoco)
    include_directories(${PROJECT_SOURCE_DIR}/ThirdParty/glfw)
    include_directories(${PROJECT_SOURCE_DIR}/MujocoInterface/include)
    include_directories(${PROJECT_SOURCE_DIR}/ThirdParty/gl/include)


    aux_source_directory(${PROJECT_SOURCE_DIR}/MujocoInterface/src dir_mujoco)


    link_directories(${PROJECT_SOURCE_DIR}/ThirdParty/mujoco/lib)
    link_directories(${PROJECT_SOURCE_DIR}/ThirdParty/glfw/lib)


    add_executable(main ${PROJECT_SOURCE_DIR}/RobotStart/mujoco_sim/main.cc
                        
                        ${dir_mujoco}
                        ${COMMON_SOURCES}

                        )
    target_link_libraries(main  PUBLIC 
                            ${TORCH_LIBRARIES}
                            mujoco
                            glfw3
                            pthread
                            yaml-cpp
                            lcm          # 新增：链接LCM库
                            glib-2.0     # 新增：LCM依赖的glib2库
                            )


elseif(WHICH_ENV AND(WHICH_ENV STREQUAL "gazebo_sim"))
    ## 查找 ROS 依赖
    # set(phybot_msgs_DIR /home/kairan/Documents/phybot_ws/src/phybot_msgs)
    set(phybot_msgs_DIR "${PROJECT_SOURCE_DIR}/../../devel/share/phybot_msgs/cmake")
    add_definitions(-DCMAKE_CURRENT_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/gazebo_sim/")

    find_package(catkin REQUIRED COMPONENTS
        phybot_msgs
        roscpp
        std_msgs
        tf
        geometry_msgs
        rospy
        gazebo_ros
        controller_manager
        joint_state_controller
        robot_state_publisher
        genmsg
        

    )
    
    catkin_package(
        CATKIN_DEPENDS roscpp std_msgs tf geometry_msgs
    )
    
    include_directories(
        ${catkin_INCLUDE_DIRS}
        ${PROJECT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/gazebo_sim/include
        ${PROJECT_SOURCE_DIR}/gazebo_sim/library/loop
        ${PROJECT_SOURCE_DIR}/gazebo_sim/library/rl_sdk

    )

    include_directories(${PROJECT_SOURCE_DIR}/gazebo_sim/include)

    add_library(rl_sdk ${PROJECT_SOURCE_DIR}/gazebo_sim/library/rl_sdk/rl_sdk.cpp)
    target_link_libraries(rl_sdk "${TORCH_LIBRARIES}" yaml-cpp)
    set_property(TARGET rl_sdk PROPERTY CXX_STANDARD 17)


    add_executable(${PROJECT_NAME} ${PROJECT_SOURCE_DIR}/gazebo_sim/src/rl_sim.cpp
                        # ${PROJECT_SOURCE_DIR}/gazebo_sim/library/rl_sdk/rl_sdk.cpp
                        # ${dir_wbc}
                        # ${dir_est}
                        # ${dir_datapackage}
                        ) 
                           

    target_link_libraries(${PROJECT_NAME}
                                # pinocchio_default 
                                # pinocchio_parsers
                                yaml-cpp
                                torch
                                rl_sdk
                                # qpOASES 
                                # urdfdom_model
                                ${catkin_LIBRARIES})



endif()





# install(TARGETS DataPackage DESTINATION ${PROJECT_SOURCE_DIR}/lib)
