cmake_minimum_required(VERSION 3.12)
project(bdml_phybot_deploy_c2 CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

enable_testing()

# ── yaml-cpp (interim: points at official ThirdParty; vendor to own third_party later) ────
# Used to load THIS tree's own SIM/policy config (below), not the official tree's yaml.
set(YAML_CPP_ROOT       "${CMAKE_CURRENT_SOURCE_DIR}/../phybot_deploy_c2/ThirdParty/yaml")
set(YAML_CPP_INCLUDE_DIR "${YAML_CPP_ROOT}/include")
set(YAML_CPP_LIB_DIR     "${YAML_CPP_ROOT}/lib")
find_library(YAML_CPP_LIB yaml-cpp PATHS "${YAML_CPP_LIB_DIR}" NO_DEFAULT_PATH REQUIRED)

# ── SIM/policy config (yaml, loaded at runtime by walk_config.cpp / zero_config.cpp) ──────
# STRICT domain separation: these YAML describe SIM-only config. sim_kp/sim_kd are the SIM
# actuator PD gains (fed to mujoco via DataPackage.sim_P/sim_D). The REAL robot uses the
# motor-driver PD (official MotorList config) and is NOT configured from these files.
# Interim: one compile-definition per directory (mirrors the GOLDEN_DIR pattern below);
# revisit once this tree vendors its own config-loading story.
set(CONFIG_DIR_PATH      "${CMAKE_CURRENT_SOURCE_DIR}/config")
set(WALK_CONFIG_DIR_PATH "${CMAKE_CURRENT_SOURCE_DIR}/tasks/walk/config")
set(ZERO_CONFIG_DIR_PATH "${CMAKE_CURRENT_SOURCE_DIR}/tasks/zero/config")

# ── Dependency-free ctests (no Torch; now need yaml-cpp for walk_config()/zero_config()) ──

add_executable(test_task_manager core/test/test_task_manager.cpp)
target_include_directories(test_task_manager PRIVATE core/include core/test)
add_test(NAME core_task_manager COMMAND test_task_manager)

add_executable(test_walk_numeric core/test/test_walk_numeric.cpp
    core/src/walk_config.cpp core/src/zero_config.cpp)
target_include_directories(test_walk_numeric PRIVATE core/include core/test
    "${YAML_CPP_INCLUDE_DIR}")
target_compile_definitions(test_walk_numeric PRIVATE
    CONFIG_DIR="${CONFIG_DIR_PATH}"
    WALK_CONFIG_DIR="${WALK_CONFIG_DIR_PATH}"
    ZERO_CONFIG_DIR="${ZERO_CONFIG_DIR_PATH}")
target_link_libraries(test_walk_numeric PRIVATE "${YAML_CPP_LIB}")
add_test(NAME walk_numeric COMMAND test_walk_numeric)

add_executable(test_obs_alignment core/test/test_obs_alignment.cpp
    core/src/walk_config.cpp core/src/zero_config.cpp)
target_include_directories(test_obs_alignment PRIVATE core/include core/test
    "${YAML_CPP_INCLUDE_DIR}")
target_compile_definitions(test_obs_alignment PRIVATE
    GOLDEN_DIR="${CMAKE_CURRENT_SOURCE_DIR}/core/test/golden"
    CONFIG_DIR="${CONFIG_DIR_PATH}"
    WALK_CONFIG_DIR="${WALK_CONFIG_DIR_PATH}"
    ZERO_CONFIG_DIR="${ZERO_CONFIG_DIR_PATH}")
target_link_libraries(test_obs_alignment PRIVATE "${YAML_CPP_LIB}")
add_test(NAME obs_alignment COMMAND test_obs_alignment)

add_executable(test_zero_task core/test/test_zero_task.cpp
    tasks/zero/zero_task.cpp core/src/zero_config.cpp)
target_include_directories(test_zero_task PRIVATE core/include core/test
    "${YAML_CPP_INCLUDE_DIR}")
target_compile_definitions(test_zero_task PRIVATE
    CONFIG_DIR="${CONFIG_DIR_PATH}"
    ZERO_CONFIG_DIR="${ZERO_CONFIG_DIR_PATH}")
target_link_libraries(test_zero_task PRIVATE "${YAML_CPP_LIB}")
add_test(NAME zero_task COMMAND test_zero_task)

# ── libtorch (interim: points at official ThirdParty; vendor to own third_party later) ──
set(CMAKE_PREFIX_PATH
    "${CMAKE_CURRENT_SOURCE_DIR}/../phybot_deploy_c2/ThirdParty/libtorch/share/cmake/Torch"
    ${CMAKE_PREFIX_PATH})
find_package(Torch REQUIRED)

# ── PolicyRunner (Torch-dependent) ────────────────────────────────────────────
add_library(policy_runner STATIC core/src/policy_runner.cpp)
target_include_directories(policy_runner PUBLIC core/include)
target_link_libraries(policy_runner PUBLIC "${TORCH_LIBRARIES}")
set_property(TARGET policy_runner PROPERTY CXX_STANDARD 17)

# ── WalkTask ──────────────────────────────────────────────────────────────────
# walk_config.cpp/zero_config.cpp compiled in once here; every consumer of walk_task
# (test_walk_e2e, task_dry_run, sim_walk, test_walk_sim) picks up walk_config()/zero_config()
# and the yaml-cpp link requirement transitively -- they don't need CONFIG_DIR themselves.
add_library(walk_task STATIC tasks/walk/walk_task.cpp
    core/src/walk_config.cpp core/src/zero_config.cpp)
target_include_directories(walk_task
    PUBLIC  core/include tasks
    PRIVATE "${YAML_CPP_INCLUDE_DIR}")
target_compile_definitions(walk_task PRIVATE
    CONFIG_DIR="${CONFIG_DIR_PATH}"
    WALK_CONFIG_DIR="${WALK_CONFIG_DIR_PATH}"
    ZERO_CONFIG_DIR="${ZERO_CONFIG_DIR_PATH}")
target_link_libraries(walk_task PUBLIC policy_runner "${YAML_CPP_LIB}")
set_property(TARGET walk_task PROPERTY CXX_STANDARD 17)

# ── E2E alignment test ────────────────────────────────────────────────────────
add_executable(test_walk_e2e core/test/test_walk_e2e.cpp)
target_include_directories(test_walk_e2e PRIVATE core/include core/test tasks)
target_compile_definitions(test_walk_e2e PRIVATE
    GOLDEN_DIR="${CMAKE_CURRENT_SOURCE_DIR}/core/test/golden")
target_link_libraries(test_walk_e2e PRIVATE walk_task)
set_property(TARGET test_walk_e2e PROPERTY CXX_STANDARD 17)
add_test(NAME walk_e2e COMMAND test_walk_e2e)

# ── Dry-run app ───────────────────────────────────────────────────────────────
add_executable(task_dry_run apps/task_dry_run.cpp)
target_include_directories(task_dry_run PRIVATE core/include tasks)
target_link_libraries(task_dry_run PRIVATE walk_task)
set_property(TARGET task_dry_run PROPERTY CXX_STANDARD 17)

# ── MuJoCo (interim: points at official ThirdParty; vendor to own third_party later) ──
set(MUJOCO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../phybot_deploy_c2/ThirdParty/mujoco")
set(MUJOCO_INCLUDE_DIR "${MUJOCO_ROOT}")        # provides <mujoco/mujoco.h>
set(MUJOCO_LIB_DIR     "${MUJOCO_ROOT}/lib")
find_library(MUJOCO_LIB mujoco PATHS "${MUJOCO_LIB_DIR}" NO_DEFAULT_PATH REQUIRED)

# ── MuJoCo backend library ────────────────────────────────────────────────────
add_library(mujoco_backend STATIC backends/mujoco/mujoco_backend.cpp)
target_include_directories(mujoco_backend
    PUBLIC  core/include backends/mujoco
    PRIVATE "${MUJOCO_INCLUDE_DIR}")
target_link_libraries(mujoco_backend PUBLIC "${MUJOCO_LIB}")
set_property(TARGET mujoco_backend PROPERTY CXX_STANDARD 17)

# ── Headless sim walk app ─────────────────────────────────────────────────────
set(PHYBOT_XML_PATH
    "${CMAKE_CURRENT_SOURCE_DIR}/../phybot_deploy_c2/RobotModel/phybot_c2/xml/phybot_c2.xml")
set(POLICY_PT_PATH
    "${CMAKE_CURRENT_SOURCE_DIR}/core/test/golden/cpg_policy.pt")

add_executable(sim_walk apps/sim_walk.cpp)
target_include_directories(sim_walk PRIVATE core/include tasks backends/mujoco
    "${MUJOCO_INCLUDE_DIR}")
target_link_libraries(sim_walk PRIVATE walk_task mujoco_backend)
set_property(TARGET sim_walk PROPERTY CXX_STANDARD 17)

# ── Walk simulation ctest ─────────────────────────────────────────────────────
add_executable(test_walk_sim core/test/test_walk_sim.cpp)
target_include_directories(test_walk_sim PRIVATE core/include tasks backends/mujoco
    "${MUJOCO_INCLUDE_DIR}")
target_compile_definitions(test_walk_sim PRIVATE
    POLICY_PATH="${POLICY_PT_PATH}"
    XML_PATH="${PHYBOT_XML_PATH}")
target_link_libraries(test_walk_sim PRIVATE walk_task mujoco_backend)
set_property(TARGET test_walk_sim PROPERTY CXX_STANDARD 17)
add_test(NAME walk_sim COMMAND test_walk_sim)

# ── JS mapping (pure logic + device discovery) ───────────────────────────────
add_executable(test_js_mapping publisher/test/test_js_mapping.cpp)
target_include_directories(test_js_mapping PRIVATE publisher/include core/test)
add_test(NAME js_mapping COMMAND test_js_mapping)

# ── JsController library (device I/O + hotplug) ───────────────────────────
add_library(phybot_publisher STATIC publisher/src/js_controller.cpp)
target_include_directories(phybot_publisher PUBLIC publisher/include)

# ── Controller probe app ──────────────────────────────────────────────────
add_executable(controller_probe publisher/apps/controller_probe.cpp)
target_link_libraries(controller_probe PRIVATE phybot_publisher)

# ── InputProtocol (wire codec for 0xeb90) ────────────────────────────────────
add_executable(test_input_protocol core/test/test_input_protocol.cpp core/src/input_protocol.cpp)
target_include_directories(test_input_protocol PRIVATE core/include core/test
    "${CMAKE_CURRENT_SOURCE_DIR}/../phybot_deploy_c2/ThirdParty/LCM/include")
add_test(NAME input_protocol COMMAND test_input_protocol)

# ── Input mapping (ControllerState → DriveCmd) ─────────────────────────────────
add_executable(test_input_mapping publisher/test/test_input_mapping.cpp)
target_include_directories(test_input_mapping PRIVATE publisher/include core/include core/test
    "${CMAKE_CURRENT_SOURCE_DIR}/../phybot_deploy_c2/ThirdParty/LCM/include")
add_test(NAME input_mapping COMMAND test_input_mapping)
