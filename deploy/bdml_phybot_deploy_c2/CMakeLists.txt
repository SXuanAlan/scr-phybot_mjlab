cmake_minimum_required(VERSION 3.12)
project(bdml_phybot_deploy_c2 CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

enable_testing()

# ── yaml-cpp (vendored in third_party/ (populate via vendor.sh; git-ignored)) ────
# Used to load THIS tree's own SIM/policy config (below), not the official tree's yaml.
set(YAML_CPP_ROOT       "${CMAKE_CURRENT_SOURCE_DIR}/third_party/yaml")
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

add_executable(test_controller_config core/test/test_controller_config.cpp core/src/controller_config.cpp)
target_include_directories(test_controller_config PRIVATE core/include core/test
    "${YAML_CPP_INCLUDE_DIR}")
target_compile_definitions(test_controller_config PRIVATE PHYBOT_TASKS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tasks")
target_link_libraries(test_controller_config PRIVATE "${YAML_CPP_LIB}")
add_test(NAME controller_config COMMAND test_controller_config)

add_executable(test_walk_numeric tasks/walk/test/test_walk_numeric.cpp
    tasks/walk/walk_config.cpp tasks/zero/zero_config.cpp)
target_include_directories(test_walk_numeric PRIVATE core/include core/test tasks
    "${YAML_CPP_INCLUDE_DIR}")
target_compile_definitions(test_walk_numeric PRIVATE
    CONFIG_DIR="${CONFIG_DIR_PATH}"
    WALK_CONFIG_DIR="${WALK_CONFIG_DIR_PATH}"
    ZERO_CONFIG_DIR="${ZERO_CONFIG_DIR_PATH}"
    PHYBOT_ROOT="${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(test_walk_numeric PRIVATE "${YAML_CPP_LIB}")
add_test(NAME walk_numeric COMMAND test_walk_numeric)

add_executable(test_obs_alignment tasks/walk/test/test_obs_alignment.cpp
    tasks/walk/walk_config.cpp tasks/zero/zero_config.cpp)
target_include_directories(test_obs_alignment PRIVATE core/include core/test tasks
    "${YAML_CPP_INCLUDE_DIR}")
target_compile_definitions(test_obs_alignment PRIVATE
    GOLDEN_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tasks/walk/test/golden"
    CONFIG_DIR="${CONFIG_DIR_PATH}"
    WALK_CONFIG_DIR="${WALK_CONFIG_DIR_PATH}"
    ZERO_CONFIG_DIR="${ZERO_CONFIG_DIR_PATH}"
    PHYBOT_ROOT="${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(test_obs_alignment PRIVATE "${YAML_CPP_LIB}")
add_test(NAME obs_alignment COMMAND test_obs_alignment)

# policy_path (WalkConfig, resolved via PHYBOT_ROOT) — see walk_config.{h,cpp}.
add_executable(test_walk_config_policy tasks/walk/test/test_walk_config_policy.cpp
    tasks/walk/walk_config.cpp tasks/zero/zero_config.cpp)
target_include_directories(test_walk_config_policy PRIVATE core/include core/test tasks
    "${YAML_CPP_INCLUDE_DIR}")
target_compile_definitions(test_walk_config_policy PRIVATE
    CONFIG_DIR="${CONFIG_DIR_PATH}"
    WALK_CONFIG_DIR="${WALK_CONFIG_DIR_PATH}"
    ZERO_CONFIG_DIR="${ZERO_CONFIG_DIR_PATH}"
    PHYBOT_ROOT="${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(test_walk_config_policy PRIVATE "${YAML_CPP_LIB}")
add_test(NAME walk_config_policy COMMAND test_walk_config_policy)

add_executable(test_zero_task tasks/zero/test/test_zero_task.cpp
    tasks/zero/zero_task.cpp tasks/zero/zero_config.cpp)
target_include_directories(test_zero_task PRIVATE core/include core/test tasks
    "${YAML_CPP_INCLUDE_DIR}")
target_compile_definitions(test_zero_task PRIVATE
    CONFIG_DIR="${CONFIG_DIR_PATH}"
    ZERO_CONFIG_DIR="${ZERO_CONFIG_DIR_PATH}")
target_link_libraries(test_zero_task PRIVATE "${YAML_CPP_LIB}")
add_test(NAME zero_task COMMAND test_zero_task)

# ── libtorch (vendored in third_party/ (populate via vendor.sh; git-ignored)) ──
set(CMAKE_PREFIX_PATH
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party/libtorch/share/cmake/Torch"
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
    tasks/walk/walk_config.cpp tasks/zero/zero_config.cpp)
target_include_directories(walk_task
    PUBLIC  core/include tasks
    PRIVATE "${YAML_CPP_INCLUDE_DIR}")
target_compile_definitions(walk_task PRIVATE
    CONFIG_DIR="${CONFIG_DIR_PATH}"
    WALK_CONFIG_DIR="${WALK_CONFIG_DIR_PATH}"
    ZERO_CONFIG_DIR="${ZERO_CONFIG_DIR_PATH}"
    PHYBOT_ROOT="${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(walk_task PUBLIC policy_runner "${YAML_CPP_LIB}")
set_property(TARGET walk_task PROPERTY CXX_STANDARD 17)

# ── phybot_tasks (all task .cpp's + their config deps) ───────────────────────
# Every REGISTER_TASK(...) registrar lives in one of these .cpp's as a file-scope
# static-init object (see task_registry.h). Inside a plain STATIC lib, the linker
# only pulls in .o's that satisfy an undefined symbol elsewhere -- since nothing
# calls into these .o's directly (registration happens via static-init side effect,
# not a referenced symbol), the linker is free to drop them entirely, silently
# emptying TaskRegistry. Whole-archiving phybot_tasks below forces every .o (and
# therefore every registrar) into the final link regardless of reference.
add_library(phybot_tasks STATIC tasks/walk/walk_task.cpp tasks/zero/zero_task.cpp
    tasks/walk/walk_config.cpp tasks/zero/zero_config.cpp)
target_include_directories(phybot_tasks
    PUBLIC  core/include tasks
    PRIVATE "${YAML_CPP_INCLUDE_DIR}")
target_compile_definitions(phybot_tasks PRIVATE
    CONFIG_DIR="${CONFIG_DIR_PATH}" WALK_CONFIG_DIR="${WALK_CONFIG_DIR_PATH}"
    ZERO_CONFIG_DIR="${ZERO_CONFIG_DIR_PATH}" PHYBOT_ROOT="${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(phybot_tasks PUBLIC policy_runner "${YAML_CPP_LIB}")
set_property(TARGET phybot_tasks PROPERTY CXX_STANDARD 17)
if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
  set(PHYBOT_TASKS_WHOLE "$<LINK_LIBRARY:WHOLE_ARCHIVE,phybot_tasks>")
else()
  set(PHYBOT_TASKS_WHOLE -Wl,--whole-archive phybot_tasks -Wl,--no-whole-archive)
endif()

# ── Link-guard test: proves REGISTER_TASK registrars survive whole-archiving ──
# Links ONLY the whole-archived phybot_tasks (not walk_task, which would double-
# define WalkTask's/ZeroState's symbols). Calls TaskRegistry::instance().size()/
# name(i) only -- never make() -- so no task is constructed (no Torch model load).
add_executable(test_task_registry_link core/test/test_task_registry_link.cpp)
target_link_libraries(test_task_registry_link PRIVATE ${PHYBOT_TASKS_WHOLE})
add_test(NAME task_registry_link COMMAND test_task_registry_link)

# ── E2E alignment test ────────────────────────────────────────────────────────
add_executable(test_walk_e2e tasks/walk/test/test_walk_e2e.cpp)
target_include_directories(test_walk_e2e PRIVATE core/include core/test tasks)
target_compile_definitions(test_walk_e2e PRIVATE
    GOLDEN_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tasks/walk/test/golden")
target_link_libraries(test_walk_e2e PRIVATE walk_task)
set_property(TARGET test_walk_e2e PROPERTY CXX_STANDARD 17)
add_test(NAME walk_e2e COMMAND test_walk_e2e)

# ── Dry-run app ───────────────────────────────────────────────────────────────
add_executable(task_dry_run apps/task_dry_run.cpp)
target_include_directories(task_dry_run PRIVATE core/include tasks)
target_link_libraries(task_dry_run PRIVATE walk_task)
set_property(TARGET task_dry_run PROPERTY CXX_STANDARD 17)

# ── MuJoCo (vendored in third_party/ (populate via vendor.sh; git-ignored)) ──
set(MUJOCO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/third_party/mujoco")
set(MUJOCO_INCLUDE_DIR "${MUJOCO_ROOT}")        # provides <mujoco/mujoco.h>
set(MUJOCO_LIB_DIR     "${MUJOCO_ROOT}/lib")
find_library(MUJOCO_LIB mujoco PATHS "${MUJOCO_LIB_DIR}" NO_DEFAULT_PATH REQUIRED)

# --- LCM vendored in third_party/lcm (populate via vendor.sh; git-ignored) ---
set(PHYBOT_LCM_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/lcm")
add_library(phybot_lcm INTERFACE)
target_include_directories(phybot_lcm INTERFACE "${PHYBOT_LCM_DIR}/include")
target_link_libraries(phybot_lcm INTERFACE
    "${PHYBOT_LCM_DIR}/lib/liblcm.so" "${PHYBOT_LCM_DIR}/lib/libglib-2.0.so.0")

# ── MuJoCo backend library ────────────────────────────────────────────────────
add_library(mujoco_backend STATIC backends/mujoco/mujoco_backend.cpp)
target_include_directories(mujoco_backend
    PUBLIC  core/include backends/mujoco
    PRIVATE "${MUJOCO_INCLUDE_DIR}")
target_link_libraries(mujoco_backend PUBLIC "${MUJOCO_LIB}")
set_property(TARGET mujoco_backend PROPERTY CXX_STANDARD 17)

# ── SimHarness (glfw-free) ─────────────────────────────────────────────────────
# Composition root: its ctor iterates TaskRegistry (populated by REGISTER_TASK
# static-init side effects inside the whole-archived phybot_tasks below) and pairs
# each self-registered task with its own tasks/<name>/config/controller.yaml --
# select_code decides TaskManager filing, the full ControllerConfig goes to
# TaskRouter. Links the whole-archived phybot_tasks, exactly like
# test_task_registry_link above -- so anything linking sim_harness must NOT also
# link walk_task (duplicate WalkTask/ZeroState symbols; see that test's comment).
# Also compiles controller_config.cpp itself (yaml-cpp include/link needed here,
# independent of phybot_tasks' own yaml-cpp usage).
add_library(sim_harness STATIC backends/mujoco/sim_harness.cpp
    core/src/task_router.cpp core/src/controller_config.cpp)
target_include_directories(sim_harness
    PUBLIC  core/include backends/mujoco tasks
    PRIVATE "${MUJOCO_INCLUDE_DIR}" "${YAML_CPP_INCLUDE_DIR}"
            "${CMAKE_CURRENT_SOURCE_DIR}/third_party/lcm/include")
target_compile_definitions(sim_harness PRIVATE
    PHYBOT_TASKS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tasks"
    PHYBOT_ROOT="${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(sim_harness PUBLIC mujoco_backend ${PHYBOT_TASKS_WHOLE} "${YAML_CPP_LIB}" pthread)
set_property(TARGET sim_harness PROPERTY CXX_STANDARD 17)

# ── Optional MuJoCo GUI (vendored `simulate` UI + MujocoViewer wrapper) ─────────
# GLFW + GL are vendored in third_party/ (populate via vendor.sh; git-ignored).
# This is a DEV-ONLY interactive viewer: only sim_walk/sim_teleop link it (via
# --gui); the ctest binaries stay headless and never pull glfw/GL/X11.
set(PHYBOT_GLFW_DIR    "${CMAKE_CURRENT_SOURCE_DIR}/third_party/glfw")
set(PHYBOT_GL_INCLUDE  "${CMAKE_CURRENT_SOURCE_DIR}/third_party/gl/include")
add_library(mujoco_gui STATIC
    backends/mujoco/gui/src/simulate.cc
    backends/mujoco/gui/src/glfw_adapter.cc
    backends/mujoco/gui/src/platform_ui_adapter.cc
    backends/mujoco/gui/src/glfw_dispatch.cc
    backends/mujoco/gui/src/lodepng.cpp
    backends/mujoco/mujoco_viewer.cpp)
target_include_directories(mujoco_gui
    PUBLIC  backends/mujoco backends/mujoco/gui/include
    PRIVATE "${MUJOCO_INCLUDE_DIR}" "${PHYBOT_GLFW_DIR}" "${PHYBOT_GL_INCLUDE}")
target_link_libraries(mujoco_gui PUBLIC
    "${MUJOCO_LIB}" "${PHYBOT_GLFW_DIR}/lib/libglfw3.so" ${CMAKE_DL_LIBS} pthread)
set_property(TARGET mujoco_gui PROPERTY CXX_STANDARD 17)

# ── RunGui harness (physics thread + MujocoViewer; apps-only link) ───────────
add_library(sim_harness_gui STATIC backends/mujoco/sim_harness_gui.cpp)
target_include_directories(sim_harness_gui PUBLIC backends/mujoco PRIVATE "${MUJOCO_INCLUDE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/third_party/lcm/include")
target_link_libraries(sim_harness_gui PUBLIC sim_harness mujoco_gui)
set_property(TARGET sim_harness_gui PROPERTY CXX_STANDARD 17)

# ── Headless sim walk app ─────────────────────────────────────────────────────
set(PHYBOT_XML_PATH
    "${CMAKE_CURRENT_SOURCE_DIR}/assets/robot/phybot_c2/xml/phybot_c2.xml")
set(POLICY_PT_PATH
    "${CMAKE_CURRENT_SOURCE_DIR}/tasks/walk/test/golden/cpg_policy.pt")

add_executable(sim_walk apps/sim_walk.cpp)
target_include_directories(sim_walk PRIVATE core/include tasks backends/mujoco
    "${MUJOCO_INCLUDE_DIR}")
target_compile_definitions(sim_walk PRIVATE
    POLICY_PATH="${POLICY_PT_PATH}"
    XML_PATH="${PHYBOT_XML_PATH}"
    PHYBOT_ENABLE_GUI=1)
target_link_libraries(sim_walk PRIVATE walk_task mujoco_backend mujoco_gui)
set_property(TARGET sim_walk PROPERTY CXX_STANDARD 17)

# ── Walk simulation ctest ─────────────────────────────────────────────────────
add_executable(test_walk_sim tasks/walk/test/test_walk_sim.cpp)
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
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party/lcm/include")
add_test(NAME input_protocol COMMAND test_input_protocol)

# ── TaskRouter (code→TaskCode routing with default fallback + axis shaping) ──
add_executable(test_task_router core/test/test_task_router.cpp core/src/task_router.cpp)
target_include_directories(test_task_router PRIVATE core/include core/test "${CMAKE_CURRENT_SOURCE_DIR}/third_party/lcm/include")
add_test(NAME task_router COMMAND test_task_router)

# ── Input mapping (ControllerState → DriveCmd) ─────────────────────────────────
add_executable(test_input_mapping publisher/test/test_input_mapping.cpp)
target_include_directories(test_input_mapping PRIVATE publisher/include core/include core/test
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party/lcm/include")
add_test(NAME input_mapping COMMAND test_input_mapping)

# ── LCM loopback (REAL-HOST transport check; NOT an automated ctest) ───────────
# Requires UDP multicast routing (startProject.sh adds `route add -net 224.0.0.0`).
# In a headless sandbox with no multicast route, the vendored liblcm aborts inside
# handleTimeout(), so this is a manual/real-host tool, not a registered test.
# The wire codec itself is proven bit-exact by the `input_protocol` ctest.
add_executable(test_lcm_loopback core/test/test_lcm_loopback.cpp core/src/input_protocol.cpp)
target_include_directories(test_lcm_loopback PRIVATE core/include core/test)
target_link_libraries(test_lcm_loopback PRIVATE phybot_lcm)
# add_test intentionally omitted (see above).

# ── Publisher process (JsController -> MapControllerToDrive -> ip_encode -> LCM) ──
# buttons_config.cpp (yaml-cpp) is compiled ONLY into this app -- input_mapping.h itself
# and test_input_mapping stay yaml-cpp-free (see buttons_config.h's own comment).
add_executable(main_publisher publisher/apps/main_publisher.cpp core/src/input_protocol.cpp
    publisher/src/buttons_config.cpp)
target_include_directories(main_publisher PRIVATE publisher/include core/include
    "${YAML_CPP_INCLUDE_DIR}")
target_compile_definitions(main_publisher PRIVATE PHYBOT_ROOT="${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(main_publisher PRIVATE phybot_publisher phybot_lcm "${YAML_CPP_LIB}")

# ── Teleop loop: command -> task-switch (ZERO<->WALK) -> live MuJoCo sim ─────────
# apps/sim_teleop.cpp (real LCM) drives backends/mujoco/teleop_loop.h's
# ApplyCmdAndTick() directly against walk_task + a directly-compiled zero_task.cpp
# (ZeroState is not part of the `walk_task` library -- only walk_config()/
# zero_config() are, transitively). The headless ctest below instead goes through
# SimHarness (registry-paired tasks; see sim_harness's own comment above), so it
# needs neither teleop_loop.h nor a directly-named Task subclass.

# Headless end-to-end smoke ctest: NO LCM (see the test_lcm_loopback note above for
# why the vendored liblcm aborts headless in this sandbox) -- a DriveCmd is set
# directly in-process instead of publishing it, through SimHarness::tick(). Links
# sim_harness ONLY: sim_harness already whole-archives phybot_tasks (walk_task and
# tasks/zero/zero_task.cpp would double-define WalkTask's/ZeroState's symbols).
add_executable(test_teleop_smoke core/test/test_teleop_smoke.cpp)
target_include_directories(test_teleop_smoke PRIVATE core/include backends/mujoco
    "${MUJOCO_INCLUDE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party/lcm/include")
target_compile_definitions(test_teleop_smoke PRIVATE
    POLICY_PATH="${POLICY_PT_PATH}"
    XML_PATH="${PHYBOT_XML_PATH}")
target_link_libraries(test_teleop_smoke PRIVATE sim_harness)
set_property(TARGET test_teleop_smoke PROPERTY CXX_STANDARD 17)
add_test(NAME teleop_smoke COMMAND test_teleop_smoke)

# Real-host teleop main: LCM subscribe Robot_ControlData -> live sim. Pairs with
# main_publisher for the Xbox demo. Requires LCM multicast on the host, so it is
# built (and must compile) but intentionally NOT registered as a ctest.
add_executable(sim_teleop apps/sim_teleop.cpp tasks/zero/zero_task.cpp core/src/input_protocol.cpp)
target_include_directories(sim_teleop PRIVATE core/include tasks backends/mujoco
    "${MUJOCO_INCLUDE_DIR}")
target_compile_definitions(sim_teleop PRIVATE
    POLICY_PATH="${POLICY_PT_PATH}"
    XML_PATH="${PHYBOT_XML_PATH}"
    PHYBOT_ENABLE_GUI=1)
target_link_libraries(sim_teleop PRIVATE walk_task mujoco_backend mujoco_gui phybot_lcm)
set_property(TARGET sim_teleop PROPERTY CXX_STANDARD 17)
