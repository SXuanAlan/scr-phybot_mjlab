# PHYBot C2 Deploy 重构设计文档

- 日期:2026-06-28
- 分支:`codex/phybot-c2-amp-walk-deploy`
- 源:`deploy/phybot_deploy_c2`(现官方单体部署代码,已从原 `bdml_phybot_deploy_c2` 改名而来)
- 目标:`deploy/bdml_phybot_deploy_c2`(空目录,新架构落地处)

---

## 1. 目标与范围

### 1.1 目标
1. 把官方单体部署重构成**可复用库 + 自注册 Task 插件**,让"加一个动作"从改 ~7 处文件降到:中心 `task_id.h` 加 1 行枚举 + 一个自包含 `tasks/<name>/` 目录(+cmake/bindings 各一行)。
2. `TaskManager` 做成**项目内可复用的任务状态机模块**(放入 `core`,枚举注册表 + 自注册),借鉴 Unitree `unitree_guide` 的成熟契约;**不**抽成跨机器人独立库。
3. **Xbox 手柄支持**:稳健设备层(SDL2/热插拔)+ 配置驱动按键映射 + 轴→连续命令。
4. **集成自己的 task**:walk_amp(**你自己的 task,非官方**;`rl_deploy_walk_amp` 不作权威参考)+ 一个空白模板。obs/动作权威来源待定,**靠后接入**(见 §15)。
5. **A1 参考/对齐基线 = 官方 CPG 走路 `rl_deploy_cpg`**(航空手柄走路键 `RL_walk` 跑的就是它;配置一致、可运行、obs=283、零位 -0.295)。mimic 因官方配置损坏(num_task_obs=201 vs 动作 CSV 591 列)暂缓;walk_amp 你自己的、靠后。

### 1.2 范围
- **做**:新架构 + `core` 内任务状态机 + InputProtocol 统一 + 自注册 Task + 配置体系 + walk_amp + 空白模板 + mimic + Xbox 输入层。
- **不做**:其余旧 task(cpg/kongfu/taichi/climb_up/crane_down/long_motion/forward_*)暂不迁移;gazebo 环境不迁;**不引入任何软件急停**。

### 1.3 不变量(硬约束)
- **电机使能/失能 + 官方急停路径**:原样保留,本次重构不修改该路径。
- **`DataPackage` 黑板的数值管线**:A1 迁移的官方走路 `cpg` 的 obs(283)/动作与官方 `rl_deploy_cpg` **逐位一致**(含其已知怪癖,见 §13)→ 实机对齐不受影响。walk_amp(你自己的)来源待定、靠后。
- **LCM 线格式**(`0xeb90` / data 偏移 / 校验和):逐字节保留 → 旧 iPad App、旧手柄仍兼容。
- **主循环骨架**:读电机→读IMU→[大脑]→写电机→等周期,只换"大脑"两块。

---

## 2. 总体架构

### 2.1 分层(依赖单向向内)

```
core/ (libphybot_core)        部署核心库(认识 DataPackage / libtorch / LCM)
  ├─ Task / TaskManager / TaskRegistry / REGISTER_TASK   ← 任务状态机(直接吃 DataPackage)
  └─ PolicyTask / PolicyRunner / HistoryBuffers / InputProtocol / RobotBackend(接口)

backends/                    RobotBackend 实现(按 WHICH_ENV 选其一)
  ├─ realrobot_mini/ (MotorList + IMU)
  └─ mujoco_sim_mini/ (MujocoInterface)

tasks/                       自注册 Task 插件(每个一目录, 自包含)
  ├─ zero/  walk_amp/  mimic/  _template/

publisher/                   独立进程: Xbox 输入层(SDL2 + 配置映射 → LCM)

apps/                        可执行: main(robot) / task_dry_run

config/                      robot.yaml / bindings.yaml
```

依赖方向:`tasks → core`;`backends → core`;`apps → core + backend + tasks`;`publisher → core(仅 InputProtocol)`。任务状态机是 `core` 的一部分,直接以 `DataPackage` 为黑板,不再做跨机器人解耦(放弃模板 `Ctx` 抽象,换取简单内聚)。

### 2.2 目录布局(目标)

```
bdml_phybot_deploy_c2/
├── core/
│   ├── include/phybot/{task_id.h, task.h, task_manager.h, task_registry.h,
│   │                   PolicyTask.h, PolicyRunner.h, HistoryBuffers.h,
│   │                   ObsBuilder.h, InputProtocol.h, RobotBackend.h}
│   └── src/...
├── backends/{realrobot_mini/, mujoco_sim_mini/}
├── tasks/
│   ├── zero/      {zero.cpp}
│   ├── walk_amp/  {walk_amp.cpp, config/rl_params.yaml, model/policy.pt}
│   ├── mimic/     {mimic.cpp,    config/rl_params.yaml, model/policy.pt, ref/}
│   └── _template/ {template.cpp(桩), config/rl_params.yaml(占位)}
├── publisher/{device/(SDL2,evdev), mapping/, main_publisher.cpp}
├── apps/{main.cpp, task_dry_run.cpp}
├── config/{robot.yaml, bindings.yaml}
├── cmake/phybot.cmake   (phybot_add_task 等)
├── third_party/         (libtorch/lcm/eigen/yaml/mujoco/glfw/SDL2)
└── CMakeLists.txt
```

---

## 3. 任务状态机(`core` 内,直接以 DataPackage 为黑板)

被注册/切换的单元叫 **`Task`**;切换器叫 **`TaskManager`**(FSM 语义:同一时刻一个 active task,按请求转换)。两者都是 `core` 的一部分,不单独抽成跨机器人库;直接操作 `DataPackage&`(去掉模板 `Ctx` 与 `DeployCtx` 包装)。

### 3.1 设计来源
借鉴 Unitree `unitree_guide`(`FSM`/`FSMState`/`checkChange`)的**行为契约**,抛弃其**绑定机制**:

| Unitree 的做法 | 本设计 |
|---|---|
| ✅ `enter/run/exit` 生命周期 | 保留 |
| ✅ 每状态 `checkChange()` 自决转换 | 保留(支持动作做完自动跳转) |
| ✅ 全局安全抢占(checkSafty→passive) | 保留为能力(`requestSafe`),当前不接软急停 |
| ❌ `enum` + 手写 `getNextState` 巨型 switch | `enum TaskId` + `std::array` 注册表(无 switch) |
| ❌ 手写 `FSMStateList` typed 指针结构 | 自注册 `REGISTER_TASK`(无手写结构) |
| ❌ 无状态级互锁 | 新增 `canExit()` |

> 与 Unitree/上一稿不同点:**不引入模板 `Ctx`**。换取简单内聚;代价是任务状态机耦合 `DataPackage`、不便整体搬去其它机器人项目。项目内仍可复用(任意多 task 共用一套 `TaskManager`)。

### 3.2 标识:`enum class TaskId`(黑板字段用 enum,鲁棒)
黑板上的 `requested_task` / `current_task` 都会被 LCM 回调线程与心跳回报线程**跨线程**访问。用 **`enum`(平凡可拷贝)**与官方 `NextState` 一样安全,避免 `std::string` 跨线程 UB(见 §5.5)。代价:一个**中心枚举清单**,加 task 在此加一行(唯一中心编辑点;名字由 `REGISTER_TASK` 与 task 放一起,不中心化)。

```cpp
// core/include/phybot/task_id.h  —— 唯一中心清单(加 task 在此加一行)
namespace phybot {
enum class TaskId { ZERO, WALK_AMP, MIMIC, /* …新增在此加一行… */ COUNT };
}
```

### 3.3 接口

```cpp
// core/include/phybot/task.h
namespace phybot {
class Task {
public:
    virtual ~Task() = default;
    explicit Task(TaskId id, std::string name) : id_(id), name_(std::move(name)) {}
    TaskId id() const { return id_; }
    const std::string& name() const { return name_; }            // 仅日志/配置可读

    virtual void enter(DataPackage&) {}                          // 进入(复位/加载)
    virtual void run  (DataPackage&) = 0;                         // 每拍执行
    virtual void exit (DataPackage&) {}                           // 退出(清零)
    virtual TaskId checkChange(DataPackage&) { return id_; }      // 自跳转, 返回自身=不变
    virtual bool canExit(DataPackage&) { return true; }           // 互锁
protected:
    TaskId id_;  std::string name_;
};

// core/include/phybot/task_manager.h
class TaskManager {
public:
    void add(std::unique_ptr<Task> t) { tasks_[(int)t->id()] = std::move(t); }
    void start(TaskId id, DataPackage& io) { current_ = tasks_[(int)id].get(); current_->enter(io); }
    TaskId current() const { return current_->id(); }

    void tick(DataPackage& io) {
        TaskId next = current_->checkChange(io);              // ① task 自决(返回自身=不变)
        if (next == current_->id()) next = io.requested_task; // ② 否则外部请求
        if (next != current_->id() && tasks_[(int)next]) {
            if (current_->canExit(io)) {                      // ③ 互锁
                current_->exit(io);
                current_ = tasks_[(int)next].get();
                current_->enter(io);
            }
        }
        io.current_task = current_->id();
        current_->run(io);                                    // ④ 跑当前
    }
private:
    std::array<std::unique_ptr<Task>, (size_t)TaskId::COUNT> tasks_;   // 按 enum 下标, 无哈希/无堆查找
    Task* current_ = nullptr;
};
} // namespace phybot
```
黑板字段:`TaskId requested_task{TaskId::ZERO};` `TaskId current_task{TaskId::ZERO};`(平凡,跨线程安全)。

### 3.4 转换语义
- 单线程、单拍内 `exit→swap→enter→run` 顺序执行,无双 run、无交错。
- 转换源优先级:task `checkChange`(返回非自身=自动跳转)> `io.requested_task`(外部手柄/App)。
- `canExit() == false` → 维持当前 task(对应官方 Blocked)。
- 切换瞬间不跑旧 task 的 `run`,只 `enter` 新 task 并跑其 `run`(单拍内完成,1kHz 下 1ms 内)。

---

## 4. 部署层:PolicyTask

### 4.1 `PolicyTask` —— RL 便利基类(吃掉通用流程)
直接继承 `phybot::Task`,以 `DataPackage` 为黑板。
```cpp
class PolicyTask : public phybot::Task {
public:
    using phybot::Task::Task;
    void enter(DataPackage&) override;     // 载入 policy.pt + rl_params.yaml + 复位 history
    void run  (DataPackage&) final;         // 模板方法: BuildObservation→Forward→ActionToTarget→IsFinite→写回
    void exit (DataPackage&) override;      // 清零控制量
protected:
    virtual void BuildObservation(ObsBuilder&, const DataPackage&) = 0;   // 必填: 观测布局
    virtual Command MapCommand(const Command& raw) { return raw; }        // 可选: 命令解释
    virtual Eigen::VectorXd ActionToTarget(const Eigen::VectorXd& a);     // 可选: 默认 a*scale+default_pos
    PolicyRunner policy_;  HistoryBuffers hist_;  PolicyConfig cfg_;
};
```

`run()` 固定流程(每 decimation 拍推理一次,其余拍复用上次 target;`IsFinite` 失败保持上一目标,不发散)。

---

## 5. 输入 / 命令层

### 5.1 进程架构:保留两进程(LCM 解耦)
```
[Publisher 进程]                                  [main 进程]
SDL2 设备层 → 归一化 ControllerState               LCM 订阅
  → mapping(读 bindings.yaml)                        → InputProtocol::decode
    → DriveCmd{model,vx,vy,wz}                          → io.requested_task(配置映射) + io.js_*（限幅）
      → InputProtocol::encode → LCM 发布  ───────────→  → TaskManager.tick 消费
```

### 5.2 `InputProtocol`(消灭"两份解析")
位置:`core/include/phybot/InputProtocol.h` + `core/src/InputProtocol.cpp`,**main 与 publisher 共用一份**。

线格式(逐字节保留官方):`header(int16)=0xeb90` · `frame_type(int16)=0x0001` · `data[59]` · `checksum(int8)`;`data[0]=model(int8)`, `data[1..4]=vx(f32)`, `data[5..8]=vy(f32)`, `data[9..12]=wz(f32)`。校验和 = 原始结构体字节 `[2,63)` 求和 `% 255`(与现有 `server.cpp`/`joystick_int.cpp` 两份完全一致)。

```cpp
namespace phybot::InputProtocol {
  struct DriveCmd { int8_t model{0}; float vx{0}, vy{0}, wz{0}; };
  int8_t checksum(const custom_lcm::system2Algorithm_packet&);
  bool   decode(const custom_lcm::system2Algorithm_packet&, DriveCmd&);  // 校验 header/校验和/帧类型
  void   encode(const DriveCmd&, custom_lcm::system2Algorithm_packet&);
}
```

### 5.3 main 侧 `input.poll()`(取代 `Server` 的 model 分支)
```cpp
void InputReceiver::poll(DataPackage& io) {
    Packet pkt;
    { std::lock_guard<std::mutex> lk(mu_); pkt = latest_; }       // 主线程取快照(见 5.5 线程安全)
    InputProtocol::DriveCmd c;
    if (!InputProtocol::decode(pkt, c)) return;                   // 解包失败丢弃(同现状)
    io.js_vx_desire     = std::clamp((double)c.vx, -0.4, 1.0);   // 保留官方限幅
    io.js_vy_desire     = std::clamp((double)c.vy, -0.1, 0.1);
    io.js_OmegaZ_desire = std::clamp((double)c.wz, -1.0, 1.0);
    if (auto it = model_map_.find(c.model); it != model_map_.end())
        io.requested_task = debounce(it->second);                 // model→TaskId(配置映射)+ 去抖(见 5.5)
}
```

### 5.4 Xbox 设备层(publisher 内)
```cpp
class IController { public:
  virtual bool poll(ControllerState& out) = 0;   // 归一化: 按键 bool + 轴 [-1,1]
  virtual bool connected() const = 0;
};  // 实现: SdlController(默认) / EvdevController(后备)
```
- 默认 SDL2 GameController:统一按键语义、热插拔、多手柄。
- 死区/缩放/平滑(slew)在 publisher 的 mapping 层完成(main 收到干净命令)。
- 兼容旧数字 model:线上仍传 `model` int;按键↔`TaskId` 两侧都用 `bindings.yaml` 的 `lcm_model_map` 互转。

### 5.5 线程安全与切换去抖(改动官方逻辑后新引入的两条真机风险)

**(N1)跨线程字段用 `enum`(已解决)。**
`requested_task` / `current_task` 会被 LCM 回调线程与心跳回报线程跨线程访问。**采用 `enum class TaskId`(平凡可拷贝)**,与官方 `NextState`(enum)一样安全,无 `std::string` 跨线程 UB。另:LCM 回调线程**只在 mutex 下存原始字节** `latest_`,`poll()` 在主线程取快照解码,避免撕裂读原始包。`current_task` 同为 enum,供心跳线程安全读取。

**(N2)切换去抖:官方 500ms 轮询天然去抖,改每拍即时后须补。**
每拍即时切换 → 一次按键抖动/瞬时误触会立刻切 task(真机行走中误切到趴下/爬起很危险)。规定:
- 输入侧做**按键边沿检测 + 去抖窗口**(`debounce()`);相同请求需稳定 N 拍才生效。
- 切换仍受 `canExit` 互锁;对"危险跳转"可在目标/当前 task 的 `canExit` 加条件。

---

## 6. 主循环集成(保留官方骨架)

官方 `run_mini.cpp` 九步里,只把中间的 `server`+`manager`(④⑤⑥⑦⑧)换成 ❶❷:

```cpp
phybot::TaskManager manager;
phybot::TaskRegistry::instance().loadInto(manager);   // 自注册 task 全部加载(开机加载所有 policy.pt)
manager.start(TaskId::ZERO, package);

while (1) {
    motorlist.GetStatesToPackage(...);     // ② 读电机    —— 官方不动
    package.getIMUdata(reader);            // ③ 读 IMU    —— 官方不动
    input.poll(package);                   // ❶ decode → requested_task + vx/vy/ω(每拍即时, 取代 500ms 轮询)
    manager.tick(package);                 // ❷ 切换 + 跑当前 task
    motorlist.SetCommandsFromPackage(...); // ⑨ 写电机    —— 官方不动
    wait_rest_of_period(&P_info);          // 周期       —— 官方不动
}
```
- 上电序列(读 50 次 → `Enable()` → sleep 10s)、电机使能/失能、**官方急停路径** 全部在 ❶❷ 之外,不碰。
- 改进:task 切换从官方"独立 500ms 轮询线程"变为每拍即时解析,延迟更低更确定。

---

## 7. 配置体系(三层,职责不重叠)

| 文件 | 归属 | 内容 | 加 task 是否改 |
|---|---|---|---|
| `config/robot.yaml` | 整机 | 关节顺序、默认零位、PD 上限、限幅 | 否 |
| `config/bindings.yaml` | 输入 | 按键→task、轴→命令通道(死区/缩放/slew)、`lcm_model_map` | 改一行加按键 |
| `tasks/<name>/config/rl_params.yaml` | 该 task 自带 | `policy_path`/obs/action 维度/decimation/scale/PD/阈值 | 新 task 自带 |

`bindings.yaml` 示例:
```yaml
deadzone: 0.08
buttons: { B: walk_amp, X: mimic, Y: zero }
axes:
  vx: { source: left_y,  scale: 0.6, invert: true, deadzone: 0.08, slew: 2.0 }
  vy: { source: left_x,  scale: 0.4,               deadzone: 0.08, slew: 2.0 }
  wz: { source: right_x, scale: 0.8,               deadzone: 0.12, slew: 3.0 }
lcm_model_map: { 1: zero, 2: cpg, 3: mimic, 9: walk_amp }
```

---

## 8. 构建与自注册

### 8.1 库切分
- `libphybot_core`(.a,**含任务状态机**)· `backend_realrobot.a` / `backend_mujoco.a` · `tasks`(OBJECT)· 可执行 `main` / `task_dry_run` / `publisher`。(不再有独立 `libfsm`。)

### 8.2 自注册防丢(关键技术点)
自注册 task 没有被显式引用,静态库 `.a` 会丢未引用目标文件 → 注册丢失。**对策:tasks 用 CMake `OBJECT` 库,目标文件直接链进可执行**(不走 `.a` 的丢弃规则);等价替代为 `alwayslink`/`--whole-archive`。

```cmake
add_library(phybot_tasks OBJECT
    tasks/zero/zero.cpp tasks/walk_amp/walk_amp.cpp tasks/mimic/mimic.cpp)
target_link_libraries(main PRIVATE phybot_core $<TARGET_OBJECTS:phybot_tasks>)
```

### 8.3 `phybot_add_task()` —— 把细节藏起来
```cmake
phybot_add_task(walk_amp)   # 收 tasks/walk_amp/*.cpp + 挂 core + 并入可执行
phybot_add_task(mimic)
phybot_add_task(my_new)     # 二次开发: 新目录 + 一个类 + 这一行
```

### 8.4 环境开关
```bash
cmake -S . -B build -DWHICH_ENV=realrobot_mini   # 链 backend_realrobot
cmake -S . -B build -DWHICH_ENV=mujoco_sim_mini  # 链 backend_mujoco
```
同一 task、同一 `policy.pt`,真机/仿真零改动切换(task 只碰黑板,碰硬件的是 backend)。

---

## 9. Task 编写规范

### 9.1 每个 task 固定编辑面
1. `core/include/phybot/task_id.h`:枚举加一行 `TaskId::<NAME>`(唯一中心编辑点)
2. `tasks/<name>/<name>.cpp`:类 + `REGISTER_TASK(TaskId::<NAME>, "<name>", <Class>)`
3. `tasks/<name>/config/rl_params.yaml`
4. `tasks/<name>/model/policy.pt`
5. `CMakeLists.txt`:`phybot_add_task(<name>)`
6. `config/bindings.yaml`:按键 + lcm model id

**永不碰**:`core` 任务状态机(`Task`/`TaskManager`)、`InputProtocol`、`server`、主循环、中心注册/分发逻辑。

### 9.2 `.cpp` 钩子(按 task 差异重写)
| 钩子 | 何时 | 必填 |
|---|---|---|
| `BuildObservation` | 总是 | ✅ |
| `MapCommand` | 手柄命令特殊解释 | 可选 |
| `ActionToTarget` | 动作换算非默认 | 可选 |
| `enter` | 进入额外准备(如加载参考动作) | 可选 |
| `checkChange` | 做完自动跳转 | 可选 |
| `canExit` | 中段不许打断 | 可选 |

### 9.3 三类 task 示例
- **A 行走类(walk_amp)**:继承 `PolicyTask`,只重写 `BuildObservation`(+可选 `MapCommand`)。
- **B 参考动作类(mimic)**:继承 `PolicyTask`,重写 `enter`(加载参考片段)+ `BuildObservation`(含 phase)+ 可选 `checkChange`(做完回 zero)。
- **C 非 RL 类(zero)**:继承 `phybot::Task`,自己写 `run`(PD 到零位),无 policy/obs。

```cpp
// 自注册宏(名字与 task 放一起, 不中心化; TaskId 引用中心枚举)
REGISTER_TASK(TaskId::WALK_AMP, "walk_amp", WalkAmpTask);
```

### 9.4 "改"已有 task
| 改什么 | 改哪 | 重编 |
|---|---|---|
| 换策略权重 | `model/policy.pt` | 否(跑 dry-run)|
| 调 PD/scale/阈值 | `rl_params.yaml` | 否 |
| 改按键 | `bindings.yaml` | 否 |
| 改观测/命令逻辑 | `<name>.cpp` 钩子 | 是(仅该 task)|

---

## 10. 安全策略

- **沿用官方原有急停与电机使能策略,本次重构不修改该路径。**
- 不引入任何软件急停(无看门狗 / seq / SAFE / estop 协议)。`phybot::TaskManager::requestSafe()` 仅作为能力保留,默认不接线。
- 真机安全依赖**硬件急停按钮**(需确认其接在电机供电/使能回路中,而非软件读取)。
- 取舍(显式接受):手柄断连 / publisher 崩 / WiFi 掉 / main 崩 时机器人不自动停,依赖人工监护 + 物理急停。
- 测试物理防护:硬件急停 = 断电 = 双足会瘫倒,测试上吊架/护具;真机测试建议 publisher 跑在机器人本机(本地组播)。
- 待确认项(不阻塞本设计):MotorList 的电机心跳保护(`HEART_BEAT_PROTECT_ON`/`SetHeartbeat`)当前是否使能。

### 10.1 SIM 与真机的 PD 链路不同(关键,已在官方代码核实)
**任务输出的 `sim_P/sim_D` 只对仿真有效;真机完全不看它。**
- **SIM**:`MujocoInterface::SetTorque` 用 `sim_P/sim_D` 算力矩 `ctrl = tau_ff + P·(pos_des−pos) + D·(vel_des−vel)`。新 `MujocoBackend` 一致,是忠实的。ZeroState 的 `1000/100`、walk 的 `100/5` 都是**仿真增益**。
- **真机**:`MotorList::SetCommandsFromPackage` 调 `SetBigparam(Pos,Vel,Cur=0)`——**签名无 kp/kd**,只发 位置+速度 目标、前馈力矩清零(`0*tor_cmd`),**从不引用 `sim_P/sim_D`**。真机 PD 增益在**电机驱动固件**里,由 `MotorList/config/phybot_mini_*.yaml`(`ControlMode:0x0B` + 每电机 `PDKp∈{20,50,100}`/`PDKd∈{5,10}`,大腿≈100/5)在初始化写入。
- **推论**:`1000/100` 是 sim-only 保真值,**不是真机安全数**;真机 ≈ 100/5(电机配置)。二者不冲突,各管一条链路。
- **B 组真机 backend 硬约束**:必须照 `SetCommandsFromPackage` —— **只发 pos/vel,增益走 MotorList 配置,不要把 `sim_P/sim_D` 发给硬件,前馈力矩清零**。
- **但捕获位姿 + 0.5s 三次斜坡对真机仍必需**:`motor_Pos_desire`(位置目标)会发给真机电机,snap 到 default = 位置阶跃,驱动器 PD(100/5)照样猛追 → 顿挫。斜坡让位置目标平滑,是真机安全关键(与增益无关)。

---

## 11. 测试与验收

### 11.1 通用 dry-run(`task_dry_run`)
取代仅针对 walk_amp 的 `walk_amp_dry_run_check`,对**任意 task** 喂一帧数据跑一次 `run`,检查:策略加载成功、obs 维度正确、输出有限、在限幅内。上电前必跑。

### 11.2 对齐保护(最高优先级)
- **A1 对齐目标 = 官方 CPG 走路**:新 `WalkTask`(端口自 `rl_deploy_cpg`)与官方 `rl_deploy_cpg` 在同一帧输入下 obs(**283**)/action(**21**)**逐位一致**(通用 dry-run 对比新旧二进制),通过后 A1 才达标。
- **必须逐位复现 cpg 的怪癖**(见 §13.1):pos 历史存 `q×scale`(**不减 default**,与当前帧不对称)、index9 `walk_flag=(|vx|<0.3?0:1)`、无 obs/action clip、无 `torque_limits`、历史 newest-at-tail、首次推理在 `step_num=10` 见全零历史;CPG 振荡器/动作·旋转·角速度历史/`num_of_heights=625`/motion CSV **全是死代码,不移植**。**不许"顺手清理"**,否则破坏对齐。
- walk_amp / mimic 对齐:来源/配置就绪后再做。
- DataPackage 字段名、限幅值、decimation、scale、PD、obs 布局保持不变。

### 11.3 单元/集成
- 任务状态机:用最小 `DataPackage` + 几个哑 `Task` 做纯逻辑单测(切换/checkChange/canExit/注册表),不依赖电机/IMU。
- InputProtocol:encode→decode 往返一致;对旧 App/手柄真实包做解码回归。

### 11.4 上真机硬门槛 Checklist(全绿才上真机;作为实现计划中的显式 review 检查点)
1. ☐ 新旧 **obs/action 逐位一致**(对齐脚本 + dry-run)
2. ☐ `robot.yaml` 关节序/方向/零位 vs 官方逐项核对
3. ☐ PD/力矩限幅/decimation/命令限幅 = 官方
4. ☐ LCM 解码对真实旧 App/手柄包回归通过
5. ☐ 启动打印已注册 task 清单 = 期望清单
6. ☐ `requested_task`/`current_task` 为 `enum`(跨线程安全),LCM 原始包读取已加锁(N1)
7. ☐ 切换去抖到位(N2)
8. ☐ 电机心跳保护确认 + 硬件急停接线确认
9. ☐ 先 mujoco 跑通 → 真机 dry-run(不使能下发)→ 吊架/低速上机
10. ☐ 官方版回退路径已验证可用

---

## 12. 分阶段落地(先做完不需真机的 A 组,再统一分步进行真机 B 组)

### 阶段组 A:不需要真机(全部先做完并验证)
- **A0** `core` 任务状态机(`Task`/`TaskManager`/`TaskRegistry`/`REGISTER_TASK`/`TaskId`)+ 纯逻辑单测(最小 `DataPackage` + 哑 `Task`:切换/checkChange/canExit/注册表)。
- **A1** `core` 其余(`PolicyTask`/`PolicyRunner`/`HistoryBuffers`/`ObsBuilder`/`InputProtocol`)+ **mujoco backend** + `DataPackage` 端口;接入 `zero` + **官方 CPG 走路 `WalkTask`(`TaskId::WALK`,端口自 `rl_deploy_cpg`)**;mujoco 跑通;通用 dry-run + **对齐校验:新 `WalkTask` vs 官方 `rl_deploy_cpg` 逐位一致(obs 283 / action 21)**。
- **A2** 主循环集成(`TaskManager` + `InputProtocol` 替换 `server`/`manager`,骨架对照官方逐行)——**在 mujoco backend 上验证,不碰真机**。
- **A3** Publisher SDL2 输入层 + `bindings.yaml`;手柄→命令→切 task 在 mujoco 上打通;**N1(enum 跨线程)/ N2(去抖)** 验证。
- **A4** 接入 **`walk_amp`(你自己的,obs/动作来源定下后)** + 补 `_template`;在 mujoco 上验证。
- **A5** 清理、文档、CI/回归(单测 + dry-run + 对齐脚本)。

> **A 组出口标准**:仿真完整可用、对齐逐位一致、手柄打通。此期间真机部署仍由官方 `phybot_deploy_c2` 承担。

### 阶段组 B:真机(统一、分步、每步可回退;前置 = A 组全绿 + §11.4 硬门槛逐条核对)
- **B1** `realrobot backend`(MotorList + IMU)实装;**不使能电机**,只读传感器→`DataPackage`,与官方读数逐项对照。
- **B2** 真机 `task_dry_run`(**不下发电机**):逐 task 验证策略加载 / obs / 输出范围。
- **B3** 主循环接真机 backend,**电机使能但低增益 + 吊架**,`zero` 站立保持;逐行 diff 确认官方上电/急停路径未改。
- **B4** `mimic` 上机:吊架→低速→放开,人工监护 + 硬件急停常备。
- **B5** `walk_amp` 上机(同上谨慎流程;前置:其 obs/动作来源已定并通过 A 组对齐)。

> 每步独立验证、可随时回退官方版;任一步不过不进入下一步。新目录在 A1 跑通前没有"半截可用"状态,故按阶段推进、避免最后一次性合龙。

---

## 13. 风险与对策

风险按"会不会伤到真机"分级:🔴 直接致动作错误/发散  🟠 行为异常/崩溃  🟡 功能缺失/工程。

### 13.1 数值/对齐(🔴 最高,直接致摔/抽搐/冲出)
| 风险 | 对策/门槛 |
|---|---|
| 🔴 cpg obs(283)/动作拆进 `PolicyTask` 时出现细微偏差(顺序/`scale`/gravity 符号/history 填充/walk_flag 阈值) | A1: 新 `WalkTask` vs 官方 `rl_deploy_cpg` **逐位一致**(dry-run 对比);不通过禁止上真机 |
| 🔴 "顺手清理"了 cpg 的怪癖 → 偏离已训练策略 | **逐位复现**:pos 历史 `q×scale` 不减 default、index9 walk_flag、无 clip/torque_limit、newest-at-tail、首推理 step10 见全零历史;死代码(振荡器/heights625/motion CSV/动作·旋转·角速度历史)不移植但也不改变数值路径 |
| 🔴 关节顺序/方向/零位映射错位(`q/dq` 序、`direction_vector`、`default_dof_pos`=-0.295) | `robot.yaml` 与官方 `MotorList/config/*` 逐项核对;沿用官方上电"读 50 次打印"对照零位 |
| 🔴 PD/力矩限幅/decimation/控制周期 漂移 | 数值沿用官方;dry-run 校验输出在 `torque_limits` 内;周期不动 |
| 🔴 命令限幅/LCM 解包 byte 偏移/校验和错位 | 限幅值逐字保留;`encode→decode` 往返一致 + 用真实旧 App/手柄包回归 |

### 13.2 改动官方逻辑新引入(🔴/🟠)
| 风险 | 对策/门槛 |
|---|---|
| ✅ **N1(已消除)**:跨线程字段用 `enum class TaskId`(平凡可拷贝),与官方 `NextState` 一样安全 | 黑板用 enum;LCM 原始包仍在 mutex 下快照(§5.5) |
| 🟠 **N2**:切换由 500ms 轮询改每拍即时,失去天然去抖,误触即切 | 按键边沿检测 + 去抖窗口;`canExit` 互锁;危险跳转加条件(§5.5) |
| 🟠 重构误删官方某处 `Enable/Disable/SetCommands`/急停 | 逐行 diff 官方 `run_mini` 上电/电机段,确保 ❶❷ 之外零改动 |

### 13.3 任务状态机/实时/流程(🟠/🟡)
| 风险 | 对策/门槛 |
|---|---|
| 🟠 自注册被链接器丢弃 → 按键"无反应" | tasks 用 OBJECT 库(或 alwayslink/--whole-archive);**启动打印已注册 task 清单与期望核对** |
| 🟠 某 `policy.pt` 缺失/损坏 | 加载校验放在 `Enable()` **之前**,失败拒绝上电 |
| 🟠 `enter` 每次复位 history 与官方语义差异 | 对照 `rl_deploy_cpg` 进出行为(其 `Exit()` 只清 pos/vel 历史,**不**复位 step_num/last_action → 重进有残留);dry-run 覆盖"切出再切入" |
| 🟠 `checkChange` 误引入自动跳转 | walk_amp/zero 默认返回自身(不变);仅 mimic 显式用并充分测试 |
| 🟠 热路径堆分配/tensor 每拍分配 → 抖动丢拍 | `enter` 预分配所有 buffer;hot path 无 malloc;测 tick 耗时分布 |
| 🟡 `REGISTER_TASK` 名称/bindings 拼错或缺文件仅运行时发现 | 启动全量校验注册表+配置+模型,缺失拒绝进入电机使能(`TaskId` 枚举本身编译期检查) |
| 🟡 `PolicyTask` 抽象对 mimic 不合身 | `run` 之外钩子足够灵活;必要时直接继承 `phybot::Task` 自写 `run` |
| 🟡 构建/autorun 跑到旧二进制(曾见 build/ 残留 walk_preset_amp) | 构建路径明确;启动打印 git hash/版本 |
| 🟡 无渐进回退、一次性大改 | 分阶段;真机阶段前仿真+dry-run+对齐全绿;**官方版随时可回退跑真机** |
| 🟡 跨主机 WiFi 丢包(无软件兜底) | 真机测试 publisher 跑本机/有线;人工监护 + 硬件急停 |

### 13.4 最重要的安全网
**官方 `deploy/phybot_deploy_c2` 完整保留、随时可编译跑真机。** 新架构在 mujoco + dry-run + 对齐**全绿之前,真机一律用官方版** → 真机部署能力零损失。

---

## 14. 关键决策记录

1. 范围:架构 + **官方 CPG 走路 cpg(A1 参考/逐位对齐基线,航空手柄走路键 `RL_walk`)** + walk_amp(你自己的,靠后,来源 TBD)+ 空白模板。mimic 官方配置损坏暂缓;`rl_deploy_walk_amp` 非官方,不作参考。
2. 形态:核心编库 + task 自注册插件(`TaskId` 枚举 + `std::array` 注册表)。
3. Xbox:稳健设备层(SDL2/热插拔)+ 配置驱动按键 + 轴→连续命令 + 接入新架构。
4. 进程:保留两进程(LCM 解耦)。
5. 环境:realrobot_mini + mujoco_sim_mini(放弃 gazebo)。
6. 命名:注册单元 = `Task`,切换器 = `TaskManager`(FSM 语义);`TaskId` / `TaskRegistry` / `REGISTER_TASK` / `enter/run/exit` / `checkChange` / `canExit`。不照抄官方 enum+switch。
7. 任务状态机**放入 `core`,不抽成跨机器人独立库**;借鉴 Unitree 行为契约 + `std::array` 注册表替代手写 switch/结构,但**直接以 `DataPackage` 为黑板**(放弃模板 `Ctx`,换取简单内聚)。
8. 标识:黑板 `requested_task`/`current_task` 用 `enum class TaskId`(跨线程安全),不用 `std::string`。
9. 安全:沿用官方急停,不加软件急停。

## 15. 待确认 / 开放项
- [ ] 物理急停接线方式(电机供电/使能回路 vs 软件读取)——决定"按急停必失能"是否成立。
- [ ] MotorList 电机心跳保护是否已使能(main 崩溃时的硬件兜底)。
- [ ] `_template` 是否同时给 PolicyTask 版与非 RL 版两个模板。
- [ ] **mimic 官方配置损坏**(`num_task_obs=201` vs 动作 CSV 591 列):要迁 mimic 需先确定正确 model+CSV 组合;暂缓。
- [ ] **walk_amp(你自己的)obs/动作权威来源未定**:训练侧 mjlab env vs 你另给规格 —— 定后才能写 `WalkAmpTask` 与其对齐校验。`rl_deploy_walk_amp` 不作权威参考。
- [ ] cpg `model/` 下有多个 `cpg_policy.pt`,当前 yaml 选 `0401_1519`;确认这就是航空手柄实际在跑的权重。
