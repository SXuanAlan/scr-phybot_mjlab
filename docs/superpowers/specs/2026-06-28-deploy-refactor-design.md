# PHYBot C2 Deploy 重构设计文档

- 日期:2026-06-28
- 分支:`codex/phybot-c2-amp-walk-deploy`
- 源:`deploy/phybot_deploy_c2`(现官方单体部署代码,已从原 `bdml_phybot_deploy_c2` 改名而来)
- 目标:`deploy/bdml_phybot_deploy_c2`(空目录,新架构落地处)

---

## 1. 目标与范围

### 1.1 目标
1. 把官方单体部署重构成**可复用库 + 自注册状态插件**,让"加一个动作"从改 ~7 处文件降到 ~1 处。
2. `StateMachine` 做成**项目内可复用的状态机模块**(放入 `core`,字符串注册表 + 自注册),借鉴 Unitree `unitree_guide` 的成熟契约;**不**抽成跨机器人独立库。
3. **Xbox 手柄支持**:稳健设备层(SDL2/热插拔)+ 配置驱动按键映射 + 轴→连续命令。
4. **集成自己的 task**:walk_amp + 一个空白模板,便于后续复制。
5. 迁移 **mimic** 一个旧 task 以验证架构通用性。

### 1.2 范围
- **做**:新架构 + `core` 内状态机 + InputProtocol 统一 + 自注册状态 + 配置体系 + walk_amp + 空白模板 + mimic + Xbox 输入层。
- **不做**:其余旧 task(cpg/kongfu/taichi/climb_up/crane_down/long_motion/forward_*)暂不迁移;gazebo 环境不迁;**不引入任何软件急停**。

### 1.3 不变量(硬约束)
- **电机使能/失能 + 官方急停路径**:原样保留,本次重构不修改该路径。
- **`DataPackage` 黑板的数值管线**:walk_amp 的 obs/动作数值逐位不变 → 实机对齐不受影响。
- **LCM 线格式**(`0xeb90` / data 偏移 / 校验和):逐字节保留 → 旧 iPad App、旧手柄仍兼容。
- **主循环骨架**:读电机→读IMU→[大脑]→写电机→等周期,只换"大脑"两块。

---

## 2. 总体架构

### 2.1 分层(依赖单向向内)

```
core/ (libphybot_core)        部署核心库(认识 DataPackage / libtorch / LCM)
  ├─ State / StateMachine / StateRegistry / REGISTER_STATE   ← 状态机(直接吃 DataPackage)
  └─ PolicyState / PolicyRunner / HistoryBuffers / InputProtocol / RobotBackend(接口)

backends/                    RobotBackend 实现(按 WHICH_ENV 选其一)
  ├─ realrobot_mini/ (MotorList + IMU)
  └─ mujoco_sim_mini/ (MujocoInterface)

states/                      自注册状态插件(每个一目录, 自包含)
  ├─ zero/  walk_amp/  mimic/  _template/

publisher/                   独立进程: Xbox 输入层(SDL2 + 配置映射 → LCM)

apps/                        可执行: main(robot) / task_dry_run

config/                      robot.yaml / bindings.yaml
```

依赖方向:`states → core`;`backends → core`;`apps → core + backend + states`;`publisher → core(仅 InputProtocol)`。状态机是 `core` 的一部分,直接以 `DataPackage` 为黑板,不再做跨机器人解耦(放弃模板 `Ctx` 抽象,换取简单内聚)。

### 2.2 目录布局(目标)

```
bdml_phybot_deploy_c2/
├── core/
│   ├── include/phybot/{state.h, state_machine.h, state_registry.h,
│   │                   PolicyState.h, PolicyRunner.h, HistoryBuffers.h,
│   │                   ObsBuilder.h, InputProtocol.h, RobotBackend.h}
│   └── src/...
├── backends/{realrobot_mini/, mujoco_sim_mini/}
├── states/
│   ├── zero/      {zero.cpp}
│   ├── walk_amp/  {walk_amp.cpp, config/rl_params.yaml, model/policy.pt}
│   ├── mimic/     {mimic.cpp,    config/rl_params.yaml, model/policy.pt, ref/}
│   └── _template/ {template.cpp(桩), config/rl_params.yaml(占位)}
├── publisher/{device/(SDL2,evdev), mapping/, main_publisher.cpp}
├── apps/{main.cpp, task_dry_run.cpp}
├── config/{robot.yaml, bindings.yaml}
├── cmake/phybot.cmake   (phybot_add_state 等)
├── third_party/         (libtorch/lcm/eigen/yaml/mujoco/glfw/SDL2)
└── CMakeLists.txt
```

---

## 3. 状态机(`core` 内,直接以 DataPackage 为黑板)

状态机是 `core` 的一部分,不单独抽成跨机器人库;`State`/`StateMachine` 直接操作 `DataPackage&`(去掉模板 `Ctx` 与 `DeployCtx` 包装)。

### 3.1 设计来源
借鉴 Unitree `unitree_guide`(`FSM`/`FSMState`/`checkChange`)的**行为契约**,抛弃其**绑定机制**:

| Unitree 的做法 | 本设计 |
|---|---|
| ✅ `enter/run/exit` 生命周期 | 保留 |
| ✅ 每状态 `checkChange()` 自决转换 | 保留(支持动作做完自动跳转) |
| ✅ 全局安全抢占(checkSafty→passive) | 保留为能力(`requestSafe`),当前不接软急停 |
| ❌ `enum` + `getNextState` 巨型 switch | 字符串 key + `unordered_map` 注册表 |
| ❌ 手写 `FSMStateList` typed 指针结构 | 自注册 `REGISTER_STATE` |
| ❌ 无状态级互锁 | 新增 `canExit()` |

> 与 Unitree/上一稿不同点:**不引入模板 `Ctx`**。换取简单内聚;代价是状态机耦合 `DataPackage`、不便整体搬去其它机器人项目。项目内仍可复用(任意多 state 共用一套 `StateMachine`)。

### 3.2 接口

```cpp
// core/include/phybot/state.h
namespace phybot {
class State {
public:
    virtual ~State() = default;
    explicit State(std::string name) : name_(std::move(name)) {}
    const std::string& name() const { return name_; }

    virtual void enter(DataPackage&) {}                          // 进入(复位/加载)
    virtual void run  (DataPackage&) = 0;                         // 每拍执行
    virtual void exit (DataPackage&) {}                           // 退出(清零)
    virtual std::string checkChange(DataPackage&) { return ""; }  // 自跳转, ""=不变
    virtual bool canExit(DataPackage&) { return true; }           // 互锁
protected:
    std::string name_;
};

// core/include/phybot/state_machine.h
class StateMachine {
public:
    void add(std::unique_ptr<State> s) { auto k = s->name(); states_[k] = std::move(s); }
    void start(const std::string& key, DataPackage& io) { current_ = states_.at(key).get(); current_->enter(io); }
    const std::string& current() const { return current_->name(); }

    void tick(DataPackage& io) {
        std::string next = current_->checkChange(io);         // ① 状态自决
        if (next.empty()) next = io.requested_state;          // ② 否则外部请求
        if (!next.empty() && next != current_->name() && states_.count(next)) {
            if (current_->canExit(io)) {                      // ③ 互锁
                current_->exit(io);
                current_ = states_.at(next).get();
                current_->enter(io);
            }
        }
        io.current_state = current_->name();
        current_->run(io);                                    // ④ 跑当前
    }
private:
    std::unordered_map<std::string, std::unique_ptr<State>> states_;
    State* current_ = nullptr;
};
} // namespace phybot
```

### 3.3 转换语义
- 单线程、单拍内 `exit→swap→enter→run` 顺序执行,无双 run、无交错。
- 转换源优先级:状态 `checkChange`(自动)> `io.requested_state`(外部手柄/App)。
- `canExit() == false` → 维持当前状态(对应官方 Blocked)。
- 切换瞬间不跑旧状态的 `run`,只 `enter` 新状态并跑其 `run`(单拍内完成,1kHz 下 1ms 内)。

---

## 4. 部署层:PolicyState

### 4.1 `PolicyState` —— RL 便利基类(吃掉通用流程)
直接继承 `phybot::State`,以 `DataPackage` 为黑板。
```cpp
class PolicyState : public phybot::State {
public:
    using phybot::State::State;
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
    → DriveCmd{model,vx,vy,wz}                          → io.requested_state(配置映射) + io.js_*（限幅）
      → InputProtocol::encode → LCM 发布  ───────────→  → StateMachine.tick 消费
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
        io.requested_state = debounce(it->second);                // model→state(配置映射)+ 去抖(见 5.5)
}
```

### 5.5 线程安全与切换去抖(改动官方逻辑后新引入的两条真机风险)

**(N1)`requested_state` 跨线程:`std::string` 非平凡,必须加锁/快照。**
官方 `NextState` 是 `enum`(平凡),回调线程写、主线程读,撕裂顶多读到旧值不崩。新设计 `requested_state` 是 `std::string`,**并发读写是数据竞争 UB(可能堆破坏 → 真机运行中崩溃)**。规定:
- LCM 回调线程**只在 mutex 下存原始字节** `latest_`,**不触碰** `DataPackage` 的 string 字段。
- `poll()` 在**主线程内**取快照、解码、写 `io.requested_state`,与 `tick()` 同线程,无竞争。

**(N2)切换去抖:官方 500ms 轮询天然去抖,改每拍即时后须补。**
每拍即时切换 → 一次按键抖动/瞬时误触会立刻切状态(真机行走中误切到趴下/爬起很危险)。规定:
- 输入侧做**按键边沿检测 + 去抖窗口**(`debounce()`);相同请求需稳定 N 拍才生效。
- 切换仍受 `canExit` 互锁;对"危险跳转"可在目标/当前 state 的 `canExit` 加条件。

### 5.4 Xbox 设备层(publisher 内)
```cpp
class IController { public:
  virtual bool poll(ControllerState& out) = 0;   // 归一化: 按键 bool + 轴 [-1,1]
  virtual bool connected() const = 0;
};  // 实现: SdlController(默认) / EvdevController(后备)
```
- 默认 SDL2 GameController:统一按键语义、热插拔、多手柄。
- 死区/缩放/平滑(slew)在 publisher 的 mapping 层完成(main 收到干净命令)。
- 兼容旧数字 model:线上仍传 `model` int;按键↔状态字符串两侧都用 `bindings.yaml` 的 `lcm_model_map` 互转。

---

## 6. 主循环集成(保留官方骨架)

官方 `run_mini.cpp` 九步里,只把中间的 `server`+`manager`(④⑤⑥⑦⑧)换成 ❶❷:

```cpp
phybot::StateMachine machine;
phybot::StateRegistry::instance().loadInto(machine);   // 自注册状态全部加载(开机加载所有 policy.pt)
machine.start("zero", package);

while (1) {
    motorlist.GetStatesToPackage(...);     // ② 读电机    —— 官方不动
    package.getIMUdata(reader);            // ③ 读 IMU    —— 官方不动
    input.poll(package);                   // ❶ decode → requested_state + vx/vy/ω(每拍即时, 取代 500ms 轮询)
    machine.tick(package);                 // ❷ 切换 + 跑当前状态
    motorlist.SetCommandsFromPackage(...); // ⑨ 写电机    —— 官方不动
    wait_rest_of_period(&P_info);          // 周期       —— 官方不动
}
```
- 上电序列(读 50 次 → `Enable()` → sleep 10s)、电机使能/失能、**官方急停路径** 全部在 ❶❷ 之外,不碰。
- 改进:状态切换从官方"独立 500ms 轮询线程"变为每拍即时解析,延迟更低更确定。

---

## 7. 配置体系(三层,职责不重叠)

| 文件 | 归属 | 内容 | 加 task 是否改 |
|---|---|---|---|
| `config/robot.yaml` | 整机 | 关节顺序、默认零位、PD 上限、限幅 | 否 |
| `config/bindings.yaml` | 输入 | 按键→state、轴→命令通道(死区/缩放/slew)、`lcm_model_map` | 改一行加按键 |
| `states/<name>/config/rl_params.yaml` | 该 state 自带 | `policy_path`/obs/action 维度/decimation/scale/PD/阈值 | 新 state 自带 |

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
- `libphybot_core`(.a,**含状态机**)· `backend_realrobot.a` / `backend_mujoco.a` · `states`(OBJECT)· 可执行 `main` / `task_dry_run` / `publisher`。(不再有独立 `libfsm`。)

### 8.2 自注册防丢(关键技术点)
自注册状态没有被显式引用,静态库 `.a` 会丢未引用目标文件 → 注册丢失。**对策:states 用 CMake `OBJECT` 库,目标文件直接链进可执行**(不走 `.a` 的丢弃规则);等价替代为 `alwayslink`/`--whole-archive`。

```cmake
add_library(phybot_states OBJECT
    states/zero/zero.cpp states/walk_amp/walk_amp.cpp states/mimic/mimic.cpp)
target_link_libraries(main PRIVATE phybot_core $<TARGET_OBJECTS:phybot_states>)
```

### 8.3 `phybot_add_state()` —— 把细节藏起来
```cmake
phybot_add_state(walk_amp)   # 收 states/walk_amp/*.cpp + 挂 core + 并入可执行
phybot_add_state(mimic)
phybot_add_state(my_new)     # 二次开发: 新目录 + 一个类 + 这一行
```

### 8.4 环境开关
```bash
cmake -S . -B build -DWHICH_ENV=realrobot_mini   # 链 backend_realrobot
cmake -S . -B build -DWHICH_ENV=mujoco_sim_mini  # 链 backend_mujoco
```
同一 state、同一 `policy.pt`,真机/仿真零改动切换(state 只碰黑板,碰硬件的是 backend)。

---

## 9. Task(state)编写规范

### 9.1 每个 state 固定编辑面(4+1 处)
1. `states/<name>/<name>.cpp`:类 + `REGISTER_STATE("<name>", <Class>)`
2. `states/<name>/config/rl_params.yaml`
3. `states/<name>/model/policy.pt`
4. `CMakeLists.txt`:`phybot_add_state(<name>)`
5. `config/bindings.yaml`:按键 + lcm model id

**永不碰**:`core` 状态机(`State`/`StateMachine`)、`InputProtocol`、`server`、主循环、enum、中心注册文件。

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
- **A 行走类(walk_amp)**:继承 `PolicyState`,只重写 `BuildObservation`(+可选 `MapCommand`)。
- **B 参考动作类(mimic)**:继承 `PolicyState`,重写 `enter`(加载参考片段)+ `BuildObservation`(含 phase)+ 可选 `checkChange`(做完回 zero)。
- **C 非 RL 类(zero)**:继承 `phybot::State`,自己写 `run`(PD 到零位),无 policy/obs。

```cpp
// 自注册宏
REGISTER_STATE("walk_amp", WalkAmpState);
```

### 9.4 "改"已有 task
| 改什么 | 改哪 | 重编 |
|---|---|---|
| 换策略权重 | `model/policy.pt` | 否(跑 dry-run)|
| 调 PD/scale/阈值 | `rl_params.yaml` | 否 |
| 改按键 | `bindings.yaml` | 否 |
| 改观测/命令逻辑 | `<name>.cpp` 钩子 | 是(仅该 state)|

---

## 10. 安全策略

- **沿用官方原有急停与电机使能策略,本次重构不修改该路径。**
- 不引入任何软件急停(无看门狗 / seq / SAFE / estop 协议)。`phybot::StateMachine::requestSafe()` 仅作为能力保留,默认不接线。
- 真机安全依赖**硬件急停按钮**(需确认其接在电机供电/使能回路中,而非软件读取)。
- 取舍(显式接受):手柄断连 / publisher 崩 / WiFi 掉 / main 崩 时机器人不自动停,依赖人工监护 + 物理急停。
- 测试物理防护:硬件急停 = 断电 = 双足会瘫倒,测试上吊架/护具;真机测试建议 publisher 跑在机器人本机(本地组播)。
- 待确认项(不阻塞本设计):MotorList 的电机心跳保护(`HEART_BEAT_PROTECT_ON`/`SetHeartbeat`)当前是否使能。

---

## 11. 测试与验收

### 11.1 通用 dry-run(`task_dry_run`)
取代仅针对 walk_amp 的 `walk_amp_dry_run_check`,对**任意 state** 喂一帧数据跑一次 `run`,检查:策略加载成功、obs 维度正确、输出有限、在限幅内。上电前必跑。

### 11.2 对齐保护(最高优先级)
- **walk_amp 数值逐位等价**:重构后用通用 dry-run + `scripts/validation/check_walk_deploy_alignment.py` 对比新旧二进制产出的 obs/action 向量,必须一致后才允许上真机。
- DataPackage 字段名、限幅值、decimation、scale、PD、obs 布局保持不变。

### 11.3 单元/集成
- 状态机:用最小 `DataPackage` + 几个哑 `State` 做纯逻辑单测(切换/checkChange/canExit/注册表),不依赖电机/IMU。
- InputProtocol:encode→decode 往返一致;对旧 App/手柄真实包做解码回归。

### 11.4 上真机硬门槛 Checklist(全绿才上真机;作为实现计划中的显式 review 检查点)
1. ☐ 新旧 **obs/action 逐位一致**(对齐脚本 + dry-run)
2. ☐ `robot.yaml` 关节序/方向/零位 vs 官方逐项核对
3. ☐ PD/力矩限幅/decimation/命令限幅 = 官方
4. ☐ LCM 解码对真实旧 App/手柄包回归通过
5. ☐ 启动打印已注册 state 清单 = 期望清单
6. ☐ `requested_state` 跨线程已加锁/快照(N1)
7. ☐ 切换去抖到位(N2)
8. ☐ 电机心跳保护确认 + 硬件急停接线确认
9. ☐ 先 mujoco 跑通 → 真机 dry-run(不使能下发)→ 吊架/低速上机
10. ☐ 官方版回退路径已验证可用

---

## 12. 分阶段落地(每阶段可独立验证)

1. **阶段 0**:搭 `core` 状态机(`State`/`StateMachine`/`StateRegistry`)+ 纯逻辑单测(最小 DataPackage + 哑 State)。
2. **阶段 1**:`core` 其余(`PolicyState`/`PolicyRunner`/`InputProtocol`)+ mujoco backend;接入 `zero` + `walk_amp`,仿真跑通,dry-run + 对齐校验通过。
3. **阶段 2**:realrobot backend + 主循环集成(保留官方上电/急停),真机 dry-run。
4. **阶段 3**:Publisher SDL2 输入层 + `bindings.yaml`,手柄打通。
5. **阶段 4**:迁移 `mimic`,验证架构通用性;补 `_template`。
6. **阶段 5**:清理、文档、回归。

> 新目录在阶段 1 跑通前没有"半截可用"状态;故按阶段推进、每阶段独立验证,避免最后一次性合龙。

---

## 13. 风险与对策

风险按"会不会伤到真机"分级:🔴 直接致动作错误/发散  🟠 行为异常/崩溃  🟡 功能缺失/工程。

### 13.1 数值/对齐(🔴 最高,直接致摔/抽搐/冲出)
| 风险 | 对策/门槛 |
|---|---|
| 🔴 walk_amp obs/动作拆进 `PolicyState` 时出现细微偏差(720 维顺序/`scale`/`Euler_ZYXToGravityVec` 符号/history 首帧填充) | 同一帧输入,新旧二进制 **obs 与 action 逐位一致**(`check_walk_deploy_alignment.py` + dry-run);不通过禁止上真机 |
| 🔴 关节顺序/方向/零位映射错位(`q/dq` 序、`direction_vector`、`default_dof_pos`=-0.295) | `robot.yaml` 与官方 `MotorList/config/*` 逐项核对;沿用官方上电"读 50 次打印"对照零位 |
| 🔴 PD/力矩限幅/decimation/控制周期 漂移 | 数值沿用官方;dry-run 校验输出在 `torque_limits` 内;周期不动 |
| 🔴 命令限幅/LCM 解包 byte 偏移/校验和错位 | 限幅值逐字保留;`encode→decode` 往返一致 + 用真实旧 App/手柄包回归 |

### 13.2 改动官方逻辑新引入(🔴/🟠)
| 风险 | 对策/门槛 |
|---|---|
| 🔴 **N1**:`requested_state` 由 enum 变 `std::string`,跨线程读写是 UB | LCM 回调只在 mutex 下存原始字节;`poll()` 主线程取快照写 string(§5.5) |
| 🟠 **N2**:切换由 500ms 轮询改每拍即时,失去天然去抖,误触即切 | 按键边沿检测 + 去抖窗口;`canExit` 互锁;危险跳转加条件(§5.5) |
| 🟠 重构误删官方某处 `Enable/Disable/SetCommands`/急停 | 逐行 diff 官方 `run_mini` 上电/电机段,确保 ❶❷ 之外零改动 |

### 13.3 状态机/实时/流程(🟠/🟡)
| 风险 | 对策/门槛 |
|---|---|
| 🟠 自注册被链接器丢弃 → 按键"无反应" | states 用 OBJECT 库(或 alwayslink/--whole-archive);**启动打印已注册 state 清单与期望核对** |
| 🟠 某 `policy.pt` 缺失/损坏 | 加载校验放在 `Enable()` **之前**,失败拒绝上电 |
| 🟠 `enter` 每次复位 history 与官方语义差异 | 对照官方 walk_amp 进出行为;dry-run 覆盖"切出再切入" |
| 🟠 `checkChange` 误引入自动跳转 | walk_amp/zero 默认返回 `""`;仅 mimic 显式用并充分测试 |
| 🟠 热路径堆分配/tensor 每拍分配 → 抖动丢拍 | `enter` 预分配所有 buffer;hot path 无 malloc;测 tick 耗时分布 |
| 🟡 字符串 id 拼错/缺文件仅运行时炸 | 启动全量校验注册表+配置+模型,缺失拒绝进入电机使能 |
| 🟡 `PolicyState` 抽象对 mimic 不合身 | `run` 之外钩子足够灵活;必要时直接继承 `phybot::State` 自写 `run` |
| 🟡 构建/autorun 跑到旧二进制(曾见 build/ 残留 walk_preset_amp) | 构建路径明确;启动打印 git hash/版本 |
| 🟡 无渐进回退、一次性大改 | 分阶段;真机阶段前仿真+dry-run+对齐全绿;**官方版随时可回退跑真机** |
| 🟡 跨主机 WiFi 丢包(无软件兜底) | 真机测试 publisher 跑本机/有线;人工监护 + 硬件急停 |

### 13.4 最重要的安全网
**官方 `deploy/phybot_deploy_c2` 完整保留、随时可编译跑真机。** 新架构在 mujoco + dry-run + 对齐**全绿之前,真机一律用官方版** → 真机部署能力零损失。

---

## 14. 关键决策记录

1. 范围:架构 + walk_amp + 空白模板 + mimic(其余旧 task 后续按需)。
2. 形态:核心编库 + state 自注册插件(字符串 id + 注册表)。
3. Xbox:稳健设备层(SDL2/热插拔)+ 配置驱动按键 + 轴→连续命令 + 接入新架构。
4. 进程:保留两进程(LCM 解耦)。
5. 环境:realrobot_mini + mujoco_sim_mini(放弃 gazebo)。
6. 命名:state machine 家族但语义化(`phybot::State` / `StateMachine` / `StateRegistry` / `enter/run/exit` / `checkChange` / `canExit`),不照抄官方 enum+switch。
7. 状态机**放入 `core`,不抽成跨机器人独立库**;借鉴 Unitree 行为契约 + 字符串注册表替代 enum+switch,但**直接以 `DataPackage` 为黑板**(放弃模板 `Ctx`,换取简单内聚)。
8. 安全:沿用官方急停,不加软件急停。

## 15. 待确认 / 开放项
- [ ] 物理急停接线方式(电机供电/使能回路 vs 软件读取)——决定"按急停必失能"是否成立。
- [ ] MotorList 电机心跳保护是否已使能(main 崩溃时的硬件兜底)。
- [ ] `_template` 是否同时给 PolicyState 版与非 RL 版两个模板。
- [ ] mimic 参考动作文件格式与加载器接口细节。
