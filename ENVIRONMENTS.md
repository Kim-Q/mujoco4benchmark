# MuJoCo Environment Types

本文档梳理了 `mujoco4benchmark` 仓库中 MuJoCo 所包含的环境（模型）类型，并标注了哪些适合用于 **多智能体（Multi-Agent）** 场景。

> **说明**：MuJoCo 本身是一个物理引擎，不提供 Gym/Gymnasium 风格的环境类。这里的"环境"指
> `model/` 目录下的 MJCF/XML 模型文件，可作为构建 RL 环境的底层场景。

---

## 图例

| 标记 | 含义 |
|------|------|
| ✅ **多智能体适用** | 包含多个独立可控智能体，天然适合 Multi-Agent RL |
| ⚠️ **可扩展** | 当前为单智能体，但可通过复制（`<replicate>`）扩展为多智能体 |
| ❌ **单智能体** | 仅包含单个可控主体，不直接适用于多智能体 |
| 🔷 **被动仿真** | 无驱动器，纯物理演示，需额外设计才能用于 RL |

---

## 1. 人形机器人（Humanoid Locomotion）

| 文件 | 模型名称 | DOF | 驱动器数 | 多智能体 | 说明 |
|------|----------|-----|---------|---------|------|
| `humanoid/humanoid.xml` | Humanoid | 27 | 21 | ❌ 单智能体 | 标准双足行走基准（DeepMind Control Suite 同款），单个人形机器人 |
| `humanoid/22_humanoids.xml` | 22 Humanoids | 27×22=594 | 21×22=462 | ✅ **多智能体适用** | 22 个完全独立的人形智能体共存于同一场景，每个均有完整的驱动器；适合合作/竞争任务 |
| `humanoid/100_humanoids.xml` | 100 Humanoids | 27×100=2700 | 21×100=2100 | ✅ **多智能体适用** | 100 个人形机器人；常用于大规模并行仿真与性能基准测试 |
| `humanoid/humanoid100.xml` | Humanoid + 100 Objects | 627 | 21 | ❌ 单智能体 | 单个人形机器人 + 100 个被动自由刚体（胶囊/椭球/箱/柱/球），智能体仅 1 个 |

### 关键特点（22/100 Humanoids）
- 使用 `<replicate>` + `<attach>` 元素将同一基础模型实例化为多个独立副本
- 每个副本拥有独立的关节、执行器和传感器命名空间（通过 `prefix` 区分）
- 可直接在 MJX（JAX 后端）上并行化，适合大规模 MARL 训练

---

## 2. 睡眠优化（Sleep / Optimization）

| 文件 | 模型名称 | 多智能体 | 说明 |
|------|----------|---------|------|
| `sleep/humanoid.xml` | Sleeping Humanoid | ❌ 单智能体 | 开启 `sleep` 标志的单人形机器人，用于演示睡眠功能对仿真效率的提升 |
| `sleep/100_humanoids.xml` | 100 Humanoids (sleep) | ✅ **多智能体适用** | 与 `humanoid/100_humanoids.xml` 相同结构，但启用了 sleep 优化；适合超大规模 MARL 场景中的效率基准 |
| `sleep/dominos.xml` | Sleeping Dominos | 🔷 被动仿真 | 多米诺骨牌链，启用 sleep 后的连锁碰撞演示，无驱动器 |

---

## 3. 操控与手臂（Manipulation）

| 文件 | 模型名称 | 多智能体 | 说明 |
|------|----------|---------|------|
| `tendon_arm/arm26.xml` | 2-Link 6-Muscle Arm | ⚠️ 可扩展 | 2 自由度、6 肌腱驱动的手臂；单智能体，可通过复制构建多臂场景 |
| `flex/gripper.xml` | Soft-Body Gripper | ⚠️ 可扩展 | 带柔性末端的抓手，结合可变形材料仿真；单智能体 |
| `flex/gripper_trilinear.xml` | Trilinear Gripper | ⚠️ 可扩展 | 三线性插值柔性末端的抓手变体；单智能体 |
| `plugin/actuator/pid.xml` | PID Actuator Demo | ❌ 单智能体 | PID 控制器插件演示 |

---

## 4. 车辆/导航（Vehicle / Navigation）

| 文件 | 模型名称 | 多智能体 | 说明 |
|------|----------|---------|------|
| `car/car.xml` | Car | ⚠️ 可扩展 | 4 轮差速驱动小车，单智能体；可通过 `<replicate>` 扩展为多车场景（竞速/编队） |

---

## 5. 复制/批量刚体（Replicate Models）

这组模型展示 `<replicate>` 元素的用法，将同一刚体批量实例化。

### 5a. 带自由度的运动体（可用于多智能体）

| 文件 | 模型名称 | 实例数量 | 多智能体 | 说明 |
|------|----------|---------|---------|------|
| `replicate/particle_free.xml` | Free Particles 3D | 27（3×3×3）| ✅ **多智能体适用** | 每个粒子有完整 6-DOF 自由关节，无驱动器；适合作为简单多体运动研究的起点 |
| `replicate/particle_free2d.xml` | Free Particles 2D | 225（15×15）| ✅ **多智能体适用** | 2D 平面内的大量粒子，每个有 3-DOF（平移×2 + 旋转×1）；适合 2D 多体场景 |
| `replicate/particle.xml` | Particles 3D | 27（3×3×3）| ✅ **多智能体适用** | 3D 粒子，自由关节；可作为简单多智能体导航基准 |
| `replicate/bunnies.xml` | Bunnies | 125（5×5×5）| ✅ **多智能体适用** | 125 个自由刚体兔子网格；可用于大规模物体堆叠/散乱场景研究 |
| `replicate/stonehenge.xml` | Stonehenge | 若干 | ✅ **多智能体适用** | 多个运动体构成的石阵结构 |
| `replicate/leaves.xml` | Leaves | 若干 | ✅ **多智能体适用** | 多个叶片运动体 |

### 5b. 静态几何/约束演示（不适合多智能体）

| 文件 | 多智能体 | 说明 |
|------|---------|------|
| `replicate/cylinder.xml` | 🔷 被动仿真 | 圆柱体静态几何阵列 |
| `replicate/bowl.xml` | 🔷 被动仿真 | 碗形静态几何 |
| `replicate/helix.xml` | 🔷 被动仿真 | 螺旋形静态几何 |
| `replicate/container.xml` | 🔷 被动仿真 | 容器静态几何 |
| `replicate/newton_cradle.xml` | 🔷 被动仿真 | 牛顿摆，腱约束演示 |
| `replicate/tendon.xml` | 🔷 被动仿真 | 腱约束阵列演示 |
| `replicate/scene.xml` | 🔷 被动仿真 | 综合场景 |

---

## 6. 柔性体（Flex / Deformable Bodies）

MuJoCo 的 Flex 系统支持软体、布料、可变形网格等。这类模型通常用于物理特性研究，**无独立可控智能体**。

| 文件 | 描述 | 多智能体 |
|------|------|---------|
| `flex/gripper.xml` | 柔性末端抓手 | ⚠️ 可扩展（抓手本身有驱动） |
| `flex/gripper_trilinear.xml` | 三线性插值抓手 | ⚠️ 可扩展 |
| `flex/cloth_sdf.xml` | 布料 + SDF 碰撞 | 🔷 被动仿真 |
| `flex/flag.xml` | 旗帜布料 | 🔷 被动仿真 |
| `flex/trampoline.xml` | 蹦床 | 🔷 被动仿真 |
| `flex/basket.xml` | 篮筐网 | 🔷 被动仿真 |
| `flex/bunny.xml` / `flex/bunny_quadratic.xml` | 柔性兔子网格 | 🔷 被动仿真 |
| `flex/jelly.xml` | 果冻软体 | 🔷 被动仿真 |
| `flex/mannequin.xml` | 人体模型柔性体 | 🔷 被动仿真 |
| `flex/softbox.xml` | 软箱 | 🔷 被动仿真 |
| `flex/pinch.xml` / `flex/press.xml` | 软体挤压演示 | 🔷 被动仿真 |
| 其余 flex 文件 | 各类软体/弹性演示 | 🔷 被动仿真 |

---

## 7. 插件（Plugins）

### 7a. 弹性插件（Elasticity）

| 文件 | 描述 | 多智能体 |
|------|------|---------|
| `plugin/elasticity/cable.xml` | 弹性绳缆 | 🔷 被动仿真 |
| `plugin/elasticity/coil.xml` | 弹簧圈 | 🔷 被动仿真 |
| `plugin/elasticity/belt.xml` | 弹性皮带 | 🔷 被动仿真 |
| `plugin/elasticity/scene.xml` | 弹性综合场景 | 🔷 被动仿真 |

### 7b. 有符号距离场（SDF）

| 文件 | 描述 | 多智能体 |
|------|------|---------|
| `plugin/sdf/nutbolt.xml` | 螺母螺栓 SDF | 🔷 被动仿真 |
| `plugin/sdf/gear.xml` | 齿轮 SDF | 🔷 被动仿真 |
| `plugin/sdf/cow.xml` / `plugin/sdf/mug.xml` | 网格 SDF | 🔷 被动仿真 |
| 其余 sdf 文件 | SDF 演示 | 🔷 被动仿真 |

### 7c. 其他插件

| 文件 | 描述 | 多智能体 |
|------|------|---------|
| `plugin/actuator/pid.xml` | PID 驱动器插件 | ❌ 单智能体 |
| `plugin/sensor/touch_grid.xml` | 触觉网格传感器 | ❌ 单智能体 |

---

## 8. 专题演示（Specialty Demos）

| 文件 | 描述 | 多智能体 |
|------|------|---------|
| `adhesion/active_adhesion.xml` | 主动粘附驱动器演示 | ❌ 单智能体 |
| `balloons/balloons.xml` | 气球物理 | 🔷 被动仿真 |
| `cards/cards.xml` | 纸牌动力学 | 🔷 被动仿真 |
| `cube/cube_3x3x3.xml` | 三阶魔方 | ❌ 单智能体（整体操控） |
| `hammock/hammock.xml` | 吊床（布料/绳索） | 🔷 被动仿真 |
| `mug/mug.xml` | 马克杯模型 | 🔷 被动仿真 |
| `slider_crank/slider_crank.xml` | 曲柄滑块机构 | ❌ 单智能体 |
| `tactile/tactile.xml` | 触觉传感器演示 | ❌ 单智能体 |

---

## 汇总：多智能体适用性一览

### ✅ 直接适合 Multi-Agent（含多个独立可控智能体）

| 文件 | 智能体数量 | 特点 |
|------|-----------|------|
| `humanoid/22_humanoids.xml` | **22** | 完整人形机器人（27 DOF × 22），适合竞争/合作任务 |
| `humanoid/100_humanoids.xml` | **100** | 大规模人形智能体，性能基准首选 |
| `sleep/100_humanoids.xml` | **100** | 同上 + sleep 优化，适合超大规模仿真效率研究 |
| `replicate/particle_free.xml` | **27** | 简单自由粒子（无驱动器），低维度多体场景 |
| `replicate/particle_free2d.xml` | **225** | 2D 大量粒子，适合平面多智能体研究 |
| `replicate/particle.xml` | **27** | 3D 自由粒子 |
| `replicate/bunnies.xml` | **125** | 网格刚体，物体堆叠/散乱场景 |
| `replicate/stonehenge.xml` | 若干 | 多运动体 |
| `replicate/leaves.xml` | 若干 | 多叶片运动体 |

### ⚠️ 可扩展为 Multi-Agent（当前单智能体，需改造）

| 文件 | 扩展方式 |
|------|---------|
| `humanoid/humanoid.xml` | 用 `<replicate>` 复制，参考 `22_humanoids.xml` |
| `car/car.xml` | 用 `<replicate>` 复制，可做多车竞速 |
| `tendon_arm/arm26.xml` | 复制多个手臂，构建多臂协作场景 |
| `flex/gripper.xml` | 复制多个抓手，构建多机器人操控 |

---

## 关于 MuJoCo 与 Multi-Agent RL 的使用建议

1. **MJX（JAX 后端）** 天然支持 `vmap` 并行化，推荐将 `22_humanoids.xml` 或 `100_humanoids.xml` 结合 MJX 使用，可在单个 GPU 上批量模拟数千个独立环境实例。

2. **`<replicate>` 元素** 是在 MJCF 中构建多智能体场景的核心工具，可将任意单智能体模型扩展为多智能体版本。

3. **外部环境包**（如 [dm_control](https://github.com/google-deepmind/dm_control)、[MuJoCo-MPC](https://github.com/google-deepmind/mujoco_mpc)）基于 MuJoCo 提供了更完整的 Gym 兼容多智能体接口，可与本仓库中的模型配合使用。

4. **注意**：`replicate/` 下的大多数模型（粒子、bunnies 等）**没有驱动器**，若要用于强化学习需要额外添加执行器（actuators）。`humanoid/` 系列模型已包含完整驱动器，是 Multi-Agent RL 基准的首选。
