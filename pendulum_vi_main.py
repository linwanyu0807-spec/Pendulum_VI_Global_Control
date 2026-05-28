import numpy as np
import matplotlib.pyplot as plt
import sys
from pydrake.examples import PendulumPlant
from pydrake.all import (
    FittedValueIteration, 
    Simulator, 
    DiagramBuilder, 
    StartMeshcat,
    LogVectorOutput,
    DynamicProgrammingOptions
)

# 1. 启动可视化器
meshcat = StartMeshcat()
print(f"Meshcat URL: {meshcat.web_url()}", flush=True)

def run_value_iteration(cost_type="quadratic"):
    builder = DiagramBuilder()
    plant = builder.AddSystem(PendulumPlant())
    
    # 2. 定义网格 (强制 Python 原生 float)
    res = 51 
    theta_grid = set([float(x) for x in np.linspace(0, 2 * np.pi, res)])
    thetadot_grid = set([float(x) for x in np.linspace(-10, 10, res)])
    input_grid = set([float(x) for x in np.linspace(-8, 8, 9)])

    state_grid = [theta_grid, thetadot_grid]
    action_grid = [input_grid]

    # 3. 定义成本函数
    def cost_function(context):
        x = context.get_continuous_state_vector().CopyToVector()
        u = plant.get_input_port().Eval(context)[0]
        # 目标点是倒立位置 (pi, 0)
        theta_error = x[0] - np.pi
        if cost_type == "quadratic":
            return theta_error**2 + 0.1 * x[1]**2 + 0.1 * u**2
        else:
            return 0.0 if (np.abs(theta_error) < 0.1 and np.abs(x[1]) < 0.1) else 1.0

    # 4. 执行价值迭代
    sim_for_vi = Simulator(plant)
    options = DynamicProgrammingOptions()
    
    print(f"正在计算 [{cost_type}] 模式价值迭代...", flush=True)
    
    policy, cost_to_go = FittedValueIteration(
        sim_for_vi, cost_function, state_grid, action_grid, 0.01, options
    )
    
    print("计算完成，正在连接闭环控制系统...", flush=True)

    # 5. 构建闭环系统
    builder.AddSystem(policy)
    # 核心修复：不但要控制输出连输入，还要把 plant 的输出连回 policy 的输入
    builder.Connect(policy.get_output_port(), plant.get_input_port())
    builder.Connect(plant.get_state_output_port(), policy.get_input_port())
    
    logger = LogVectorOutput(plant.get_state_output_port(), builder)
    
    diagram = builder.Build()
    simulator = Simulator(diagram)
    sim_context = simulator.get_mutable_context()
    sim_context.SetContinuousState([0.0, 0.0]) # 从底部静止点开始
    
    print("正在执行8秒仿真测试...", flush=True)
    simulator.AdvanceTo(8.0)
    
    log = logger.FindLog(sim_context)
    return log, cost_to_go, np.linspace(0, 2 * np.pi, res), np.linspace(-10, 10, res)

# 执行实验
try:
    cost_mode = "min_time" 
    log, J_star, th_g, thd_g = run_value_iteration(cost_type=cost_mode)

    file_suffix = f"_{cost_mode}"
    
    # --- 绘图 1: 状态响应曲线 ---
    plt.figure(figsize=(10, 5))
    times = log.sample_times()
    data = log.data()
    plt.plot(times, data[0, :], label='Angle (theta)')
    plt.plot(times, data[1, :], label='Angular Velocity', alpha=0.7)
    plt.axhline(y=np.pi, color='r', linestyle='--', label='Target (pi)')
    plt.title(f"Swing-up Result ({cost_mode})")
    plt.xlabel("Time (s)")
    plt.ylabel("State Value")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"state_response{file_suffix}.png")
    print(f"响应曲线已生成: state_response{file_suffix}.png")

    # --- 绘图 2: 3D 价值地形图 ---
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(th_g, thd_g)
    # 适配矩阵形状
    Z = J_star.flatten().reshape(len(th_g), len(thd_g)).T
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
    ax.set_title(f"Optimal Cost-to-Go Landscape ({cost_mode})")
    ax.set_xlabel("Theta")
    ax.set_ylabel("Theta_dot")
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    plt.savefig(f"cost_3d{file_suffix}.png")
    print(f"3D地形图已生成: cost_3d{file_suffix}.png")

    print("\n实验全部成功！请在左侧下载图片。")

except Exception as e:
    import traceback
    traceback.print_exc()