import numpy as np
import matplotlib.pyplot as plt
from pydrake.examples import PendulumPlant
from pydrake.all import (
    FittedValueIteration, 
    Simulator, 
    DiagramBuilder, 
    LogVectorOutput,
    DynamicProgrammingOptions
)

def run_failure_experiment(torque_limit=3.0):
    builder = DiagramBuilder()
    plant = builder.AddSystem(PendulumPlant())
    
    # 1. 设置较低的力矩网格，制造“动力不足”的失败场景
    res = 51 
    theta_grid = set([float(x) for x in np.linspace(0, 2 * np.pi, res)])
    thetadot_grid = set([float(x) for x in np.linspace(-10, 10, res)])
    # 将限幅设为 torque_limit (例如 3.0)
    input_grid = set([float(x) for x in np.linspace(-torque_limit, torque_limit, 9)])

    state_grid = [theta_grid, thetadot_grid]
    action_grid = [input_grid]

    # 2. 成本函数使用标准的二次型成本
    def cost_function(context):
        x = context.get_continuous_state_vector().CopyToVector()
        u = plant.get_input_port().Eval(context)[0]
        theta_error = x[0] - np.pi
        return theta_error**2 + 0.1 * x[1]**2 + 0.1 * u**2

    # 3. 执行价值迭代
    sim_for_vi = Simulator(plant)
    options = DynamicProgrammingOptions()
    
    print(f"正在计算失败案例 [u_max = {torque_limit}]...", flush=True)
    policy, _ = FittedValueIteration(
        sim_for_vi, cost_function, state_grid, action_grid, 0.01, options
    )
    
    # 4. 闭环仿真
    builder.AddSystem(policy)
    builder.Connect(policy.get_output_port(), plant.get_input_port())
    builder.Connect(plant.get_state_output_port(), policy.get_input_port())
    
    logger = LogVectorOutput(plant.get_state_output_port(), builder)
    diagram = builder.Build()
    simulator = Simulator(diagram)
    sim_context = simulator.get_mutable_context()
    sim_context.SetContinuousState([0.0, 0.0]) # 从底部开始
    
    simulator.AdvanceTo(8.0)
    return logger.FindLog(sim_context), torque_limit

# 执行并绘图
try:
    log, u_limit = run_failure_experiment(torque_limit=3.0) # 设置为3.0通常会导致失败

    # --- 绘图: 失败案例的状态响应 ---
    plt.figure(figsize=(10, 5))
    times = log.sample_times()
    data = log.data()
    plt.plot(times, data[0, :], label='Angle (theta)', color='blue')
    plt.plot(times, data[1, :], label='Angular Velocity', color='orange', alpha=0.6)
    plt.axhline(y=np.pi, color='r', linestyle='--', label='Target (pi)')
    
    plt.title(f"Failure Case: Swing-up with Insufficient Torque (u_max={u_limit})")
    plt.xlabel("Time (s)")
    plt.ylabel("State Value")
    plt.legend()
    plt.grid(True)
    
    # 命名为建议的文件名
    plt.savefig("failure_low_torque_response.png")
    print("失败案例响应图已生成: failure_low_torque_response.png")
    plt.show()

except Exception as e:
    print(f"实验失败: {e}")
