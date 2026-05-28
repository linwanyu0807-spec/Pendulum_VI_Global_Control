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

def run_complete_experiment(cost_mode="min_time"):
    # 1. 求解阶段
    temp_builder = DiagramBuilder()
    temp_plant = temp_builder.AddSystem(PendulumPlant())
    res = 51 
    theta_grid = set([float(x) for x in np.linspace(0, 2 * np.pi, res)])
    thetadot_grid = set([float(x) for x in np.linspace(-10, 10, res)])
    input_grid = set([float(x) for x in np.linspace(-8, 8, 9)])
    
    print(f"正在计算 [{cost_mode}] 策略...")
    
    # 定义代价函数
    def min_time_cost(context):
        x = context.get_continuous_state_vector().CopyToVector()
        u = temp_plant.get_input_port().Eval(context)[0]
        theta_error = np.abs(np.mod(x[0] - np.pi + np.pi, 2 * np.pi) - np.pi)
        # 如果不在目标邻域（误差 < 0.1），则代价为 1.0 (计时)
        if theta_error < 0.1 and np.abs(x[1]) < 0.1:
            return 0.0
        return 1.0

    options = DynamicProgrammingOptions()
    policy, _ = FittedValueIteration(
        Simulator(temp_plant), 
        min_time_cost,
        [theta_grid, thetadot_grid], [input_grid], 0.01, options
    )

    # 2. 仿真系统构建
    builder = DiagramBuilder()
    plant = builder.AddSystem(PendulumPlant())
    ctrl = builder.AddSystem(policy)
    builder.Connect(ctrl.get_output_port(), plant.get_input_port())
    builder.Connect(plant.get_state_output_port(), ctrl.get_input_port())
    
    test_points = [[0.0, 0.0], [1.57, 0.0], [-3.14, 2.0], [-2.0, 0.5]]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    loggers = [LogVectorOutput(plant.get_state_output_port(), builder) for _ in test_points]
    
    diagram = builder.Build()
    
    # 3. 运行仿真并绘图
    plt.figure(figsize=(10, 7))
    print(f"\n--- 实验结果表格 ({cost_mode}) ---")

    for i, x0 in enumerate(test_points):
        simulator = Simulator(diagram)
        context = simulator.get_mutable_context()
        context.SetContinuousState(x0)
        simulator.AdvanceTo(8.0)
        
        log = loggers[i].FindLog(context)
        th = log.data()[0, :]
        thd = log.data()[1, :]
        t = log.sample_times()
        
        # 计算调节时间
        err = np.abs(np.mod(th - np.pi + np.pi, 2 * np.pi) - np.pi)
        idx = np.where(err > 0.1)[0]
        ts = t[idx[-1]] if len(idx) > 0 else 0.0
        print(f"起点 {x0} | 调节时间: {ts:.2f}s")
        
        plt.plot(th, thd, color=colors[i], label=f"Start {x0}", linewidth=2)

    plt.axvline(x=np.pi, color='gray', linestyle='--', alpha=0.5)
    plt.xlabel("Theta (rad)")
    plt.ylabel("Theta_dot (rad/s)")
    plt.title(f"Phase Portrait: Global Stability ({cost_mode})")
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    
    # 注意这里保存文件名的变化
    plt.savefig(f"phase_portrait_{cost_mode}.png")
    print(f"\n图像已保存为: phase_portrait_{cost_mode}.png")
    plt.show()

if __name__ == "__main__":
    run_complete_experiment(cost_mode="min_time")