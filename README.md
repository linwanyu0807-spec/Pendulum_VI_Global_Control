# 倒立摆全局最优控制 (Value Iteration)

本项目基于 Drake 工具链，利用价值迭代（Value Iteration）算法实现了倒立摆的全局起摆控制。

## 文件说明
- `pendulum_vi_main.py`: 主仿真程序，支持二次型(Quadratic)与最短时间(Min-time)模式，生成 3D 价值函数地形图。
- `pendulum_vi_benchmarks.py`: 性能基准测试，分析不同起点的调节时间，绘制相平面轨迹。
- `pendulum_vi_failure_analysis.py`: 失败案例分析，演示力矩受限时的物理边界。

## 实验结论
通过对比发现，Min-time 策略在起摆速度上优于 Quadratic 策略，但 Quadratic 策略在目标点附近的稳定性更好，震荡更小。
