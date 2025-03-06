import numpy as np
import matplotlib.pyplot as plt

def grad_f(x):
    """分段函数的梯度计算"""
    if x < 1:
        return 25 * x         # x < 1 时的导数
    elif 1 <= x < 2:
        return x + 24         # 1 ≤ x < 2 时的导数
    else:
        return 25 * x - 24    # x ≥ 2 时的导数

# 超参数设置
beta = 4/9    # 动量系数
eta = 1/9     # 学习率
x_prev = 3.3  # 初始值 x_{k-1}
x_current = 3.3
max_iters = 100
history = [x_current]

# 动量梯度下降迭代
for _ in range(max_iters):
    grad = grad_f(x_current)
    
    # 动量更新公式：x_{k+1} = x_k - η∇f(x_k) + β(x_k - x_{k-1})
    if _ == 0:
        next_x = x_current - eta * grad
    else:
        next_x = x_current - eta * grad + beta * (x_current - x_prev)
    
    # 更新状态
    x_prev, x_current = x_current, next_x
    history.append(next_x)

# 可视化结果
plt.figure(figsize=(10, 6))
plt.plot(history, marker='o', linestyle='--')
plt.axhline(y=1, color='r', linestyle=':', label='Boundary x=1')  # 函数分段边界
plt.axhline(y=2, color='g', linestyle=':', label='Boundary x=2')
plt.title("Momentum GD Trajectory (x₀=3.3)")
plt.xlabel("Iteration")
plt.ylabel("x value")
plt.legend()
plt.grid(True)
# plt.show()
plt.savefig('momentum_gd.png')

# 输出最后5个值观察趋势
print("Last 5 x values:", history[-5:])
