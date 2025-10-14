import numpy as np
from scipy.linalg import solve_continuous_are

# 系统矩阵
A = np.array([[0, 1, 0, 0],
              [0, 0, -0.98, 0],
              [0, 0, 0, 1],
              [0, 0, 21.56, 0]])
B = np.array([[0], [1.0], [0], [-2.0]])

# LQR 权重
Q = np.diag([1.0, 1.0, 10.0, 1.0])
R = np.array([[0.001]])

# 解连续时间代数黎卡提方程 (CARE)
P = solve_continuous_are(A, B, Q, R)

# 求最优反馈增益 K
K = np.linalg.inv(R) @ B.T @ P
print("Optimal LQR gain K:")
print(K)


A_cl = A - B @ K
eigvals = np.linalg.eigvals(A_cl)
print("Closed-loop eigenvalues:", eigvals)

import gymnasium as gym
import gymnasium as gym

env = gym.make("CartPole-v1", render_mode="human")  # 先创建环境
obs, info = env.reset(seed=42)
done = False

while not done:
    x = obs.reshape(-1,1)
    u = -K @ x
    action = 1 if u > 0 else 0
    obs, reward, terminated, truncated, info = env.step(action)
    
    if terminated:
        print("回合结束：杆子/小车越界")
    if truncated:
        print("回合结束：到达最大步数")
    
    done = terminated or truncated

env.close()  # 回合结束后关闭窗口