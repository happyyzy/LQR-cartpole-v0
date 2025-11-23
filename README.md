# LQR-cartpole-v0
最近学习强化学习在用cartpole环境，想从控制论出发看一看效果对比（脚本在[这](LQR_Cartpole_v1.py)），看到控制论也用Riccita方程，想起了一些本科学的废料，遂从Riccita方程出发，做个控制论/强化学习/数学/物理杂记在[这里](summary.md),整理的pdf在[这里](LQR.pdf)。

## 1️⃣ 系统变量定义

设：

- $x$：小车位置（向右为正）  
- $\dot{x}$：小车速度  
- $\theta$：杆子相对于竖直向上的角度（向右倾为正）  
- $\dot{\theta}$：杆子角速度  
- $F = u$：施加在小车上的水平推力（控制输入）  

状态向量与控制输入为：

$$
s = \begin{bmatrix} x \\ \dot{x} \\ \theta \\ \dot{\theta} \end{bmatrix}, \quad u = F
$$

---

## 2️⃣ 非线性动力学方程

根据牛顿第二定律和刚体转动方程，CartPole 的非线性动力学为：

$$
\begin{cases}
\ddot{x} = \dfrac{F + m_p l \dot{\theta}^2 \sin\theta - m_p g \sin\theta \cos\theta}{m_c + m_p \sin^2\theta} \\
\ddot{\theta} = \dfrac{g \sin\theta - \cos\theta \dfrac{F + m_p l \dot{\theta}^2 \sin\theta}{m_c + m_p}}{l \left( \dfrac{4}{3} - \dfrac{m_p \cos^2\theta}{m_c + m_p} \right)}
\end{cases}
$$

其中：

- $m_c$：小车质量  
- $m_p$：杆子质量  
- $l$：杆长一半（质心到枢轴距离）  
- $g = 9.8 \, \mathrm{m/s^2}$  

---

## 3️⃣ 线性化（平衡点 $\theta \approx 0$）

在杆子竖直向上附近线性化：

$$
\sin\theta \approx \theta, \quad \cos\theta \approx 1, \quad \text{忽略二阶小量}
$$

线性化后得到：

$$
\ddot{x} \approx \frac{F - m_p g \theta}{m_c}, \quad
\ddot{\theta} \approx \frac{(m_c + m_p) g \theta - F}{l m_c}
$$

---

## 4️⃣ 矩阵形式

定义状态向量 $s = [x, \dot{x}, \theta, \dot{\theta}]^T$：

$$
\dot{s} = 
\begin{bmatrix} \dot{x} \\ \ddot{x} \\ \dot{\theta} \\ \ddot{\theta} \end{bmatrix} 
= A s + B u
$$

线性化后的矩阵为：

$$
A = 
\begin{bmatrix}
0 & 1 & 0 & 0 \\
0 & 0 & -\frac{m_p g}{m_c} & 0 \\
0 & 0 & 0 & 1 \\
0 & 0 & \frac{(m_c + m_p) g}{l m_c} & 0
\end{bmatrix}, \quad
B =
\begin{bmatrix}
0 \\ \frac{1}{m_c} \\ 0 \\ -\frac{1}{l m_c}
\end{bmatrix}
$$

---

## 5️⃣ 代入 Gym 默认参数

| 参数 | 含义 | 默认值 |
|------|------|------|
| $m_c$ | 小车质量 | 1.0 kg |
| $m_p$ | 杆子质量 | 0.1 kg |
| $l$ | 半杆长 | 0.5 m |
| $g$ | 重力加速度 | 9.8 m/s² |

代入后：

$$
A = 
\begin{bmatrix}
0 & 1 & 0 & 0 \\
0 & 0 & -0.98 & 0 \\
0 & 0 & 0 & 1 \\
0 & 0 & 21.56 & 0
\end{bmatrix}, \quad
B = 
\begin{bmatrix}
0 \\ 1.0 \\ 0 \\ -2.0
\end{bmatrix}
$$

---

## 6️⃣ LQR 控制设计

选择二次型性能指标：

$$
J = \int_0^\infty (x^T Q x + u^T R u) \, dt
$$

示例权重矩阵：

$$
Q = \mathrm{diag}([1.0, 1.0, 10.0, 1.0]), \quad R = [[0.001]]
$$

求解连续时间代数黎卡提方程（CARE）得到最优反馈增益：

$$
K = \begin{bmatrix} -31.62 & -51.69 & -269.79 & -64.62 \end{bmatrix}
$$

控制律：

$$
u = -K s
$$

---

## 7️⃣ 闭环系统稳定性

闭环系统矩阵：

$$
A_\mathrm{cl} = A - B K
$$

- 如果 $A_\mathrm{cl}$ 的特征值实部全为负 → 系统稳定  
- Python 示例计算：

```python
import numpy as np

A_cl = A - B @ K
eigvals = np.linalg.eigvals(A_cl)
print("Closed-loop eigenvalues:", eigvals)
```

结果：
>Closed-loop eigenvalues: [-70.90197968+0.j          -2.79727985+0.61481643j
  -2.79727985-0.61481643j  -1.06570493+0.j        ]

系统确实稳定。

