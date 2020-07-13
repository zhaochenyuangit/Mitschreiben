# Robotic motion planning

workspace：工作空间，真实空间

configuration space: 机器可设置的空间

表示机器人的两种方法：

1. Point represent

障碍物: 将工作空间中障碍物投射到c-space中（discrete colision test)

参数：c-space中的$\theta_1,\theta_2,\cdots$

适用于：机械臂

2. padding of obstacles

参数：$x,y,[\theta]$

障碍物：在原障碍物上附加机器人的尺寸(geometry)

适用于：点状机器人





## 一、经典RMP方法

### （一）、BUG算法

假设机器人为一个点，利用零距离传感器（zero range sensor）前往目标。

一定能到达终点，但是只适用于简单的二维地图

#### Bug1

遇到障碍物时记下遭遇位置$q_i^H$：Hit point

绕障碍物一圈确定障碍物上离终点最近的一点$q_i^L$：Leave point

m-line: 连接$q_i^L$与$q_{goal}$

如果m-line与当前障碍物相交说明在被障碍物封在内部，即可确定没有可行的路线

> 若前进路线只是蹭过障碍则没有必要执行算法，直接继续前进即可

$$
L_{Bug_1}\le d(q_{start},q_{goal})+\underbrace{1.5\sum_{i=1}^np_i}_{探索一倍周长+最坏情况下再走一半周长}
$$

<img src="img/BUG1.PNG" style="zoom:60%;" />

#### Bug2

m-line：固定为$q_{start}$到$q_{goal}$

相当于是在同一条m-line上找离终点更近的点，只要再次碰到m-line就离开障碍物。

如果再次遭遇m-line与障碍物相交的同一点则可确定不可行
$$
L_{Bug_2}\le d+0.5\sum_{i=1}^n \underbrace{n_i}_{同一个障碍物\\遭遇n次}p_i
$$


<img src="img/BUG2.PNG" style="zoom:60%;" />

#### Tagent Bug

假设不再是零距离传感器而是有一点探测范围的传感器

这个传感器360度都可探测 $\rho_R(x,\theta)$

<img src="img/tagentBug.PNG" style="zoom: 80%;" />

当传感器刚刚探测到障碍物时，传感器半径与障碍物相切 → Tagent正切

之后相切点分裂为相交段曲线，有两个端点$O_i$

<img src="img/tagentBug2.PNG" style="zoom:60%;" />

**状态一 Motion to go**

先前往终点，

如果前往终点的路径受阻，则会按Heuristic决定前往哪一个$O_i$作为subgoal

Heuristic比如：$h(x)=d(x,O_i)+d(O_i,q_{goal})$

直到无法再减小$h(x)$说明正要远离终点，此刻来到了local minimum，记为$M_i$，由motion-to-go算法转为follow-boundary 算法

**状态二 follow boundary**

沿障碍物表面法线方向$n(x)^\perp$前进

记录$d_{followed}$：目前这个障碍物边界上**曾经记录过**离终点最近的距离

$d_{reach}=\min_{c\in\Lambda}d(q_{goal},c)$：目前探测范围内障碍物上到终点最近距离

当$d_{reach}\lt d_{followed}$时离开障碍物，leave point 记为$L_i$，由follow boundary 算法变回motion to go 算法

### （二）、Roadmap

Grid View：画出图中每个像素可达性，但要任作连续运动需要考虑一条运动路线上所有的点

* shortest-path

算法wave-front planner

<img src="img/wavefront.PNG" style="zoom: 33%;" />

* maximum clearance 离障碍物最大距离

算法brushfire-planner: 得到Voronoi diagram

<img src="img/brushfire.PNG" style="zoom:60%;" />

<img src="img/voronioDiagram.PNG" style="zoom:60%;" />

### （三）、Cell Decomposition

* trapezoidal decomposition

每次遇到障碍物的端点都画竖线。

<img src="img/trap_decomposition.PNG" style="zoom:60%;" />

*   Boustrophedon Decomposition → Canny\`s Method 

<img src="img/cannysMethod.PNG" style="zoom:60%;" />

只在存在岔路的地方画竖线（线段两端都能沿伸）

### （四）、Potential Field

把机器人当作是在gradient vector filed中移动的微粒

障碍：positive charge

目标：negative charge

能量的梯度为机器人受到的力$\dot c(t)=-\nabla U(c(t))$，当梯度=0时导航结束，梯度为0的点称为critical point

一般只考虑Hessian矩阵非奇异的势能方程，这种情况下所有的critical points均孤立。意思是一个点要么是极值点，要么是鞍点，不会出现成片梯度为0的区域。

> Hessian正定（有极小）还是负定（有极大）不重要，因为势能法下机器人总是往势能最低处走，总是能找到一个极小值







## 二、 PRM



## 三、Kalman滤波

确定机器人的位置状态pose=position+orientation

观测手段：1.外部传感器测量当前位置 2.惯性导航Odometry由之前位置预测当前位置

> 合并两个高斯分布：$z_1=\underbrace{r}_{真值}+\underbrace{\mathcal N(0,R_1)}_{传感器噪音},z_2=r+\underbrace{N(0,R_2)}_{R=\sigma^2}$
>
> 方差大的权重小一些，就能得到接近真值r的预测值
>
> 设误差为$e=\sum w_i(\hat r-z_i)^2$，则$\frac\partial{\partial \hat r}=2\sum w_i(\hat r-z_i)\doteq0\implies \hat r=\frac{\sum w_iz_i}{\sum w_i}$，权重$w=\frac1R$
>
> 在只有两个数值的情况下，$\hat r=\frac{\frac1{R_1}z_1+\frac1{R_2}z_2}{\frac{1}{R_1}+\frac1{R_2}}=\frac{R_2}{R_1+R_2}z_1+\frac{R_1}{R_1+R_2}z_2=z_1+\underbrace{\frac{R_1}{R_1+R_2}}_{融合系数K}(z_2-z_1)$
>
> 分析：当$\sigma_1=0$，即第一个值误差为0时，$K=0,\hat r\to z_1$
>
> 当$\sigma_1=\infty$，即第一个值完全无效时，$K=\frac{R_1}{R_1+R_2}=\lim_{R_1\to\infty}\frac11=1,\hat r\to z_2$
>
> 当$\sigma_2=0$，$K=1,\hat r\to z_2 $，完全信任第二个数值

Kalman滤波就是两个高斯分布的合并：

* 假设状态误差为高斯分布
* 假设测量误差也为高斯分布
* 假设新$\leftarrow$旧状态转移矩阵$A$为线性关系
* 假设测量$\to$状态矩阵$H$也为线性关系

**Prediction**
$$
\begin{array}{}
预测新状态均值&x_{k+1}^-=Ax_k+\underbrace{B\vec u}_{外部控制力}
\\预测新状态方差&P_{k+1}^-=AP_kA^{-1}+\underbrace{Q}_{系统误差：A的可靠度}

\end{array}
$$

> 转移矩阵A是否可靠? 如果系统模型和实际出入较大就要适当加大Q

**Update**
$$
\begin{array}{}
更新系数&K=P_k^-H^T(HP_k^-H^T+\underbrace{R}_{测量误差})^{-1}
\\
融合预测状态与测量状态&x_k=x_k^-+K(z_k-Hx_k^-)
\\
更新状态方差&P_k=P_k^--KHP_k^-=(I-KH)P_k^-
\end{array}
$$

> H矩阵是对测量方法的建模，建立测量值与状态之间的关系
>
> 比如：电子称来称重，读数并非直接是重量

例题：

 <img src="img/Kalman.PNG" style="zoom:60%;" />

①Static Mode：水面高度$L=C$为常数

> 即已知水面高度不变，测量到上下浮动的浮标绳长，要怎么确定这个高度L? 

设水面实际高度$L=1$, 系统误差$q=0.0001$，测量误差$r=0.1$。此处$Q，R$均小写为$q，r$是因为它们均为标量。

1. 系统建模，测量建模

$$
x_{k+1}=Ax_k=1\cdot x_k
\\
z_k=Hx_k=1\cdot x_k
$$

2. 设初值$x_0=0$，方差为1000（初值完全错误）。第一次测量值$z_1=0.9$

$$
\left\{
\begin{array}{}
x_1^-=Ax=0
\\
p_1^-=p_0+Q=1000+0.0001
\end{array}
\right.
\\
\left\{
\begin{array}{}
K_1=\frac p{p+r}=1000.001(1000.0001+0.1)^{-1}=0.9999
\\x_1=z_1+K(z_2-z_1)=0+0.9999(0.9-0)=0.8999
\\p_1=(1-KH)p=(1-0.9999)\times1000.0001=0.1
\end{array}
\right.
$$

第一次滤波后，状态方差由1000降为0.1，状态值由0变为0.8999，十分接近第一次测量值。

3. 第二次测量$z_2=0.8$

$$
\left\{
\begin{array}{}
x_2^-=x_1=0.8999
\\p_2^-=p_1+q=0.1+0.0001=0.1001
\end{array}
\right.\\
\left\{
\begin{array}{}
K_2=0.1001(0.1001+\underbrace{0.1}_{此时0.1的\\测量误差就太大})^{-1}=0.5002
\\
x_2=0.8999+0.5002(0.8-0.8999)=0.8499
\\p_2=(1-0.5002)0.1001=0.05
\end{array}
\right.
$$

<img src="img/Kalmanexample.PNG" style="zoom:60%;" />

#### Extended Kalman Filter

当状态转移矩阵A，测量矩阵H不为线性时，利用泰勒展开一次近似线性。

* 状态转移非线性：$x_{k+1}^-=f(\hat x_k,\hat u_k,\underbrace{\hat w_k)}_{噪声}$，

> 不确定度也经过非线性转变：$y=f(x)=f(\underbrace{\bar x+\epsilon}_{均值+高斯噪音}) $
>
>  泰勒展开：$f(x)=f(\bar x+\epsilon)\approx f(\bar x)+f(\bar x)'\epsilon= f(\bar x)+J\epsilon$
>
> y的均值为$E[\vec y]\approx E[f(x)+J\epsilon]=f(\bar x) $
>
> $\therefore \vec y-\bar y=f(x)+J\epsilon-f(x)=J\epsilon$
>
> y的协方差矩阵与x协方差矩阵关系为$C_y=E[(y-\bar y)(y-\bar y)^T]\approx E(J\epsilon\epsilon^TJ^T)=JC_xJ^T$

EKF: $P_{k+1}^-=\underbrace{A}_{\downarrow\\\left[\frac{\partial f}{\partial x}\right]}P_kA^T+\underbrace{W}_{\downarrow\\\left[\frac{\partial f}{\partial w}\right]}QW^T$

例题：
$$
f(\vec x)=\begin{bmatrix}f_1(x)\\f_2\\f_3\end{bmatrix}
=\begin{bmatrix}x\\y\\\theta\end{bmatrix}_k+
\begin{bmatrix}(v+w_v)\cos\theta\Delta t
\\(v+w_v)\sin\theta\Delta t
\\(\omega+w_\omega)\Delta t
\end{bmatrix}
\\A=\frac{\partial f_i}{\partial x_i}=
\begin{bmatrix}1&0&-v\sin\theta\Delta t
\\0&1&v\cos\theta\Delta t
\\0&0&1
\end{bmatrix}
\\W=\frac{\partial f_i}{\partial w_i}=
\begin{bmatrix}\Delta t\cos\theta&0
\\\Delta t\sin\theta&0
\\0&\Delta t
\end{bmatrix}
$$

* 测量值与状态值之间也非线性：$z=h(x,\underbrace{\nu}_{测量误差})$

$$
K_k=P_k^-H_k^T(H_kP_k^-H_k^T+V_kRV_k^T)^{-1}
\\\hat x_k=\hat x_k^-+K_k(z_k-h(\hat x_k^-,0))
\\P_k=(I-K_kH_k)P_k^-
$$

设测量值为机器人到一个地标的距离$z=\sqrt{(x-x_L)^2+(y-y_L)^2}$
$$
H=\frac{\partial z}{\partial x}=
\begin{bmatrix}
\frac{x-x_L}{\sqrt{(x-x_L)^2+(y-y_L)^2}}&\frac{y-y_L}{\sqrt{(x-x_L)^2+(y-y_L)^2}}&0
\end{bmatrix}
\\V=1
$$

>  第三个元素为0，说明角度$\theta$在只有一个地标的情况下无法测量。

#### SLAM

在状态向量中加入地标Landmarks，扩展了状态向量
$$
x_k=\begin{bmatrix}\vec x_R=\begin{pmatrix}x\\y\\\theta\end{pmatrix}
\\\vec x_{L_1}=\begin{pmatrix}x\\y\end{pmatrix}
\\\vec x_{L_2}
\\\vdots
\\\vec x_{L_N}
\end{bmatrix}
$$
协方差矩阵也融入这些信息
$$
P=\begin{pmatrix}P_{RR}&P_{RL_1}&\cdots&P_{RL_N}
\\&P_{L_1L_1}&&\vdots
\\&&\ddots
\\&&&P_{L_NL_N}
\end{pmatrix}
$$
测量矩阵
$$
H=\begin{bmatrix}H_R&0&0&\cdots&H_{L_i}&0&0\cdots\end{bmatrix}
$$
机器人自身的测量值$H_R$总更新，没见到的地标权重为0