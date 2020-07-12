# Robotic motion planning



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