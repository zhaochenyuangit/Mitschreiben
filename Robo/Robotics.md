## Forward Kinematics

描述空间中位置变化，Forward 由参数求位置

####　一、机器人运动两种方式

1. 平移 translation $\bold {\ ^1p}=\ ^0\bold p+\ ^1\bold t$

2. 旋转 Rotation $\bold {\ ^1p}=\ ^1_0\bold R\ ^0\bold p$

旋转矩阵$\bold R$为正交阵，有三个约束条件：1. 每一行为单位向量；2. 任意两行点积为0；3. 行列式值为1。这样$\bold R$ 只改变向量的方向，不改变长度。$||^1\bold p||=||^0\bold p||$三个条件实际上就是标准正交阵的定义。

#### 二、旋转矩阵推导方法

#### 三、 DH变换

==空间变换原来需要6个参数，降到4个==

步骤1：一个joint一个frame，为每个frame选取z，x坐标轴方向。
$$
\begin{array}{}
z：&1.若是旋转轴，右手定则确定z轴方向。
\\&2. 滑动轴，z沿着滑动方向
\\
x:&x_{i-1}垂直于z_{i-1},z_i，因为DH定义x方向为两条z轴间最短距离
\end{array}
$$
步骤2：定4个参数
$$
\begin{array}{}
a_{i-1}&沿x_{i-1}平移
\\\alpha_{i-1}&绕x_{i-1}旋转
\\\hline 
d_i&沿z_i平移
\\\theta_i&绕z_i旋转，\textcolor{red}{顺时针为正方向，与通常相反}
\end{array}
\\
\ ^{i-1}_i\bold T=\left(\begin{array}{ccc|c}
\cos\theta_i&-\sin\theta_i&\underbrace0_{\textcolor{blue}{不可以绕y轴旋转}}&\alpha_{i-1}
\\\sin\theta_i\cos\alpha_{i-1}&\cos\theta_i\cos\alpha_{i-1}&-\sin\alpha_{i-1}&-\sin\alpha_{i-1}\cdot d_i
\\\sin\theta_i\sin\alpha_{i-1}&\cos\theta_i\sin\alpha_{i-1}&\cos\alpha_{i-1}&\cos\alpha_{i-1}\cdot d_i
\\\hline 
0&0&0&1
\end{array}\right)\triangleq
\tiny{\left(\begin{array}{ccc|c}
&&&
\\&R&&t
\\&&&
\\\hline 
0&0&0&1
\end{array}\right)}
$$
$\ ^x\bold p=\ ^x_y\bold T\cdot\ ^y\bold p$ 指y在x中的坐标。将多个$\bold T$依次迭代相乘，得到机械臂末端在原点坐标系中的坐标。

<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/d/d8/DHParameter.png/779px-DHParameter.png" alt="DH" style="zoom: 67%;" />

##### 特殊情况

一般情况下，空间中两条z轴既不平行也不相交

1. 当多条z轴平行：有无数条$x_i$可选，最好让$x_i$和$x_{i-1}$共线，这样$d_i=0$

2. 当两条z轴相交：$x_{i-1}$必须同时垂直于$z_{i-1}$和$z_i$，因此$\bold x_{i-1}=\Large{\frac{\bold z_{i-1}\times\bold z_i}{||\bold z_{i-1}\times\bold z_i||}}$

##### 平面机器人

<img src="E:\git_zhao\Mitschreiben\Robo\img\3R planar robot.PNG" alt="3R planar robot" style="zoom:50%;" />

平面机器人可以一眼看出末端坐标，不需要写标准DH变换矩阵
$$
\ ^0_e\bold X=\begin{pmatrix}l_1\cos\theta_1\\l_1\sin\theta_1\\\theta_1\end{pmatrix}
+\begin{pmatrix}l_2\cos(\theta_1+\theta_2)
\\l_2\sin(\theta_1+\theta_2)
\\\theta_2
\end{pmatrix}
+\begin{pmatrix}l_3\cos(\theta_1+\theta_2+\theta_3)
\\l_3\sin(\theta_1+\theta_2+\theta_3)
\\\theta_3
\end{pmatrix}
\\=\begin{pmatrix}
l_3\cos(\theta_1+\theta_2+\theta_3)+l_2\cos(\theta_1+\theta_2)+l_1\cos\theta_1
\\l_3\sin(\theta_1+\theta_2+\theta_3)+l_2\sin(\theta_1+\theta_2)+l_1\sin\theta_1
\\\theta_3+\theta_2+\theta_1
\end{pmatrix}
$$
三个坐标分别为$x,y,\theta$，对应平面机器人三个自由度：在X-Y平面内平移，以及末端360°旋转

#### 四、Jacobian 矩阵

在某个位置，joint变换导致坐标位置怎样变换

##### 1. 定义

$$
J(\bold q)=
\large
\frac{\partial \overrightarrow p(\bold \theta)}{\partial \overrightarrow \theta}=\left[\frac{\partial f_i}{\partial x_j}\right]=
\begin{bmatrix}\partial f_1\over\partial\theta_1&f_1\over\partial\theta_2&\cdots&\cdots&f_1\over\partial\theta_6
\\f_2\over\partial\theta_1&
\\f_3\over\partial\theta_1&
\\\vdots
\\f_6\over\partial\theta_1&\cdots&\cdots&\cdots&f_6\over\partial\theta_6
\end{bmatrix}
$$

==求出的是在原点坐标系下的Jacobian矩阵，记作$\ ^0\bold J$==

##### 2. 速度推导

另外，Jacobian矩阵也可以由速度得出
$$
\begin{pmatrix}\bold v\\\bold \omega\end{pmatrix}=\bold J\cdot\dot \theta=\begin{pmatrix}\bold J_v\\\hline \bold J_{\omega}\end{pmatrix}\dot\theta
$$
假设有6个joint：
$$
\ ^0_e\bold v=\begin{pmatrix}k_1\dot\theta_1+k_2\dot\theta_2+\cdots+k_6\dot\theta_6
\\\vdots
\\\vdots
\end{pmatrix}=\bold J_v\dot\theta
\\
\ ^0_e\bold \omega=\begin{pmatrix}r_1\dot\theta_1+r_2\dot\theta_2+\cdots+r_6\dot\theta_6
\\\vdots
\\\vdots
\end{pmatrix}=\bold J_{\omega}\dot\theta
$$
速度递推公式：(angular&linear)

$\ ^{i+1}\omega_{i+1}=\ ^{i+1}\hat Z_{i+1}\cdot \dot\theta_{i+1}+\ ^{i+1}_i\bold R\cdot\ ^i\omega_i$

>  $\ ^i\hat Z_n$：$Z_n$在坐标系$\{1\}$中的方向

$\ ^{i+1}v_{i+1}=\ ^{i+1}_i\bold R(\ ^iv_i+\ ^i\omega_i\times\ ^ip_{i+1})$

==求出的是末端的Jacobian矩阵，记作$\ ^n\bold J$==

 ##### 3. 坐标系间转换

平面坐标：
$$
\ ^0\bold J(\theta)=\ ^0_3\bold R\cdot\bold {\ ^3J}(\theta)
\iff\ ^3\bold J(\theta)=\underbrace{\ ^0_3\bold R^{T}}_{\equiv\bold R^{-1}}\cdot\bold {\ ^0J}(\theta)
$$
空间坐标：
$$
\begin{pmatrix}\ ^Av\\\ ^A\omega\end{pmatrix}
=\begin{pmatrix}\ ^A_B\bold R&0\\0&\ ^A_B\bold R\end{pmatrix}\begin{pmatrix}\ ^Bv\\\ ^B\omega\end{pmatrix}
\\\therefore\ ^A\bold J(\theta)=\begin{pmatrix}\ ^A_B\bold R&0\\0&\ ^A_B\bold R\end{pmatrix}\ ^B\bold J(\theta)
$$
只有当$det(J)\neq0$时才可以转换。==确定奇点时，计算最后一个frame中的$|\ ^eJ(\theta)|$，算式最简便。==

#### 五、奇点 Singularity

计算det(J)=0

> 这里有一件很奇怪的事情：由于只有方阵可以求行列式，而Jacobian矩阵又$\in\mathbb R^{n\times m}$，就导致n必须等于m，既joints（自由度）个数必须等于坐标个数。

#### 六、isotropic points

