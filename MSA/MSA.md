#  微传感器与微执行器

## 一、硅的晶格

密勒指数：用原子在晶胞内截距的倒数之整数比来描述晶面和晶向

由于硅是六面体晶胞，故共只有三大类晶面（及晶向）。

![miller index](img/cvdd-cutter-1.jpg)

* (h k l ) 圆括号表示一个特定的晶面

* {h k l} 花括号表示一组性质相同的晶面，

  > {1 0 0} 包括6个面
  >
  > {1 1 0} 包括12个面
  >
  > {1 1 1} 包括8个面

* [h k l] 方括号表示一个特定方向
* \<h k l\>尖括号表示一组晶面，和花括号类似

>  （例）练习1.3： 画出晶面 (332)
>
> 同除最大公约数为 $\frac36,\frac36,\frac26=\frac12,\frac12,\frac13$
>
> 再取倒数：$x_0=2,y_0=2,z_0=3$

#### 算晶面夹角

两个晶面的夹角可以直接用密勒指数得出，$\cos(\alpha)=\frac{\vec{n_1}\vec{n_2}}{|\vec{n_1}||\vec{n_2}|}$

> （例）练习1.4：求晶面组{111} 和 (100) 的夹角
>
> $\cos(\alpha) = \frac{\pm1\cdot1+\pm1\cdot0+\pm1\cdot0}{\sqrt{(\pm1^2+\pm1^2+\pm1^2)\cdot(1^2+0^2+0^2)}}=\frac{\pm1}{\sqrt3}$
>
> ∴$\alpha =\arccos(\frac{\pm1}{\sqrt3})=54.74^\circ$

## 二、硅的腐蚀

分为异性Anisotropic（蚀刻速度在各个晶面不同）与同性Isotropic（蚀刻速度在各个晶面相同）

其中，异性腐蚀在{111}晶面组的腐蚀速度约为其它晶面的$\frac1{400}$

> 在{111}晶面，每个原子和另外三个原子相连，化学键相较其它面更加牢固。所以腐蚀更慢
>
> <img src="img/111planeslower.PNG" style="zoom:50%;" />

这也导致在异性腐蚀时，不论掩膜自身形状如何，只要时间一长，蚀刻出的区域总是与{111}面横平竖直。

![](img/111echtlingshape.PNG)

因为六面晶体一共只有三种类型的面：$\{1,0,0\},\{1,1,0\},\{1,1,1\}$, 为了利用$\{1,1,1\}$抗腐蚀的特性，往往以$\{1,0,0\}或\{1,1,0\}$ 为表面

<img src="img/anisoaetzen.png" alt="aniso" style="zoom:67%;" />

## 三、压电陶瓷

Direct-piezo 效应：挤压形变产生电势

Reziproker piezo效应：电热势产生形变

 <img src="img/piezoeffect.PNG" alt="piezo" style="zoom:60%;" />

压电材料产生的力，与施加的电压、材料的形变有关。
$$
\frac{\Delta l}{l}=\frac1E\cdot\underbrace{\frac FA}_{压强}
+d_{ij}\cdot\underbrace{\frac Ud}_{电场强度}
$$

> E为材料的弹性系数，由于机械挤压产生形变是所有材料都有的特性
>
> 而$d_{ij}$是由于施加电势产生的形变之系数，$d_{33}$是沿长度方向形变，最重要。

1. 求材料的劲度系数（类似于弹簧的$F=kx$的$k$）

   劲度系数只和材料本身有关，和施加的电势无关，∴$\frac{\Delta l}{l}=\frac 1E\cdot\frac FA$可得$c=\frac F{\Delta l}=\frac Al\cdot E\triangleq\frac A{l\cdot s_{33}}$, 其中$s_{33}=\frac 1E$

2. 无应力状态下，施加电势求**长度方向**形变

   $\frac{\Delta l}{l}=d_{33}\cdot\frac Ud\implies\Delta l =d_{33}\cdot\frac Ud\cdot l$，匀强电场中距离$d$也会写成$t_{El}$

3. 施加电场情况下，要多少应力才能使长度方向上的形变相互抵消

   $0 \mathop{=}\limits^{!} \frac 1E\cdot \frac FA+d_{33}\cdot\frac Ud\implies F = -d_{33}\cdot \frac {U}{t_{El}}\cdot\frac A{s_{33}} $， 外力的符号为负

4. 在压电材料上加上一个初始状态松弛的弹簧，其劲度系数为$c_F$，再施加电场，问压电材料可以沿长度方向形变多少?
   $$
   \frac{\Delta l}{l}=\frac {1}{E}\cdot\frac {F}{A}+d_{33}\cdot \frac Ud
   \\F_{弹簧}=-c_F\cdot\Delta l
   $$
   ∴$\large\Delta l \cdot(\frac1l+\frac{s_{33}\cdot c_F}{A})=d_{33}\cdot\frac Ud\implies\Delta l =\frac{d_{33}\cdot\frac{U}{t_{El}}\cdot l}{1+\frac{c_F\cdot s_{33}\cdot l}{A}}$

## 四、 记忆合金

镍-钛合金（Nickel-Titan）、铜锌铝合金（Kupfer-Zink-Aluminium）

这些合金在低温时是马氏体，柔软且容易变形; 而在高温时变为奥氏体，变得坚硬难变形

![memoryalloy](img/memoryalloy.PNG)

单程效果：低温时加工变形，加热后现原型，而且降温后不回复

双程效果：低温加工变形，加热后现原型，但是降温后可以**大概**回到加工后形状

## 八、光纤

<img src="img/glasfiber.PNG" alt="光纤原理" style="zoom: 67%;" />

折射角和入射角的关系：$n_k\cdot\sin\beta=n_m\cdot\sin\delta$，所以不发生折射的极限情况为当$\delta=90^\circ$时，即$n_k\cdot\sin\beta=n_m\cdot 1\implies \sin(\beta_{极限})=\frac{n_m}{n_k}$

信号发射机从**空气**向光纤发射信号，其角度与光纤切面夹角为$\gamma$，所以$n_0\cdot\sin\gamma=n_k\cdot\sin\alpha=n_k\cdot\cos\beta$ 而空气中折射率$n_0=1$, 信号入射的极限角度为$\sin\gamma_{极限}=n_k\cdot\cos\beta_{极限}=n_k\sqrt{1-\sin^2\beta}=\sqrt{n_k^2-(n_k\cdot\sin\beta)^2}=\sqrt{n_k^2-n_m^2}$。

即信号入射角度$\sin\gamma\lt\underbrace{\sqrt{n_k^2-n_m^2}}_{numerische\ Apetur}$

**计算最大模数**： $k=\frac{Z^2}{2},Z=\pi\cdot d\cdot \frac1\lambda\cdot\sqrt{n_k^2-n_m^2}$

$Z\gt2.4$称为多模光纤，$Z\le2.4$称为单模光纤

**计算不同模态之间最大时间差：**

最快是直线传播，最慢是沿极限入射角传播

<img src="img/direct&amp;grenz.PNG" alt="传播" style="zoom:67%;" />

直线传播所用时间：$\tau_{直线}=\frac{l}{v}=\frac{l}{\frac{c}{n_k}}=\frac{l\cdot n_k}{c}$ _单位：秒s_

极限入射角所用时间：$\tau_\max=\frac{l}{v\cdot\sin(\beta_{极限})}=\frac{l}{\frac{c}{n_k}\cdot\frac{n_m}{n_k}}=\frac{l\cdot n_k^2}{c\cdot n_m} $

两者的时间差为：$\Delta\tau =\tau_\max-\tau_{直线}=\frac lc\cdot\frac{n_k}{n_m}(n_k-n_m) $

**计算最大带宽：** $B=\frac{1}{2\cdot\Delta\tau}=\frac{1}{2\cdot(\tau_\max-\tau_{直线})}$ _单位：$\frac 1s=Hz$_

