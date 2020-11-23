# 经济学原理

考试：120分钟，120分值，共40道选择题

## 第一章 引入

绝对优势(absolute advantage)：消耗更少资源

相对优势(comparative ~)：更少机会成本

**转换曲线transformation curve**: 假设一个人只生产两种货物A与B，而且时间是有限的，那么多生产一个A就少生产几个B，并且呈线性关系

<img src="img/Production_Possibilities_Set_in_Robinson_Crusoe_Economy_with_two_commodities.jpg" alt="transCurve" style="zoom: 33%;" />

> （例）练习1题1：设每一小时，Jack生产1kg鱼或者$\frac14$kg酒; Will 生产$\frac15$kg鱼或$\frac15$kg酒 
>
> $$
> \begin{array}{}
> 绝对成本（时间）&Fish&Rum
> \\\hline 
> Jack&1&4\\
> Will&5&5
> \end{array}
> $$
> $$
> \begin{array}{}
> 机会成本&Fish&Rum
> \\\hline 
> Jack&\frac14&4\\
> &\wedge&\vee\\
> Will&1&1
> \end{array}
> $$
>
> 虽然Jack全方位碾压Will, 但是在每个人时间有限的情况下，让Will去做Jack相对不擅长的事（Will机会成本小），这样jack可以省出时间做价值更高的事(Fish)

通过交易可以把饼做大，共同获益

**Joint Curve**: 只有两个人的情况下，折点为两人只做自己机会成本小的事。最后出来的图一定是bend-out的，代表总生产力提高了。

<img src="img/Joint_production_possibilities_in_the_Robinson_Crusoe_Economy.jpg" alt="JointCurve" style="zoom:50%;" />

两种货物、多个人的情况下，先假设所有人都只生产货物A，然后让生产B机会成本低的先耗尽时间。

**怎么判断交易是否合理?** 看交易价格是否落在两人机会成本之间（造不如买）。



## 第二章 消费与要求（消费者角度）

假设有两种货物，$q$为数量，$p$为价格。而且消费者没有议价能力。预算$y=p_1q_1+p_2q_2$

个人最优解inidividual optimal: 花完所有的钱，达到最大效益(Utility)
$$
\begin{array}{}最大化&U(q_1,q_2)
\\s.t.&y=p_1q_1+p_2q_2
\end{array}
$$

#### Utility function

引入效益函数(Utility function)模拟用户的喜好

> 数字只有序数意义（顺序有意义），没有基数意义（数字本身大小无意义）, e.g. 效益10比效益5要好，但不能说效益10是5效益的两倍那么好

![indifferenceCurve](img/19494147.cms)

效益函数画成图像就是等优曲线(indifference curve)。等优曲线总是一条下陷的曲线。

> 心理学假设：一种货物多了就会想要另一种。

两个必须特征：

* Complete: 不同的两个Perference Bundle 总能两两比较，并给出优先级$A\prec B,A\succ B或A\sim B$。图形上每个消费组合(comsumption bundle)均在某条等优曲线上。
* Consistent/Transitive: 逻辑上通顺，如$A\succ B,B\sim C\implies A\succ C$。反应到图形上就是等优曲线不能相交。

还有两个特征不是必须的，甚至可能互斥：

* Monotonicity: 货物的数量越多总是越好。但必须是两种货物都比原来多（soft assumption），只多一种不算。推论：等优曲线总是负斜率的。
* Convexity: 均数*至少*和两个原数一样好。即$C=\lambda A+(1-\lambda)B\succeq A或B,\lambda\in(0,1)$.

> （例）练习2题3：$A=(8,2),B=(2.8)$,则两者的均数为$D=(5,5)$，由convexity得$D\succeq B$。若已知$C=(6,6)$且$B\succ C$，则由transitivity得$D\succ C$。但是monotonicity得$D(5,5)\prec C(6,6)$. 可知在这种情况下，monotonicity和convexity互斥。

#### Optimal Consumption Bundle

最优的消费组合：等优曲线勉强和预算线(Budget Line)相切，称为tagent solution

<img src="img/consumer-choice.png" alt="optimalBundle" style="zoom:50%;" />

如何求这个切点（求此时$q_1,q_2$分别为多少）? 这可以转化为拉格朗日优化问题
$$
\begin{array}{rl}
\begin{split}\mathop{maximize}_{q_1,q_2}\end{split} & U(q_1,q_2)\\
s.t.&y-(p_1q_1+p_2q_2)\ge 0\\
\end{array}
\\转为
\mathcal L=U(q_1,q_2)+\lambda(y-p_1q_1-p_2q_2)\\
\left\{\begin{array}{}
\frac{\partial \mathcal L}{\partial q_1}=\frac{\partial U}{\partial q_1}-\lambda p_1\triangleq 0
\\\frac{\partial \mathcal L}{\partial q_2}=\frac{\partial U}{\partial q_2}-\lambda p_2\triangleq0
\end{array}\right.
\implies\left\{\begin{array}{}
\lambda=\frac{\partial U}{\partial q_1}/p_1
\\\lambda=\frac{\partial U}{\partial q_2}/p_2
\end{array}\right.
$$

> 可得：$\large{\frac{\partial U}{\partial q_1}\over{p_1}}={\frac{\partial U}{\partial q_2}\over p_2}$，即在这个平衡点上，要是多花一块钱在任何一种货物上，所获的效益都是相同的。若此条件不成立，就一定不是最优点（会减少一种货物而去买另一种）

切点处具有两个特征：

1. 落在预算线上$\therefore y=p_1q_1+p_2q_2$
2. 斜率相同$\therefore \frac{\partial U/\partial q_1}{\partial U/\partial q_2}\triangleq MRS_{1,2}=\frac{p_1}{p_2}$ expense rate=utility rate (marginal rate of substitution)

#### 货物的特征Charactise

1. Normal/Inferior

normal: 预算多时消费也多，$\frac{dq_i}{dy}\gt0$

inferior: 预算更多时消费反而少，$\frac{dq_i}{dy}\lt0$

> 比如垃圾食品、公共交通、主食（不吃肉只吃土豆）

2. Ordinary/Giffen

Ordinary: 价格上涨时消费量下降，$\frac{dq_i}{dp_i}\lt0$

Giffen: 价格上涨时消费量反常上升，$\frac{dq_i}{dp_i}\gt 0$

> 为何会出现Giffen商品? 因为代偿作用总是用相对廉价的货物来代替贵的货物。一种商品涨价导致了总购买力降低，而当该商品代偿作用过强时就反而超过原来消费量。比如，土豆涨价吃不起肉了，为了吃饱要买更多比例的土豆→土豆消费量上升。
>
> Giffen商品一定是Inferiro商品

3. Subsitution/Complement

Substitution代替商品: 货物1涨价会消耗更多货物2，$\frac{dq_j}{dp_i}\gt0$

> 两种货物其实无所谓要哪一种，比如葵花油和菜籽油。一种涨价了就去买另一种。

Complement配对商品：货物1涨价使货物2消费量也下降，$\frac{dq_j}{dp_i}\lt0$

> 两种货物一定是配对地使用的，比如当网球涨价，网球拍就会买得少了。

==分析三种属性用一个公式即可：写出$\underbrace{q_i}_{数量}\left(\underbrace{p_i}_{自己价格},\underbrace{p_j}_{另一货物价格},\underbrace{y}_{总预算}\right)$，然后求对三个元素的导数==

#### 价格变化效果的分解 Price Effect Decomposition

货物涨价带来两个作用：

1. 收入效应——购买力减少
2. 代偿效应——用廉价物品代替昂贵物品

<img src="img/decomposition.PNG" alt="decomposition" style="zoom:60%;" />

分解步骤：

1. 先看代偿作用——假设预算增加到能达到原来的效益Ulitity，用现在的预算线平移出去和原效益曲线相切。从$C$点到$\tilde C$。
2. 再看收入作用——从$\tilde C$到$C'$

> 定量计算$\tilde C,C'$的具体位置需要用到效益函数。
>
> （例）练习2题4：$U(q_1,q_2)=(q_1\cdot q_2)^{\frac12}$，则可以联立公式$\left\{\begin{array}{}y=p_1q_1+p_2q_2\\\frac{\frac12q_1^{-\frac12}q_2^\frac12}{\frac12q_1^{\frac12}q_2^{-\frac12}}=\frac{p_1}{p_2}\implies\frac{q_2}{q_1}=\frac{p_1}{p_2}\end{array}\right.$
>
> 得$y=p_1q_1+p_2\cdot\frac{p_1}{p_2}q_1=2p_1q_1\implies q_1=\frac{y}{2p_1}$，同理$q_2=\frac{y}{2p_2}$，代入具体数值可以求$C$和$C'$点
>
> $\tilde C$点是一个辅助点，效益和$C$点一样。$U'(\tilde q_1,\frac{p_1}{p_2}\tilde q_1)=原来U$，即算出现在$\tilde q_1$,同理算出$\tilde q_2$

## 第三章 生产与供应（生产者角度）

## 第四章 完美竞争

## 第五章 市场失灵

## 第六章 宏观经济指标

## 第七章 经济增长

## 第八章 经济波动

