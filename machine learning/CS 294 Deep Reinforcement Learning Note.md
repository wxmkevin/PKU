# CS 294: Deep Reinforcement Learning Note

# **Berkeley深度强化学习**

[TOC]

# 第一章：简介+模仿学习

本专题记录了在自学伯克利的深度学习课程时的笔记，主要目的是供个人学习用和同道交流，因此有一些个人的观点和感想。文章主要是对CS 294课程视频的翻译与整理，附加一些相关知识。课程地址：[http://rll.berkeley.edu/deeprlcoursesp17/](https://link.zhihu.com/?target=http%3A//rll.berkeley.edu/deeprlcoursesp17/)，[课程视频地址](https://link.zhihu.com/?target=https%3A//www.youtube.com/watch%3Fv%3D8jQIKgTzQd4%26index%3D1%26list%3DPLkFD6_40KJIwTmSbCv9OVJB3YaO4sFwkX) 。（这是我第一次写文章，文章有什么问题请指教。）

预备要求：学过机器学习，深度学习，知道一点优化，知道一点线性代数。会用python。

## **课程笔记导读：**

1. **简介+模仿学习（Imitation Learning）**
2. **有模型的优化控制（Optimal Control）**
3. **基于模型的强化学习（model-based reinforcement learning）**
4. **对优化控制的模仿学习**
5. **MDP，值迭代，策略迭代（model-free RL）**
6. **策略梯度（policy gradient）**
7. **Q-learning**
8. **DQN**
9. **Advanced Policy Gradient - Variance Reduction（GAE）**
10. **Advanced Policy Gradient - pathwise derivative（SVG and DPG）**
11. **Advanced Policy Gradient：TRPO**

**注：2-4节为model-based RL，5-10节为model-free RL，两者可以分开看。**

## **一、深度强化学习简介**

## **1.深度强化学习名词简介**

深度指的是**深度学习**（deep learning，DL），通过多层的神经网络而有十分强大的表示能力（representation learning），能接受学习高维的复杂的输入信息，能表示高维复杂函数，在学习之后深度网络能把原始的高维信息转化为有利于学习目标的高层特征，例如cifar10的图像分类任务。

**强化学习**（RL）的学习任务是机器（agent）学习策略（policy）：即对于外界环境给定的观测（observation）或者状态（state）给出理想的动作（action）（对这种“理想的”界定往往通过奖励函数（reward）），在通过学习后可进行复杂的决策过程。而且强化学习是**序列性的**（sequential），即前一个动作会通过影响后一个状态来影响后一动作的决策，因此在做一步决策时，不光要考虑当前的奖励最大，而且要考虑之后的总奖励累积。

为简化分析，对这样一个过程，常做**马尔科夫（Markov）假设**，即当前状态仅与前一状态和动作有关，而与更早的状态无关。（这种序列性是强化学习与其他机器学习最大的不同）。例如AlphaGo学习下围棋，每一步不但要考虑这步的收益最大，而且要考虑如何获得后续的收益。

## **2.深度网络在强化学习中的作用：可用来表征以下三个映射**

1. **策略函数**（给出观测，经过神经网络，得到理想的动作）
2. **值函数**（Value functions）（用来衡量状态的好坏，或者状态—动作对的好坏）（可用来选择最优策略）
3. **动态模型**（用来预测下一状态和奖励）

## **3.深度强化学习与人脑的相像之处：**

1. 两者都能接受不同的复杂的感知信息
2. 对于不同的感知信息的加工和反应任务都使用同一套学习算法（人类的舌头经过学习之后能“看见”物体）
3. 人类在做一些任务时会假定一个奖励函数并以此来检验行为或作为目标（大脑中的基底核似乎与奖励机制有关）
4. 而且无模型的强化学习过程和实验中的动物适应过程有一定一致性（可能会用一些探索——利用的策略之类的）等。

## **4.强化学习面临的挑战：**

1. 人可以在**短时间，少样本**的条件下学习得很好，而深度强化学习需要大量的样本和时间。
2. 人可以利用之前的相关知识和经验，而**迁移学习**（Transfer learning）在强化学习中还是在研究中的问题。
3. 时常无法确定一个明确的有利于学习的**奖励函数**。（有时可以用逆强化学习（Inverse RL）：推测奖励函数或目标 或模仿学习（imitation learning））
4. 人能预测之后会发生什么以此来帮助决策，但是在强化学习中决策时还无法很好利用这种对未来走向的预测（Prediction）。

## **5.强化学习对其他机器学习任务的作用：**

1. 在图像分类任务中强化学习可以决策图像哪块区域值得注意，然后对该区域更精细地加工，来提升分类效果。
2. 在机器翻译中，可用强化学习方法来进行序列性的预测，产生目标语言词语。

## **强化学习术语、记号：**

强化学习的结果是机器接受观测或状态，经过一定策略来产生行为的，用数学式表示：

![o_t](https://www.zhihu.com/equation?tex=o_t) 表示t时刻的观测， ![x_t](https://www.zhihu.com/equation?tex=x_t) 表示t时刻的状态， ![u_t](https://www.zhihu.com/equation?tex=u_t) 表示t时刻机器所采取的动作， ![\pi_{\theta}(u_t|o_t)](https://www.zhihu.com/equation?tex=%5Cpi_%7B%5Ctheta%7D%28u_t%7Co_t%29) 表示带参数的策略（表示在t时刻的观测下采取各种行为的条件概率，而可以是神经网络的参数），若策略为： ![\pi_{\theta}(o_t) \to a_t](https://www.zhihu.com/equation?tex=%5Cpi_%7B%5Ctheta%7D%28o_t%29+%5Cto+a_t) ，则是确定型策略。其中 ![o_t](https://www.zhihu.com/equation?tex=o_t) 和 ![x_t](https://www.zhihu.com/equation?tex=x_t) 不同为： ![o_t](https://www.zhihu.com/equation?tex=o_t) 往往表示机器所感知到的信息（如由像素组成的图像），而 ![x_t](https://www.zhihu.com/equation?tex=x_t) 表示外界环境实际所处的状态（如物体运动的速度，物体相对位置等）（不一定能观测到，而且是高维像素的低维内在表示），而且状态是满足马尔科夫性质的，但观测就不一定满足了，但在课程中有时不太需要十分区别两者，都可以作为输入。

我们之后所用的是马尔可夫决策过程（Markov Decision Processes，MDP）即假设当前状态只与上一状态和动作有关，用图表示：

![img](https://pic3.zhimg.com/80/v2-d17d6d5173f20f1c72b7db7b6a60f0ae_hd.jpg)

![p(x_{t+1}|x_t,u_t)](https://www.zhihu.com/equation?tex=p%28x_%7Bt%2B1%7D%7Cx_t%2Cu_t%29) 表示状态转移函数： ![x_{t+1}](https://www.zhihu.com/equation?tex=x_%7Bt%2B1%7D) 可见仅与 ![x_t,u_t](https://www.zhihu.com/equation?tex=x_t%2Cu_t) 有关，而与 ![x_{t-1},u_{t-1}](https://www.zhihu.com/equation?tex=x_%7Bt-1%7D%2Cu_%7Bt-1%7D) 条件不相关。

## **二、模仿学习（Imitation Learning）**

想让机器学会某样操作（比如开车），一个很直接的想法是观察人类行为，并且模仿人类，在相应观测下做出人类所做行为。将这个想法实现起来也很简单，只需要收集该任务的一些观测（路面的画面），以及每个观测人类会做出的反应（转动方向盘），然后像监督学习一样训练一个神经网络，以观测为输入，人类行为为标签，其中行为是离散时是分类任务，连续时是回归任务：

![img](https://pic4.zhimg.com/80/v2-ee4085040d8cded1f9f42cb1b091bd74_hd.jpg)

然而这简单的监督学习理论上并不可行，一个直观的原因是由于现实的随机性或者复杂性，使得机器所采用的动作和人类的动作有偏差或者动作所产生的结果有偏差，这样在有偏差的下一状态，机器还会做出有偏差的动作，使得之后状态的偏差积累，导致机器遇到监督学习时没有碰到过的状态，那机器就完全不知道该怎么做了。

![img](https://pic1.zhimg.com/80/v2-cdc2049ca175848e391180c45d6793e2_hd.jpg)

严格的讲，就是**学习时和实际操作时的** ![o_t](https://www.zhihu.com/equation?tex=o_t) **的分布不同**，实际操作时的 ![o_t](https://www.zhihu.com/equation?tex=o_t) 的情况由于存在偏差而会比学习时的要糟糕许多。因此我们希望 ![p_{data}(o_t)=p_{\pi_{\theta}}(o_t)](https://www.zhihu.com/equation?tex=p_%7Bdata%7D%28o_t%29%3Dp_%7B%5Cpi_%7B%5Ctheta%7D%7D%28o_t%29) ，这也就是**DAgger**算法的想法。在

## **稳定化实际分布**

DAgger算法之前，我们先来看看简单的模仿学习在实际中的表现：

![img](https://pic2.zhimg.com/80/v2-1cc908bcf1f98bb389349add70398dc0_hd.jpg)

有关汽车的自动驾驶的论文：《[End to End Learning for Self-Driving Cars](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1604.07316)》中，在采集数据时，汽车中间和两侧都放有摄像头，将三张图片作为观测，而相应的人类的正常驾驶行为作为标记，将这组数据打乱喂给CNN，监督学习出模仿人类的策略。出人意料的是这种简单的模仿学习实际上能成功。这是因为论文中所用的一个技巧：在标记左边摄像机画面时，**它的的标记为人类正常动作加上小量的右转，而汽车右边摄像机图像的标记是人类正常动作加上小量的左转**。这样机器行为在产生的向左偏差时，机器所接受到的画面就和正常情况下的左边摄像机的画面相似，而汽车左边摄像机图像的标记是人类正常动作加上小量的右转，因此机器进行有小量的右转。这样**能在偏差时，检测到偏差并作出修复**，也就是对的实际分布 ![o_t](https://www.zhihu.com/equation?tex=o_t) 起了稳定的作用。

对于这种想法的拓展，就是希望 ![o_t](https://www.zhihu.com/equation?tex=o_t) 的实际分布能相对学习时的分布稳定，一种方法是，学习用的数据不光是某种条件下的一条人类所走的路径，而是希望正确的路径有一个明确的概率分布tj，作为模仿学习的数据。因为实在正确路径的概率分布中取路径，因此其包含了许多偏差路径修正的例子作为学习数据，可从下图看出：

![img](https://pic4.zhimg.com/80/v2-36a19a1aa06524ffc2afa3aefd598f0d_hd.jpg)

这样实际操作中，机器由于学习了许多面对偏差的修正行为，能让实际路径分布相对学习时的分布稳定。而正确路径的概率分布的生成可以用下一节会讲的iLQR方法。

## **DAgger方法**

为了让 ![p_{data}(o_t)=p_{\pi_{\theta}}(o_t)](https://www.zhihu.com/equation?tex=p_%7Bdata%7D%28o_t%29%3Dp_%7B%5Cpi_%7B%5Ctheta%7D%7D%28o_t%29) ，我们从数据来源出发，想让数据来源和实际操作的分布相似，那么很直接的一个想法是直接从 ![p_{\pi_{\theta}}(o_t)](https://www.zhihu.com/equation?tex=p_%7B%5Cpi_%7B%5Ctheta%7D%7D%28o_t%29) 采样学习数据，做法就是实际运行策略 ![\pi_{\theta}](https://www.zhihu.com/equation?tex=%5Cpi_%7B%5Ctheta%7D) ，但是需要对策略 ![\pi_{\theta}](https://www.zhihu.com/equation?tex=%5Cpi_%7B%5Ctheta%7D) 的运行结果做标记，使其成为训练数据，然后更新策略 ![\pi_{\theta}](https://www.zhihu.com/equation?tex=%5Cpi_%7B%5Ctheta%7D) ，以此循环：

![img](https://pic2.zhimg.com/80/v2-5e67e987ff7824b5e3e42eb1adbb52db_hd.jpg)

但DAgger方法的主要问题在第三步，即需要人来对策略 ![\pi_{\theta}](https://www.zhihu.com/equation?tex=%5Cpi_%7B%5Ctheta%7D) 的运行结果做上标记，这是十分耗费人力的事情。

## **总结**

总而言之，直接用人类数据做监督学习，来让机器模仿学习会遇到训练数据和实际操作时的分布不匹配的问题，因此不可行。

但是用一些方法能让模仿学习学的不错：

1. 和学习任务相关的技巧（如汽车自动驾驶时在两边装摄像头）
2. **生成训练数据的分布**，然后大量采样，使实际分布稳定（需要另外的学习算法）
3. 直接从 ![p_{\pi_{\theta}}(o_t)](https://www.zhihu.com/equation?tex=p_%7B%5Cpi_%7B%5Ctheta%7D%7D%28o_t%29) 采样学习数据（DAgger）

但即使用了一些方法，模仿学习始终有这些问题：

1. 需要人类提供的**大量数据**（尤其是深度学习，然而常常没有多人类数据）
2. 人类对一些任务也做的不太好，对于一些复杂任务，人类能做出的动作有限（比如操作无人直升机）
3. 我们希望机器能**自动学习**，即能不断地在错误中自我完善，而不需要人类的指导。

## **作业指导**

关于课程的第一次作业，需要从专家数据中模仿学习，以及用DAgger方法。

完整的tensorflow代码已经上交到我的Github上了：[https://github.com/futurebelongtoML/homework.git](https://link.zhihu.com/?target=https%3A//github.com/futurebelongtoML/homework.git)

仅做参考。

<div STYLE="page-break-after: always;"></div>

#**第二章：有模型的优化控制**

本专题记录了在自学伯克利的深度学习课程时的笔记，主要目的是供个人学习用，因此有一些个人的观点和感想。文章主要是对CS 294课程视频的翻译与整理，附加一些相关知识。课程地址：[http://rll.berkeley.edu/deeprlcoursesp17/](https://link.zhihu.com/?target=http%3A//rll.berkeley.edu/deeprlcoursesp17/)，课程视频地址：[https://www.youtube.com/watch?v=8jQIKgTzQd4&index=1&list=PLkFD6_40KJIwTmSbCv9OVJB3YaO4sFwkX](https://link.zhihu.com/?target=https%3A//www.youtube.com/watch%3Fv%3D8jQIKgTzQd4%26index%3D1%26list%3DPLkFD6_40KJIwTmSbCv9OVJB3YaO4sFwkX)

*提示：这一节的优化控制以及之后的model-based RL主要解决的是\**连续动作空间**的强化学习问题，比如**机器人控制**，所以它的环境模型相对简单，但这两个内容没有在Sutton的Reinforcement Learning：An Introduction上详细写出，Sutton书主要还是处理**离散动作空间**，用Q-leaning或policy gradient等（PG可以应用到**连续动作空间**），所以之前如果读过此书的，应该区别对待两种研究对象。*

**\*想要直接看离散动作空间问题，如用Q-leaning玩Atari Game，可直接跳到第（五）节，2-4节的model-based RL和5-10节的model-free RL知识关联度不大，可分开看。***

*然后这节的\**蒙特卡洛树搜索则是AlphaGo的核心算法之一。***

## **优化控制（Optimal Control）（有模型的）**

现在我们不想只让机器模仿人类行为，而是想要让机器自己根据目的和情形做出最优的行动。没有了人类的标准后，那么要衡量行为的好坏需要规定一个客观的**损失函数** ![c(x_t, u_t)](https://www.zhihu.com/equation?tex=c%28x_t%2C+u_t%29) 或者**奖励函数** ![r(x_t, u_t)](https://www.zhihu.com/equation?tex=r%28x_t%2C+u_t%29) 。而且强化学习是一个序列决策过程，也就是说学习的目标是要选择一系列的动作来让总体的损失最低（或奖励最高），用公式表示即（这里取总体损失函数为优化目标）：

![\min_{u_1,...,u_T}\sum^T_{t=1}c(x_t, u_t) , s.t.x_t=f(x_{t-1}, u_{t-1})\quad(3.1)](https://www.zhihu.com/equation?tex=%5Cmin_%7Bu_1%2C...%2Cu_T%7D%5Csum%5ET_%7Bt%3D1%7Dc%28x_t%2C+u_t%29+%2C+s.t.x_t%3Df%28x_%7Bt-1%7D%2C+u_%7Bt-1%7D%29%5Cquad%283.1%29) 

其中 ![f(x_{t-1}, u_{t-1})](https://www.zhihu.com/equation?tex=f%28x_%7Bt-1%7D%2C+u_%7Bt-1%7D%29) 是状态转移函数， ![f](https://www.zhihu.com/equation?tex=f) 根据上一状态和动作决定了下一状态。在这一节中我们假设我们对这个模型有完全的了解，即我们知道 ![f](https://www.zhihu.com/equation?tex=f) 的具体表达式，在这种假设情况下的强化学习称为有模型的（known dynamics）。在这种假设下，学习过程不需要任何的数据，更不需要用到CNN，而可以当一个纯优化问题对待。

> 这里优化的目标采用周期性回报，而非有折扣的连续回报： ![G_t=\sum_{k=0}^{\infty}\gamma^kR_{t+k+1}](https://www.zhihu.com/equation?tex=G_t%3D%5Csum_%7Bk%3D0%7D%5E%7B%5Cinfty%7D%5Cgamma%5EkR_%7Bt%2Bk%2B1%7D) ，之后的model-free leaning会采用后者。

我们可以把(3.1)式看作一个有限制的**优化问题**，称为**路径优化**（Trajectory optimization）。很显然的，我们把限制条件带进去可得等价的无限制的优化问题：

![\min_{u_1,...,u_T}c(x_1, u_1) +c(f(x_1,u_1), u_2) +...+c(f(f(...)...),u_T) \quad (3.2)](https://www.zhihu.com/equation?tex=%5Cmin_%7Bu_1%2C...%2Cu_T%7Dc%28x_1%2C+u_1%29+%2Bc%28f%28x_1%2Cu_1%29%2C+u_2%29+%2B...%2Bc%28f%28f%28...%29...%29%2Cu_T%29+%5Cquad+%283.2%29) 

对于这种问题，常常用梯度下降法，以及导数的反向传播（神经网络的知识）。

*注：但在实际中可能使用二阶梯度下降的方法（如牛顿法）更有效，因为如果这个过程持续时间很长，那么早些时候的* ![u_t](https://www.zhihu.com/equation?tex=u_t) *会遭受梯度弥散问题，这样标准的梯度下降法就不太有效（这种问题在大型的RNN中也有）。在优化中，如果优化（3.2），即仅优化* ![u_t](https://www.zhihu.com/equation?tex=u_t) *，称为打靶法，其优化路径受初值的影响较大，此外也可以同时优化* ![c_t](https://www.zhihu.com/equation?tex=c_t) *和* ![u_t](https://www.zhihu.com/equation?tex=u_t) *，需要（3.1）的限制条件，称为协同法。这里我们用较简单的打靶法，即优化（3.2）式即可。*

## **线性情况：LQR（linear quadratic regulator）**

现在讨论一种简单的情形：我们的模型是线性的，即是 ![f(x_{t}, u_{t})](https://www.zhihu.com/equation?tex=f%28x_%7Bt%7D%2C+u_%7Bt%7D%29) 线性函数，是 ![c(x_{t}, u_{t})](https://www.zhihu.com/equation?tex=c%28x_%7Bt%7D%2C+u_%7Bt%7D%29) 二阶函数：

![img](https://pic2.zhimg.com/80/v2-084139fa7eb97f035feb03b34c73783f_hd.jpg)

关于这个模型有一个对应的例子：假设我们的模型是二维平面里的一个小球的物理运动， ![x_t](https://www.zhihu.com/equation?tex=x_t) 是小球的（坐标向量，速度向量），则由物理知识知：

![\begin{pmatrix} x_{t+1} \\ y_{t+1} \\ x'_{t+1} \\ y'_{t+1} \\ \end{pmatrix}= \begin{pmatrix} x_{t}+\delta t x'_t + 1/2\delta t^2 f_x\\ y_{t}+\delta t y'_t + 1/2\delta t^2 f_y \\ x'_{t}+\delta t f_x \\ y'_{t} +\delta t f_y\\ \end{pmatrix}](https://www.zhihu.com/equation?tex=%5Cbegin%7Bpmatrix%7D+x_%7Bt%2B1%7D+%5C%5C+y_%7Bt%2B1%7D+%5C%5C+x%27_%7Bt%2B1%7D+%5C%5C+y%27_%7Bt%2B1%7D+%5C%5C+%5Cend%7Bpmatrix%7D%3D+%5Cbegin%7Bpmatrix%7D+x_%7Bt%7D%2B%5Cdelta+t+x%27_t+%2B+1%2F2%5Cdelta+t%5E2+f_x%5C%5C+y_%7Bt%7D%2B%5Cdelta+t+y%27_t+%2B+1%2F2%5Cdelta+t%5E2+f_y+%5C%5C+x%27_%7Bt%7D%2B%5Cdelta+t+f_x+%5C%5C+y%27_%7Bt%7D+%2B%5Cdelta+t+f_y%5C%5C+%5Cend%7Bpmatrix%7D) 

![\delta t](https://www.zhihu.com/equation?tex=%5Cdelta+t) 是时间间隔是固定的，因此下一状态是上一状态和施力 ![ (f_x, f_y)](https://www.zhihu.com/equation?tex=+%28f_x%2C+f_y%29) 的线性函数。而这个模型的可能是小球与某一目标点的距离，则我们的学习任务就是对这个小球施加一定的力（比如控制机器臂），让他达到目标位置。

现在我们在这种线性条件下优化（3.2）式：

## **1).优化最后一个动作** ![\large u_T](https://www.zhihu.com/equation?tex=%5Clarge+u_T) 

首先我们固定其他动作，仅优化最后一个动作 ![u_T](https://www.zhihu.com/equation?tex=u_T) ，那么我们的整体优化目标函数变为 ![Q(x_T,u_T)](https://www.zhihu.com/equation?tex=Q%28x_T%2Cu_T%29) ，（可自行利用线性条件和固定先前动作，带入（3.1）式中检验下式）：

![img](https://pic1.zhimg.com/80/v2-a7755bb480f259a2453467f4521a20b2_hd.jpg)

然后把 ![C_T](https://www.zhihu.com/equation?tex=C_T) 和 ![c_T](https://www.zhihu.com/equation?tex=c_T) 写成分块矩阵：

![img](https://pic1.zhimg.com/80/v2-8a081b3aa8f87015cc457405d76067b3_hd.jpg)

有了这个准备后，求 ![Q(x_T,u_T)](https://www.zhihu.com/equation?tex=Q%28x_T%2Cu_T%29) 对 ![u_T](https://www.zhihu.com/equation?tex=u_T) 的梯度，取梯度为零的点（极值小点），由于是 ![Q(x_T,u_T)](https://www.zhihu.com/equation?tex=Q%28x_T%2Cu_T%29) 凸函数，极小值点就是最小值，这样我们就得到了，（在固定了 ![x_T ](https://www.zhihu.com/equation?tex=x_T+) 后）让 ![Q(x_T,u_T)](https://www.zhihu.com/equation?tex=Q%28x_T%2Cu_T%29) 最小的 ![u_T](https://www.zhihu.com/equation?tex=u_T) 值：

![img](https://pic4.zhimg.com/80/v2-6b754de8e2c8690acb2fab967faa7331_hd.jpg)

**结果为** ![u_T=K_Tx_T+k_T](https://www.zhihu.com/equation?tex=u_T%3DK_Tx_T%2Bk_T) **，其中**

![img](https://pic1.zhimg.com/80/v2-7a0064c12de0ad38a97c86c322b489ba_hd.jpg)

可见最优的 ![u_T](https://www.zhihu.com/equation?tex=u_T) 完全由 ![x_T](https://www.zhihu.com/equation?tex=x_T) 决定，那我们把最优的 ![u_T=K_Tx_T+k_T](https://www.zhihu.com/equation?tex=u_T%3DK_Tx_T%2Bk_T) 代入到 ![Q(x_T,u_T)](https://www.zhihu.com/equation?tex=Q%28x_T%2Cu_T%29) 中，可得到仅与 ![x_T](https://www.zhihu.com/equation?tex=x_T) 有关的优化目标函数 ![V(x_T)(=\min_{u_T}Q(x_T,u_T))](https://www.zhihu.com/equation?tex=V%28x_T%29%28%3D%5Cmin_%7Bu_T%7DQ%28x_T%2Cu_T%29%29) :

![img](https://pic4.zhimg.com/80/v2-069f1a18b7acd94f43f6689e9583efe4_hd.jpg)

进一步计算得

![img](https://pic4.zhimg.com/80/v2-8c7374a4b1db3b98f7d4430ddeb1a3b6_hd.jpg)

其中 ![V_T](https://www.zhihu.com/equation?tex=V_T) 和 ![v_T](https://www.zhihu.com/equation?tex=v_T)

![img](https://pic3.zhimg.com/80/v2-06f5180e55ac9bd3a167e3251e5b0d20_hd.jpg)

## **2).在此基础上优化**

在代入最优的 ![u_T](https://www.zhihu.com/equation?tex=u_T) 后，我们再把T-1之前的动作和状态固定，考虑对 ![u_{T-1}](https://www.zhihu.com/equation?tex=u_%7BT-1%7D) 的优化，这种情况下，优化目标（3.2）式（或者说就是 ![V(x_T)](https://www.zhihu.com/equation?tex=V%28x_T%29) ，其中的const变为与 ![(x_{T-1},u_{T-1})](https://www.zhihu.com/equation?tex=%28x_%7BT-1%7D%2Cu_%7BT-1%7D%29) 有关）就化为了 ![Q(x_{T-1},u_{T-1})](https://www.zhihu.com/equation?tex=Q%28x_%7BT-1%7D%2Cu_%7BT-1%7D%29) 式：

![img](https://pic2.zhimg.com/80/v2-fbb807cb8ce1c26bbd412c821f2f4743_hd.jpg)

我们把 ![x_T](https://www.zhihu.com/equation?tex=x_T) 用 ![f(x_{T-1},u_{T-1})](https://www.zhihu.com/equation?tex=f%28x_%7BT-1%7D%2Cu_%7BT-1%7D%29) 代入，还记得 ![f](https://www.zhihu.com/equation?tex=f) 是一个线性函数：

![img](https://pic3.zhimg.com/80/v2-f11269093441641a212fd29c66732828_hd.jpg)

把 ![V(x_T)](https://www.zhihu.com/equation?tex=V%28x_T%29) 化开后，将 ![Q(x_{T-1},u_{T-1})](https://www.zhihu.com/equation?tex=Q%28x_%7BT-1%7D%2Cu_%7BT-1%7D%29) 式进行化简得：

![img](https://pic3.zhimg.com/80/v2-5560cffb03ce6bbae9702875d0b5b30d_hd.jpg)

求 ![Q(x_{T-1},u_{T-1})](https://www.zhihu.com/equation?tex=Q%28x_%7BT-1%7D%2Cu_%7BT-1%7D%29) 关于 ![u_{T-1}](https://www.zhihu.com/equation?tex=u_%7BT-1%7D) 的梯度，并取极小值点可得与 ![u_{T-1}](https://www.zhihu.com/equation?tex=u_%7BT-1%7D) 的优化结果类似的优化结果：

![img](https://pic2.zhimg.com/80/v2-fc3d1f2f532a256cd6c34dc4f183dbf7_hd.jpg)

其中 ![K_{T-1}](https://www.zhihu.com/equation?tex=K_%7BT-1%7D) 和 ![k_{T-1}](https://www.zhihu.com/equation?tex=k_%7BT-1%7D) 为

![img](https://pic2.zhimg.com/80/v2-33c6432c9bcabc6d1a24e286d4807300_hd.jpg)

## **3).完整的优化算法：**

类似求 ![u_{T-1}](https://www.zhihu.com/equation?tex=u_%7BT-1%7D) 的过程，可以得到反向递归的优化算法：

![img](https://pic2.zhimg.com/80/v2-e78fb5b1d92682d993038cb11ad58771_hd.jpg)

但注意到我们的**优化结果为** ![u_t=K_tx_t+k_t](https://www.zhihu.com/equation?tex=u_t%3DK_tx_t%2Bk_t) ， ![u_t ](https://www.zhihu.com/equation?tex=u_t+) 随着当前状态 ![x_t ](https://www.zhihu.com/equation?tex=x_t+) 变动，因此要想求最优的 ![u_t ](https://www.zhihu.com/equation?tex=u_t+) ，必须先知道当前的状态 ![x_t](https://www.zhihu.com/equation?tex=x_t) ，而 ![x_t](https://www.zhihu.com/equation?tex=x_t) 又是由前一状态和前一最优动作决定的。由此我们可以想到一个前向递归生成 ![x_t](https://www.zhihu.com/equation?tex=x_t) 的算法：

![img](https://pic2.zhimg.com/80/v2-0011dbc96512d41ba819139053121922_hd.jpg)

至此我们就能通过计算得到线性的已知模型的最优动作序列。

但如果状态转移函数是带有随机噪音的呢，即

![img](https://pic3.zhimg.com/80/v2-a2f2a0c612deca1ea65f05dda97ba95e_hd.jpg)

实际上它的优化算法和确定情况的算法一模一样，这是因为随机性只影响了 ![Q(x_{T-1},u_{T-1})](https://www.zhihu.com/equation?tex=Q%28x_%7BT-1%7D%2Cu_%7BT-1%7D%29) 式中的 ![x_T](https://www.zhihu.com/equation?tex=x_T) 用 ![f(x_{T-1},u_{T-1})](https://www.zhihu.com/equation?tex=f%28x_%7BT-1%7D%2Cu_%7BT-1%7D%29) 代入，对其做期望即 ![E_{p(x_t|x_t,u_t)}V(x_{T})](https://www.zhihu.com/equation?tex=E_%7Bp%28x_t%7Cx_t%2Cu_t%29%7DV%28x_%7BT%7D%29) ，结果就是确定情况的 ![V(f(x_{T-1},u_{T-1}))](https://www.zhihu.com/equation?tex=V%28f%28x_%7BT-1%7D%2Cu_%7BT-1%7D%29%29) 。

## **非线性情况（DDP/iLQR）**

对于更一般的非线性情况，一个常用的想法是对它做线性近似，然后用LQR求解。那 ![f](https://www.zhihu.com/equation?tex=f) 和 ![c](https://www.zhihu.com/equation?tex=c) 在任意初始点： ![(\hat x_t,\hat u_t)](https://www.zhihu.com/equation?tex=%28%5Chat+x_t%2C%5Chat+u_t%29) 处做泰勒展开，可以近似得：（ ![f](https://www.zhihu.com/equation?tex=f) 做一阶近似， ![c](https://www.zhihu.com/equation?tex=c) 做二阶近似）

![img](https://pic3.zhimg.com/80/v2-2cf75ad7ad6d0d55a12e92f16b504b97_hd.jpg)

设 ![\delta x_t=x_t -\hat x_t](https://www.zhihu.com/equation?tex=%5Cdelta+x_t%3Dx_t+-%5Chat+x_t) ， ![\delta u_t=u_t -\hat u_t](https://www.zhihu.com/equation?tex=%5Cdelta+u_t%3Du_t+-%5Chat+u_t) ，那么现在的**非线性模型可以近似为有关** ![\delta x_t](https://www.zhihu.com/equation?tex=%5Cdelta+x_t) **,** ![\delta u_t](https://www.zhihu.com/equation?tex=%5Cdelta+u_t) **的线性模型:**

![img](https://pic4.zhimg.com/80/v2-ec5ab6572d20a24af3450ef690156cd3_hd.jpg)

![img](https://pic4.zhimg.com/80/v2-e6a802d5928f5e155075d1e1a83b9168_hd.jpg)

现在我们便可以对 ![\delta x_t](https://www.zhihu.com/equation?tex=%5Cdelta+x_t) , ![\delta u_t](https://www.zhihu.com/equation?tex=%5Cdelta+u_t) 的线性模型做之前的LQR算法：

![img](https://pic1.zhimg.com/80/v2-530ae53d0fa4d4cb08eeecf70b6b5890_hd.jpg)

这样最优的近似动作为 ![u_t=K_t(x_t-\hat x_t)+k_t+\hat u_t](https://www.zhihu.com/equation?tex=u_t%3DK_t%28x_t-%5Chat+x_t%29%2Bk_t%2B%5Chat+u_t) （原图少了 ![\hat u_t](https://www.zhihu.com/equation?tex=%5Chat+u_t) ），还需要注意在倒数第二步获得的前向过程应该使用真实的非线性模型，而不使用近似的线性模型（因为在运行多步后，线性模型和真实模型就差的很多了）。用最优的近似路径 ![x_t](https://www.zhihu.com/equation?tex=x_t) 和 ![u_t](https://www.zhihu.com/equation?tex=u_t) 来代替 ![\hat x_t](https://www.zhihu.com/equation?tex=%5Chat+x_t) 和 ![\hat u_t](https://www.zhihu.com/equation?tex=%5Chat+u_t) ，重复以上的LQR算法。多次之后我们期望其收敛到一个局部最优的路径。

如果学过一点优化的能感受到这个iLQR方法和牛顿法很相似，都是对目标函数做了二阶近似，但略有不同的是iLQR对只做了 ![f](https://www.zhihu.com/equation?tex=f) 一阶近似，对 ![c](https://www.zhihu.com/equation?tex=c) 才做了二阶近似。

不过和标准的牛顿法一样，iLQR也有局部收敛（即初值在局部最小值点的小领域内能保证收敛），而整体不能保证收敛。因此我们需要调节学习率 ![\alpha](https://www.zhihu.com/equation?tex=%5Calpha) ，比如在下降方向做一个线性搜索，找到下降方向上的最小值，来保证收敛。在这样的调节下，最优近似动作变为： ![u_t=K_t(x_t-\hat x_t)+\alpha k_t+\hat u_t](https://www.zhihu.com/equation?tex=u_t%3DK_t%28x_t-%5Chat+x_t%29%2B%5Calpha+k_t%2B%5Chat+u_t) 。

## **案例分析：重计划（replanning）**

在论文《Synthesis and Stabilization of Complex Behavior through Online Trajectory Optimization》中采用了重计划（replanning）的技巧，即在每一步都用iLQR得到一系列的近似最优动作，但是只执行第一个，舍弃剩余动作，到下一状态时，重新做iLQR，再只执行第一步，以此类推，每一步都做一个iLQR：

![img](https://pic1.zhimg.com/80/v2-f378b0e506929773bc5cbcc7730e130d_hd.jpg)

这个技巧的作用是当模型比较复杂不能很好用线性近似或者本身就有一定随机性时，如果仅在初始状态做一个iLQR，得到一系列近似最优动作，在实际执行时，实际路径可能会和模型的iLQR的最优路径有些偏差，这样仅执行这一系列近似最优动作的话，到最后会有较大的累计偏差。而重计划能在每一步都接收到这种偏差，基于这些偏差来产生最优动作，能及时修正偏差。

## **离散情况：蒙特卡洛树搜索(Monte Carlo tree search (MCTS))**

对于有模型的离散情况，即状态空间和动作空间都只能取离散的值，于是和都是不可导的，但是是已知的。对于这种情况，一般用搜索算法在状态的完全树中找出最优的动作序列，然而如果要进行全搜索，那复杂度是指数级的，肯定不可行，因此可以用蒙特卡洛树搜索来找到较优路径。

离散情况的例子如Atari上的一些小游戏，动作空间只有上下左右等有限的键盘键，状态空间由图像的有限个像素块组成，奖励函数为游戏分数：

![img](https://pic4.zhimg.com/80/v2-9e3c6b6c1efb44550e8bb44ef8500e22_hd.jpg)

较简单的搜索树（二叉树）如：

![img](https://pic2.zhimg.com/80/v2-ee14bb893cdb503f029c3856b60ac968_hd.jpg)

由于课上讲的蒙特卡洛方法有些细节不太清楚，以下对其的讲解将结合这篇论文：《[Browne, Powley, Whitehouse, Lucas, Cowling, Rohlfshagen, Tavener, Perez, Samothrakis, Colton. (2012). A Survey of Monte Carlo Tree Search Methods](https://link.zhihu.com/?target=http%3A//www.cameronius.com/cv/mcts-survey-master.pdf)》。

蒙特卡洛树搜索的完整算法为：

![img](https://pic3.zhimg.com/80/v2-1fdad6539c13773c7db23c58cab64d06_hd.jpg)

首先我们有一个初始节点 ![v_0](https://www.zhihu.com/equation?tex=v_0) （其对应的初始状态为 ![s_0](https://www.zhihu.com/equation?tex=s_0) ），对其可进行一个动作 ![a_1](https://www.zhihu.com/equation?tex=a_1) ， ![a_1](https://www.zhihu.com/equation?tex=a_1) 在（有限的）动作空间中取值，为简单起见，假设搜索树为二叉树。对于 ![a_1](https://www.zhihu.com/equation?tex=a_1) =0或1，分别到达了下一状态 ![s_2=s_{2,0}](https://www.zhihu.com/equation?tex=s_2%3Ds_%7B2%2C0%7D) 或 ![s_{2,1}](https://www.zhihu.com/equation?tex=s_%7B2%2C1%7D) 。现在需要对 ![s_2](https://www.zhihu.com/equation?tex=s_2) 的两个取值做出评估，来说明那个相应动作更应该被执行。评估方法是从 ![s_2=s_{2,0}](https://www.zhihu.com/equation?tex=s_2%3Ds_%7B2%2C0%7D) 或开始 ![s_{2,1}](https://www.zhihu.com/equation?tex=s_%7B2%2C1%7D) ，执行**默认策略** ![\pi](https://www.zhihu.com/equation?tex=%5Cpi) （Default Policy）（一般取**随机策略**就够用了），直到达到终止状态，然后取终止状态的奖励值作为 ![s_2=s_{2,0}](https://www.zhihu.com/equation?tex=s_2%3Ds_%7B2%2C0%7D) 或 ![s_{2,1}](https://www.zhihu.com/equation?tex=s_%7B2%2C1%7D) 的评估值，比如分别为10和12（图上15改为12）。

![img](https://pic2.zhimg.com/80/v2-90bd13167bd106094a68e869b1d6a69a_hd.jpg)

这样我们对执行 ![a_1](https://www.zhihu.com/equation?tex=a_1) 后的 ![s_2](https://www.zhihu.com/equation?tex=s_2) 有了了解，但想要继续搜索下一动作 ![a_2](https://www.zhihu.com/equation?tex=a_2) 和状态 ![s_3](https://www.zhihu.com/equation?tex=s_3) 就面临了选择困难：因为不能进行全搜索，我们需要在 ![s_{2,0}](https://www.zhihu.com/equation?tex=s_%7B2%2C0%7D) 和 ![s_{2,1}](https://www.zhihu.com/equation?tex=s_%7B2%2C1%7D) 中选择一个来进行下一步动作。选择搜索路径采用**树搜索策略**（Tree Policy），其大致思想是尽量选择累计奖励高的子节点，因为这样看起来会成功，但是也希望能对较少搜索过的子节点有些关注，因为其有未发掘的潜力。具体而言：

![img](https://pic4.zhimg.com/80/v2-d15d7f61d8e8d6d3c49c231062c9b4a3_hd.jpg)

如果子节点中有还未探索过的，那优先探索未知子节点。如果子节点都探索过了（完全展开），那就根据上面的得分函数Score( ![s_{t+1}](https://www.zhihu.com/equation?tex=s_%7Bt%2B1%7D) )，来选择得分高的子节点探索。公式中 ![Q(s_t)](https://www.zhihu.com/equation?tex=Q%28s_t%29) 表示子状态 ![s_t](https://www.zhihu.com/equation?tex=s_t) 的累计评估值（由之前的评估方法得到）， ![N(s_t)](https://www.zhihu.com/equation?tex=N%28s_t%29) 表示子状态 ![s_t](https://www.zhihu.com/equation?tex=s_t) 被探索过的次数（可以看出探索次数增多，得分会降低）。

在此例中 ![s_{2,1}](https://www.zhihu.com/equation?tex=s_%7B2%2C1%7D) 的得分高，因此下一次搜索时选择 ![s_{2,1}](https://www.zhihu.com/equation?tex=s_%7B2%2C1%7D) ，并随机取其一个子节点 ![s_{3,0}](https://www.zhihu.com/equation?tex=s_%7B3%2C0%7D) ，然后对子节点 ![s_{3,0}](https://www.zhihu.com/equation?tex=s_%7B3%2C0%7D) 进行评估。同时在评估完子节点 ![s_{3,0}](https://www.zhihu.com/equation?tex=s_%7B3%2C0%7D) 后，把 ![s_{3,0}](https://www.zhihu.com/equation?tex=s_%7B3%2C0%7D) 的评估值（比如10）通过搜索路径**反向传播**给它的父节点，使父节点的累计评估值增加（Q+10），以及搜索次数加一，有利于下一次的树搜索策略：

![img](https://pic1.zhimg.com/80/v2-0deb5a552a8395be809270b1b2f7e72f_hd.jpg)

类似的，以后的搜索都按照上述过程，通过树搜索策略来搜索路径，直到达到一个未完全展开的叶节点或者到了终止状态，然后对该未完全展开的叶节点取一个未探索过的子节点，评估子节点的值，并把评估值反向传播到搜索路径上的节点：

![img](https://pic3.zhimg.com/80/v2-f9239a9d2889d9f113a4eef359687177_hd.jpg)

重复这一整个过程直到达到**计算预算**（computational budget）（如时间，储存量）。这样之后，我们对这整个树有了很好的了解，于是可以选择从 ![s_0](https://www.zhihu.com/equation?tex=s_0) 开始的最优路径（或者是相应的最优动作序列），每一步只需要选择平均累计奖励最大的子状态（也可以选择被搜索最多次的子状态）。

## **案例分析：向MCTS模仿学习**

在论文《[Deep Learning For Real-time Atari Game Play Using Offline Monte-Carlo Tree Search Planning](https://link.zhihu.com/?target=https%3A//papers.nips.cc/paper/5421-deep-learning-for-real-time-atari-game-play-using-offline-monte-carlo-tree-search-planning.pdf)》中，他将蒙特卡洛树搜索算法得到的动作序列作为专家动作，取代了之前模仿学习DAgger中的人类行为，让深度神经网络学习MCTS的输出，具体而言：

![img](https://pic3.zhimg.com/80/v2-0dba1a1866b57279f1585dc5704b2351_hd.jpg)

仅仅把第三步的人类会选择的动作改为了MCTS会选择的动作。

这样做的主要原因是MCTS对于实时游戏而言太慢了，而把策略“存储”在神经网络里，能更快速的根据观测得出结果，而不需要对当前状态做多次的树搜索才得最优路径。同时用神经网络也有利于泛化能力，可以对新状态（模型的微调）做出快速反应。

## **总结**

在本节的优化控制中我们学习了有模型的强化学习，根据模型的类型分为了

1.线性模型：LQR

2.非线性模型：DDP/iLQR

3.离散情况：蒙特卡洛树搜索 MCTS

对于线性模型，学习任务相当于解一个优化问题，通过反向递推，依次求出最优的 ![u_t](https://www.zhihu.com/equation?tex=u_t) ，为 ![u_t=K_tx_t+k_t](https://www.zhihu.com/equation?tex=u_t%3DK_tx_t%2Bk_t) ，再做一遍前向传播 ![x_t](https://www.zhihu.com/equation?tex=x_t) 便可得出最优路径。对于非线性模型，我们用线性模型进行近似，反复多次LQR，使其收敛到局部最优路径。为了能及时检测到偏差，可以用重计划的技巧，每一步都做一个iLQR。对于离散情况，我们用蒙特卡洛树搜索 MCTS，利用树搜索策略寻找较好的叶节点，用默认（随机）策略评估叶节点，并反向传播评估值，最终通过这样的树搜索得到最优路径。实际中可以用神经网络模仿学习MCTS来加快推测过程。

但是要知道在现实中，很多模型的运行方式都是未知的，甚至是很复杂的，对于这种情况，一个方法（基于模型的学习）是让机器先学习模型，再做决策，也就是下一节的**基于模型的学习**，而有模型的学习的知识为其做了铺垫。

<div STYLE="page-break-after: always;"></div>

#**第三章：模型学习**

之前两节分别讲了通过模仿人类行为来学习动作，以及在已知模型的情况下，通过优化控制（反向传播近似最优动作）来计划动作序列。本节将讨论无模型情况下的学习，基本思想是先学习模型（整体模型或局部模型），在对近似模型做优化控制得到动作序列。

> *提示：之前的优化控制以及这一节的model-based RL主要解决的是\**连续动作空间**的强化学习问题，比如**机器人控制**，所以它的环境模型相对简单。* **\*想要直接看model-free离散动作空间问题，如用Q-leaning或policy gradient玩Atari Game，可直接跳到第（五）节，2-4节的model-based RL和5-10节的model-free RL知识关联度不大，可分开看。***

## **基于模型的强化学习（model-based reinforcement learning）：**

首先来看为什么需要学习模型，没有模型对之前的优化控制有什么影响：

1. 在连续情况，转化为一个无限制的优化问题，通过梯度下降和反向传播来求解，但是在反向传播时需要计算![\frac{\partial f}{\partial x_t}](https://www.zhihu.com/equation?tex=%5Cfrac%7B%5Cpartial+f%7D%7B%5Cpartial+x_t%7D)和![\frac{\partial f}{\partial u_t}](https://www.zhihu.com/equation?tex=%5Cfrac%7B%5Cpartial+f%7D%7B%5Cpartial+u_t%7D)，而这是需要知道模型的（即状态转移函数）
2. 在离散情况，做蒙特卡洛树搜索时，需要知道每执行一个动作，下一个状态是什么。虽然可以在与环境交互中进行蒙特卡洛树搜索来寻找最优子状态，但是肯定不如在已知树中搜索来得快，会大大降低学习速度。

现在假设我们知道了![f(x_t,u_t)](https://www.zhihu.com/equation?tex=f%28x_t%2Cu_t%29)（或者![p(x_{t+1}|x_t,u_t)](https://www.zhihu.com/equation?tex=p%28x_%7Bt%2B1%7D%7Cx_t%2Cu_t%29)对于随机情况）那我们就可以用上节讲的一些优化控制方法来求解。因此对于无模型的情况，只需要从环境（数据）中学习出，就可以通过它反向传播。

## **版本0.5：随机探索**

一个最直接的想法学习![f(x_t,u_t)](https://www.zhihu.com/equation?tex=f%28x_t%2Cu_t%29)就是直接当监督学习做，以一定策略在环境中运行，搜集状态转移的数据，再喂给一些常用的监督学习的模型。这就是我们的**基于模型的强化学习的版本0.5**：

![img](https://pic3.zhimg.com/80/v2-3b55b3d509624c9c3484ac55919daf27_hd.jpg)

它采用**随机策略**![\pi_0(u_t|x_t)](https://www.zhihu.com/equation?tex=%5Cpi_0%28u_t%7Cx_t%29)来从环境中收集信息。在实际上我们的版本0.5在特定情况下能表现的很好（在一些经典机器控制中），这需要：1.有一个比较好的探索策略（随机策略在一些情况下还是能发挥好的）2.最好有一些先验知识，先建立一个大概的模型，然后学习时只需要确定几个参数就可以了（比如先确立物理运动规则模型，然后一些参数：如长度，重量等可以学习得到）

但是在一般情形下，就不一定好了。举一个爬山的例子，学习目标是爬到最高处，而模型（山的形状）是未知的，现在我们用版本0.5学习模型，并基于其来做iLQR得到爬山的动作序列对：但假设这座山有一处悬崖，那么我们经过版本0.5学习的模型会表示这座山的形状好像是越往左走，越高，因此对其优化控制的结果就是一直向左走，结果就跌落悬崖：

![img](https://pic4.zhimg.com/80/v2-344323252e13c1be0f5e1a48273010d7_hd.jpg)

可以从图上看出，跌落悬崖的根源是在探索期间，机器一直采用随机策略使得它一直在一个小范围内打转，学习出山的一些局部特征，无法知道山的整体形状。但是在实际运行中，它仅利用局部的信息来对山建模做的最优策略，最后超过探索过的范围，达到未知领域（那么它在悬崖边的时候并不会正确预测到下一步（如果采用iLQR控制优化那就完全是依据局部特征来制定完整动作序列，连悬崖边都不会检测到）（如果采用replanning技巧，它每一步都重计划，那它虽然能看到悬崖边，但无法正确预测到下一步，还是会倾向于一直向左走））。

严格来讲，**根源是**![p_{\pi_f}(x_t)\ne p_{\pi_0}(x_t)](https://www.zhihu.com/equation?tex=p_%7B%5Cpi_f%7D%28x_t%29%5Cne+p_%7B%5Cpi_0%7D%28x_t%29)，即探索时的状态分布与基于探索模型后（用iLQR）采用的策略运行得到的状态分布不匹配。而且这种**不匹配现象**在用表示能力更强的学习模型（如CNN）时更严重，因为其泛化能力更差，只要模型有一点改变，就影响很大。

## **版本1.0：用iLQR策略的迭代模型学习**

那针对这个问题，我们希望![p_{\pi_f}(x_t)= p_{\pi_0}(x_t)](https://www.zhihu.com/equation?tex=p_%7B%5Cpi_f%7D%28x_t%29%3D+p_%7B%5Cpi_0%7D%28x_t%29)，回想一下模仿学习中的DAgger也是解决类似问题，因此做法就是让![\pi_f](https://www.zhihu.com/equation?tex=%5Cpi_f)跑起来，收集它的数据，并迭代提升对模型的估计![f(x_t,u_t)](https://www.zhihu.com/equation?tex=f%28x_t%2Cu_t%29)，也就提升了策略![\pi_f](https://www.zhihu.com/equation?tex=%5Cpi_f)。这样我们就得到了**基于模型的强化学习的版本1.0**：

![img](https://pic1.zhimg.com/80/v2-954203dbac97017b9b88417b85069dfe_hd.jpg)

这种迭代学习模型并执行iLQR的算法看起来已经不错了，但是有一个小问题，是出于iLQR本身的，即iLQR是基于模型后把整个动作序列都一次规划出来了，但如果模型估计的不完美，或者本身环境就有随机性，那么每一次都带有偏差，执行完iLQR的一套动作后，偏差会越来越大。

## **版本1.5：用replanning技巧的迭代模型学习**

针于这种偏差问题，可以用上节讲的重计划（replanning）技巧，每一步都根据当前状态做一次iLQR，并且只执行动作序列中的第一步，这样能及时检测到偏差：**基于模型的强化学习的版本1.5**：

![img](https://pic3.zhimg.com/80/v2-465dce09ea319caf65cb509806f0a99c_hd.jpg)

也就只是把版本1.0的第四步中的iLQR来计划一组动作改成用replanning技巧来分步实现一组动作。

## **版本2.0：用学习模型作为策略**

但是每一步都要做一个iLQR（或者MCTS，离散情况），那计算的负担是很大的，如果做实时的反应将会运行得很慢。这种情况其实在我们上一节讲MCTS时已经提到过了，所采用的针对方法是，以MCTS为导师，对其进行模仿学习，把策略“存储”在一个深度神经网络中。因此在这里也用一样的想法：用一个监督学习的模型（比如RNN）：![u_t=\pi_{\theta}(x_t)](https://www.zhihu.com/equation?tex=u_t%3D%5Cpi_%7B%5Ctheta%7D%28x_t%29)来代替iLQR（或者MCTS）来作为策略产生相应动作，那么对其的优化就是将多步的损失依次通过近似的模型函数![f(x_t,u_t)](https://www.zhihu.com/equation?tex=f%28x_t%2Cu_t%29)来反向传播（便于理解用的是确定性模型和策略，随机性模型和策略后面有案例会说（至少可以用采样的方法））：

![img](https://pic2.zhimg.com/80/v2-0ef2a07035f0f288f338b6a1851c1254_hd.jpg)

然后像之前的算法一样，用当前优化后的策略收集数据，迭代优化对模型的估计![f(x_t,u_t)](https://www.zhihu.com/equation?tex=f%28x_t%2Cu_t%29)，这就得到了**基于模型的强化学习的版本2.0**：

![img](https://pic4.zhimg.com/80/v2-47d5dddf3dda3ba88382814ff1776d71_hd.jpg)

## **基于模型的RL的四个版本小结：**

**版本0.5：随机探索取样，学习模型，优化控制**

优点：简单，不需要迭代过程

缺点：会有训练数据与运行策略的数据分布不匹配问题：![p_{\pi_f}(x_t)\ne p_{\pi_0}(x_t)](https://www.zhihu.com/equation?tex=p_%7B%5Cpi_f%7D%28x_t%29%5Cne+p_%7B%5Cpi_0%7D%28x_t%29)

**版本1.0：用iLQR得到的策略的迭代模型学习**

优点：简单，能解决不匹配问题

缺点：无法修复偏差，应对随机情况表现不佳

**版本1.5：用replanning的迭代模型学习**

优点：能及时修正偏差

缺点：计算开销大

**版本2.0：用学习模型作为策略，将损失直接反向传播到策略模型中**

优点：实时运行时计算得快

缺点：不太稳定，尤其是随机情况（下一节会讲到）

## **学习模型的选择：**

列出和比较常用的机器学习模型用来学习环境的动态模型（之后案例会看到更多模型，并比较它们的实际效果）：

![img](https://pic3.zhimg.com/80/v2-bc53779daa68183fd8e795428b8b082b_hd.jpg)

1. 高斯过程（GP）：输入![(x_t,u_t)](https://www.zhihu.com/equation?tex=%28x_t%2Cu_t%29)，根据学习数据来建立一个高斯分布![p(x'_t|x_t,u_t)](https://www.zhihu.com/equation?tex=p%28x%27_t%7Cx_t%2Cu_t%29)。优点：能有效利用所有数据（data-efficient），简单高效。缺点：对于不光滑的模型拟合效果不太好，当数据集很大时计算得很慢。
2. 神经网络（如RNN，autoencoder等）输入![(x_t,u_t)](https://www.zhihu.com/equation?tex=%28x_t%2Cu_t%29)，输出![x'_t](https://www.zhihu.com/equation?tex=x%27_t)（也可以是一个高斯分布的均值），因此它的损失函数就是实际的![x'_t](https://www.zhihu.com/equation?tex=x%27_t)与输出的欧几里得距离。（也可以更复杂一点，输出的是一个混合高斯分布的参数）优点：表示能力很强，能在大量数据上学习。缺点：在少量数据时的学习效果不佳
3. 其他模型：混合高斯分布模型（GMM）：是对于![(x_t,u_t,x'_t)](https://www.zhihu.com/equation?tex=%28x_t%2Cu_t%2Cx%27_t%29)对的生成式模型，训练得到混合高斯分布![p(x_t,u_t,x'_t)](https://www.zhihu.com/equation?tex=p%28x_t%2Cu_t%2Cx%27_t%29)，然后取条件分布，得到![p(x'_t|x_t,u_t)](https://www.zhihu.com/equation?tex=p%28x%27_t%7Cx_t%2Cu_t%29)。带有先验知识的专家模型：训练时只需要优化几个参数即可。

如果对高斯过程和混合高斯分布模型不太清楚的，可以简单看一下这几篇博客来初步了解一下：

高斯过程:[http://www.kuqin.com/shuoit/20150508/345958.html](https://link.zhihu.com/?target=http%3A//www.kuqin.com/shuoit/20150508/345958.html)

混合高斯分布：[http://www.cnblogs.com/CBDoctor/archive/2011/11/06/2236286.html](https://link.zhihu.com/?target=http%3A//www.cnblogs.com/CBDoctor/archive/2011/11/06/2236286.html)

## **案例分析1：用高斯过程的基于模型的强化学习**

在论文《[Learning to Control a Low-Cost Manipulator using Data-Efficient Reinforcement Learning](https://link.zhihu.com/?target=http%3A//www.roboticsproceedings.org/rss07/p08.pdf)》中，采用的算法和**基于模型的强化学习的版本2.0**大致一样，只不过是随机的情况：

![img](https://pic3.zhimg.com/80/v2-6b8dcb36f9ae01c13b952bf8a603d3fb_hd.jpg)

在第二步中采用了GP（高斯过程）来作为环境学习模型![p(x'|x,u)](https://www.zhihu.com/equation?tex=p%28x%27%7Cx%2Cu%29)，极大拟然作为学习的目标。而策略学习模型![\pi_{\theta}(u_t|x_t)](https://www.zhihu.com/equation?tex=%5Cpi_%7B%5Ctheta%7D%28u_t%7Cx_t%29)也是随机的。可以从原论文中的一样的算法（称为PILCO）看更多的细节：

![img](https://pic2.zhimg.com/80/v2-5fba4392ff688349498d3fed3bda5391_hd.jpg)

注意到在第三步的策略模型的学习时，对策略模型的评估函数（也就是学习的损失函数）为![J^{\pi}(\psi)](https://www.zhihu.com/equation?tex=J%5E%7B%5Cpi%7D%28%5Cpsi%29)，具体来看：

![J^{\pi}(\psi)=\sum_{t=0}^{T} \mathbb{E}_{x_t}(c(x_t))](https://www.zhihu.com/equation?tex=J%5E%7B%5Cpi%7D%28%5Cpsi%29%3D%5Csum_%7Bt%3D0%7D%5E%7BT%7D+%5Cmathbb%7BE%7D_%7Bx_t%7D%28c%28x_t%29%29)

其中![c(x_t)](https://www.zhihu.com/equation?tex=c%28x_t%29)是任务目标的损失函数。对![J^{\pi}(\psi)](https://www.zhihu.com/equation?tex=J%5E%7B%5Cpi%7D%28%5Cpsi%29)的表达式的解释为：注意到![\mathbb{E}_{x_t}(c(x_t))=\mathbb{E}_{(x_t,x_{t-1},u_{t-1})}(c(x_t))](https://www.zhihu.com/equation?tex=%5Cmathbb%7BE%7D_%7Bx_t%7D%28c%28x_t%29%29%3D%5Cmathbb%7BE%7D_%7B%28x_t%2Cx_%7Bt-1%7D%2Cu_%7Bt-1%7D%29%7D%28c%28x_t%29%29)，其中![p(x_t,x_{t-1},u_{t-1})=p(x_t|x_{t-1},u_{t-1})\pi_{\theta}(u_{t-1}|x_{t-1})p(x_{t-1})](https://www.zhihu.com/equation?tex=p%28x_t%2Cx_%7Bt-1%7D%2Cu_%7Bt-1%7D%29%3Dp%28x_t%7Cx_%7Bt-1%7D%2Cu_%7Bt-1%7D%29%5Cpi_%7B%5Ctheta%7D%28u_%7Bt-1%7D%7Cx_%7Bt-1%7D%29p%28x_%7Bt-1%7D%29)，这可以直接验证得到。因此，![J^{\pi}(\psi)=\sum_{t=0}^{T} \mathbb{E}_{x_t}(c(x_t))=\mathbb{E}_{\tau}(\sum_{t=0}^{T}c(x_t))](https://www.zhihu.com/equation?tex=J%5E%7B%5Cpi%7D%28%5Cpsi%29%3D%5Csum_%7Bt%3D0%7D%5E%7BT%7D+%5Cmathbb%7BE%7D_%7Bx_t%7D%28c%28x_t%29%29%3D%5Cmathbb%7BE%7D_%7B%5Ctau%7D%28%5Csum_%7Bt%3D0%7D%5E%7BT%7Dc%28x_t%29%29)，其中![\tau=(x_0,u_0...x_T)](https://www.zhihu.com/equation?tex=%5Ctau%3D%28x_0%2Cu_0...x_T%29)为轨迹，![J^{\pi}(\psi)](https://www.zhihu.com/equation?tex=J%5E%7B%5Cpi%7D%28%5Cpsi%29)的含义就是在当前近似的环境模型下，采用策略函数![\pi](https://www.zhihu.com/equation?tex=%5Cpi)，其T步路径（服从由![\pi](https://www.zhihu.com/equation?tex=%5Cpi)生成的路径分布）的累计损失的期望。

论文里的策略函数就很简单地采用![u_t=\pi_{\theta}(x_t)=Ax_t+B](https://www.zhihu.com/equation?tex=u_t%3D%5Cpi_%7B%5Ctheta%7D%28x_t%29%3DAx_t%2BB)，这种线性函数能让![p(x_t,u_{t})](https://www.zhihu.com/equation?tex=p%28x_t%2Cu_%7Bt%7D%29)仍为高斯分布。在课上教授说也可以用高斯径向基网络（RBF network）（想了解一下的可以看博客：[http://blog.csdn.net/zouxy09/article/details/13297881](https://link.zhihu.com/?target=http%3A//blog.csdn.net/zouxy09/article/details/13297881)

在具体计算![J^{\pi}(\psi)](https://www.zhihu.com/equation?tex=J%5E%7B%5Cpi%7D%28%5Cpsi%29)的梯度时，需要知道![p(x_{t})](https://www.zhihu.com/equation?tex=p%28x_%7Bt%7D%29)，但是其不是一个高斯分布，可从GP的结果中看出（右上）：

![img](https://pic1.zhimg.com/80/v2-a8f335f33e84f282a850be691237f54f_hd.jpg)

那我们就用一个高斯分布来近似它。这样计算![\mathbb{E}_{x_t}(c(x_t))](https://www.zhihu.com/equation?tex=%5Cmathbb%7BE%7D_%7Bx_t%7D%28c%28x_t%29%29)就简单许多，当c是容易计算的，![p(x_{t})](https://www.zhihu.com/equation?tex=p%28x_%7Bt%7D%29)是高斯分布。得到![J^{\pi}(\psi)](https://www.zhihu.com/equation?tex=J%5E%7B%5Cpi%7D%28%5Cpsi%29)的梯度后，可以用[CG（共轭梯度法）](https://link.zhihu.com/?target=http%3A//www.cnblogs.com/walccott/p/4956966.html)或[L-BFGS](https://link.zhihu.com/?target=http%3A//www.hankcs.com/ml/l-bfgs.html)优化算法优化策略。

## **案例分析2：用循环网络近似模型做预测**

在论文《[Recurrent Network Model for Human Dynamic](https://link.zhihu.com/?target=http%3A//www.cv-foundation.org/openaccess/content_iccv_2015/papers/Fragkiadaki_Recurrent_Network_Models_ICCV_2015_paper.pdf)》中，他想要预测人类走路动作，而不是学习控制策略。使用的预测模型为Encoder-Recurrent-Decoder（和RNN差不多）（图一）：

![img](https://pic1.zhimg.com/80/v2-6f674d6e88169a05616cc5a71970e9ce_hd.jpg)

在图二中，预测线条人的网络模型中encoder为两层全连接层，循环层为两层LSTM，decoder为两层全连接层。输出为混合高斯分布的参数，然后在混合高斯分布上采样，得到预测图作为下一网络的输入。

在论文中还将ERD比较了一些常见的机器学习模型：

1. 条件受限玻尔兹曼机（[Conditional Restricted Boltzmann Machines，CRBM](https://link.zhihu.com/?target=http%3A//www.uoguelph.ca/%7Egwtaylor/publications/icml2009/fcrbm_supplementary.pdf)）（如果没接触过，可以先了解一下受限玻尔兹曼机：[http://www.cnblogs.com/kemaswill/p/3203605.html](https://link.zhihu.com/?target=http%3A//www.cnblogs.com/kemaswill/p/3203605.html)（博客））
2. GPs（高斯过程）或GPLVMs（可自行找一下国外的博客）
3. 三层的LSTM（可以学习下NLP）
4. 简单的n-gram（和NLP中word2vec所用的一样）

![img](https://pic3.zhimg.com/80/v2-f9890303a74663b039626b58af35638b_hd.jpg)

效果来看ERD最好，其他几个模型都有各自犯错的地方：如GP容易保持均值的动作，LSTM容易困在某一姿势上，而CRBM会倾向于做出一些夸张的动作。

## **整体模型的问题：**

之前我们用一个环境学习模型（常用大型的神经网络）来对整个环境建模，可以给环境中任何一个状态和动作得出下一状态。但是这种整体的建模存在一些问题：

1. 如果我们有一个比较好的决策模型，那么它会倾向于去探索那些环境模型表示过于乐观的地方：比如之前的爬山模型，环境模型表示向左爬更高一些，则机器会探索到悬崖。虽然能检测错误很好，但是在高维中用很多的坑坑洼洼的地方，机器不可能把所有的坑都探索一遍，再得出很好的策略。（实际上很多情况下，错误的路径比正确的路径要多得多）
2. 需要在大多数情景中建立很好的环境模型，那样策略才能收敛到最优的策略。但有时候，只需要有一在某一情景最好的模型就够我们在这里寻找最优策略（即使这个模型在另一些情况下是有错误的）。
3. （出现以上的情况的原因是）在很多情况下，环境的模型远比策略复杂得多（如拿起杯子的任务中，环境模型的物理模型很复杂，比如需要知道肌肉的运动情况，杯子的材质导致的弯曲等很难完全表述的物理模型，而拿起杯子的动作蛮简单的）。

## **局部模型（local model）：**

既然对环境的整体建模存在问题，那么我们想就对我们现在策略所处的位置进行一个局部的建模。先来看在进行iLQR或者直接反向传播给策略时，需要知道由环境模型给出的![\frac{\partial f}{\partial x_t}](https://www.zhihu.com/equation?tex=%5Cfrac%7B%5Cpartial+f%7D%7B%5Cpartial+x_t%7D)和![\frac{\partial f}{\partial u_t}](https://www.zhihu.com/equation?tex=%5Cfrac%7B%5Cpartial+f%7D%7B%5Cpartial+u_t%7D)，若采用局部建模的想法，那也就是只要知道在局部的![\frac{\partial f}{\partial x_t}](https://www.zhihu.com/equation?tex=%5Cfrac%7B%5Cpartial+f%7D%7B%5Cpartial+x_t%7D)和![\frac{\partial f}{\partial u_t}](https://www.zhihu.com/equation?tex=%5Cfrac%7B%5Cpartial+f%7D%7B%5Cpartial+u_t%7D)（即![f](https://www.zhihu.com/equation?tex=f)局部的变化情况）就可以了进行反向传播了。那对于每一个时刻![t](https://www.zhihu.com/equation?tex=t)，我们都对（当前路径处的）![\frac{\partial f}{\partial x_t}](https://www.zhihu.com/equation?tex=%5Cfrac%7B%5Cpartial+f%7D%7B%5Cpartial+x_t%7D)和![\frac{\partial f}{\partial u_t}](https://www.zhihu.com/equation?tex=%5Cfrac%7B%5Cpartial+f%7D%7B%5Cpartial+u_t%7D)进行估计，使得![f](https://www.zhihu.com/equation?tex=f)变为关于![(x_t,u_t)](https://www.zhihu.com/equation?tex=%28x_t%2Cu_t%29)的线性函数（就是一阶的泰勒展开对![f](https://www.zhihu.com/equation?tex=f)的线性近似）。同时，我们取策略函数![p(u_t|x_t)](https://www.zhihu.com/equation?tex=p%28u_t%7Cx_t%29)为随时间![t](https://www.zhihu.com/equation?tex=t)而变的**线性高斯策略**。：

![img](https://pic3.zhimg.com/80/v2-c77003a58a400e8f917cc2dd82627278_hd.jpg)

然后我们对局部模型和策略函数的训练方法和**基于模型的强化学习的版本2.0**很相近：

1. 运行当前策略![p(u_t|x_t)](https://www.zhihu.com/equation?tex=p%28u_t%7Cx_t%29)，得到路径的数据集![D={\tau_i}](https://www.zhihu.com/equation?tex=D%3D%7B%5Ctau_i%7D)。
2. 基于路径数据优化局部模型![p(x_{t+1}|x_t,u_t)](https://www.zhihu.com/equation?tex=p%28x_%7Bt%2B1%7D%7Cx_t%2Cu_t%29)。
3. 根据局部模型来训练策略![p(u_t|x_t)](https://www.zhihu.com/equation?tex=p%28u_t%7Cx_t%29)

![img](https://pic2.zhimg.com/80/v2-a46e370e5335b236b296549cac7c8dfc_hd.jpg)

下面我们对2，3步中间的训练过程详细展开。首先来看

## **第3步对策略![p(u_t|x_t)](https://www.zhihu.com/equation?tex=p%28u_t%7Cx_t%29)的训练：**

因为是线性策略，和之前一样使用iLQR：假设在![t](https://www.zhihu.com/equation?tex=t)步，iLQR的结果为：![\hat x_t,\hat u_t,K_t,k_t](https://www.zhihu.com/equation?tex=%5Chat+x_t%2C%5Chat+u_t%2CK_t%2Ck_t)，那么近似的最优动作为![u_t=K_t(x_t-\hat x_t)+k_t+\hat u_t](https://www.zhihu.com/equation?tex=u_t%3DK_t%28x_t-%5Chat+x_t%29%2Bk_t%2B%5Chat+u_t)。因此照iLQR的结果，我们的策略函数应该是：

**版本0.5**：![p(u_t|x_t)=\delta(u_t=\hat u_t)](https://www.zhihu.com/equation?tex=p%28u_t%7Cx_t%29%3D%5Cdelta%28u_t%3D%5Chat+u_t%29)，即只能取iLQR得到的动作

但就像我们之前常说的iLQR是输出一套动作，无法应对偏差，要用replanning的技巧。因此：

**版本1.0**：![p(u_t|x_t)=\delta(u_t=K_t(x_t-\hat x_t)+k_t+\hat u_t)](https://www.zhihu.com/equation?tex=p%28u_t%7Cx_t%29%3D%5Cdelta%28u_t%3DK_t%28x_t-%5Chat+x_t%29%2Bk_t%2B%5Chat+u_t%29)

虽然这样输出的就是近似的最优策略，但是这种策略是确定性的，导致每一次从同一出发点出发时的采样路径都相同，不利于之后的局部模型近似。因此我们每次决策时加一点噪声，使得采样路径多样：

**版本2.0**：![p(u_t|x_t)=\mathcal N(K_t(x_t-\hat x_t)+k_t+\hat u_t,\varSigma_t)](https://www.zhihu.com/equation?tex=p%28u_t%7Cx_t%29%3D%5Cmathcal+N%28K_t%28x_t-%5Chat+x_t%29%2Bk_t%2B%5Chat+u_t%2C%5CvarSigma_t%29)

那么有个问题是我们怎么取![\varSigma_t](https://www.zhihu.com/equation?tex=%5CvarSigma_t)（即噪声大小）呢？有论文表示，我们应该取![\varSigma_t=Q_{u_t,u_t}^{-1}](https://www.zhihu.com/equation?tex=%5CvarSigma_t%3DQ_%7Bu_t%2Cu_t%7D%5E%7B-1%7D)，其中

![img](https://pic3.zhimg.com/80/v2-bde3283845a3622dde69829ed2e62561_hd.jpg)

这种取法也是有实际意义的：如果![Q_{u_t,u_t}](https://www.zhihu.com/equation?tex=Q_%7Bu_t%2Cu_t%7D)很大，那表示改变![u_t](https://www.zhihu.com/equation?tex=u_t)的话，会对积累损失：![Q](https://www.zhihu.com/equation?tex=Q)值有较大影响。这样的话，为了让我们的策略的正确性有保障，我们在加噪声时要很小心，加一点就够了。反之，![Q_{u_t,u_t}](https://www.zhihu.com/equation?tex=Q_%7Bu_t%2Cu_t%7D)很小时，反正![u_t](https://www.zhihu.com/equation?tex=u_t)对Q的影响很小，我们就可以尽情地去探索而且不影响结果，噪声加大，因此策略就显得随机。

实际上有论文表示：我们在用标准的LQR或者iLQR时，优化的目标问题是![\min \sum_{t=1}^{T}c(x_t,u_t)](https://www.zhihu.com/equation?tex=%5Cmin+%5Csum_%7Bt%3D1%7D%5E%7BT%7Dc%28x_t%2Cu_t%29)，而在使用版本2.0线性高斯策略（有噪声的iLQR），优化的目标问题变为：![\min \sum_{t=1}^{T} E_{(x_t,u_t)\sim p(x_t,u_t)}[c(x_t,u_t)-\mathcal H(p(u_t|x_t))]](https://www.zhihu.com/equation?tex=%5Cmin+%5Csum_%7Bt%3D1%7D%5E%7BT%7D+E_%7B%28x_t%2Cu_t%29%5Csim+p%28x_t%2Cu_t%29%7D%5Bc%28x_t%2Cu_t%29-%5Cmathcal+H%28p%28u_t%7Cx_t%29%29%5D)，其中![\mathcal H(p(u_t|x_t))=E_{p(u_t|x_t)}(\frac{1}{log(p(u_t|x_t))})](https://www.zhihu.com/equation?tex=%5Cmathcal+H%28p%28u_t%7Cx_t%29%29%3DE_%7Bp%28u_t%7Cx_t%29%7D%28%5Cfrac%7B1%7D%7Blog%28p%28u_t%7Cx_t%29%29%7D%29)为熵。那么之前的解有噪声的iLQR且![\varSigma_t=Q_{u_t,u_t}^{-1}](https://www.zhihu.com/equation?tex=%5CvarSigma_t%3DQ_%7Bu_t%2Cu_t%7D%5E%7B-1%7D)就是让熵最大的解：不损害累计损失同时尽可能地做出随机的动作。

## **第2步对局部环境模型的拟合：**

对于收集到的路径数据集，每一个路径的每一个时刻的状态转移为![\{(x_t,u_t,x_{t+1})_i\}](https://www.zhihu.com/equation?tex=%5C%7B%28x_t%2Cu_t%2Cx_%7Bt%2B1%7D%29_i%5C%7D)，对每一个时刻的状态转移（局部模型）可以做简单的线性拟合：

**版本1.0：**在每一步![t](https://www.zhihu.com/equation?tex=t)都用线性回归进行拟合：![p(x_{t+1}|x_t,u_t)=\mathcal N(A_tx_t+B_tu_t+c,N_t)](https://www.zhihu.com/equation?tex=p%28x_%7Bt%2B1%7D%7Cx_t%2Cu_t%29%3D%5Cmathcal+N%28A_tx_t%2BB_tu_t%2Bc%2CN_t%29)，其中![A_t,B_t](https://www.zhihu.com/equation?tex=A_t%2CB_t)的含义是![\frac{\partial f}{\partial x_t}](https://www.zhihu.com/equation?tex=%5Cfrac%7B%5Cpartial+f%7D%7B%5Cpartial+x_t%7D)和![\frac{\partial f}{\partial u_t}](https://www.zhihu.com/equation?tex=%5Cfrac%7B%5Cpartial+f%7D%7B%5Cpartial+u_t%7D)。

事实上我们进一步地可以做贝叶斯线性回归（也就是对模型参数做极大后验拟然）：

**版本2.0：**在每一步![t](https://www.zhihu.com/equation?tex=t)，进行贝叶斯线性回归：![\hat \theta=arg\max_{\theta}p(\theta|\{(x_t,u_t,x_{t+1})_i\})](https://www.zhihu.com/equation?tex=%5Chat+%5Ctheta%3Darg%5Cmax_%7B%5Ctheta%7Dp%28%5Ctheta%7C%5C%7B%28x_t%2Cu_t%2Cx_%7Bt%2B1%7D%29_i%5C%7D%29)

但在计算时，需要参数的先验概率，而这个可以用之前的整体模型（如GP，NN，GMM）得到。

## **保持与原路径的距离：**

现在我们对局部模型进行了线性高斯拟合，但问题是原来的模型是一个非线性的，我们对它的线性近似也就是在路径点上的一阶泰勒展开近似，只是在路径点的局部领域内能比较好的线性近似，如果离路径点远的话，模型的偏差就较大，而且这种偏差会累积：

![img](https://pic2.zhimg.com/80/v2-d25b81ddfc130f5e198ba71b03774868_hd.jpg)

因此我们希望基于局部模型训练后的策略或者说是最优路径能保持与原路径的距离，在原路径点的局部领域内，在这样领域中对局部模型的线性近似是比较准确的，做出的策略也是比较符合真实模型的。

那么如何来衡量原始的路径分布![\overline p(\tau)](https://www.zhihu.com/equation?tex=%5Coverline+p%28%5Ctau%29)与现在最优的路径![p(\tau)](https://www.zhihu.com/equation?tex=p%28%5Ctau%29)分布间的差别呢？实际上是用KL散度：![D_{KL}(p(\tau)||\overline p(\tau))=E_{p(\tau)}(log(p(\tau))-log(\overline p(\tau)))](https://www.zhihu.com/equation?tex=D_%7BKL%7D%28p%28%5Ctau%29%7C%7C%5Coverline+p%28%5Ctau%29%29%3DE_%7Bp%28%5Ctau%29%7D%28log%28p%28%5Ctau%29%29-log%28%5Coverline+p%28%5Ctau%29%29%29)，其中![p(\tau)=p(x_1)\prod_{t=1}^Tp(u_t|x_t)p(x_{t+1}|x_t,u_t)](https://www.zhihu.com/equation?tex=p%28%5Ctau%29%3Dp%28x_1%29%5Cprod_%7Bt%3D1%7D%5ETp%28u_t%7Cx_t%29p%28x_%7Bt%2B1%7D%7Cx_t%2Cu_t%29)，而最优策略![p(x_{t+1}|x_t,u_t)=\mathcal N(A_tx_t+B_tu_t+c,N_t)](https://www.zhihu.com/equation?tex=p%28x_%7Bt%2B1%7D%7Cx_t%2Cu_t%29%3D%5Cmathcal+N%28A_tx_t%2BB_tu_t%2Bc%2CN_t%29)。

将![p(\tau)](https://www.zhihu.com/equation?tex=p%28%5Ctau%29)和![\overline p(\tau)](https://www.zhihu.com/equation?tex=%5Coverline+p%28%5Ctau%29)的展开式代入KL散度，计算可得：

![img](https://pic1.zhimg.com/80/v2-6db74d86494e13b22b9fbc41230c98b8_hd.jpg)

即为：

![img](https://pic1.zhimg.com/80/v2-debba1004e5c746009e232a2468bbd2e_hd.jpg)

可以看到KL散度的这个表示与线性高斯所优化的目标：![\min \sum_{t=1}^{T} E_{(x_t,u_t)\sim p(x_t,u_t)}[c(x_t,u_t)-\mathcal H(p(u_t|x_t))]](https://www.zhihu.com/equation?tex=%5Cmin+%5Csum_%7Bt%3D1%7D%5E%7BT%7D+E_%7B%28x_t%2Cu_t%29%5Csim+p%28x_t%2Cu_t%29%7D%5Bc%28x_t%2Cu_t%29-%5Cmathcal+H%28p%28u_t%7Cx_t%29%29%5D)相似，只是损失函数变成了![-log\overline p(u_t|x_t)](https://www.zhihu.com/equation?tex=-log%5Coverline+p%28u_t%7Cx_t%29)。那么要让当前最优路径与原路径差别不大，只需要在优化线性高斯策略时，增加一个约束条件：![D_{KL}(p(\tau)||\overline p(\tau)) \le \epsilon](https://www.zhihu.com/equation?tex=D_%7BKL%7D%28p%28%5Ctau%29%7C%7C%5Coverline+p%28%5Ctau%29%29+%5Cle+%5Cepsilon)。

对于有约束条件的优化问题，可以用拉格朗日乘子法（dual gradient descent，DGD）做：

![img](https://pic1.zhimg.com/80/v2-5f4a7608daa8bf62d0fcf4563ee55836_hd.jpg)

算法：交替更新![p](https://www.zhihu.com/equation?tex=p)和![\lambda](https://www.zhihu.com/equation?tex=%5Clambda)：

![img](https://pic4.zhimg.com/80/v2-0b43f2497e1bf8a2c9d9847efe3d80ba_hd.jpg)

其中第一，二步是下面优化问题的解（和之前的线性高斯策略的目标有一样的形式）：

![img](https://pic4.zhimg.com/80/v2-7f5514dee3d1edf515863a18f733cdd3_hd.jpg)

限制两个策略之间的KL散度，无论是对线性高斯或者更复杂的（神经网络）都有很大帮助。而且两个策略之间的KL散度和由其产生的路径分布之间的KL散度是等价的。这种限制策略间的KL距离在之后的**无模型学习**中也是十分有用的。

## **案例分析：用局部模型的机器控制**

在论文《[Learning Contact-Rich Manipulation Skill with Guided Policy Search](https://link.zhihu.com/?target=http%3A//ieeexplore.ieee.org/xpls/icp.jsp%3Farnumber%3D7138994)》中仅仅采用了局部模型的强化学习算法（使用线性高斯策略），来控制机器完成指定的物体插入任务，结果为：

![img](https://pic1.zhimg.com/80/v2-f66f9b6aea853abab0f3f8cfb68555ad_hd.jpg)

可以看出局部模型只需要较少的学习次数，就可以达到比较好的结果。

## **总结：**

本节讨论了**基于模型的强化学习**：在未知模型情况下，想通过对模型的学习与近似，然后使用上节有模型的方法来得到最优策略。关于模型训练数据的收集以及策略的训练方法，我们循序渐进地得到**基于模型的强化学习**的四个版本：

**版本0.5：随机探索取样，学习模型，优化控制**

**版本1.0：用iLQR得到的策略的迭代模型学习**

**版本1.5：用replanning的迭代模型学习**

**版本2.0：用学习模型作为策略，将损失直接反向传播到策略模型中**

关于策略学习模型，可以用高斯线性模型或者RBF等。而环境学习模型可以用GPs，NN，GMM等。

以上是对整体模型的学习，但是整体模型存在问题：有大量的错误情形，真实模型比策略复杂得多。因此希望能直接在当前路径下每一时刻进行局部环境的建模，在对局部环境模型的拟合：

**版本1.0：**在每一步都用线性回归进行拟合：![p(x_{t+1}|x_t,u_t)=\mathcal N(A_tx_t+B_tu_t+c,N_t)](https://www.zhihu.com/equation?tex=p%28x_%7Bt%2B1%7D%7Cx_t%2Cu_t%29%3D%5Cmathcal+N%28A_tx_t%2BB_tu_t%2Bc%2CN_t%29).

**版本2.0：**：在每一步![t](https://www.zhihu.com/equation?tex=t)，进行贝叶斯线性回归：![\hat \theta=arg\max_{\theta}p(\theta|\{(x_t,u_t,x_{t+1})_i\})](https://www.zhihu.com/equation?tex=%5Chat+%5Ctheta%3Darg%5Cmax_%7B%5Ctheta%7Dp%28%5Ctheta%7C%5C%7B%28x_t%2Cu_t%2Cx_%7Bt%2B1%7D%29_i%5C%7D%29).

在对策略的训练：

**版本0.5**：![p(u_t|x_t)=\delta(u_t=\hat u_t)](https://www.zhihu.com/equation?tex=p%28u_t%7Cx_t%29%3D%5Cdelta%28u_t%3D%5Chat+u_t%29)，即只能取iLQR得到的动作

**版本1.0**：![p(u_t|x_t)=\delta(u_t=K_t(x_t-\hat x_t)+k_t+\hat u_t)](https://www.zhihu.com/equation?tex=p%28u_t%7Cx_t%29%3D%5Cdelta%28u_t%3DK_t%28x_t-%5Chat+x_t%29%2Bk_t%2B%5Chat+u_t%29)

**版本2.0**：![p(u_t|x_t)=\mathcal N(K_t(x_t-\hat x_t)+k_t+\hat u_t,\varSigma_t)](https://www.zhihu.com/equation?tex=p%28u_t%7Cx_t%29%3D%5Cmathcal+N%28K_t%28x_t-%5Chat+x_t%29%2Bk_t%2B%5Chat+u_t%2C%5CvarSigma_t%29)，![\varSigma_t=Q_{u_t,u_t}^{-1}](https://www.zhihu.com/equation?tex=%5CvarSigma_t%3DQ_%7Bu_t%2Cu_t%7D%5E%7B-1%7D)

但是如果优化后的策略的路径与原路径相差比较大时，局部环境的偏差就较大，因此还希望两个路径分布间的距离要小，那就对原优化问题增加约束条件：![D_{KL}(p(\tau)||\overline p(\tau)) \le \epsilon](https://www.zhihu.com/equation?tex=D_%7BKL%7D%28p%28%5Ctau%29%7C%7C%5Coverline+p%28%5Ctau%29%29+%5Cle+%5Cepsilon)。用拉格朗日乘子法可解。

在下一节中，我们会将之前学的模仿学习，优化控制，以及模型学习整合起来。

<div STYLE="page-break-after: always;"></div>

#**第四章：对优化控制的模仿学习**

先来回顾一下之前一节我们所学内容：1.训练整体模型（e.g.GPs），基于整体模型的RL的四个版本。2.训练局部模型（线性模型），以及线性高斯策略，和有KL距离限制的路径优化。3.我们也可以用贝叶斯线性回归来将整体模型和局部模型结合起来。

之前的策略大多是用LQR得出的，但是**直接的策略函数**（模型）![\pi_{\theta}(u_t|x_t)](https://www.zhihu.com/equation?tex=%5Cpi_%7B%5Ctheta%7D%28u_t%7Cx_t%29)会有更好的性质：1.不需要replanning，也就更快速。2.有更好的泛化能力。（教授讲了一个接球的例子：如果完全要考虑环境模型，那球的旋转，风速等都是复杂计算的，而行动策略相对简单，只要盯着球，让视线和球保持直线就可以了。而且学会了这种策略，也可以用来接其他的东西。）

## **对策略优化的配点法（Collocation method）**

## **直接梯度传入的问题**

关于得到直接的策略函数，在上一节的**基于模型的强化学习的版本2.0**（直接将梯度传入策略）时，已经提及了：

![img](https://pic2.zhimg.com/80/v2-0ef2a07035f0f288f338b6a1851c1254_hd.jpg)

但是这样的梯度反向传播是存在一些小问题的：前面的梯度由于是后面所有惩罚函数的梯度累加，因此会比后面的梯度大（尤其当![f(x_t,u_t)](https://www.zhihu.com/equation?tex=f%28x_t%2Cu_t%29)梯度不小时）；而当![f(x_t,u_t)](https://www.zhihu.com/equation?tex=f%28x_t%2Cu_t%29)的梯度较小时，又会出现梯度弥散现象，即后面的惩罚函数对前面的梯度影响很小。这种现象在大型的RNN中也存在（用LSTM结构可解决）。前一种情况其实和之前在优化控制的LQR时的**shooting method（打靶法）**是相似的，即路径受初始动作的影响很大，如下图。但现在我们不能再使用LQR类似的二阶方法了，因为我们的策略参数随步骤是倍数增加。

![img](https://pic3.zhimg.com/80/v2-2a6499fd826f45c37ee593d8e57942bf_hd.jpg)

## **配点法（Collocation method）**

之前提到过的**配点法（Collocation method）**能够解决打靶法（shooting method）的对初值敏感的问题。在路径优化的具体形式为：

![img](https://pic1.zhimg.com/80/v2-ebf5953236943aa9db95c11cdf28ad39_hd.jpg)

即将原来的对![\{u_t\}](https://www.zhihu.com/equation?tex=%5C%7Bu_t%5C%7D)进行优化的无限制优化问题转变为，对![\{u_t\}\{x_t\}](https://www.zhihu.com/equation?tex=%5C%7Bu_t%5C%7D%5C%7Bx_t%5C%7D)联合优化的限制问题。转换后的效果为：

![img](https://pic2.zhimg.com/80/v2-6d1c300b7416c2febed11bac3539b9aa_hd.jpg)

现在我们希望能拥有一个策略模型![\pi_{\theta}(u_t|x_t)](https://www.zhihu.com/equation?tex=%5Cpi_%7B%5Ctheta%7D%28u_t%7Cx_t%29)，那么优化问题的Collocation method就变为：

![img](https://pic3.zhimg.com/80/v2-115082823499f9b8e046affddb3c74f3_hd.jpg)

对于有限制的优化问题，可以用上节的**拉格朗日乘子法（对偶梯度下降，DGD）**，其一般的优化过程为：交替更新 ![x](https://www.zhihu.com/equation?tex=x) 和 ![\lambda](https://www.zhihu.com/equation?tex=%5Clambda) 

![img](https://pic1.zhimg.com/80/v2-0ee621de892573d16c4987d8bcaa354f_hd.jpg)

实际上我们还可以对拉格朗日函数做一些小调整（加入一个平方项），来让它能更稳定地收敛（这个技巧和BADMM augmented Lagrangians相似）：

![img](https://pic1.zhimg.com/80/v2-fcd69ae077c04af28d99f4ad1cec58f1_hd.jpg)

现在回到我们的路径优化问题：

![img](https://pic1.zhimg.com/80/v2-398ff1ceb50beede3904dba96978c0dc_hd.jpg)

其中路径![\tau=(x_1,u_1...x_T,u_T)](https://www.zhihu.com/equation?tex=%5Ctau%3D%28x_1%2Cu_1...x_T%2Cu_T%29)。那么一般的优化过程具体为：

![img](https://pic4.zhimg.com/80/v2-8d698fefd9a6e47ec99c166f837474f0_hd.jpg)

即交替地更新路径![\tau](https://www.zhihu.com/equation?tex=%5Ctau)和策略模型的参数![\theta](https://www.zhihu.com/equation?tex=%5Ctheta)，以及拉格朗日函数参数。其中在选择最优路径时，和以往一样使用iLQR，只是累积惩罚函数变为![\overline L(\tau,\theta,\lambda)](https://www.zhihu.com/equation?tex=%5Coverline+L%28%5Ctau%2C%5Ctheta%2C%5Clambda%29)。而优化策略模型参数![\theta](https://www.zhihu.com/equation?tex=%5Ctheta)时，就用SGD（梯度的反向传播）即可。

## **另一种视角看路径优化过程**

现在我们对这个路径优化过程换一种眼光来看：在第2步，观察![\overline L(\tau,\theta,\lambda)](https://www.zhihu.com/equation?tex=%5Coverline+L%28%5Ctau%2C%5Ctheta%2C%5Clambda%29)对![\theta](https://www.zhihu.com/equation?tex=%5Ctheta)的表达式，可以看出，优化![\theta](https://www.zhihu.com/equation?tex=%5Ctheta)就是缩小![\pi_{\theta}(x_t)](https://www.zhihu.com/equation?tex=%5Cpi_%7B%5Ctheta%7D%28x_t%29)与控制优化结果![u_t](https://www.zhihu.com/equation?tex=u_t)之间差异，因此这一步和监督学习（或是模仿学习）十分相像，以第一步的优化控制得到的![x_t](https://www.zhihu.com/equation?tex=x_t)和标记![u_t](https://www.zhihu.com/equation?tex=u_t)作为专家数据，来让策略模型进行监督学习。

然后再仔细观察一下第一步：在优化控制的目标![\overline L(\tau,\theta,\lambda)](https://www.zhihu.com/equation?tex=%5Coverline+L%28%5Ctau%2C%5Ctheta%2C%5Clambda%29)中不光有之前优化控制的目标![c(\tau)](https://www.zhihu.com/equation?tex=c%28%5Ctau%29)，而且有优化控制优化的![u_t](https://www.zhihu.com/equation?tex=u_t)和策略模型给出的在路径点![x_t](https://www.zhihu.com/equation?tex=x_t)下的动作之间的距离，也就是说第一步优化时不但要让累积惩罚最小，而且也要顾及到它的“学生”：策略模型，需要采取的动作和学生会采取的动作不能相差太多，这样有利于增加模仿学习步的可学习性。

教授讲了个例子来理解这种可学习性：假如我们目标是走出大厅，然后专家会选择走右边的门，因为左边的门有人站着，但是那个人是被墙挡住的，不能直接观测到。那么如果专家径直地选择往右走的话，得到的数据集和学生的路径不匹配，不易学习：在学生学习时会不明所以，因为学生并看不到左边的人，看起来左边和右边情况一样，不知道为什么往右走，而且专家当前的最优动作，放到学生的路径分布中，可能反而累积损失大，这样的准确度和泛化能力都很弱。因此专家应该顾及到学生，可以采取先向左走一点来看看情况之类的动作，来增加可学习性。

## **更一般的有限制的路径优化（Guided policy search，GPS）**

将之前的路径优化过程的要点提取出来，可以得到一般的路径优化步骤：

![img](https://pic1.zhimg.com/80/v2-c925565f52f305a3d5b6c6195f0f994e_hd.jpg)

在具体问题时，其中需要明确的是：![p(\tau)](https://www.zhihu.com/equation?tex=p%28%5Ctau%29)的形式（e.g.线性函数，线性高斯，GPs），替代惩罚函数![\tilde c(x_t,u_t)](https://www.zhihu.com/equation?tex=%5Ctilde+c%28x_t%2Cu_t%29)（e.g.拉格朗日函数），![p(\tau)](https://www.zhihu.com/equation?tex=p%28%5Ctau%29)的优化方法（e.g.iLQR），给![\pi_{\theta}(u_t|x_t)](https://www.zhihu.com/equation?tex=%5Cpi_%7B%5Ctheta%7D%28u_t%7Cx_t%29)的监督学习目标（e.g.对![p(\tau)](https://www.zhihu.com/equation?tex=p%28%5Ctau%29)采样）。

比如说，对于随机情况，**Stochastic (Gaussian) GPS with local models**：

![img](https://pic1.zhimg.com/80/v2-fc28c15472f8120142db244fdb45198b_hd.jpg)

那么对它的训练（第一步）可以完全按照局部模型的优化过程来：

![img](https://pic1.zhimg.com/80/v2-d2623f3dc61fb313856b77c6aa48ab84_hd.jpg)

其中多的部分是对![p(\tau)](https://www.zhihu.com/equation?tex=p%28%5Ctau%29)采样得到的数据也用来训练策略模型![\pi_{\theta}(u_t|o_t)](https://www.zhihu.com/equation?tex=%5Cpi_%7B%5Ctheta%7D%28u_t%7Co_t%29)（第2步），而在优化![p(u_t|x_t)](https://www.zhihu.com/equation?tex=p%28u_t%7Cx_t%29)（线性高斯）也会顾及策略模型（通过调整替代惩罚函数）。

**Input Remapping Trick：**细心的同学会发现，策略模型中的输入时![o_t](https://www.zhihu.com/equation?tex=o_t)（观测值），而非优化控制中的输入![x_t](https://www.zhihu.com/equation?tex=x_t)（状态）。这是为了让机器能处理真实的情况，以像素作为输入，但是在训练阶段，它学习的专家是以较低维的状态作为输入（因此专家行为能较容易地做到正确），这样在模仿学习后的真实测试环节，根据真实像素来选择行为，实现实时的端到端的控制，而且这种输入转换也能适应一些泛化情形。

![img](https://pic4.zhimg.com/80/v2-032cb6080900d42f8fb0a3e5c0d33359_hd.jpg)

**案例分析：**论文《[End-to-End Training of Deep Visuomotor Policies](https://link.zhihu.com/?target=http%3A//www.jmlr.org/papers/volume17/15-522/source/15-522.pdf)》就是用了这个过程，其中讲了更多这个过程的细节部分：在做优化控制时（训练期间），需要用到![\pi_{\theta}(u_t|x_t)](https://www.zhihu.com/equation?tex=%5Cpi_%7B%5Ctheta%7D%28u_t%7Cx_t%29)的均值，但是我们的策略模型（CNN）只输出了![\pi_{\theta}(u_t|o_t)](https://www.zhihu.com/equation?tex=%5Cpi_%7B%5Ctheta%7D%28u_t%7Co_t%29)的均值和方差，而状态与观测的关系![p(o_t|x_t)](https://www.zhihu.com/equation?tex=p%28o_t%7Cx_t%29)其实是不知道的，这时他假定![x_t](https://www.zhihu.com/equation?tex=x_t)和![\pi_{\theta}(u_t|o_t)](https://www.zhihu.com/equation?tex=%5Cpi_%7B%5Ctheta%7D%28u_t%7Co_t%29)的均值成线性关系，那么只需要像训练局部模型一样，以![\{x_t^i,E_{\pi_{\theta}(u_t|o_t)^i}(u_t)\}](https://www.zhihu.com/equation?tex=%5C%7Bx_t%5Ei%2CE_%7B%5Cpi_%7B%5Ctheta%7D%28u_t%7Co_t%29%5Ei%7D%28u_t%29%5C%7D)为数据，训练![\pi_{\theta}(u_t|x_t)](https://www.zhihu.com/equation?tex=%5Cpi_%7B%5Ctheta%7D%28u_t%7Cx_t%29)得到一个线性高斯的策略模型，而数据完全可以通过实际采样得到。此外他还讲了策略模型（CNN）的预训练（第一层卷积层用分类任务的，然后让CNN去做一些预测任务来预训练前几层）和优化控制路径的预训练技巧（用之前讲的局部模型过程，而先不考虑CNN）。以及对路径初始点的多次随机采样来得到![p(\tau)](https://www.zhihu.com/equation?tex=p%28%5Ctau%29)，有助于策略模型的泛化能力。

## **用DAgger模仿学习优化控制（Imitating optimal control with DAgger）**

在之前的模仿学习中的蒙特卡洛树搜索案例分析时，已经看到过可以用DAgger方法来对MCTS的结果进行模仿学习，具体的DAgger过程复述：

![img](https://pic3.zhimg.com/80/v2-8160a67cabf31f088589140617620340_hd.jpg)

**DAgger方法中的问题**：在第2步中，如果模型仅训练了少量的循环次数，那运行![\pi_{\theta}(u_t|o_t)](https://www.zhihu.com/equation?tex=%5Cpi_%7B%5Ctheta%7D%28u_t%7Co_t%29)时，机器会经常犯错，包括一些惩罚值很高的错误，能直接使训练回合结束。这样在运行时（比如自动驾驶），需要人来立马改正严重的错误，因此会消耗人力，而且学习效果不佳（有效的训练时长较短）。因此我们对DAgger方法进行改善，使得第2步所运行得策略有更高的正确性，但是为了避免路径分布不匹配的问题，所运行的策略应该与![\pi_{\theta}(u_t|o_t)](https://www.zhihu.com/equation?tex=%5Cpi_%7B%5Ctheta%7D%28u_t%7Co_t%29)相差小。为此我们对第2步做出以下修改：

**Imitating MPC: PLATO algorithm**

![img](https://pic4.zhimg.com/80/v2-8daa6d67e7b1eab05718c9e276a07090_hd.jpg)

其中第2步实际运行采样的策略![\hat \pi_{\theta}(u_t|x_t)=\mathcal N(K_tx_t+k_t,\varSigma_t)](https://www.zhihu.com/equation?tex=%5Chat+%5Cpi_%7B%5Ctheta%7D%28u_t%7Cx_t%29%3D%5Cmathcal+N%28K_tx_t%2Bk_t%2C%5CvarSigma_t%29)，为线性高斯策略，是以下优化问题的解：

![img](https://pic4.zhimg.com/80/v2-25e4bf5393abd48d9195883b2ec71a89_hd.jpg)

这个其实和局部模型中有限制的优化问题的形式是一样的，因此完全可以用上节有限制的优化问题的方法来求解。而且注意到这里也用了**Input Remapping Trick**，策略模型的输入是真实图像，而优化问题的策略是以状态为输入。

比如说在自动驾驶的实际运行中（训练阶段），在t时刻，机器接受到状态![x_t](https://www.zhihu.com/equation?tex=x_t)和观测![o_t](https://www.zhihu.com/equation?tex=o_t)，那我们的策略模型会给出动作分布![\pi_{\theta}(u_t|o_t)](https://www.zhihu.com/equation?tex=%5Cpi_%7B%5Ctheta%7D%28u_t%7Co_t%29)，然而![\hat \pi_{\theta}(u_t|x_t)](https://www.zhihu.com/equation?tex=%5Chat+%5Cpi_%7B%5Ctheta%7D%28u_t%7Cx_t%29)会根据当前状态和之后的总体损失来做决策，但是其动作分布也要和![\pi_{\theta}(u_t|o_t)](https://www.zhihu.com/equation?tex=%5Cpi_%7B%5Ctheta%7D%28u_t%7Co_t%29)保持一定的一致性，比如前方遇到一棵树，![\pi_{\theta}(u_t|o_t)](https://www.zhihu.com/equation?tex=%5Cpi_%7B%5Ctheta%7D%28u_t%7Co_t%29)说要直行，但![\hat \pi_{\theta}(u_t|x_t)](https://www.zhihu.com/equation?tex=%5Chat+%5Cpi_%7B%5Ctheta%7D%28u_t%7Cx_t%29)知道前方有树需绕行，但是不能直接拐弯，而是渐渐地偏离直行，使得能避免很高的惩罚，同时与![\pi_{\theta}(u_t|o_t)](https://www.zhihu.com/equation?tex=%5Cpi_%7B%5Ctheta%7D%28u_t%7Co_t%29)的路径分布有一定匹配。

## **DAgger vs GPS**

DAgger方法中不需要有适应性的专家，因为只是在第二步中改善了运行策略![\hat \pi_{\theta}(u_t|x_t)](https://www.zhihu.com/equation?tex=%5Chat+%5Cpi_%7B%5Ctheta%7D%28u_t%7Cx_t%29)，但是真正标记![D_{\pi}](https://www.zhihu.com/equation?tex=D_%7B%5Cpi%7D)数据集的专家系统并不会顾及到策略模型![\pi_{\theta}(u_t|o_t)](https://www.zhihu.com/equation?tex=%5Cpi_%7B%5Ctheta%7D%28u_t%7Co_t%29)。这样的好处是：任何的专家系统都可以用来标记样本（如人类，优化控制）。但坏处是：它总是假设专家行为对于策略模型是可学习的，即策略模型在进行模仿学习时的loss不会很大。但是这一点有时候是无法保证的，尤其是有遮挡的情况（像之前解释可学习性用的例子，专家的![x_t](https://www.zhihu.com/equation?tex=x_t)，和学习用的![o_t](https://www.zhihu.com/equation?tex=o_t)不一致）。

而GPS的专家行为能体现出对“学生”的适应性，这样专家会不断地因学生而改变，也就不需要假设开始时的模仿学习时的loss不会很大。

## **向优化控制的模仿学习的优势（Why imitate optimal control?）**

1. 相对的更加简单，方便使用。监督学习有很好的效果，优化控制经常能有好效果，将两者结合也经常能有较好的效果。
2. Input remapping trick:在训练期间能充分利用“状态”所提供的额外信息，有利于策略模型在实际像素图像上做决策。
3. 能够克服梯度直接传入策略时的问题（梯度过大或过小，对初始动作的敏感性）
4. 往往能十分有效地进行采样（e.g.比无模型学习），在真实的物理环境中能够实时地做出行动。

## **总结：基于模型的强化学习的局限**

1. 需要某种动态模型，但往往无法学习得到，因为如果是直接对真实图像建立模型那会十分困难，而且一些模型的物理背景可能十分复杂（比如接球任务）比策略要复杂许多。
2. 学习模型需要时间，数据：比如表示能力很强的神经网络需要大量的数据，而一些data-efficient模型（e.g.GP，线性模型）快速高效但是表示能力不强。
3. 对于模型常常做出一些额外的假设：如希望模型是线性或连续地；对于局部模型，我们需要能够重置系统，即总是能从同一初始状态出发；对于GP模型，我们假定模型函数是光滑的。

因此就引出了无模型的强化学习（Model-free RL），是在尝试和错误中学习，而不需要对环境建模。将会看到无模型RL学习地更慢，但是能有更好的泛化能力。

## **小结：**

本节对优化控制的模仿学习将前几节：模仿学习，优化控制，模型学习结合在了一起。从两个角度出发：

1. 考虑到梯度直接传入策略时的问题，采用配点法（collocation method）将原无限制的优化问题转化为有限制的优化问题，用对偶梯度下降法求解。对求解过程分析，发现第二步其实是策略模型对优化控制结果的监督学习，而第一步的优化控制也考虑到要缩短与策略模型的路径分布之间的距离。
2. 从DAgger方法出发，对第二步进行调整，使得运行策略能够有较好的正确性，防止过大损失，但也保持与学习策略距离。

之后比较了GPS和DAgger：主要看专家策略是否对学生有适应性；说明了对优化控制的模仿学习的优势，以及model-based RL的局限。下一节就开始model-free RL的学习笔记。

注：还有一节课是请Open AI的专家更深入地讲解collocation method，是用inverse dynamic model（反动态模型），即![u_t=f^{-1}(x_t,x_{t+1})](https://www.zhihu.com/equation?tex=u_t%3Df%5E%7B-1%7D%28x_t%2Cx_%7Bt%2B1%7D%29)。但已经是最前沿model-based RL，所以我将不会写那节的笔记，有兴趣的同学可自行了解，我们最好还是快点开始model-free RL的学习。

<div STYLE="page-break-after: always;"></div>

# 第五章：MDP，值迭代，策略迭代

之前的四节讲了优化控制，有模型的强化学习（model-based RL），在最后也看出了有模型的RL的一些弊端：动态模型可能很复杂，模型需要时间和数据进行学习，在建模时人为建立了很多额外假设。因此从这节开始，我们从另个角度出发，用无模型的强化学习（model-free RL）。不再关注对外部环境的建模，而仅关注值函数，策略函数，以此来做迭代优化。

> 这节开始，将部分与Sutton的Reinforcement Learning：An Introduction第二版联合起来做笔记。（如果网上找不到资源可私信我）
> 这节的MDP在Sutton第二版书中的第3节，可以看到![\gamma](https://www.zhihu.com/equation?tex=%5Cgamma)折扣反馈和![T](https://www.zhihu.com/equation?tex=T)步反馈的差别。
> 这节的**值迭代，策略迭代**也称为**动态规划**（dynamic programming）在书第四节，与**蒙特卡洛方法**和**TD方法**成为经典RL的三种基本方法。

在这一节中，我们先考虑一种比较简单的情形：**已知动态模型，而且状态和动作都是离散有限的**。而这个简单的情形，为之后深入的model-free RL的求解过程和思想打下基础。在求解之前，我们先重述一些MDP的术语：

## **马尔科夫决策过程（Markov Decision Process，MDP）**

![img](https://pic3.zhimg.com/80/v2-d17d6d5173f20f1c72b7db7b6a60f0ae_hd.jpg)

![p(x_{t+1}|x_t,u_t)](https://www.zhihu.com/equation?tex=p%28x_%7Bt%2B1%7D%7Cx_t%2Cu_t%29) 表示状态转移函数： ![x_{t+1}](https://www.zhihu.com/equation?tex=x_%7Bt%2B1%7D) 仅与 ![x_t,u_t](https://www.zhihu.com/equation?tex=x_t%2Cu_t) 有关，而与条件 ![x_{t-1},u_{t-1}](https://www.zhihu.com/equation?tex=x_%7Bt-1%7D%2Cu_%7Bt-1%7D) 不相关。

MDP在第一节中就讲过一些了，现在再定义一些记号：

- ![\mathcal{S}](https://www.zhihu.com/equation?tex=%5Cmathcal%7BS%7D)：状态空间
- ![\mathcal{A}](https://www.zhihu.com/equation?tex=%5Cmathcal%7BA%7D)：动作空间
- ![P(r,s'|s,a)](https://www.zhihu.com/equation?tex=P%28r%2Cs%27%7Cs%2Ca%29)：状态转移概率，其中r是奖励值，s'是下一状态。（也可以用![P(s'|s,a)](https://www.zhihu.com/equation?tex=P%28s%27%7Cs%2Ca%29)，以及![R(s)](https://www.zhihu.com/equation?tex=R%28s%29)或![R(a)](https://www.zhihu.com/equation?tex=R%28a%29)或![R(s,a,s')](https://www.zhihu.com/equation?tex=R%28s%2Ca%2Cs%27%29)表示）
- ![a=\pi(s)](https://www.zhihu.com/equation?tex=a%3D%5Cpi%28s%29)确定性策略，或![a\sim \pi(a|s)](https://www.zhihu.com/equation?tex=a%5Csim+%5Cpi%28a%7Cs%29)不确定性策略。

部分观测的MDP（POMDP）：机器只能接收到观测![y](https://www.zhihu.com/equation?tex=y)，而非完整的状态s的信息，其中观测与状态关系：![y\sim P(y|s)](https://www.zhihu.com/equation?tex=y%5Csim+P%28y%7Cs%29)。这种情况，观测y不满足Markov 性质。但POMDP可以转化为MDP：通过重定义：![\tilde s_0=\{y_0\}](https://www.zhihu.com/equation?tex=%5Ctilde+s_0%3D%5C%7By_0%5C%7D)，![\tilde s_1=\{y_0,y_1\}…](https://www.zhihu.com/equation?tex=%5Ctilde+s_1%3D%5C%7By_0%2Cy_1%5C%7D%E2%80%A6)但是这样定义的状态在步数比较多后，就太大了，可以对其窗口进行一些限制：比如只与前n个有关（n-gram）或者用LSTM。

MDP的例子：gym上的FrozenLake-v0，也是这次的作业题，![\mathcal{S}](https://www.zhihu.com/equation?tex=%5Cmathcal%7BS%7D)只有4个状态：起始状态，目标状态，冰面，洞。当达到目标或洞时，回合结束。而达到目标状态，r=1，未达到r=0。![\mathcal{A}](https://www.zhihu.com/equation?tex=%5Cmathcal%7BA%7D)为上下左右四个动作，但是状态转移概率为：有50%概率朝错误方向前进。

## **与MDP相关的问题：**

**策略优化**：选取一种策略使累积奖励最大：![\max_{\pi} E[\sum_{t=0}^{\infty}r_t]](https://www.zhihu.com/equation?tex=%5Cmax_%7B%5Cpi%7D+E%5B%5Csum_%7Bt%3D0%7D%5E%7B%5Cinfty%7Dr_t%5D)

**策略评估**：对于某个策略![\pi](https://www.zhihu.com/equation?tex=%5Cpi)，基于之后一段时间的奖励来给出合理的评估反馈（return）：

- 反馈可以分两种：![\gamma](https://www.zhihu.com/equation?tex=%5Cgamma)折扣反馈：![r_t+\gamma r_{t+1}+\gamma^2 r_{t+2}+…,\gamma \in (0,1)](https://www.zhihu.com/equation?tex=r_t%2B%5Cgamma+r_%7Bt%2B1%7D%2B%5Cgamma%5E2+r_%7Bt%2B2%7D%2B%E2%80%A6%2C%5Cgamma+%5Cin+%280%2C1%29)![\qquad\qquad\qquad\quad](https://www.zhihu.com/equation?tex=%5Cqquad%5Cqquad%5Cqquad%5Cquad)![T](https://www.zhihu.com/equation?tex=T)步反馈：![r_t+r_{t+1}+…+r_{T-1}+V(s_T)](https://www.zhihu.com/equation?tex=r_t%2Br_%7Bt%2B1%7D%2B%E2%80%A6%2Br_%7BT-1%7D%2BV%28s_T%29)，![V(s_T)](https://www.zhihu.com/equation?tex=V%28s_T%29)是已知的对最终状态作出评估的函数，可以是![r_T](https://www.zhihu.com/equation?tex=r_T)
- 基于上面的反馈，可以对策略的表现作出评估：

![\qquad\eta(\pi)= E[\sum_{t=0}^{\infty}\gamma^tr_t]](https://www.zhihu.com/equation?tex=%5Cqquad%5Ceta%28%5Cpi%29%3D+E%5B%5Csum_%7Bt%3D0%7D%5E%7B%5Cinfty%7D%5Cgamma%5Etr_t%5D)

- 但用的比较多的是，关于某一状态的策略![\pi](https://www.zhihu.com/equation?tex=%5Cpi)的值函数，用于**策略评估**：

![\qquad V^{\pi}(s)=E[\sum_{t=0}^{\infty}\gamma^tr_t|s_0=s]](https://www.zhihu.com/equation?tex=%5Cqquad+V%5E%7B%5Cpi%7D%28s%29%3DE%5B%5Csum_%7Bt%3D0%7D%5E%7B%5Cinfty%7D%5Cgamma%5Etr_t%7Cs_0%3Ds%5D)

- 以及在该策略下的状态—动作对的值函数，用在**策略优化（选取greedy策略）时会很好用**：

![\qquad Q^{\pi}(s,a)=E[\sum_{t=0}^{\infty}\gamma^tr_t|s_0=s,a_0=a]](https://www.zhihu.com/equation?tex=%5Cqquad+Q%5E%7B%5Cpi%7D%28s%2Ca%29%3DE%5B%5Csum_%7Bt%3D0%7D%5E%7B%5Cinfty%7D%5Cgamma%5Etr_t%7Cs_0%3Ds%2Ca_0%3Da%5D)

## **值迭代（Value Iteration, VI）：**

我们希望能够通过对策略的评估，在T步反馈时是简单的，然后取最优的策略/动作（贪心）；对于![\gamma](https://www.zhihu.com/equation?tex=%5Cgamma)折扣反馈，我们需要在迭代中提升对策略的评估的正确性，同时取最优策略。

## **有限视野：**

先考虑有限视野的情形，即采用T步反馈：那么问题变为：

![img](https://pic4.zhimg.com/80/v2-c5d82febac53961e9f0e87aa5ca3e9c3_hd.jpg)

这里策略![\pi_t](https://www.zhihu.com/equation?tex=%5Cpi_t)是随时间t而变的，![V(s_T)](https://www.zhihu.com/equation?tex=V%28s_T%29)是已知的。在做出这样假设后，![\pi_t](https://www.zhihu.com/equation?tex=%5Cpi_t)与t之前的动作无关：

![img](https://pic3.zhimg.com/80/v2-874696ca9bc73efe5d7a5cd85e66e4fa_hd.jpg)

可依次求解得到单步最大的动作和值函数：

![img](https://pic3.zhimg.com/80/v2-cf798940455fea5d5c3a233be21167aa_hd.jpg)

其中maximize的意思是得到argmax,max对，因此每一步可得到![V_{t-1}(s)](https://www.zhihu.com/equation?tex=V_%7Bt-1%7D%28s%29)用来做上一时间的求解。

用算法表示：

![img](https://pic3.zhimg.com/80/v2-a4cb1c47ac9c542e4d12b151b6cd4c64_hd.jpg)

如果仔细回忆一下，便发现**这个过程其实和第二节笔记的优化控制的LQR一模一样，只是LQR是线性的连续状态空间，能得到![\pi_t(s)](https://www.zhihu.com/equation?tex=%5Cpi_t%28s%29)为s的线性函数，是直接的解析表达式**。而离散有限情况，往往不能有这么好的显式解析表达式。以上算法，循环一次就可以得到分步最优的策略，以及对分步最优策略的正确的T步反馈评估。

## **无限视野有折扣：**

现在我们希望能找到一个策略函数能对任一状态出发最大化有折扣![(\gamma \in[0,1))](https://www.zhihu.com/equation?tex=%28%5Cgamma+%5Cin%5B0%2C1%29%29)的策略评估值![V^{\pi}(s)](https://www.zhihu.com/equation?tex=V%5E%7B%5Cpi%7D%28s%29)。实际上有折扣的评估可以通过以下模型得到：原模型中每一步额外地会以![1-\gamma](https://www.zhihu.com/equation?tex=1-%5Cgamma)的概率达到沉没状态（sink state）（终止，无奖励）。直接先来看算法：

![img](https://pic4.zhimg.com/80/v2-d8d30cc459d39f63803115f66721faf7_hd.jpg)

可以任一选取初始值函数![V^{(0)}(s)](https://www.zhihu.com/equation?tex=V%5E%7B%280%29%7D%28s%29)（选取的影响会随着n增大而减小）。然后将更新步骤与有限视野的Algorithm 1对比，可以发现几乎是一样的更新步骤，因此Algorithm 2第n次循环其实就是n步评估下![V_0](https://www.zhihu.com/equation?tex=V_0)的值，而![V_T(s_T)](https://www.zhihu.com/equation?tex=V_T%28s_T%29)为![V^{(0)}(s)](https://www.zhihu.com/equation?tex=V%5E%7B%280%29%7D%28s%29)。不同的是不同步骤的策略函数![\pi](https://www.zhihu.com/equation?tex=%5Cpi)是同一策略了，在每次迭代中更新；而且每次迭代值函数有折扣。

因此Algorithm 2可以理解为：在第n步循环时，抛弃掉![r_n,r_{n+1},…](https://www.zhihu.com/equation?tex=r_n%2Cr_%7Bn%2B1%7D%2C%E2%80%A6)，只考虑前n步奖励，这样就变成了有限视野的问题。而且由于折扣![\gamma \in[0,1)](https://www.zhihu.com/equation?tex=%5Cgamma+%5Cin%5B0%2C1%29)，这样抛弃的误差为![\epsilon \le r_{max}\gamma^n/(1-\gamma)](https://www.zhihu.com/equation?tex=%5Cepsilon+%5Cle+r_%7Bmax%7D%5Cgamma%5En%2F%281-%5Cgamma%29)，随着迭代步数n的增加而指数阶减小。因此当![n\to\infty](https://www.zhihu.com/equation?tex=n%5Cto%5Cinfty)时，n步评估的![\pi_0,V_0](https://www.zhihu.com/equation?tex=%5Cpi_0%2CV_0)会收敛到最优的策略和值函数。

**从算子的视角看：**

对于有限状态空间，V是![\mathcal{S}](https://www.zhihu.com/equation?tex=%5Cmathcal%7BS%7D)上的函数，其可以表示成![|\mathcal{S}|](https://www.zhihu.com/equation?tex=%7C%5Cmathcal%7BS%7D%7C)维的向量，因此![V\in\mathbb R^{|\mathcal{S}|}](https://www.zhihu.com/equation?tex=V%5Cin%5Cmathbb+R%5E%7B%7C%5Cmathcal%7BS%7D%7C%7D)。

那么对于VI更新步骤，就可以看成![R^{|\mathcal{S}|}](https://www.zhihu.com/equation?tex=R%5E%7B%7C%5Cmathcal%7BS%7D%7C%7D)到自身的映射：![\mathcal{T}:\mathbb R^{|\mathcal{S}|}\to \mathbb R^{|\mathcal{S}|}](https://www.zhihu.com/equation?tex=%5Cmathcal%7BT%7D%3A%5Cmathbb+R%5E%7B%7C%5Cmathcal%7BS%7D%7C%7D%5Cto+%5Cmathbb+R%5E%7B%7C%5Cmathcal%7BS%7D%7C%7D)，称为backup算子（backup operator），具体为：

![\qquad[\mathcal T V ](s) = \max_a E_{s′ | s,a} [r + \gamma V (s′)]](https://www.zhihu.com/equation?tex=%5Cqquad%5B%5Cmathcal+T+V+%5D%28s%29+%3D+%5Cmax_a+E_%7Bs%E2%80%B2+%7C+s%2Ca%7D+%5Br+%2B+%5Cgamma+V+%28s%E2%80%B2%29%5D)

因此更新步骤为：![V^{(n+1)} = \mathcal T V^{(n)}](https://www.zhihu.com/equation?tex=V%5E%7B%28n%2B1%29%7D+%3D+%5Cmathcal+T+V%5E%7B%28n%29%7D)。

容易看出：![||\mathcal T V − \mathcal T W||_{\infty} \le \gamma||V − W||_{\infty}](https://www.zhihu.com/equation?tex=%7C%7C%5Cmathcal+T+V+%E2%88%92+%5Cmathcal+T+W%7C%7C_%7B%5Cinfty%7D+%5Cle+%5Cgamma%7C%7CV+%E2%88%92+W%7C%7C_%7B%5Cinfty%7D)，因此由压缩映射定理，![V^{(n)}](https://www.zhihu.com/equation?tex=V%5E%7B%28n%29%7D)最终会收敛到不动点：对最优策略的正确无限视野折扣评估。

## **策略迭代（Policy Iteration，PI）：**

之前的值迭代的结果是得到了最优的策略，以及对该策略的正确评估。是因为在更新步骤中策略和值函数（评估正确性）是同时提升的。但现在我们想对一般的策略进行评估：

![img](https://pic2.zhimg.com/80/v2-23406a2d917ff46028164e1bec80cbdd_hd.jpg)

使用与之前一样的方法，可以先考虑有限视野情形，发现迭代步骤为：![V_{t} = \mathcal T^{\pi} V_{t+1}](https://www.zhihu.com/equation?tex=V_%7Bt%7D+%3D+%5Cmathcal+T%5E%7B%5Cpi%7D+V_%7Bt%2B1%7D)，其中：

![img](https://pic2.zhimg.com/80/v2-ea4e232c64be60e81f25ec85fa308e13_hd.jpg)

与值迭代相比，只是每一步取最优动作改为了取评估对象![\pi](https://www.zhihu.com/equation?tex=%5Cpi)所给的动作。而且也![\mathcal T^{\pi}](https://www.zhihu.com/equation?tex=%5Cmathcal+T%5E%7B%5Cpi%7D)是满足压缩映射定理的，因此最终会收敛到正确评估。值得注意的是这个不动点![V = \mathcal T^{\pi} V](https://www.zhihu.com/equation?tex=V+%3D+%5Cmathcal+T%5E%7B%5Cpi%7D+V)是一个![|\mathcal{S}|](https://www.zhihu.com/equation?tex=%7C%5Cmathcal%7BS%7D%7C)维线性方程，因此可以有显式解：

![img](https://pic3.zhimg.com/80/v2-f925cdbadd78ec9ab111d3214a2e7cb5_hd.jpg)

从这一点出发，我们避免了对策略![\pi](https://www.zhihu.com/equation?tex=%5Cpi)评估时要做多次循环，而只需要解方程组即可。

这样我们对任一策略![\pi](https://www.zhihu.com/equation?tex=%5Cpi)都能正确评估，那么我们想得到最优策略，只需要在策略![\pi](https://www.zhihu.com/equation?tex=%5Cpi)的评估基础上，每一状态都取能让![V^{\pi}](https://www.zhihu.com/equation?tex=V%5E%7B%5Cpi%7D)最大的动作（贪心）即可：

![img](https://pic2.zhimg.com/80/v2-3bfa73b08a65659b5b172db693d7c618_hd.jpg)

其中![\mathcal G V^{\pi^{(n-1)}}](https://www.zhihu.com/equation?tex=%5Cmathcal+G+V%5E%7B%5Cpi%5E%7B%28n-1%29%7D%7D)为![\pi^{(n)}(s)=argmax_a E_{s′ | s,a}V^{\pi^{(n-1)}}(s')](https://www.zhihu.com/equation?tex=%5Cpi%5E%7B%28n%29%7D%28s%29%3Dargmax_a+E_%7Bs%E2%80%B2+%7C+s%2Ca%7DV%5E%7B%5Cpi%5E%7B%28n-1%29%7D%7D%28s%27%29)。

容易证明![V^{\pi^{(0)}} \le V^{\pi^{(1)}}\le V^{\pi^{(2)}}\le…](https://www.zhihu.com/equation?tex=V%5E%7B%5Cpi%5E%7B%280%29%7D%7D+%5Cle+V%5E%7B%5Cpi%5E%7B%281%29%7D%7D%5Cle+V%5E%7B%5Cpi%5E%7B%282%29%7D%7D%5Cle%E2%80%A6)，因此策略是单调提升的，而且极限情况就是最优策略。

实际上策略迭代是比值迭代的收敛速度要快的（在这本书中有证明：M. L. Puterman. Markov decision processes: discrete stochastic dynamic programming. John Wiley & Sons, 2014.），但这点也会在这次的作业里面很明显地看出来（而且从作业中可以看出，冰面模型的随机性导致了这种差异（扩大了感受野），策略迭代能很敏锐地发现值函数的变化，而值迭代只能一格一格地更新）。

**策略迭代修改版：**

有时候维数大时，或者更一般的无限情况，就无法解线性方程组了，为此我们结合值迭代和策略迭代，可以得到：

![img](https://pic2.zhimg.com/80/v2-d3407788fb8cd8dce7e7348f5c1456c0_hd.jpg)

即在策略评估时，用迭代k步的方法进行评估。实际上，当![k=1](https://www.zhihu.com/equation?tex=k%3D1)时，这个就是值迭代；当![k=\infty](https://www.zhihu.com/equation?tex=k%3D%5Cinfty)时，就是策略迭代。

> 这节动态规划内容我也挺欣赏Sutton书上的条理：
> 他把RL的学习过程分为两块：**策略估计**（policy evaluation）和**策略提升**（policy improvement）。
> 对于第一块策略估计，可以用Bellman等式得到Bellman backup（第7节也会说）：![V_{\pi}(s) \gets \sum_a\pi(a|s)\sum_{s',r}p(s',r|s,a)[r+\gamma V_{\pi}(s')]=\mathcal T^{\pi} V_{\pi}(s)](https://www.zhihu.com/equation?tex=V_%7B%5Cpi%7D%28s%29+%5Cgets+%5Csum_a%5Cpi%28a%7Cs%29%5Csum_%7Bs%27%2Cr%7Dp%28s%27%2Cr%7Cs%2Ca%29%5Br%2B%5Cgamma+V_%7B%5Cpi%7D%28s%27%29%5D%3D%5Cmathcal+T%5E%7B%5Cpi%7D+V_%7B%5Cpi%7D%28s%29)（其中![\gets](https://www.zhihu.com/equation?tex=%5Cgets)表示更新步骤）我们之前证明了这个能收敛到正确的对![\pi](https://www.zhihu.com/equation?tex=%5Cpi)的值估计。
> 对于第二块策略提升，可以证明策略提升定理：
> 如果![Q_{\pi}(s,\pi'(s)) \ge V_{\pi}(s)](https://www.zhihu.com/equation?tex=Q_%7B%5Cpi%7D%28s%2C%5Cpi%27%28s%29%29+%5Cge+V_%7B%5Cpi%7D%28s%29)对任意s成立，那么![\pi'](https://www.zhihu.com/equation?tex=%5Cpi%27)的值函数优于![\pi](https://www.zhihu.com/equation?tex=%5Cpi)的，因此根据提升定理，想要提升策略![\pi](https://www.zhihu.com/equation?tex=%5Cpi)，取贪心策略![\mathcal G V^{\pi^{(n-1)}}](https://www.zhihu.com/equation?tex=%5Cmathcal+G+V%5E%7B%5Cpi%5E%7B%28n-1%29%7D%7D)为![\pi^{(n)}(s)=argmax_a E_{s′ | s,a}V^{\pi^{(n-1)}}(s')](https://www.zhihu.com/equation?tex=%5Cpi%5E%7B%28n%29%7D%28s%29%3Dargmax_a+E_%7Bs%E2%80%B2+%7C+s%2Ca%7DV%5E%7B%5Cpi%5E%7B%28n-1%29%7D%7D%28s%27%29)。两块结合这就是我们的**策略迭代**啊！
> 然而策略迭代需要第一步策略估计比较准确，那么需要做很多步的Bellman backup。
> 那么考虑极端情况，只做一步策略估计，然后立马做greedy策略提升，那么就是我们的**值迭代**了啊！
> 因此Sutton通过一个general的学习过程**策略估计+策略提升，**把策略迭代和值迭代整合起来了，实际上这个general的学习过程将贯穿整个RL基本。

随便说一句，这里虽然策略迭代比值迭代的效果好，但是当环境模型未知时，值迭代（Q-learning）比策略迭代（Sarsa）有一个天然优势，第七节将会讲到。

## **总结**

这一节为无模型的强化学习开了个头，介绍了Markov决策过程（MDP）。引入值函数概念，以此出发来优化策略。对于策略有两种评估方法：![\gamma](https://www.zhihu.com/equation?tex=%5Cgamma)折扣反馈，T步反馈。这节假设有限的状态空间和已知模型。

- 对于有限视野的评估，我们引出了值迭代的思想和方法（**根据对策略的评估：值函数，来选择当前最优的策略，而经过迭代来提升值函数的正确性**）。
- 并且推广到无限的![\gamma](https://www.zhihu.com/equation?tex=%5Cgamma)折扣评估，采用迭代更新值函数和策略的方法，最后收敛到最优策略和对其正确的评估。
- 同时我们从算子的视角更清楚地来看这个过程![V^{(n+1)} = \mathcal T V^{(n)}](https://www.zhihu.com/equation?tex=V%5E%7B%28n%2B1%29%7D+%3D+%5Cmathcal+T+V%5E%7B%28n%29%7D)。

但我们并不满足只对最优策略进行评估，或者不想策略和值函数同时提升。因此我们先对任一策略能做正确评估：相似的迭代的方法，或者解线性方程组![V = \mathcal T^{\pi} V](https://www.zhihu.com/equation?tex=V+%3D+%5Cmathcal+T%5E%7B%5Cpi%7D+V)。

- 基于对上一个策略的正确评估（解线性方程组），用贪心法来得到当前最好的策略。最后也能收敛到最优策略。
- 为了解决更一般的情况，在对当前策略评估时，用迭代的方法得到较优评估。

对于无限状态空间和未知模型情形，下一节会讲另一个无模型RL的方法：策略梯度方法。

这节的作业已经上传到我的GitHub上了：[https://github.com/futurebelongtoML/homework.git](https://link.zhihu.com/?target=https%3A//github.com/futurebelongtoML/homework.git)

是一个很简单的MDP问题：FrozenLake-v0

<div STYLE="page-break-after: always;"></div>

#第六章：策略梯度（Policy Gradient）

在上一节中，我们研究了比较简单的情况下的model-free RL，即已知模型，状态空间和动作空间有限情况。分别对T步反馈和折扣反馈，给出值迭代（VI）算法来得到最优策略和对其正确的值函数，然后以此为基础，提出收敛更快的策略迭代（PI）。

这一节中，我们讨论更一般的情形：未知模型，状态空间和动作空间可以是无限的。讨论这种情况下的model-free RL，而这一节中，将采用策略梯度的方法（Policy Gradient Methods）。将会看到这种方法与下一节的Q-learning是两种十分重要的model-free RL方法。

## **优化问题及一些记号：**

强化学习想解决的问题最终都可以化为一个优化问题，其一般形式为：

![\qquad\operatorname*{maximize}_{\pi}E_{\pi}[expression]](https://www.zhihu.com/equation?tex=%5Cqquad%5Coperatorname%2A%7Bmaximize%7D_%7B%5Cpi%7DE_%7B%5Cpi%7D%5Bexpression%5D)

其中expression可以是：

- 有限视野T步奖励：![\sum_{t=0}^{T-1}r_t](https://www.zhihu.com/equation?tex=%5Csum_%7Bt%3D0%7D%5E%7BT-1%7Dr_t)
- 平均奖励：![\lim_{T\to\infty}\frac{1}{T}\sum_{t=0}^{T-1}r_t](https://www.zhihu.com/equation?tex=%5Clim_%7BT%5Cto%5Cinfty%7D%5Cfrac%7B1%7D%7BT%7D%5Csum_%7Bt%3D0%7D%5E%7BT-1%7Dr_t)
- 无限视野![\gamma](https://www.zhihu.com/equation?tex=%5Cgamma)折扣：![\sum_{t=0}^{\infty}\gamma^t r_t](https://www.zhihu.com/equation?tex=%5Csum_%7Bt%3D0%7D%5E%7B%5Cinfty%7D%5Cgamma%5Et+r_t)
- 可变长度无折扣：![\sum_{t=0}^{T_{terminal}-1}r_t](https://www.zhihu.com/equation?tex=%5Csum_%7Bt%3D0%7D%5E%7BT_%7Bterminal%7D-1%7Dr_t)
- 无限视野无折扣：![\sum_{t=0}^{\infty}r_t](https://www.zhihu.com/equation?tex=%5Csum_%7Bt%3D0%7D%5E%7B%5Cinfty%7Dr_t)

但实际上我们只需考虑有限视野T步奖励就可以了，如果要用其他的标准，只需要对T步奖励做出一些小改动，以及取极限即可（如![\gamma](https://www.zhihu.com/equation?tex=%5Cgamma)折扣就是加上sink state后，T趋于无穷，在上一节中已经看过）。

这样我们的目标为：![\operatorname*{maximize} \eta(\pi)](https://www.zhihu.com/equation?tex=%5Coperatorname%2A%7Bmaximize%7D+%5Ceta%28%5Cpi%29)，其中

![\qquad\eta(\pi)=E[r_0+r_1+...+r_{T-1}|\pi]](https://www.zhihu.com/equation?tex=%5Cqquad%5Ceta%28%5Cpi%29%3DE%5Br_0%2Br_1%2B...%2Br_%7BT-1%7D%7C%5Cpi%5D)

其中路径的概率分布可分解为：

![s_0\sim\mu(s_0);\\a_0\sim\pi(a_0|s_0);\\s_1,r_0\sim P(s_1,r_0|a_0,s_0);\\…;\\ a_{T-1}\sim\pi(a_{T-1}|s_{T-1});\\s_T,r_{T-1}\sim P(s_T,r_{T-1}|a_{T-1},s_{T-1})](https://www.zhihu.com/equation?tex=s_0%5Csim%5Cmu%28s_0%29%3B%5C%5Ca_0%5Csim%5Cpi%28a_0%7Cs_0%29%3B%5C%5Cs_1%2Cr_0%5Csim+P%28s_1%2Cr_0%7Ca_0%2Cs_0%29%3B%5C%5C%E2%80%A6%3B%5C%5C+a_%7BT-1%7D%5Csim%5Cpi%28a_%7BT-1%7D%7Cs_%7BT-1%7D%29%3B%5C%5Cs_T%2Cr_%7BT-1%7D%5Csim+P%28s_T%2Cr_%7BT-1%7D%7Ca_%7BT-1%7D%2Cs_%7BT-1%7D%29)

这一节中（策略梯度方法中），我们设策略是一个带有参数的策略模型（e.g.神经网络）：

对于确定型策略：![a=\pi(s,\theta)](https://www.zhihu.com/equation?tex=a%3D%5Cpi%28s%2C%5Ctheta%29)；对于随机型策略：![a\sim\pi(a|s,\theta)](https://www.zhihu.com/equation?tex=a%5Csim%5Cpi%28a%7Cs%2C%5Ctheta%29)

对于离散（有限）情况：输出每一个动作的概率（用softmax得到）；对于连续情况：输出一个高斯分布的均值和方差（对角元）。

**策略梯度法的整体思想**是对于上述优化问题，我们sample出许多路径，然后想通过梯度下降（上升）法来使好的动作发生的可能性更大。为此我们先要知道怎么得到它的梯度：

## **期望函数的梯度估计（Score Function Gradient Estimator）：**

考虑期望函数：![E_{x\sim p(x|\theta)}(f(x))](https://www.zhihu.com/equation?tex=E_%7Bx%5Csim+p%28x%7C%5Ctheta%29%7D%28f%28x%29%29)，想要计算![\theta](https://www.zhihu.com/equation?tex=%5Ctheta)的梯度：

![img](https://pic1.zhimg.com/80/v2-1e1b7f04138187b9cebdbba43a687a92_hd.jpg)

利用最后一个表达式，对x采样：![x_i\sim p(x|\theta)](https://www.zhihu.com/equation?tex=x_i%5Csim+p%28x%7C%5Ctheta%29)，然后计算![\hat g_i=f(x_i)\nabla_{\theta}\log p(x_i|\theta)](https://www.zhihu.com/equation?tex=%5Chat+g_i%3Df%28x_i%29%5Cnabla_%7B%5Ctheta%7D%5Clog+p%28x_i%7C%5Ctheta%29)，最后对sample的梯度取平均即可（假设p关于![\theta](https://www.zhihu.com/equation?tex=%5Ctheta)连续可导，如果p是神经网络，则这个导数计算过程，其实和分类任务中以cross-entropy或极大拟然为损失函数的梯度是极其相似的）。

实际上这个公式也可以用importance sampling得到（T. Jie and P. Abbeel. 《On a connection between importance sampling and the likelihood ratio policy gradient》. Advances in Neural Information Processing Systems. 2010, pp. 1000–1008.）

我们再对上面得到的公式![\hat g_i=f(x_i)\nabla_{\theta}\log p(x_i|\theta)](https://www.zhihu.com/equation?tex=%5Chat+g_i%3Df%28x_i%29%5Cnabla_%7B%5Ctheta%7D%5Clog+p%28x_i%7C%5Ctheta%29)多做一点解读：![f(x_i)](https://www.zhihu.com/equation?tex=f%28x_i%29)表示了取样点有多好，而![\hat g_i](https://www.zhihu.com/equation?tex=%5Chat+g_i)方向就是按照采样点的好坏来提高sample点的对数概率值。值得注意的是这个公式对于离散的f和x也是有效的，甚至f的具体表达式也不需要知道（黑箱f，无法知道其导数）。

## **策略的梯度估计**

现在把上面的结果运用到我们的强化学习优化问题中，对带参数策略求梯度：

![\qquad\nabla_{\theta}E_{\tau}[R(\tau)]=E_\tau[R(\tau)\nabla_{\theta}\log p(\tau|\theta)]](https://www.zhihu.com/equation?tex=%5Cqquad%5Cnabla_%7B%5Ctheta%7DE_%7B%5Ctau%7D%5BR%28%5Ctau%29%5D%3DE_%5Ctau%5BR%28%5Ctau%29%5Cnabla_%7B%5Ctheta%7D%5Clog+p%28%5Ctau%7C%5Ctheta%29%5D)

其中路径为![\tau=(s_0,a_0,r_0,s_1,a_1,r_1,…,s_{T-1},a_{T-1},r_{T-1},s_{T})](https://www.zhihu.com/equation?tex=%5Ctau%3D%28s_0%2Ca_0%2Cr_0%2Cs_1%2Ca_1%2Cr_1%2C%E2%80%A6%2Cs_%7BT-1%7D%2Ca_%7BT-1%7D%2Cr_%7BT-1%7D%2Cs_%7BT%7D%29)。而路径分布可以分解，从而得到：

![img](https://pic2.zhimg.com/80/v2-532f932393a05e5cea3ae705ea736661_hd.jpg)

可以看到最后的表达式中没有初始分布![\mu(s_0)](https://www.zhihu.com/equation?tex=%5Cmu%28s_0%29)和环境模型![P(s_t,r_{t-1}|a_{t-1},s_{t-1})](https://www.zhihu.com/equation?tex=P%28s_t%2Cr_%7Bt-1%7D%7Ca_%7Bt-1%7D%2Cs_%7Bt-1%7D%29)，因为它们与参数![\theta](https://www.zhihu.com/equation?tex=%5Ctheta)无关，因此我们**根本不需要知道这两者的显示表达式！**因此策略梯度方法是一种model-free的RL。

注：在之前的有模型的强化学习中，也有带参数的策略函数，当时把梯度直接传入策略中，但其讨论的主要对象是确定型策略，因此梯度的求解方法不一样，而且求解时需要知道环境模型P以及奖励函数r的梯度。（实际上策略梯度和把梯度直接传入策略是两种梯度求解方法，两者区别会在第九节深入讨论）

现在我们仅对t'时间的奖励![r_{t'}](https://www.zhihu.com/equation?tex=r_%7Bt%27%7D)做以上过程，可以得到：

![img](https://pic1.zhimg.com/80/v2-45a1afd8b9ac680a91a7b3c3eec651eb_hd.jpg)

把所有的![r_{t'}](https://www.zhihu.com/equation?tex=r_%7Bt%27%7D)累加起来，得：

![img](https://pic2.zhimg.com/80/v2-6dbccef3231fee7e2c5725df7015a232_hd.jpg)

这样实现了求和符号的交换，使得我们可以单独对一个时间点进行讨论： ![t](https://www.zhihu.com/equation?tex=t) 时刻sample的动作会根据之后的return ![\sum_{t'=t}^{T-1} r_{t'}](https://www.zhihu.com/equation?tex=%5Csum_%7Bt%27%3Dt%7D%5E%7BT-1%7D+r_%7Bt%27%7D) 的大小来提升。

## **策略梯度：基线(Baseline)**

实际上，如果用上面公式得到梯度来进行优化，会导致最后策略的方差太大，同时不易收敛到最大值，这是因为公式对每一个采样点都做正向的梯度，想增加每一个采样点的概率。为此我们引入基线![b(s)](https://www.zhihu.com/equation?tex=b%28s%29)，使得比基线（平均水平）好的动作得到鼓励，而低于平均的动作将会抑制：（如果继续往后上课，会发现这个方差减小的原理和reward reshaping有很大关系：![\tilde r(s,a,s')=r(s,a,s')+\gamma V(s')-V(s)](https://www.zhihu.com/equation?tex=%5Ctilde+r%28s%2Ca%2Cs%27%29%3Dr%28s%2Ca%2Cs%27%29%2B%5Cgamma+V%28s%27%29-V%28s%29)）

![img](https://pic3.zhimg.com/80/v2-cae9ce44aa81443e9428ef5bb391d887_hd.jpg)

其中基线为：![b(s_t)\approx E[r_t+r_{t+1}+…+r_{T-1}]](https://www.zhihu.com/equation?tex=b%28s_t%29%5Capprox+E%5Br_t%2Br_%7Bt%2B1%7D%2B%E2%80%A6%2Br_%7BT-1%7D%5D)，是对平均累积奖励的近似（之后会更具体给出）（这里用约等号的原因是，这个平均累积奖励并不是使方差最小的最优基线）

值得注意的是，无论![b(s)](https://www.zhihu.com/equation?tex=b%28s%29)的表达式是怎样，这个减去基线的梯度估计任然是无偏估计，不影响梯度的均值。这是由于可证![E_{\tau}[\nabla_{\theta}\log\pi(a_t|s_t,\theta)b(s_t)]=0](https://www.zhihu.com/equation?tex=E_%7B%5Ctau%7D%5B%5Cnabla_%7B%5Ctheta%7D%5Clog%5Cpi%28a_t%7Cs_t%2C%5Ctheta%29b%28s_t%29%5D%3D0)。但由方差定义可知，减去基线后括号中的平方的均值减小了，因此方差变小了。

接着，我们再对这个梯度估计做一些修正：加入![\gamma](https://www.zhihu.com/equation?tex=%5Cgamma)折扣，来减少延迟的奖励对动作决策的影响（从实验中发现，这样做能让采样的数据更有效）：

![img](https://pic1.zhimg.com/80/v2-06f6514a3a7cb4ef18a144a907cad45d_hd.jpg)

同时基线也要做相应修改：![b(s_t)\approx E[r_t+\gamma r_{t+1}+…+\gamma^{T-1-t}r_{T-1}]](https://www.zhihu.com/equation?tex=b%28s_t%29%5Capprox+E%5Br_t%2B%5Cgamma+r_%7Bt%2B1%7D%2B%E2%80%A6%2B%5Cgamma%5E%7BT-1-t%7Dr_%7BT-1%7D%5D)。

有了这些工作，就可以把策略梯度的算法完整写出：

## **“Vanilla” Policy Gradient Algorithm：**

![img](https://pic2.zhimg.com/80/v2-7e3518a7694888ad8f060259588648b7_hd.jpg)

其中![b(s)](https://www.zhihu.com/equation?tex=b%28s%29)其实是个神经网络，做回归任务，每一步目标是最小化与![R_t](https://www.zhihu.com/equation?tex=R_t)的距离，整体来看就是之前策略累计奖励的动态平均。![\hat A_t=R_t-b(s_t)](https://www.zhihu.com/equation?tex=%5Chat+A_t%3DR_t-b%28s_t%29)被称为优势估计（advantage estimate），衡量了累计回馈与平均水平的差距。之后用梯度来更新策略是很常规的做法。

在实际tensorflow的代码编写中，只需要定义loss function为：

![img](https://pic2.zhimg.com/80/v2-6cbfbb1dffda86798e20b84fa4778262_hd.jpg)

然后让tensorflow自动求导即可。而实际上，有的loss function会同时优化值函数![V(s_t)](https://www.zhihu.com/equation?tex=V%28s_t%29)（如果策略网络与值函数网络有**共用参数**，这是因为策略和值函数都是在读取状态的信息/特征，建立状态和奖励之间联系，这之后会在A3C看到）：

![img](https://pic2.zhimg.com/80/v2-65e35dccbbeb1ae128d9400ed8ad5c0f_hd.jpg)

但对这个式子提到了一个小问题：在更新策略的梯度时，每一步都只会更新一次（否则取样点的概率会比其他高很多，而且迭代一次后策略就改变了），而对值函数![V(s_t)](https://www.zhihu.com/equation?tex=V%28s_t%29)的优化是回归任务，需要迭代多次。（一种方法是两个网络更新采用不同的学习率）

> 仅采用 ![\nabla_{\theta}E[R]=E[\sum_{t=0}^{T-1}\nabla_{\theta}\log\pi(a_t|s_t,\theta)\sum_{t'=t}^{T-1}r_{t'}]](https://www.zhihu.com/equation?tex=%5Cnabla_%7B%5Ctheta%7DE%5BR%5D%3DE%5B%5Csum_%7Bt%3D0%7D%5E%7BT-1%7D%5Cnabla_%7B%5Ctheta%7D%5Clog%5Cpi%28a_t%7Cs_t%2C%5Ctheta%29%5Csum_%7Bt%27%3Dt%7D%5E%7BT-1%7Dr_%7Bt%27%7D%5D) 作为策略梯度的算法称为REINFORCE，它有很好的理论上的收敛性质，但由于高varience，所以学习得很慢。
> 而加入基线baseline后的**Vanilla**算法也被称为REINFORCE with baseline，由于采用 ![\sum_{t'=t}^{T-1} r_{t'}](https://www.zhihu.com/equation?tex=%5Csum_%7Bt%27%3Dt%7D%5E%7BT-1%7D+r_%7Bt%27%7D) 作为return，所以被认为是一种蒙特卡洛方法。

## **值函数角度看策略梯度：**

现在对上面提出的策略梯度公式用更加理论性的眼光来看，首先回顾一下几个值函数的定义：

- Q-function或状态-动作-值函数：

![\qquad Q^{\pi,\gamma}(s,a)=E_{\pi}[r_0+\gamma r_1+...|s_0=s,a_0=a]](https://www.zhihu.com/equation?tex=%5Cqquad+Q%5E%7B%5Cpi%2C%5Cgamma%7D%28s%2Ca%29%3DE_%7B%5Cpi%7D%5Br_0%2B%5Cgamma+r_1%2B...%7Cs_0%3Ds%2Ca_0%3Da%5D)

- 状态-值函数（state-value function）

![\qquad V^{\pi,\gamma}(s)=E_{\pi}[r_0+\gamma r_1+...|s_0=s]](https://www.zhihu.com/equation?tex=%5Cqquad+V%5E%7B%5Cpi%2C%5Cgamma%7D%28s%29%3DE_%7B%5Cpi%7D%5Br_0%2B%5Cgamma+r_1%2B...%7Cs_0%3Ds%5D)

- 优势函数（advantage function）

![\qquad A^{\pi,\gamma}(s,a)=Q^{\pi,\gamma}(s,a)-V^{\pi,\gamma}(s)](https://www.zhihu.com/equation?tex=%5Cqquad+A%5E%7B%5Cpi%2C%5Cgamma%7D%28s%2Ca%29%3DQ%5E%7B%5Cpi%2C%5Cgamma%7D%28s%2Ca%29-V%5E%7B%5Cpi%2C%5Cgamma%7D%28s%29)

之前的优势估计![\hat A_t=R_t-b(s_t)](https://www.zhihu.com/equation?tex=%5Chat+A_t%3DR_t-b%28s_t%29)是对优势函数![A^{\pi,\gamma}(s,a)](https://www.zhihu.com/equation?tex=A%5E%7B%5Cpi%2C%5Cgamma%7D%28s%2Ca%29)的估计。定义了值函数之后就可以将之前结果用值函数（无限的评估视野）来表示。

回想一下**之前的策略梯度：**

![img](https://pic1.zhimg.com/80/v2-59f0e63962dfc470d96058ff1305899c_hd.jpg)

其中第一行是原来推导的策略梯度，带上基线项（仅减小了方差），第二行是加入![\gamma](https://www.zhihu.com/equation?tex=%5Cgamma)折扣。

现在**用值函数表示：**（只是将上式用另一种方式表达，可以证明上面和下面的式子是相等的）

![img](https://pic4.zhimg.com/80/v2-79a1dda878948d3cf9e1aceb2396990a_hd.jpg)

第一行是原始策略梯度用无折扣值函数 ![Q^{\pi}](https://www.zhihu.com/equation?tex=Q%5E%7B%5Cpi%7D) 表示，第二行是加入了基线（用![V^{\pi}(s)](https://www.zhihu.com/equation?tex=V%5E%7B%5Cpi%7D%28s%29)作为基线），于是就用无折扣优势函数 ![A^{\pi}](https://www.zhihu.com/equation?tex=A%5E%7B%5Cpi%7D) 表示，第三行是加入![\gamma](https://www.zhihu.com/equation?tex=%5Cgamma)折扣，就用有折扣的优势函数 ![A^{\pi,\gamma}](https://www.zhihu.com/equation?tex=A%5E%7B%5Cpi%2C%5Cgamma%7D) 表示。

值得注意的几点：

- 优势函数![A^{\pi,\gamma}(s,a)](https://www.zhihu.com/equation?tex=A%5E%7B%5Cpi%2C%5Cgamma%7D%28s%2Ca%29)可以用有限情形的![\hat A_t=R_t-b(s_t)](https://www.zhihu.com/equation?tex=%5Chat+A_t%3DR_t-b%28s_t%29)估计
- 优势函数的一般形式为：![return-V(s)](https://www.zhihu.com/equation?tex=return-V%28s%29)
- 基线的作用其实是取掉之前动作的影响，因为我们想知道的是当前状态下采用某个动作的好坏，而不是当前状态本身的好坏（有之前动作得到），因此要减去状态自身的好坏来去掉之前动作的影响。
- 实际上我们还可以利用值函数来去掉未来动作的影响：用 ![V^{\pi}(s)](https://www.zhihu.com/equation?tex=V%5E%7B%5Cpi%7D%28s%29) 代替之后的反馈（使用下面的return）：

![img](https://pic2.zhimg.com/80/v2-d187f1ec75019ab176f1c6d2ccfb35b7_hd.jpg)

- 因此用这些return可以得到对优势函数的一个好的估计：

![img](https://pic3.zhimg.com/80/v2-741579fdb26dd7d6fa473c5152a38db8_hd.jpg)

- 可以看到，![\hat A^{(1)}_t](https://www.zhihu.com/equation?tex=%5Chat+A%5E%7B%281%29%7D_t)对![A^{\pi,\gamma}](https://www.zhihu.com/equation?tex=A%5E%7B%5Cpi%2C%5Cgamma%7D)的估计的偏差较大（毕竟它把1步之后的reward都用平均的![V](https://www.zhihu.com/equation?tex=V)来去掉了，而baselineV又不是准确的值函数），但是它的方差很小（因为只取采样的![r_t,s_t,s_{t+1}](https://www.zhihu.com/equation?tex=r_t%2Cs_t%2Cs_%7Bt%2B1%7D)三个变动量）（而且实际上在有延迟的reward和V准确的情况下，![\hat A^{(1)}_t](https://www.zhihu.com/equation?tex=%5Chat+A%5E%7B%281%29%7D_t)有利于去除其他动作的影响，准确指出当前动作的好坏）；而![\hat A^{(\infty)}_t](https://www.zhihu.com/equation?tex=%5Chat+A%5E%7B%28%5Cinfty%29%7D_t)可以说是对![A^{\pi,\gamma}](https://www.zhihu.com/equation?tex=A%5E%7B%5Cpi%2C%5Cgamma%7D)的无偏估计，但是它取整条路径的值，采样的方差自然就很大。
- 常常用一些中间值来平衡偏差和方差（如k=20）

> 采用 ![\hat A^{(\infty)}_t](https://www.zhihu.com/equation?tex=%5Chat+A%5E%7B%28%5Cinfty%29%7D_t) 的策略梯度就是Vanilla算法或REINFORCE with baseline，是一种蒙特卡洛方法，即return ![\sum_{t'=t}^{T-1} r_{t'}](https://www.zhihu.com/equation?tex=%5Csum_%7Bt%27%3Dt%7D%5E%7BT-1%7D+r_%7Bt%27%7D) 中不带有V值。
> 而我们把上述采用 ![\hat A^{(n)}_t(1\le n < \infty)](https://www.zhihu.com/equation?tex=%5Chat+A%5E%7B%28n%29%7D_t%281%5Cle+n+%3C+%5Cinfty%29)  的策略梯度称为actor-critic算法，其中actor就是策略函数 ![\pi(a|s,\theta)](https://www.zhihu.com/equation?tex=%5Cpi%28a%7Cs%2C%5Ctheta%29) （用神经网络表示），而critc可分这两种：
> \1. ![r_t+ \gamma V(s_{t+1})-V(s_t)](https://www.zhihu.com/equation?tex=r_t%2B+%5Cgamma+V%28s_%7Bt%2B1%7D%29-V%28s_t%29) 称为TD-error，是一种bootstrapping的方法（与蒙特卡洛方法相对，且一般优于蒙特卡洛方法），即return ![r_t+ \gamma V(s_{t+1})](https://www.zhihu.com/equation?tex=r_t%2B+%5Cgamma+V%28s_%7Bt%2B1%7D%29) 中包含V值。
> \2. ![r_t+ \gamma r_{t+1}+...+\gamma^{n-1}r_{t+n-1}+\gamma^nV(s_{t+n})-V(s_t)](https://www.zhihu.com/equation?tex=r_t%2B+%5Cgamma+r_%7Bt%2B1%7D%2B...%2B%5Cgamma%5E%7Bn-1%7Dr_%7Bt%2Bn-1%7D%2B%5Cgamma%5EnV%28s_%7Bt%2Bn%7D%29-V%28s_t%29) 称为n步TD-error，是介于TD方法和蒙特卡洛方法之间，而且通过调节超参数n，可以得到比TD方法和蒙特卡洛方法都好的成果，即能更快学到正确的 ![V(s)](https://www.zhihu.com/equation?tex=V%28s%29) 。
> 想要彻底搞懂TD方法和蒙特卡洛方法，还得看Sutton书的5、6节，但不影响本节学习。

## **与MPC（model predictive control）之前的联系：**

尝试回忆一下在有模型学习中我们一开始做的优化过程（如LQR）或者上一节中一开始的值迭代过程，可以发现MPC是一步一步从最后动作向前进行优化的，在向前的每一步中取的都是最优的策略/动作，用公式表示就是：

![img](https://pic3.zhimg.com/80/v2-93530738665e14b334e8dcacc56c327a_hd.jpg)

其中MPC用有限无折扣视野来近似无限有折扣的Q-function，而且对动作的评估都是按照最优策略来的，即![Q^{\pi,\gamma}=Q^{*,\gamma}](https://www.zhihu.com/equation?tex=Q%5E%7B%5Cpi%2C%5Cgamma%7D%3DQ%5E%7B%2A%2C%5Cgamma%7D)，因为在向前的每一步中取得是最优策略。

而策略梯度是同时对整一个过程所采用的整个策略进行梯度优化，而不是分步进行的，因此它不是采用最优策略来进行Q-function评估而是当前策略，它的优化目标用公式表示为：

![img](https://pic1.zhimg.com/80/v2-fa355ff23991e5d18a4024f6af9affec_hd.jpg)

达到这个优化目标时，就是取最优策略时（关于这个claim，在学完后面两节就清楚了）。

## **案例分析：A2C/A3C（（Asynchronous）Advantage Actor-Critic）**

其实策略梯度方法已经是比较老的方法了，在2000年就提出了Vanilla算法（以往人们认为这种方法的方差太大了），但是最近的一篇DeepMind的paper又让这个算法重新流行起来。他们对Vanilla算法做了一些改进得到A2C/A3C（（Asynchronous）Advantage Actor-Critic）（其中异步策略（Asynchronous，在后面几节会解释“异步”）只是缩短了训练时间，不影响结果）。在DeepMind的paper：《[Asynchronous Methods for Deep Reinforcement Learning](https://link.zhihu.com/?target=http%3A//proceedings.mlr.press/v48/mniha16.pdf)》中，用了以上对优势函数的估计来改进Vanilla算法得到：

**Asynchronous advantage actor-critic:**

![img](https://pic4.zhimg.com/80/v2-acb2a7b3c7712094ea052a1273bb9de9_hd.jpg)

这个幻灯片上的优势估计函数![\hat A_t](https://www.zhihu.com/equation?tex=%5Chat+A_t)和原文有些不同，原paper：![A(s_t,a_t;\theta,\theta_v)](https://www.zhihu.com/equation?tex=A%28s_t%2Ca_t%3B%5Ctheta%2C%5Ctheta_v%29)（即![\hat A_t](https://www.zhihu.com/equation?tex=%5Chat+A_t)）![=\sum_{i=0}^{k-1}\gamma^ir_{t+i}+\gamma^kV(s_{t+k};\theta_v)-V(s_t;\theta_v)](https://www.zhihu.com/equation?tex=%3D%5Csum_%7Bi%3D0%7D%5E%7Bk-1%7D%5Cgamma%5Eir_%7Bt%2Bi%7D%2B%5Cgamma%5EkV%28s_%7Bt%2Bk%7D%3B%5Ctheta_v%29-V%28s_t%3B%5Ctheta_v%29)，其中![\pi(a_t|s_t,\theta)](https://www.zhihu.com/equation?tex=%5Cpi%28a_t%7Cs_t%2C%5Ctheta%29)（actor网络）和![V(s_t;\theta_v)](https://www.zhihu.com/equation?tex=V%28s_t%3B%5Ctheta_v%29)（critic网络）在实践中常共用参数：卷积层和全连接层都共用，只是在最后的输出层中![\pi(a_t|s_t,\theta)](https://www.zhihu.com/equation?tex=%5Cpi%28a_t%7Cs_t%2C%5Ctheta%29)用softmax的输出，而![V(s_t;\theta_v)](https://www.zhihu.com/equation?tex=V%28s_t%3B%5Ctheta_v%29)用线性函数的输出。而且在最后的loss function中还加入了entropy项（策略的熵）：![\beta H(\pi(a_t|s_t,\theta))](https://www.zhihu.com/equation?tex=%5Cbeta+H%28%5Cpi%28a_t%7Cs_t%2C%5Ctheta%29%29)，用来防止策略收敛为确定型策略，有利于探索过程和一些需要高层次的决策行为。 (Williams& Peng, 1991)

然后可以来看一下实验结果，将A3C与一些Q-leaning比较（CPU上结果），可以看出A3C还是有竞争力的，最终能超越Q-leaning的表现：

![img](https://pic3.zhimg.com/80/v2-2d38cd3bbbbd89b77f6b391df73f4421_hd.jpg)

## **总结**

从这节开始，我们真正地进入到model-free RL，承接了上节的MDP和值函数内容，讨论一般的无模型，无限状态/动作空间情形。从优化目标：T步累计奖励出发，想用梯度方法求解（策略梯度法）。

为了求期望的梯度，首先推导了期望梯度的估计，得到：![\hat g_i=f(x_i)\nabla_{\theta}\log p(x_i|\theta)](https://www.zhihu.com/equation?tex=%5Chat+g_i%3Df%28x_i%29%5Cnabla_%7B%5Ctheta%7D%5Clog+p%28x_i%7C%5Ctheta%29)（需要p对![\theta](https://www.zhihu.com/equation?tex=%5Ctheta)可导）。

- 将这个结果运用到策略优化中，并做些变换，就得到了策略梯度：

![\qquad\nabla_{\theta}E[R]=E[\sum_{t=0}^{T-1}\nabla_{\theta}\log\pi(a_t|s_t,\theta)\sum_{t'=t}^{T-1}r_{t'}]](https://www.zhihu.com/equation?tex=%5Cqquad%5Cnabla_%7B%5Ctheta%7DE%5BR%5D%3DE%5B%5Csum_%7Bt%3D0%7D%5E%7BT-1%7D%5Cnabla_%7B%5Ctheta%7D%5Clog%5Cpi%28a_t%7Cs_t%2C%5Ctheta%29%5Csum_%7Bt%27%3Dt%7D%5E%7BT-1%7Dr_%7Bt%27%7D%5D)

- 为了减小采样的方差，引入了基线![b(s)](https://www.zhihu.com/equation?tex=b%28s%29)，代表了状态s的平均累积奖励。然后每一步累积奖励减去基线就是优势估计![\hat A_t](https://www.zhihu.com/equation?tex=%5Chat+A_t)，表示策略动作和平均比有多好。于是用![\hat A_t](https://www.zhihu.com/equation?tex=%5Chat+A_t)代入到策略梯度中，就是Vanilla算法，每一步都对路径采样，每一步都需要更新策略梯度和基线（两者常共用参数）。
- 之后我们用值函数的眼光来看上述过程，发现优势估计![\hat A_t](https://www.zhihu.com/equation?tex=%5Chat+A_t)就是对优势函数![A^{\pi,\gamma}(s,a)](https://www.zhihu.com/equation?tex=A%5E%7B%5Cpi%2C%5Cgamma%7D%28s%2Ca%29)的估计，而基线就是状态-值函数![V^{\pi}(s)](https://www.zhihu.com/equation?tex=V%5E%7B%5Cpi%7D%28s%29)，于是得到策略梯度式：

![\qquad\nabla_{\theta}E[R]=E[\sum_{t=0}^{T-1}\nabla_{\theta}\log\pi(a_t|s_t,\theta)A^{\pi,\gamma}(s,a)]](https://www.zhihu.com/equation?tex=%5Cqquad%5Cnabla_%7B%5Ctheta%7DE%5BR%5D%3DE%5B%5Csum_%7Bt%3D0%7D%5E%7BT-1%7D%5Cnabla_%7B%5Ctheta%7D%5Clog%5Cpi%28a_t%7Cs_t%2C%5Ctheta%29A%5E%7B%5Cpi%2C%5Cgamma%7D%28s%2Ca%29%5D)

- 而对优势函数![A^{\pi,\gamma}(s,a)](https://www.zhihu.com/equation?tex=A%5E%7B%5Cpi%2C%5Cgamma%7D%28s%2Ca%29)可以有多种估计方式根据之后多少步开始用![V(s)](https://www.zhihu.com/equation?tex=V%28s%29)来替代，取折中的步数即可。这样，就改进得到A2C算法，从实验可以看出它的结果还不错。

之后一节就要讲之前提过多次的Q-learning了，考察它的来源与它的变种。

顺便一提，这节的所有证明可以在教授自己写的讲义中找到：[http://joschu.net/docs/thesis.pdf](https://link.zhihu.com/?target=http%3A//joschu.net/docs/thesis.pdf)

<div STYLE="page-break-after: always;"></div>

#第七章：Q-learning

上一节讲了策略梯度方法来处理model-free，无限状态空间情形，整体思路为：让策略用一个神经网络表示，用梯度下降的方法进行训练，并且加入baseline来减小方差。这一节将延续上上节的值迭代和策略迭代的思路，引出更好用的Q-learning，并且这种Q-learning进一步可以解决model-free的情形。

## ![\large Q^{\pi}](https://www.zhihu.com/equation?tex=%5Clarge+Q%5E%7B%5Cpi%7D)**的Bellman等式和Bellman backup算子：**

在上一节中我们给出了Q-function![Q^{\pi}(s,a)](https://www.zhihu.com/equation?tex=Q%5E%7B%5Cpi%7D%28s%2Ca%29)，状态-值函数![V^{\pi}(s)](https://www.zhihu.com/equation?tex=V%5E%7B%5Cpi%7D%28s%29)，以及优势函数![A^{\pi}(s,a)](https://www.zhihu.com/equation?tex=A%5E%7B%5Cpi%7D%28s%2Ca%29)的定义：![Q^{\pi,\gamma}(s,a)=E_{\pi}[r_0+\gamma r_1+...|s_0=s,a_0=a]](https://www.zhihu.com/equation?tex=Q%5E%7B%5Cpi%2C%5Cgamma%7D%28s%2Ca%29%3DE_%7B%5Cpi%7D%5Br_0%2B%5Cgamma+r_1%2B...%7Cs_0%3Ds%2Ca_0%3Da%5D)，![V^{\pi,\gamma}(s)=E_{\pi}[r_0+\gamma r_1+...|s_0=s]](https://www.zhihu.com/equation?tex=V%5E%7B%5Cpi%2C%5Cgamma%7D%28s%29%3DE_%7B%5Cpi%7D%5Br_0%2B%5Cgamma+r_1%2B...%7Cs_0%3Ds%5D)，![A^{\pi,\gamma}(s,a)=Q^{\pi,\gamma}(s,a)-V^{\pi,\gamma}(s)](https://www.zhihu.com/equation?tex=A%5E%7B%5Cpi%2C%5Cgamma%7D%28s%2Ca%29%3DQ%5E%7B%5Cpi%2C%5Cgamma%7D%28s%2Ca%29-V%5E%7B%5Cpi%2C%5Cgamma%7D%28s%29)。

从上面的定义式，我们可以很容易地验证下面的

**Bellman等式**：

![img](https://pic2.zhimg.com/80/v2-06289d24ae314ad4b243e3883cf34568_hd.jpg)

现在定义**Bellman backup算子（Backup Operator）**：

![img](https://pic4.zhimg.com/80/v2-a6f4ba4179ae7d3bf7d95e0636445428_hd.jpg)

作用在Q函数上。与Bellman等式作对比，可以发现，Bellman backup算子就是把Bellman等式中的![Q^{\pi}](https://www.zhihu.com/equation?tex=Q%5E%7B%5Cpi%7D)替换为被作用的![Q](https://www.zhihu.com/equation?tex=Q)函数。因此，显然有![Q^{\pi}](https://www.zhihu.com/equation?tex=Q%5E%7B%5Cpi%7D)就是它的不动点，即![\mathcal T^{\pi} Q^{\pi}=Q^{\pi}](https://www.zhihu.com/equation?tex=%5Cmathcal+T%5E%7B%5Cpi%7D+Q%5E%7B%5Cpi%7D%3DQ%5E%7B%5Cpi%7D)。

可以看到，这里的Q函数的![\mathcal T^{\pi}](https://www.zhihu.com/equation?tex=%5Cmathcal+T%5E%7B%5Cpi%7D)与之前策略迭代的![\mathcal T^{\pi}](https://www.zhihu.com/equation?tex=%5Cmathcal+T%5E%7B%5Cpi%7D)：![[\mathcal T^{\pi}V](s)=E_{s_1\sim P(s_0,\pi(s_0))}[r_0+\gamma V(s_1)]](https://www.zhihu.com/equation?tex=%5B%5Cmathcal+T%5E%7B%5Cpi%7DV%5D%28s%29%3DE_%7Bs_1%5Csim+P%28s_0%2C%5Cpi%28s_0%29%29%7D%5Br_0%2B%5Cgamma+V%28s_1%29%5D)十分相似，因此类似之前对收敛性的证明（用压缩映射定理），可得：

![img](https://pic1.zhimg.com/80/v2-d0dfcdbb2be8095bd536bae3243afb61_hd.jpg)

对任一Q-function作用无限多次![\mathcal T^{\pi}](https://www.zhihu.com/equation?tex=%5Cmathcal+T%5E%7B%5Cpi%7D)可收敛到![Q^{\pi}](https://www.zhihu.com/equation?tex=Q%5E%7B%5Cpi%7D)。

## **Q-值迭代和Q-策略迭代算法：**

## **引入![\large Q^\*](https://www.zhihu.com/equation?tex=%5Clarge+Q%5E%2A)：**

- 设![\pi^*](https://www.zhihu.com/equation?tex=%5Cpi%5E%2A)为最优策略，即使得累积奖励最大的策略![\pi^*=argmax_{\pi}V^{\pi}(s),\forall s](https://www.zhihu.com/equation?tex=%5Cpi%5E%2A%3Dargmax_%7B%5Cpi%7DV%5E%7B%5Cpi%7D%28s%29%2C%5Cforall+s)
- 设![Q^*=Q^{\pi^*}](https://www.zhihu.com/equation?tex=Q%5E%2A%3DQ%5E%7B%5Cpi%5E%2A%7D)，即![Q^*(s,a)=max_{\pi}Q^{\pi}(s,a)](https://www.zhihu.com/equation?tex=Q%5E%2A%28s%2Ca%29%3Dmax_%7B%5Cpi%7DQ%5E%7B%5Cpi%7D%28s%2Ca%29)
- 可以验证：![\pi^*(s)=argmax_aQ^*(s,a)](https://www.zhihu.com/equation?tex=%5Cpi%5E%2A%28s%29%3Dargmax_aQ%5E%2A%28s%2Ca%29)（也就是说如果求得最优的Q函数，那反过来很容易能求得最优策略，而且是一个确定策略了）

对于![Q^*](https://www.zhihu.com/equation?tex=Q%5E%2A)的Bellman等式：

![\qquad \begin{align} Q^*(s_0,a_0)&=E_{s_1}[r_0+\gamma V^*(s_1)]\\ &=E_{s_1}[r_0+\gamma \max_{a_1}Q^*(s_1,a_1)] \end{align}](https://www.zhihu.com/equation?tex=%5Cqquad+%5Cbegin%7Balign%7D+Q%5E%2A%28s_0%2Ca_0%29%26%3DE_%7Bs_1%7D%5Br_0%2B%5Cgamma+V%5E%2A%28s_1%29%5D%5C%5C+%26%3DE_%7Bs_1%7D%5Br_0%2B%5Cgamma+%5Cmax_%7Ba_1%7DQ%5E%2A%28s_1%2Ca_1%29%5D+%5Cend%7Balign%7D)

相应的对于![Q^*](https://www.zhihu.com/equation?tex=Q%5E%2A)的Bellman backup算子:

![\qquad [\mathcal T Q](s_0,a_0)=E_{s_1}[r_0+\gamma \max_{a_1}Q(s_1,a_1)]](https://www.zhihu.com/equation?tex=%5Cqquad+%5B%5Cmathcal+T+Q%5D%28s_0%2Ca_0%29%3DE_%7Bs_1%7D%5Br_0%2B%5Cgamma+%5Cmax_%7Ba_1%7DQ%28s_1%2Ca_1%29%5D)

与之前一样的有：![Q^*](https://www.zhihu.com/equation?tex=Q%5E%2A)为![\mathcal T](https://www.zhihu.com/equation?tex=%5Cmathcal+T)的不动点，且有收敛结果：

![img](https://pic2.zhimg.com/80/v2-03e3c83ae883159358364b68bcb4bec1_hd.jpg)

因为![Q^{\pi}(s,a)](https://www.zhihu.com/equation?tex=Q%5E%7B%5Cpi%7D%28s%2Ca%29)表示了在当前策略![\pi](https://www.zhihu.com/equation?tex=%5Cpi)下，对某一状态s时采用动作a的好坏评估，而![V^{\pi}(s)](https://www.zhihu.com/equation?tex=V%5E%7B%5Cpi%7D%28s%29)表示了在当前策略![\pi](https://www.zhihu.com/equation?tex=%5Cpi)下某一状态s的好坏。总体想法都是通过对策略的值函数，来找到当前策略下的更优策略。因而遵循或者效仿之前V-值迭代和V-策略迭代的算法，可以很容易得到

**Q-值迭代算法：**

![img](https://pic4.zhimg.com/80/v2-d397004199bcfa1f9d1839e09c3f0f48_hd.jpg)

**以及Q-策略迭代算法：**

![img](https://pic3.zhimg.com/80/v2-0a5d3391a4865afa7255dba6746e6913_hd.jpg)

其中策略迭代时的，![\mathcal G](https://www.zhihu.com/equation?tex=%5Cmathcal+G)表示在当前Q-function![Q^{(n)}](https://www.zhihu.com/equation?tex=Q%5E%7B%28n%29%7D)下选择最优的策略![\pi^{(n+1)}(s)=argmax_aQ^{(n)}(s,a)](https://www.zhihu.com/equation?tex=%5Cpi%5E%7B%28n%2B1%29%7D%28s%29%3Dargmax_aQ%5E%7B%28n%29%7D%28s%2Ca%29)。![Q^{\pi^{(n)}}](https://www.zhihu.com/equation?tex=Q%5E%7B%5Cpi%5E%7B%28n%29%7D%7D)表示解线性方程组：![\mathcal T^{\pi} Q^{\pi}=Q^{\pi}](https://www.zhihu.com/equation?tex=%5Cmathcal+T%5E%7B%5Cpi%7D+Q%5E%7B%5Cpi%7D%3DQ%5E%7B%5Cpi%7D)

类似得也可以得到**改进的Q-策略迭代算法**：

![img](https://pic3.zhimg.com/80/v2-cba85b2dbe71ca1a412f2ba3024064ec_hd.jpg)

注意：以上和下一段的Q-learning都是假设了在有限状态和动作空间中的，也就是说Q-function可以用一张状态-动作表来表示。

## **基于采样的算法（Sampling-Based Algorithms）：**

## **基于采样的估计：**

以上的算法需要计算![\mathcal T](https://www.zhihu.com/equation?tex=%5Cmathcal+T)或![\mathcal T^{\pi}](https://www.zhihu.com/equation?tex=%5Cmathcal+T%5E%7B%5Cpi%7D)：

![img](https://pic3.zhimg.com/80/v2-56279e3dfbc1ec836ab41ecd18d81017_hd.jpg)

然而对于一般的model-free无模型情形，状态转移概率![P(s_1|s_0,a_0)](https://www.zhihu.com/equation?tex=P%28s_1%7Cs_0%2Ca_0%29)是未知的，则上面两式无法直接计算。

但是我们可以通过对![s_1](https://www.zhihu.com/equation?tex=s_1)的采样来得到对上式期望的无偏估计，而且仅采一次样即可：

![img](https://pic1.zhimg.com/80/v2-f65b30805cbdadf02e3a1ec23ff8362b_hd.jpg)

值得注意的是这个![s_1](https://www.zhihu.com/equation?tex=s_1)的采样过程本质上与用来选择动作的策略无关，是环境本身的模型，这就给异步策略off-policy提供可能（之后会讲）。

而且可以证明多次作用这种采样的backup，任然可以分别收敛到![Q^{\pi},Q^*](https://www.zhihu.com/equation?tex=Q%5E%7B%5Cpi%7D%2CQ%5E%2A)。

> T. Jaakkola, M. I. Jordan, and S. P. Singh. “On the convergence of stochastic iterative dynamic programming algorithms”. (1994); D. P. Bertsekas. Dynamic programming and optimal control. Athena Scientific, 2012.Neural computation

可以将上面过程推广到**多步采样**：（即n步TD）

![img](https://pic4.zhimg.com/80/v2-9f88eecb98320c39a3d433c03bda82e2_hd.jpg)

可以通过采集路径![(s_0,a_0,r_0,s_1,a_1,r_1,…,s_{k-1},a_{k-1},r_{k-1},s_{k})](https://www.zhihu.com/equation?tex=%28s_0%2Ca_0%2Cr_0%2Cs_1%2Ca_1%2Cr_1%2C%E2%80%A6%2Cs_%7Bk-1%7D%2Ca_%7Bk-1%7D%2Cr_%7Bk-1%7D%2Cs_%7Bk%7D%29)来得到对![[(\mathcal T^{\pi})^k Q](s_0,a_0)](https://www.zhihu.com/equation?tex=%5B%28%5Cmathcal+T%5E%7B%5Cpi%7D%29%5Ek+Q%5D%28s_0%2Ca_0%29)的无偏估计。

## **Q-function vs. V-function:**

可以看到之前的Q-值迭代或者Q-策略迭代只是把上一节的V-值迭代和V-策略迭代中的值函数改变了，算法本身几乎没有改变。那么我们为什么要用Q-function而不是V-function呢？再来看一下V-function的backups：

![img](https://pic2.zhimg.com/80/v2-954bf7d879c3e7d1a0f6ded6cdfe374d_hd.jpg)

可以看到与Q-function的backups相比，对![s_1](https://www.zhihu.com/equation?tex=s_1)的期望![E_{s_1}](https://www.zhihu.com/equation?tex=E_%7Bs_1%7D)与对行动的选择![\max_{a_1}](https://www.zhihu.com/equation?tex=%5Cmax_%7Ba_1%7D)或![E_{a_0\sim\pi}](https://www.zhihu.com/equation?tex=E_%7Ba_0%5Csim%5Cpi%7D)调换了位置，这样的调换会导致在采样时，对V-function的backups采样不是无偏估计（对先![E_{s_1}](https://www.zhihu.com/equation?tex=E_%7Bs_1%7D)再![\max_{a_1}](https://www.zhihu.com/equation?tex=%5Cmax_%7Ba_1%7D)采样时，由于max函数的非线性性，![s_1](https://www.zhihu.com/equation?tex=s_1)又依赖于![a_1](https://www.zhihu.com/equation?tex=a_1)，那么对每个动作采样一次得到的![max_{a_1}[r_0+\gamma V^{\pi}(s_1(a_1))]](https://www.zhihu.com/equation?tex=max_%7Ba_1%7D%5Br_0%2B%5Cgamma+V%5E%7B%5Cpi%7D%28s_1%28a_1%29%29%5D)并非无偏估计），而对Q-function的确是无偏估计（先![max_{a_1}](https://www.zhihu.com/equation?tex=max_%7Ba_1%7D)再![E_{s_1}](https://www.zhihu.com/equation?tex=E_%7Bs_1%7D)，而且给出了![a_0](https://www.zhihu.com/equation?tex=a_0)值，让![s_1](https://www.zhihu.com/equation?tex=s_1)可单独进行采样，那就不会有这样问题）。

总结来说，**使用Q-function的好处在于：**

- 能够利用![\max_a Q(s,a)](https://www.zhihu.com/equation?tex=%5Cmax_a+Q%28s%2Ca%29)很轻松地计算当前最Greedy的动作（而用![V(s)](https://www.zhihu.com/equation?tex=V%28s%29)计算需要知道环境P）。
- 能够只用一步采样来得到对![\mathcal T^{\pi}Q(s,a)](https://www.zhihu.com/equation?tex=%5Cmathcal+T%5E%7B%5Cpi%7DQ%28s%2Ca%29)的无偏估计。
- 能够用异步策略off-policy采样来得到一样的无偏估计。

## **基于采样的算法：**

现在我们只需要把之前的Q-值迭代或者Q-策略迭代中的backups![[\mathcal T^{\pi}Q](s,a)](https://www.zhihu.com/equation?tex=%5B%5Cmathcal+T%5E%7B%5Cpi%7DQ%5D%28s%2Ca%29)替换成采样得到的![\widehat{[\mathcal T^{\pi} Q]}(s,a)](https://www.zhihu.com/equation?tex=%5Cwidehat%7B%5B%5Cmathcal+T%5E%7B%5Cpi%7D+Q%5D%7D%28s%2Ca%29)，即可得到model-free的Q-learning算法：

**基于采样的Q-值迭代：**

![img](https://pic3.zhimg.com/80/v2-a4afbd1f18daa71bc29ea1a1f0a42693_hd.jpg)

其中Q-function更新步可看作![Q^{(n+1)}=\mathcal T Q^{(n)}+noise](https://www.zhihu.com/equation?tex=Q%5E%7B%28n%2B1%29%7D%3D%5Cmathcal+T+Q%5E%7B%28n%29%7D%2Bnoise)，增加了由采样产生的noise项。

**从最小二乘的眼光来看：**

因为注意到有这样的等式：![mean\{x_i\}=argmin_x\sum_i ||x_i-x||^2](https://www.zhihu.com/equation?tex=mean%5C%7Bx_i%5C%7D%3Dargmin_x%5Csum_i+%7C%7Cx_i-x%7C%7C%5E2)。

因此在上述算法中的Q值更新步：![Q^{(n+1)}(s,a)=mean\{\widehat{\mathcal TQ_t}, \forall t \quad s.t. (s_t,a_t)=(s,a)\}](https://www.zhihu.com/equation?tex=Q%5E%7B%28n%2B1%29%7D%28s%2Ca%29%3Dmean%5C%7B%5Cwidehat%7B%5Cmathcal+TQ_t%7D%2C+%5Cforall+t+%5Cquad+s.t.+%28s_t%2Ca_t%29%3D%28s%2Ca%29%5C%7D)，用最小二乘的眼光来看就是：

![\qquad Q^{(n+1)}(s,a)=argmin_{Q(s,a)} \sum_{t,s.t. (s_t,a_t)=(s,a)}||\widehat{\mathcal TQ_t}-Q(s,a)||^2](https://www.zhihu.com/equation?tex=%5Cqquad+Q%5E%7B%28n%2B1%29%7D%28s%2Ca%29%3Dargmin_%7BQ%28s%2Ca%29%7D+%5Csum_%7Bt%2Cs.t.+%28s_t%2Ca_t%29%3D%28s%2Ca%29%7D%7C%7C%5Cwidehat%7B%5Cmathcal+TQ_t%7D-Q%28s%2Ca%29%7C%7C%5E2)

那么对于函数Q本身而言（不只是对![Q(s,a)](https://www.zhihu.com/equation?tex=Q%28s%2Ca%29)值），在总共进行K步后，Q函数的更新是以下最小二乘的结果：

![\qquad Q^{(n+1)}(s,a)=argmin_Q \sum_{t=1}^K||\widehat{\mathcal TQ_t}-Q(s_t,a_t)||^2](https://www.zhihu.com/equation?tex=%5Cqquad+Q%5E%7B%28n%2B1%29%7D%28s%2Ca%29%3Dargmin_Q+%5Csum_%7Bt%3D1%7D%5EK%7C%7C%5Cwidehat%7B%5Cmathcal+TQ_t%7D-Q%28s_t%2Ca_t%29%7C%7C%5E2)

因此可用新的表达式来表达（没有改变本质）**基于采样的Q-值迭代算法**：

![img](https://pic1.zhimg.com/80/v2-384c6ae6c177d1b1e93c0e82911a6358_hd.jpg)

这种表达式的好处是为之后Q-function用神经网络来表示时，网络的更新步骤（梯度下降）奠定基础。

## **部分支持算子（Partial Backups）：**

- Full Backups：![Q \gets \widehat{\mathcal TQ_t}](https://www.zhihu.com/equation?tex=Q+%5Cgets+%5Cwidehat%7B%5Cmathcal+TQ_t%7D)是之前的Q值更新步骤
- Partial Backups:![Q \gets \epsilon \widehat{\mathcal TQ_t}+(1-\epsilon)Q](https://www.zhihu.com/equation?tex=Q+%5Cgets+%5Cepsilon+%5Cwidehat%7B%5Cmathcal+TQ_t%7D%2B%281-%5Cepsilon%29Q)，按照![\epsilon](https://www.zhihu.com/equation?tex=%5Cepsilon)比例进行更新
- 可以把上面的Partial Backups与对最小二乘的**梯度下降法（步长为![\epsilon](https://www.zhihu.com/equation?tex=%5Cepsilon)）等同起来**：

![img](https://pic3.zhimg.com/80/v2-55a5202c4cc2a4724a929885d6232000_hd.jpg)

- 那么对于足够小的步长![\epsilon](https://www.zhihu.com/equation?tex=%5Cepsilon)，可以期望损失函数![||\widehat{\mathcal TQ_t}-Q||^2](https://www.zhihu.com/equation?tex=%7C%7C%5Cwidehat%7B%5Cmathcal+TQ_t%7D-Q%7C%7C%5E2)能减小。

利用Partial Backups的更新步，**基于采样的Q-值迭代算法（或者称Q-learning）**可变为：

![img](https://pic3.zhimg.com/80/v2-d58a80108d6383c874eb2e7e225ea3c3_hd.jpg)

上面的Q值更新步骤，也可以做这样解释：

![img](https://pic2.zhimg.com/80/v2-c7cdc73730ccd11f85949b0c7949b3bc_hd.jpg)

可以看出，当![\epsilon](https://www.zhihu.com/equation?tex=%5Cepsilon)比较小时，采样产生的noise的影响也会比较小，这样就能保证了更新步骤一些收敛性。

此外，当K=1时，上述算法称为 Watkins’ Q-learning（1992），当n足够大时能遍历所有状态-动作对，由此能收敛到最优值函数。（是一种on-line算法，即每与环境互动一次，就进行更新）

当K比较大时，称为batch Q-value iteration，是近年比较常用的Q-learning。（off-line算法，能加快速度）

**收敛性证明：**

有关上面算法的收敛性的证明并不简单，因为其本身不完全是一个最小二乘法，即损失函数![L(Q ) = ||\mathcal T Q − Q ||^2 /2](https://www.zhihu.com/equation?tex=L%28Q+%29+%3D+%7C%7C%5Cmathcal+T+Q+%E2%88%92+Q+%7C%7C%5E2+%2F2)并不是固定的，由于目标![\mathcal T Q^{(n)}](https://www.zhihu.com/equation?tex=%5Cmathcal+T+Q%5E%7B%28n%29%7D)在不断地改变。

但利用Partial Backups，取比较合适的步长，e.g.![\epsilon = 1/n](https://www.zhihu.com/equation?tex=%5Cepsilon+%3D+1%2Fn)，则有![\lim_{n\to\infty}Q^{(n)}=Q^*](https://www.zhihu.com/equation?tex=%5Clim_%7Bn%5Cto%5Cinfty%7DQ%5E%7B%28n%29%7D%3DQ%5E%2A).

> 证明：T. Jaakkola, M. I. Jordan, and S. P. Singh. “On the convergence of stochastic iterative dynamic programming algorithms”. Neural computation (1994).

**Sarsa算法**

> 实际上类似地可以做**基于采样的Q-策略迭代算法（使用**Partial Backups**），其backup公式为：** ![Q \gets (1-\alpha)Q+ \alpha \widehat{\mathcal T^{\pi}Q_t}](https://www.zhihu.com/equation?tex=Q+%5Cgets+%281-%5Calpha%29Q%2B+%5Calpha+%5Cwidehat%7B%5Cmathcal+T%5E%7B%5Cpi%7DQ_t%7D) ，即：

![img](https://pic2.zhimg.com/80/v2-25f7005b0ea47f39b64d9240f2e6ad62_hd.jpg)

> 与**基于采样的Q-值迭代算法**不同的地方就只在**：**值迭代使用最优策略的![\mathcal T](https://www.zhihu.com/equation?tex=%5Cmathcal+T) backup算子，Q-策略迭代使用当前策略 ![\pi](https://www.zhihu.com/equation?tex=%5Cpi) 的 ![\mathcal T^{\pi}](https://www.zhihu.com/equation?tex=%5Cmathcal+T%5E%7B%5Cpi%7D) backup算子。
> **基于采样的Q-策略迭代算法**也称为**Sarsa算法，**其命名是由于它的Q更新式中用到了 ![(S_t,A_t,R_{t+1},S_{t+1},A_{t+1})](https://www.zhihu.com/equation?tex=%28S_t%2CA_t%2CR_%7Bt%2B1%7D%2CS_%7Bt%2B1%7D%2CA_%7Bt%2B1%7D%29) ，包含了一个状态-动作对转移到下一个状态-动作对的过程。
> 其完整算法为：

![img](https://pic4.zhimg.com/80/v2-ee7113c71deec01e207e2a07fd95dca2_hd.jpg)

> 而且Sarsa是一种on-policy的算法，而Q-learning是off-policy的算法。其中on-policy指的是执行动作（在环境中探索）的策略和作为Q-值更新对象的策略是一样的，off-policy指的是两个策略不一样，比如Q-learning中执行动作的策略是 ![\epsilon](https://www.zhihu.com/equation?tex=%5Cepsilon) -greedy的策略，但是更新的对象是 ![Q^*](https://www.zhihu.com/equation?tex=Q%5E%2A) 也就是最greedy策略的Q值，两个策略不同（当 ![\epsilon](https://www.zhihu.com/equation?tex=%5Cepsilon) 不为0时）。
> off-policy的算法比on-policy的算法优势在于：可以利用任意不同的策略来收集数据，而始终能用来更新 ![Q^*](https://www.zhihu.com/equation?tex=Q%5E%2A) 值，这是一个极好的性质。（可以利用下一节讲的replay memory）

## **参数化Q-function：**

可以用一个神经网络（参数![\theta](https://www.zhihu.com/equation?tex=%5Ctheta)）来表示Q-function：![Q_{\theta}](https://www.zhihu.com/equation?tex=Q_%7B%5Ctheta%7D)，来应对无限状态空间情形，以及提高泛化能力。

那么之前算法的Q值更新步骤（Full Backups）：![Q \gets \widehat{\mathcal TQ_t}](https://www.zhihu.com/equation?tex=Q+%5Cgets+%5Cwidehat%7B%5Cmathcal+TQ_t%7D)，就变为：

![img](https://pic2.zhimg.com/80/v2-69cba5d69507eda88e884f591dafdfd9_hd.jpg)

就得到用神经网络Q-learning的最早的雏形算法：

![img](https://pic2.zhimg.com/80/v2-e0e012db1383e105864c69fd3819387b_hd.jpg)

> M. Riedmiller. “Neural fitted Q iteration–first experiences with a data efficient neural reinforcement learning method”. ECML 2005. Springer, 2005. Machine Learning:

其中采样策略![\pi^{(n)}](https://www.zhihu.com/equation?tex=%5Cpi%5E%7B%28n%29%7D)可以是异步策略，比如说![\epsilon](https://www.zhihu.com/equation?tex=%5Cepsilon)贪心策略。而![\widehat{\mathcal TQ_t}=r_t+\gamma \max_{a_{t+1}}Q(s_{t+1},a_{t+1})](https://www.zhihu.com/equation?tex=%5Cwidehat%7B%5Cmathcal+TQ_t%7D%3Dr_t%2B%5Cgamma+%5Cmax_%7Ba_%7Bt%2B1%7D%7DQ%28s_%7Bt%2B1%7D%2Ca_%7Bt%2B1%7D%29)。

## **总结**：

这一节在前面值迭代和策略迭代的基础上引出了Q-learning。

- 首先介绍了Q-function的Bellman等式，以及Bellman backup算子![\mathcal T^{\pi}Q](https://www.zhihu.com/equation?tex=%5Cmathcal+T%5E%7B%5Cpi%7DQ)，利用任意初始![Q](https://www.zhihu.com/equation?tex=Q)不断作用backups![\mathcal T^{\pi}](https://www.zhihu.com/equation?tex=%5Cmathcal+T%5E%7B%5Cpi%7D)会收敛到![Q^{\pi}](https://www.zhihu.com/equation?tex=Q%5E%7B%5Cpi%7D)的性质，导出了Q-值迭代和Q-策略迭代算法（本质上只是把之前的算法用Q-function来代入）。
- 为了应对model-free和无限情形，需要对路径采样来得到对![[\mathcal T^{\pi}Q](s,a)](https://www.zhihu.com/equation?tex=%5B%5Cmathcal+T%5E%7B%5Cpi%7DQ%5D%28s%2Ca%29)的估计![\widehat{[\mathcal T^{\pi} Q]}(s,a)](https://www.zhihu.com/equation?tex=%5Cwidehat%7B%5B%5Cmathcal+T%5E%7B%5Cpi%7D+Q%5D%7D%28s%2Ca%29)，对于Q-function这事一个无偏估计，但对V-function的backups就不是无偏估计，这就是采用Q-function的一个原因。
- 在基于采样的Q-值迭代算法中，![Q^{(n)}](https://www.zhihu.com/equation?tex=Q%5E%7B%28n%29%7D)的更新步骤![Q^{(n+1)}(s,a)=mean\{\widehat{\mathcal TQ_t}, \forall t \quad s.t. (s_t,a_t)=(s,a)\}](https://www.zhihu.com/equation?tex=Q%5E%7B%28n%2B1%29%7D%28s%2Ca%29%3Dmean%5C%7B%5Cwidehat%7B%5Cmathcal+TQ_t%7D%2C+%5Cforall+t+%5Cquad+s.t.+%28s_t%2Ca_t%29%3D%28s%2Ca%29%5C%7D)，可以用最小二乘的眼光来看：![Q^{(n+1)}(s,a)=argmin_Q \sum_{t=1}^K||\widehat{\mathcal TQ_t}-Q(s_t,a_t)||^2](https://www.zhihu.com/equation?tex=Q%5E%7B%28n%2B1%29%7D%28s%2Ca%29%3Dargmin_Q+%5Csum_%7Bt%3D1%7D%5EK%7C%7C%5Cwidehat%7B%5Cmathcal+TQ_t%7D-Q%28s_t%2Ca_t%29%7C%7C%5E2)。
- 为了保证采样算法的收敛性，提出了Partial Backups，是等价于对最小二乘的梯度下降（步长为![\epsilon](https://www.zhihu.com/equation?tex=%5Cepsilon)）步骤的。
- 最后用神经网络来参数化Q-function，就是deep Q-learning的雏形了。

下一节将介绍DQN（deep Q-network），让以上的Q-learning算法能在实际（如Atari game上）运行起来，而且会介绍相关的前沿的技巧，如replay buffers，double Q-learning等，而这些算法已经能在实际中有很好的效果了，在下次作业中也能感受到。

<div STYLE="page-break-after: always;"></div>

# 第八章：DQN

在上一节中，已经介绍了基本的Q-learning的思想和算法。在此基础上，要在实际中能达到很好的效果，还需要使用一些技巧，这节就主要解读**deep Q-Network（DQN）**所采用的方法：**Replay memory、Target network**，同时还会介绍其他有效的方法：**Double DQN、Dueling net、Prioritized Replay**，用来解决Q-learning中的问题，以及增强实际效果。

实验成果预示：

​            

gym上的Atari game中的Pong，其中右边绿色的是我们的agency，采用DQN决定的动作，而左边黄色的是游戏hard code的动作。

## **Q-learning 回顾：**

首先来回顾一下上一节提出的Q-learning的几个算法，最基本的**Q-值迭代算法**：

![img](https://pic2.zhimg.com/80/v2-bd09df68ab064e5e878a61f99bdf1e5b_hd.jpg)

其中更新步骤的backup算子为：

![img](https://pic2.zhimg.com/80/v2-85012d81f5592fbf12551fdf124bbcb8_hd.jpg)

之后为了应对model-free情形，1.通过采样来得到对![\mathcal TQ](https://www.zhihu.com/equation?tex=%5Cmathcal+TQ)的无偏估计![\widehat{\mathcal TQ}](https://www.zhihu.com/equation?tex=%5Cwidehat%7B%5Cmathcal+TQ%7D)；2.采用Partial Backups来类比最小二乘的梯度下降过程，提高收敛性；3.参数化Q-function，用神经网络来表示Q-function。经过这样的调整，就得到了**NFQ算法**：

![img](https://pic4.zhimg.com/80/v2-ce174dceb309a8de8224d3f8c430e82a_hd.jpg)

其中它的Q更新步骤为直接取minimize，但在大型的神经网络或者在线学习中，常常采用梯度下降法：**Watkins’ Q-learning / Incremental Q-Value Iteration算法**：

![img](https://pic1.zhimg.com/80/v2-2421173b480548efe01d4d727b921d32_hd.jpg)

## **参数化Q-learning的问题：**

在上一节中，已经指出了在**非参数化Q-learning**中，无论是Watkins’ Q-learning（K=1）还是batch Q-value iteration（K较大）只要更新步长![\epsilon](https://www.zhihu.com/equation?tex=%5Cepsilon)取得合适，趋向于0，Q-learning算法都是能收敛的。

然而在**参数化Q-learning**中，就没有理论能保证算法收敛了。而参数化Q-learning的误差主要来源于两个方面：**近似（投影）、噪声**。近似（投影）的影响可从下图看出：

![img](https://pic2.zhimg.com/80/v2-a9288872c5c468e3ed6f862ad34391a6_hd.jpg)

其中虚线![Q_{\theta}](https://www.zhihu.com/equation?tex=Q_%7B%5Ctheta%7D)表示参数化的Q-function（神经网络）在Q值空间中所能表示的超曲面，那么在算法运行中，Q值更新到第n步时变为![Q^{(n)}](https://www.zhihu.com/equation?tex=Q%5E%7B%28n%29%7D)，作用上backup算子得到![\mathcal TQ^{(n)}](https://www.zhihu.com/equation?tex=%5Cmathcal+TQ%5E%7B%28n%29%7D)，可以看出它离最优的![Q^*](https://www.zhihu.com/equation?tex=Q%5E%2A)更近了一步，然而我们的Q神经网络并不能表示![\mathcal TQ^{(n)}](https://www.zhihu.com/equation?tex=%5Cmathcal+TQ%5E%7B%28n%29%7D)，而是采用最小二乘![||\mathcal TQ^{(n)}-Q^{(n+1)}||^2](https://www.zhihu.com/equation?tex=%7C%7C%5Cmathcal+TQ%5E%7B%28n%29%7D-Q%5E%7B%28n%2B1%29%7D%7C%7C%5E2)，在![Q_{\theta}](https://www.zhihu.com/equation?tex=Q_%7B%5Ctheta%7D)超曲面上找到与![\mathcal TQ^{(n)}](https://www.zhihu.com/equation?tex=%5Cmathcal+TQ%5E%7B%28n%29%7D)最近的投影点![Q^{(n+1)}](https://www.zhihu.com/equation?tex=Q%5E%7B%28n%2B1%29%7D)。但从图中可粗略地看出，更新后的![Q^{(n+1)}](https://www.zhihu.com/equation?tex=Q%5E%7B%28n%2B1%29%7D)反而离![Q^*](https://www.zhihu.com/equation?tex=Q%5E%2A)更远了。这就会导致参数化Q-learning最终发散或者收敛到很差的结果。

用更加数学的说法：投影的Bellman backup更新为：![Q \to \Pi \mathcal TQ](https://www.zhihu.com/equation?tex=Q+%5Cto+%5CPi+%5Cmathcal+TQ)，其中backup算子![\mathcal T](https://www.zhihu.com/equation?tex=%5Cmathcal+T)是在![||\cdot||_{\infty}](https://www.zhihu.com/equation?tex=%7C%7C%5Ccdot%7C%7C_%7B%5Cinfty%7D)无穷范数（因为![\mathcal T](https://www.zhihu.com/equation?tex=%5Cmathcal+T)是取了max）下的压缩映射，而投影映射![\Pi](https://www.zhihu.com/equation?tex=%5CPi)是在![||\cdot||_2](https://www.zhihu.com/equation?tex=%7C%7C%5Ccdot%7C%7C_2)![l_2](https://www.zhihu.com/equation?tex=l_2)范数下的压缩映射。两个不同范数下的压缩映射就不一定再是压缩映射了，那么就不能用压缩映射定理来保证参数化Q-learning的收敛性了。

噪声产生的影响能更容易看出：

![img](https://pic4.zhimg.com/80/v2-203a46ec0781582e9045edb3aeb19494_hd.jpg)

图中的涡旋线是理想的Q值迭代过程，最终收敛到涡旋的中心：![Q^*](https://www.zhihu.com/equation?tex=Q%5E%2A)，然而参数化Q-learning是基于采样的算法，采用![\widehat{\mathcal TQ}](https://www.zhihu.com/equation?tex=%5Cwidehat%7B%5Cmathcal+TQ%7D)来估计backup，那么采样就会带来噪声noise，因此真正地更新步骤是![Q^{(n+1)}=\mathcal T Q^{(n)}+noise](https://www.zhihu.com/equation?tex=Q%5E%7B%28n%2B1%29%7D%3D%5Cmathcal+T+Q%5E%7B%28n%29%7D%2Bnoise)，在图中就是![\mathcal TQ^{(n)}](https://www.zhihu.com/equation?tex=%5Cmathcal+TQ%5E%7B%28n%29%7D)为圆心的圆中随机取值更新，那么更新后的Q值有很大概率是离![Q^*](https://www.zhihu.com/equation?tex=Q%5E%2A)越来越远的。这就要求更新步长![\epsilon](https://www.zhihu.com/equation?tex=%5Cepsilon)取得合适，能让noise圆的半径越来越小，才有可能收敛。

总的来说，在调整Q-learning算法时，需要注意网络近似问题（用更有表示力的网络，但也要注意Q网络更新过程），以及减少噪声（或者方差），提高收敛性。然而实际问题中也很难说是近似问题还是噪声导致的。

> 在Sutton书的11节对这个问题进行了深入的讨论，得出的结论是参数化的Q值函数加上off-policy算法无法在理论上能保证其收敛，实际上可以给出参数化Q-leaning不收敛的反例。因此对于收敛性，我们能做的是尽量保持Q-learning的稳定性，以及探索策略与学习策略不能相差太多。

## **Deep Q-Network (DQN)：**

在13年的paper：《[Playing Atari with Deep Reinforcement Learning](https://link.zhihu.com/?target=https%3A//www.cs.toronto.edu/%7Evmnih/docs/dqn.pdf)》首次提出并成功利用Deep Q-Network在Atari游戏上取得突破性成绩，轰动一时，可以说是真正让深度强化学习进入大多数人视野。因此本节就着重讨论这篇论文的所用的方法：

总体来说，DQN算法是结合了之前说的online and batch Q-value iteration，用深度神经网络（AlexNet来参数化Q-function）。但是其创新之处在于提出了两个实用的方法：

- 经验重现 Replay memory![\mathcal D](https://www.zhihu.com/equation?tex=%5Cmathcal+D)：把之前最近的N个采样得到的状态转移![(s_t,a_t,r_t,s_{t+1})](https://www.zhihu.com/equation?tex=%28s_t%2Ca_t%2Cr_t%2Cs_%7Bt%2B1%7D%29)都平等地储存在记忆![\mathcal D](https://www.zhihu.com/equation?tex=%5Cmathcal+D)中。
- 目标网络 Target network：旧的Q-function![Q^{(n)}](https://www.zhihu.com/equation?tex=Q%5E%7B%28n%29%7D)会被固定很长时间（大约10000步）才会被更新，在此期间![Q^{(n)}](https://www.zhihu.com/equation?tex=Q%5E%7B%28n%29%7D)都会用来计算：![Q\to\mathcal TQ^{(n)}](https://www.zhihu.com/equation?tex=Q%5Cto%5Cmathcal+TQ%5E%7B%28n%29%7D)，更新内部的Q-function。（下面还会详解）

现在先给出DQN的算法，再详细说明和分析：

![img](https://pic2.zhimg.com/80/v2-7f4d27643c9297508cc43e1beff6b53d_hd.jpg)

**按顺序分条详解：**

- 预处理过程![\phi](https://www.zhihu.com/equation?tex=%5Cphi)，只是一个技术上的处理：把RGB图像转为2D灰白图，并切割成84x84大小，只是为了满足AlexNet的标准输入，然后由于Atari游戏在像素层面是个POMDP问题，因此状态最好设置为前k个图像，即将前k个图像作为Q-function的输入（k取4），那么预处理后每个状态就是84x84x4的像素，输入AlexNet，输出为所有动作的Q值。（AlexNet: 16 8 x 8 filters with stride 4, 32 4 x 4 filters with stride 2, 256 fc, outputs fc ; activation function=Relu)
- 用![\epsilon](https://www.zhihu.com/equation?tex=%5Cepsilon)-贪心策略搜集路径：以![\epsilon](https://www.zhihu.com/equation?tex=%5Cepsilon)的概率随机选动作，以![1-\epsilon](https://www.zhihu.com/equation?tex=1-%5Cepsilon)概率取当前Q-function下的最优动作。
- Replay memory![\mathcal D](https://www.zhihu.com/equation?tex=%5Cmathcal+D)，储存了最近的N个采样得到的状态转移![(\phi_t,a_t,r_t,\phi_{t+1})](https://www.zhihu.com/equation?tex=%28%5Cphi_t%2Ca_t%2Cr_t%2C%5Cphi_%7Bt%2B1%7D%29)，训练网络时，随机从memory中取一个小bach作为更新样本。
- 注意用来计算更新步骤的backup![\mathcal TQ](https://www.zhihu.com/equation?tex=%5Cmathcal+TQ)的 Q 是Target network![Q^{target}=Q^{(n)}](https://www.zhihu.com/equation?tex=Q%5E%7Btarget%7D%3DQ%5E%7B%28n%29%7D)，将会被固定很长时间才更新（因此要做一个copy），在此期间，Q值的更新为![Q(s,a) \gets r + \gamma\max_{a'}Q^{(target)}(s,a)](https://www.zhihu.com/equation?tex=Q%28s%2Ca%29+%5Cgets+r+%2B+%5Cgamma%5Cmax_%7Ba%27%7DQ%5E%7B%28target%29%7D%28s%2Ca%29)，和之前的Q-learning算法更新步骤：用上一步的Q-function来计算更新下一步不同。

**原因分析：**

对于Replay memory，

- 首先由于在前一节说过对![\mathcal TQ](https://www.zhihu.com/equation?tex=%5Cmathcal+TQ)的采样![\widehat{\mathcal TQ}](https://www.zhihu.com/equation?tex=%5Cwidehat%7B%5Cmathcal+TQ%7D)是与采样所用的策略无关的（off-policy），因此用先前采样得到的状态转移任然可以用来计算之后的Q-值更新，因此这个方法是合理的。
- 而且每一个状态转移![(\phi_t,a_t,r_t,\phi_{t+1})](https://www.zhihu.com/equation?tex=%28%5Cphi_t%2Ca_t%2Cr_t%2C%5Cphi_%7Bt%2B1%7D%29)往往是重复遇到的，也就是说很可能在参数更新时用多次同一状态转移来更新，因此采用Replay memory是有效利用了数据
- 而且有一些状态的Q-值的更新增长是比较缓慢的，但很稳定，因此需要多次更新才能达到真实Q值。
- 直接从连续地采样中学习是很低效的，因为在连续采样时数据之间有很强的依赖性，而将他们放入到 memory![\mathcal D](https://www.zhihu.com/equation?tex=%5Cmathcal+D)中，并随机选取，可以打破这种依赖性，让数据分布得更均匀，就减小了更新的方差，防止了发散或振荡，（强的依赖性会导致网络在一段时间内仅往一个方向进行更新，然后再换一个方向，这样方差很大）
- 或者说是历史数据包含了不同策略（由![Q^{(n)},Q^{(n-1)},...](https://www.zhihu.com/equation?tex=Q%5E%7B%28n%29%7D%2CQ%5E%7B%28n-1%29%7D%2C...)导出）搜集到的数据，通过放入到大的统一的memory![\mathcal D](https://www.zhihu.com/equation?tex=%5Cmathcal+D)中，可以让不同策略下更新所用的数据差不多，那么Q值的更新就变得更加稳定，虽然更慢了。

对于 Target network，而不是直接采用当前 Q 来计算backup，原因是想模仿batch Q-value iteration，希望能是固定的目标![\mathcal TQ^{target}](https://www.zhihu.com/equation?tex=%5Cmathcal+TQ%5E%7Btarget%7D)而不是移动的目标，毕竟和回归任务相比，loss function中最大的不同是![||\mathcal TQ^{(n)}-Q^{(n+1)}||^2](https://www.zhihu.com/equation?tex=%7C%7C%5Cmathcal+TQ%5E%7B%28n%29%7D-Q%5E%7B%28n%2B1%29%7D%7C%7C%5E2)中目标是移动的，如果能固定下来的话，能增加算法的收敛性。（而且一步梯度下降往往更新得不彻底，需要多次更新才能达到![\mathcal TQ^{(n)}](https://www.zhihu.com/equation?tex=%5Cmathcal+TQ%5E%7B%28n%29%7D)）

## **Q-value有意义？**

在论文中还实验了多次更新步骤后，Q-值的变化情况：

![img](https://pic4.zhimg.com/80/v2-b8c449cae7092b43e655f0bcf41f23fb_hd.jpg)

这是在两个游戏中，机器得到的平均reward和神经网络给出的某一状态的Q-值的变化曲线，可以看出虽然reward的振荡（左两图）很剧烈，但是Q-值（右两图）是很稳定地改变，这说明了算法的稳定性（尽管无理论证明）。同时进一步地实验，发现网络的确在更优的动作上给出了更高的Q-值：

![img](https://pic1.zhimg.com/80/v2-5b9391340ff1010727972c4767e97462_hd.jpg)

然而在新的实验中，发现DQN网络给出的Q-值往往偏高：

![img](https://pic1.zhimg.com/80/v2-37a1d1753e2b44d4e999a5c368f260fc_hd.jpg)

其中红色振荡线为DQN给出的Q-值，红色直线为真实Q-值，蓝线为Double DQN（之后会讲）。这说明了虽然DQN给出的Q-值明显偏高了，但是从它的表现上来看，这种偏差并不影响它做正确的动作，也就是说只是基线提高了，动作之间的差别还是对的。然而我们很自然地会期望能不能估计出正确的Q-值，也许这么做会有利于结果，Double Q-learning给出了解答。

## **Double Q-learning**

在2010年的论文《[Double Q-learning](https://link.zhihu.com/?target=https%3A//papers.nips.cc/paper/3964-double-q-learning.pdf)》中指出单estimator的偏差问题：

对于一组随机变量![{X_1,…,X_M}](https://www.zhihu.com/equation?tex=%7BX_1%2C%E2%80%A6%2CX_M%7D)想要考察：![\max_iE[X_i]](https://www.zhihu.com/equation?tex=%5Cmax_iE%5BX_i%5D)。

对于每个![E[X_i]](https://www.zhihu.com/equation?tex=E%5BX_i%5D)可以用采样集![S_i](https://www.zhihu.com/equation?tex=S_i)来近似：![E[X_i]=E[\mu_i]\approx \mu_i(S_i)=\frac{1}{|S_i|}\sum_{s\in S_i}s](https://www.zhihu.com/equation?tex=E%5BX_i%5D%3DE%5B%5Cmu_i%5D%5Capprox+%5Cmu_i%28S_i%29%3D%5Cfrac%7B1%7D%7B%7CS_i%7C%7D%5Csum_%7Bs%5Cin+S_i%7Ds)，其中![\mu_i](https://www.zhihu.com/equation?tex=%5Cmu_i)表示对![X_i](https://www.zhihu.com/equation?tex=X_i)的估计，容易得出这个估计是无偏估计。

然而在论文中说明了![max_i\mu_i(S)](https://www.zhihu.com/equation?tex=max_i%5Cmu_i%28S%29)是对![E[max_i\mu_i]](https://www.zhihu.com/equation?tex=E%5Bmax_i%5Cmu_i%5D)，即![E[max_iX_i]](https://www.zhihu.com/equation?tex=E%5Bmax_iX_i%5D)的无偏估计。另一方面，容易证明![E[max_iX_i]\ge max_iE[X_i]](https://www.zhihu.com/equation?tex=E%5Bmax_iX_i%5D%5Cge+max_iE%5BX_i%5D)。因此通过采样得到的![max_i\mu_i(S)](https://www.zhihu.com/equation?tex=max_i%5Cmu_i%28S%29)是对![max_iE[X_i]](https://www.zhihu.com/equation?tex=max_iE%5BX_i%5D)**过于积极的估计**，估计偏高。

反观Q-learning，Q-值的噪声是很大的，因此含有采样的![r+\gamma\max_{a'}Q(s,a')](https://www.zhihu.com/equation?tex=r%2B%5Cgamma%5Cmax_%7Ba%27%7DQ%28s%2Ca%27%29)是对Q-值的偏高估计。

**解决方法**：采用两个网络 双estimator![Q_A,Q_B](https://www.zhihu.com/equation?tex=Q_A%2CQ_B)，互相计算另一个的argmax：

![img](https://pic3.zhimg.com/80/v2-1b5313381aa9672656d1efab6f61fd28_hd.jpg)

在论文中证明了双estimator能有效降低偏差，这个也在上一段的实验（图中蓝线）中体现了。

**与Q-learning联系：**

上一篇paper是在DQN之前发表的，那么自然地就想用Double Q-learning的思想对DQN进行改进，于是在2015年便有paper：《[Deep reinforcement learning with double Q-learning](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1509.06461)》，其中就把Double Q-learning的思想应用到DQN中：

- 原始的DQN：

![img](https://pic1.zhimg.com/80/v2-582505fad3cfb0cd80be8522b2826b68_hd.jpg)

- Double DQN（或者说是half DQN）

![img](https://pic3.zhimg.com/80/v2-3efe9b56569227252934bd2c01b97c70_hd.jpg)

用当前的Q-function来计算最优动作，而用target network来计算Q值，利用的Double Q-learning折中了原始的Q-learning与target network，并且有效地减小了估计偏差。

再介绍两个有用的技巧：

## **Dueling net：**

在2016年的paper《[Dueling network architectures for deep reinforcement learning](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1511.06581)》中，在DDQN基础上对Q-function的深度神经网络结构进行了改进：

![img](https://pic4.zhimg.com/80/v2-bc79696823da35cff7b74eacf2a1b2b3_hd.jpg)

上半网络是Value网络，估计![V(s)](https://www.zhihu.com/equation?tex=V%28s%29)，下半网络是Advantage网络，估计![A(s,a)](https://www.zhihu.com/equation?tex=A%28s%2Ca%29)的值，两者合并得到![Q(s,a)](https://www.zhihu.com/equation?tex=Q%28s%2Ca%29)值。（其思想和之前A3C的Actor和Critic参数共享有些相似）

Dueling net的动机是：想要分开估计Q-function中的V-value和Advantage：

![Q(s,a)=V(s)+A(s,a)](https://www.zhihu.com/equation?tex=Q%28s%2Ca%29%3DV%28s%29%2BA%28s%2Ca%29)，其中![V(s)](https://www.zhihu.com/equation?tex=V%28s%29)表示了状态本身的好坏，而![A(s,a)](https://www.zhihu.com/equation?tex=A%28s%2Ca%29)表示采取的动作的好坏，所以![A(s,a)](https://www.zhihu.com/equation?tex=A%28s%2Ca%29)决定了策略。

然而![V(s)](https://www.zhihu.com/equation?tex=V%28s%29)的量级会比![A(s)](https://www.zhihu.com/equation?tex=A%28s%29)大很多（由于![A^*(s,argmax_aQ^*(s,a))=0](https://www.zhihu.com/equation?tex=A%5E%2A%28s%2Cargmax_aQ%5E%2A%28s%2Ca%29%29%3D0)，![V(s)+A(s,a)=Q(s,a)\approx r+\gamma V^{(target)}(s')](https://www.zhihu.com/equation?tex=V%28s%29%2BA%28s%2Ca%29%3DQ%28s%2Ca%29%5Capprox+r%2B%5Cgamma+V%5E%7B%28target%29%7D%28s%27%29)，因此V大约是A的![1/(1-\gamma)](https://www.zhihu.com/equation?tex=1%2F%281-%5Cgamma%29)倍），这导致在权值更新中不平衡，网络会更关心![V(s)](https://www.zhihu.com/equation?tex=V%28s%29)值有没有估计对，然而这对做决策没有帮助（![A(s,a)](https://www.zhihu.com/equation?tex=A%28s%2Ca%29)之间的小差异才是决策关键）。

另一方面，从下图中可以看出：

![img](https://pic2.zhimg.com/80/v2-282546a838a5bf56de6d6bf150f25155_hd.jpg)

（红色表示机器注意的位置，通过求输入图像的Jacobi得到）上图显示了Value网络关注路况（尤其是海平线车子出现的位置）以及得分，而Advantage网络更关心前面车子的情况，来以免撞车。在很多情况下（上半图），动作的选择并不影响Q-值，Q由V完全决定，那么此时用duel net可以单独地训练Value网络，这样Value网络就能估计地更好。而在一些紧急情况（下半图），动作的Advantage就很重要了，分离网络就有助于单独地确定这些动作的Advantage的差别。

然而直接用式子![Q(s,a;\theta,\alpha,\beta)=V(s;\theta,\alpha)+A(s,a;\theta,\beta)](https://www.zhihu.com/equation?tex=Q%28s%2Ca%3B%5Ctheta%2C%5Calpha%2C%5Cbeta%29%3DV%28s%3B%5Ctheta%2C%5Calpha%29%2BA%28s%2Ca%3B%5Ctheta%2C%5Cbeta%29)来把Q分离成两块，效果并不好，因为它不具有可辨识性：通过![Q(s,a)](https://www.zhihu.com/equation?tex=Q%28s%2Ca%29)不能反过来确认V和A（相差一个常数项）。为此我们可以把注意力放在Advantage项，希望这项能在当前最优动作时取0，即将Q改写为：![Q(s,a;\theta,\alpha,\beta)=V(s;\theta,\alpha)+(A(s,a;\theta,\beta)-\max_{a'}A(s,a';\theta,\beta))](https://www.zhihu.com/equation?tex=Q%28s%2Ca%3B%5Ctheta%2C%5Calpha%2C%5Cbeta%29%3DV%28s%3B%5Ctheta%2C%5Calpha%29%2B%28A%28s%2Ca%3B%5Ctheta%2C%5Cbeta%29-%5Cmax_%7Ba%27%7DA%28s%2Ca%27%3B%5Ctheta%2C%5Cbeta%29%29)，则当![a^*=argmax_{a'}Q(s,a';\theta,\alpha,\beta)](https://www.zhihu.com/equation?tex=a%5E%2A%3Dargmax_%7Ba%27%7DQ%28s%2Ca%27%3B%5Ctheta%2C%5Calpha%2C%5Cbeta%29)时，![Q(s,a^*;\theta,\alpha,\beta)=V(s;\theta,\alpha)](https://www.zhihu.com/equation?tex=Q%28s%2Ca%5E%2A%3B%5Ctheta%2C%5Calpha%2C%5Cbeta%29%3DV%28s%3B%5Ctheta%2C%5Calpha%29)，因此Value网络给出了最优V-值的估计，那么相应的Advantage网络也可出了Advantage的估计。

在实际操作中，则采用减去A的平均：![Q(s,a;\theta,\alpha,\beta)=V(s;\theta,\alpha)+(A(s,a;\theta,\beta)-\frac{1}{|\mathcal A|}\sum_{a'}A(s,a';\theta,\beta))](https://www.zhihu.com/equation?tex=Q%28s%2Ca%3B%5Ctheta%2C%5Calpha%2C%5Cbeta%29%3DV%28s%3B%5Ctheta%2C%5Calpha%29%2B%28A%28s%2Ca%3B%5Ctheta%2C%5Cbeta%29-%5Cfrac%7B1%7D%7B%7C%5Cmathcal+A%7C%7D%5Csum_%7Ba%27%7DA%28s%2Ca%27%3B%5Ctheta%2C%5Cbeta%29%29)，虽然这样得到的V和A不在是有真实意义的Value和Advantage，但是能增加稳定性（因为采用![-\max_{a'}A(s,a';\theta,\beta)](https://www.zhihu.com/equation?tex=-%5Cmax_%7Ba%27%7DA%28s%2Ca%27%3B%5Ctheta%2C%5Cbeta%29)在更新时会损害最优A的值）。

这样把Value和Advantage网络分离，不同量级的梯度值能分别传入各自网络（通过RMSProp / ADAM改进SGD能实现）。

实验结果还是有一些提升的：

![img](https://pic1.zhimg.com/80/v2-33d296df263f521280b617348d6f32f3_hd.jpg)

## **Prioritized Replay：**

在2015年paper《[Prioritized Experience Replay](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1511.05952)》中通过重要性采样来对Replay Memory采样，提升训练速度。

之前用的loss function为Bellman error loss：![\sum_i||Q_{\theta}(s_i,a_i)-\hat Q_t||^2/2](https://www.zhihu.com/equation?tex=%5Csum_i%7C%7CQ_%7B%5Ctheta%7D%28s_i%2Ca_i%29-%5Chat+Q_t%7C%7C%5E2%2F2)。

可以根据memory![\mathcal D](https://www.zhihu.com/equation?tex=%5Cmathcal+D)中的transition i的梯度大小来进行重要性采样。这样梯度大的样本有更大的概率采样到，因此Q-网络更新会加快。

具体来讲，将采用Bellman error：![|\delta_i|=|Q_{\theta}(s_i,a_i)-\hat Q_t|](https://www.zhihu.com/equation?tex=%7C%5Cdelta_i%7C%3D%7CQ_%7B%5Ctheta%7D%28s_i%2Ca_i%29-%5Chat+Q_t%7C)作为优先级![p_i](https://www.zhihu.com/equation?tex=p_i)的评判标准（因为对Bellman error loss求梯度，容易看出梯度与![|\delta_i|](https://www.zhihu.com/equation?tex=%7C%5Cdelta_i%7C)成正比），然后用概率分布：![P(i)=\frac{p_i^{\alpha}}{\sum_k p_k^{\alpha}}](https://www.zhihu.com/equation?tex=P%28i%29%3D%5Cfrac%7Bp_i%5E%7B%5Calpha%7D%7D%7B%5Csum_k+p_k%5E%7B%5Calpha%7D%7D)进行采样。其中优先级![p_i](https://www.zhihu.com/equation?tex=p_i)有两种方式确定：

- proportional prioritization:![p_i=|\delta_i|+\epsilon](https://www.zhihu.com/equation?tex=p_i%3D%7C%5Cdelta_i%7C%2B%5Cepsilon)
- rank:![p_i=\frac{1}{rank_i}](https://www.zhihu.com/equation?tex=p_i%3D%5Cfrac%7B1%7D%7Brank_i%7D)，其中![rank_i](https://www.zhihu.com/equation?tex=rank_i)表示样本按照![|\delta_i|](https://www.zhihu.com/equation?tex=%7C%5Cdelta_i%7C)进行排序的顺序。

![img](https://pic3.zhimg.com/80/v2-576554475253b2d58d78f2f9c1a16ade_hd.jpg)

从实验中可以看出两种优先级方式都能有效地加速DQN。

## **实用的小tips：**

- DQN十分依赖于任务类型（如下图），因此先在一些较稳定的任务（如Pong and Breakout）中进行尝试，如果没得到较高的分数，那表示算法有问题。

![img](https://pic3.zhimg.com/80/v2-32f423644f76df75b53691e7df593b10_hd.jpg)

- 大的Replay Memory能有效提高DQN的稳定性，因此如何高效储存Memory很重要。（如使用uint8的图像）
- DQN收敛得十分缓慢，在GPU上都需要几个小时到几天时间才能发现得分明显高于随机策略。
- 使用Bellman loss的Huber loss：（类似clip gradient的作用）

![img](https://pic2.zhimg.com/80/v2-b462dcad46d912a02d9267a925ee59c1_hd.jpg)

- 使用DDQN能有效提高效果
- 测试数据的预处理过程，如down sample是否有损失过多的信息
- 使用至少两个seed，避免偶然情况
- 在一开始时使用大的learning rate
- 调整![\epsilon](https://www.zhihu.com/equation?tex=%5Cepsilon)-贪心策略的![\epsilon](https://www.zhihu.com/equation?tex=%5Cepsilon)来达到更好效果

## **总结：**

在这节中，我们承接了上一节Q-learning的基础内容，展示了很多先进的DQN技巧。

- 首先回顾了神经网络参数化Q-learning，指出投射与噪声带来的误差和不稳定。
- 介绍了著名的DQN算法，其核心技术是Replay Memory和Target Network，有助于减小振荡，提升效果。
- 在实验中发现DQN对Q-值的估计偏高，再经过理论分析，提出了Double Q-learning，并将它与DQN结合得到DDQN，来让网络能正确估计Q-值，以提升效果。
- 使用Dueling net来分离Value和Advantage，各自分离更新，来提升各自更新效果。使用Prioritized Replay来对重要的（梯度大的）样本多采样，以提升训练速度。
- 最后提出一些实用的tips来应对实际操作，提升实际质量。

到这节为止，已经把RL里的一些经典知识、主体知识介绍完毕了，至少已经学会了让自己的电脑玩Atari game的理论知识，要让自己的电脑在OpenAI的gym上得到高分还是需要多编程实践。之后的课程内容都是在这些知识基础上更为先进的研究结果，包括inverse learning, transfer learning等。

## **作业：**

[https://github.com/futurebelongtoML/homework/tree/master/hw3](https://link.zhihu.com/?target=https%3A//github.com/futurebelongtoML/homework/tree/master/hw3)

实验结果如下：

Pong：

​            

![img](https://pic3.zhimg.com/80/v2-8f3bda84cc2f12e0150880e469a7fd06_hd.jpg)

Breakout：

​            

![img](https://pic4.zhimg.com/80/v2-593f304ee627a3cbb03f57f9809fc4a7_hd.jpg)

Enduro：（视频太大了..）

![img](https://pic4.zhimg.com/80/v2-1518afa8fede52170d7e260ef5ceefd7_hd.jpg)

可以看出对于不同难度的游戏，DQN都能在1e7步左右达到较好水平。

另外我还做了Duel net和Double Q的实验，发现Duel net和Double Q对游戏成果的提升并不太大，但是通过tensorboard，可以看出使用Double Q后Q值有明显下降，说明原来的Q-learning对Q-值有偏高估计。

<div STYLE="page-break-after: always;"></div>

#第九章：Advanced Policy Gradient - Variance Reduction

## 降低策略梯度的方差

这节主要对第六节的**Policy Gradient**进行深入展开**，**讲了用reward reshaping来降低策略梯度的方差，以及更一般的GAE算法。由于Variance Reduction for Policy Gradient这节比较短，所以会附加讲一下实验效果。

## **reward reshaping:**

在很多情形下，环境的reward可能要很多步才会有改变，那么某一时间执行的动作需要很多步之后才会得到反馈，而且对于零散的奖励，从动作发生(t=0)到得到反馈(t=T)这一段时间中，路径中的动作几乎都是等价的，因为它们都共同在t=T得到几乎一样的反馈，**那么就无法单独评判路径上各个动作的好坏**，也就是说如果整体路径得到正向结果，路径中有些不太好的动作也会被鼓励了。

举一简单例子：对于一维的随机游动情形：状态空间为![S=\{-m,-m+1,…,n-1,n\}](https://www.zhihu.com/equation?tex=S%3D%5C%7B-m%2C-m%2B1%2C%E2%80%A6%2Cn-1%2Cn%5C%7D)，起始从0出发，每次动作为向左或向右(-1or+1)，若最后达到n奖励为1，达到-m奖励为-1，并达到两端后停止。

![img](https://pic2.zhimg.com/80/v2-b3b74db0be6e5b8ea599cd605bc4a84b_hd.jpg)

一次采样产生的路径如图1所示，最后达到了n，因此路径在最后才会有+1的奖励（图2）。为了解决奖励延迟，想法是在每一步都告诉agency哪一个方向会更好。在上例中，可以把reward更改（reward reshaping）为![\tilde r=r+s'-s](https://www.zhihu.com/equation?tex=%5Ctilde+r%3Dr%2Bs%27-s)，那么向右走，更改后的reward是1，向左-1，更改后路径的reward如图3所示。每一个时刻动作的好坏一目了然，让agency能更方便地学习。

实际上可以用任一函数![\phi](https://www.zhihu.com/equation?tex=%5Cphi)进行**reward reshaping**：![\tilde r(s,a,s')=r(s,a,s')+\phi(s')-\phi(s)](https://www.zhihu.com/equation?tex=%5Ctilde+r%28s%2Ca%2Cs%27%29%3Dr%28s%2Ca%2Cs%27%29%2B%5Cphi%28s%27%29-%5Cphi%28s%29)，而且能证明使用reshaping后的reward的最优策略和原来的reward的最优策略是一样的。

如果在policy gradient时使用reward reshaping：

![\large E[\nabla_{\theta}\log \pi_{\theta}(a_0|s_0)(\tilde r_0+\gamma\tilde r_1+…+\gamma^T\tilde r_T)]= \\ \large E[\nabla_{\theta}\log \pi_{\theta}(a_0|s_0)( r_0+\gamma r_1+…+\gamma^Tr_T+\gamma^{T+1}\phi(s_T)-\phi(s_0))]](https://www.zhihu.com/equation?tex=%5Clarge+E%5B%5Cnabla_%7B%5Ctheta%7D%5Clog+%5Cpi_%7B%5Ctheta%7D%28a_0%7Cs_0%29%28%5Ctilde+r_0%2B%5Cgamma%5Ctilde+r_1%2B%E2%80%A6%2B%5Cgamma%5ET%5Ctilde+r_T%29%5D%3D+%5C%5C+%5Clarge+E%5B%5Cnabla_%7B%5Ctheta%7D%5Clog+%5Cpi_%7B%5Ctheta%7D%28a_0%7Cs_0%29%28+r_0%2B%5Cgamma+r_1%2B%E2%80%A6%2B%5Cgamma%5ETr_T%2B%5Cgamma%5E%7BT%2B1%7D%5Cphi%28s_T%29-%5Cphi%28s_0%29%29%5D)

因此![\phi](https://www.zhihu.com/equation?tex=%5Cphi)实际上就是policy gradient中的baseline，而且在第6节中已经说明了**加入baseline后policy gradient是不改变的，只会改变方差**。这里不妨取![\phi=V](https://www.zhihu.com/equation?tex=%5Cphi%3DV)，这样的![\phi](https://www.zhihu.com/equation?tex=%5Cphi)有意义且能减小方差。

现在假设采用reward reshaping（且![\phi=V](https://www.zhihu.com/equation?tex=%5Cphi%3DV)）后令外部的![\gamma=0](https://www.zhihu.com/equation?tex=%5Cgamma%3D0)，那么就得到常用的**一步估计的policy gradient**：

![img](https://pic2.zhimg.com/80/v2-01c439787e7dd62602f17a0fb0c52ee8_hd.jpg)

> 多步估计的policy gradient，以及具体代码见第6节笔记的A2C/A3C部分

更一般的可以用![\gamma \lambda](https://www.zhihu.com/equation?tex=%5Cgamma+%5Clambda)，![(\lambda\in(0,1))](https://www.zhihu.com/equation?tex=%28%5Clambda%5Cin%280%2C1%29%29)作为外部的discount：（因为如果reward reshaping已经能根据V-值变化表示某一时刻的动作好坏，那么之后累积reward对这一时刻的作用就没那么大了）

![img](https://pic4.zhimg.com/80/v2-b17575f055d7da279459a46a735447f4_hd.jpg)

这个其实就是![TD(\lambda)](https://www.zhihu.com/equation?tex=TD%28%5Clambda%29)算法了。（在早年的RL研究中就有提出）

> 严格来说这个只是 ![\lambda](https://www.zhihu.com/equation?tex=%5Clambda) -return算法，真正的 ![TD(\lambda)](https://www.zhihu.com/equation?tex=TD%28%5Clambda%29) 还需要利用eligibility traces，eligibility traces只是计算上的差别（将 ![\lambda](https://www.zhihu.com/equation?tex=%5Clambda) -return算法转化为真正on-line的算法），具体实现细节见Sutton书的12节eligibility traces。

其等价于**generalized advantage estimation (GAE)，是在15年paper 《**[High-dimensional continuous control using generalized advantage estimation](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1506.02438)**》中被重新探究**：

之前定义的k步advantage估计：![\hat A_t^{(k)}=r_t+\gamma r_{t+1}+…+\gamma^{k-1} r_{t+k-1}+\gamma^k V(s_{t+k})-V(s_t)](https://www.zhihu.com/equation?tex=%5Chat+A_t%5E%7B%28k%29%7D%3Dr_t%2B%5Cgamma+r_%7Bt%2B1%7D%2B%E2%80%A6%2B%5Cgamma%5E%7Bk-1%7D+r_%7Bt%2Bk-1%7D%2B%5Cgamma%5Ek+V%28s_%7Bt%2Bk%7D%29-V%28s_t%29)

现在定义TD error：![\delta_t=r_t+\gamma V(s_{t+1})-V(s_t)](https://www.zhihu.com/equation?tex=%5Cdelta_t%3Dr_t%2B%5Cgamma+V%28s_%7Bt%2B1%7D%29-V%28s_t%29)（就是这节中reward reshaping）

容易看出：![\hat A_t^{(k)}= \delta_t+\gamma \delta_{t+1}+…+\gamma^{k-1} \delta_{t+k-1}](https://www.zhihu.com/equation?tex=%5Chat+A_t%5E%7B%28k%29%7D%3D+%5Cdelta_t%2B%5Cgamma+%5Cdelta_%7Bt%2B1%7D%2B%E2%80%A6%2B%5Cgamma%5E%7Bk-1%7D+%5Cdelta_%7Bt%2Bk-1%7D)

现在定义一般化的advantage估计，就是各步advantage的exponential平均：

![\qquad\hat A_t^{\lambda}=\hat A_t^{1}+\lambda\hat A_t^{2}+\lambda^2\hat A_t^{3}...](https://www.zhihu.com/equation?tex=%5Cqquad%5Chat+A_t%5E%7B%5Clambda%7D%3D%5Chat+A_t%5E%7B1%7D%2B%5Clambda%5Chat+A_t%5E%7B2%7D%2B%5Clambda%5E2%5Chat+A_t%5E%7B3%7D...)

![\qquad\hat A_t^{\lambda}=\delta_t+\gamma \lambda\delta_{t+1}+(\gamma \lambda)^2\delta_{t+2}+...](https://www.zhihu.com/equation?tex=%5Cqquad%5Chat+A_t%5E%7B%5Clambda%7D%3D%5Cdelta_t%2B%5Cgamma+%5Clambda%5Cdelta_%7Bt%2B1%7D%2B%28%5Cgamma+%5Clambda%29%5E2%5Cdelta_%7Bt%2B2%7D%2B...)

实验成果：

![img](https://pic3.zhimg.com/80/v2-08c3fe9ea7a78128405cb9f72a260858_hd.jpg)

可以看出适当调节超参数![\gamma](https://www.zhihu.com/equation?tex=%5Cgamma)和![\lambda](https://www.zhihu.com/equation?tex=%5Clambda)后结果有明显提升。

## **作业：**

homework4：[https://github.com/futurebelongtoML/homework/tree/master/hw4](https://link.zhihu.com/?target=https%3A//github.com/futurebelongtoML/homework/tree/master/hw4)

## 实验**：**

除了作业本身，我也有比较过A3C和DQN的好坏：

[futurebelongtoML/homework](https://link.zhihu.com/?target=https%3A//github.com/futurebelongtoML/homework/tree/master/hw3)

运行run_dqn_atari.py得到DQN的结果，运行run_A3C.py得到A3C的结果（我参考了这篇blog [https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-8-asynchronous-actor-critic-agents-a3c-c88f72a5e9f2](https://link.zhihu.com/?target=https%3A//medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-8-asynchronous-actor-critic-agents-a3c-c88f72a5e9f2)）

**在同一实验环境中：gym的Atari Game的Pong游戏**

​            

**A3C的学习曲线**：

![img](https://pic2.zhimg.com/80/v2-9ac70526861fdd3c55fa759e9c9cb99e_hd.jpg)

**实验配置**：使用了GPU的四个线程，即有四个agency同时平行地与游戏环境交互，收集到的数据都用来训练一个global的网络，因此可以期望这个平行处理过程会比只用一个agency快4倍。使用了这节的GAE，其中 ![\gamma=0.99,\lambda=0.97](https://www.zhihu.com/equation?tex=%5Cgamma%3D0.99%2C%5Clambda%3D0.97) 。

A3C的网络采用A3C论文《[Asynchronous Methods for Deep Reinforcement Learning](https://link.zhihu.com/?target=http%3A//proceedings.mlr.press/v48/mniha16.pdf)》的网络，即卷积网络+LSTM结构，而LSTM的输出，经过一个全连接层得到各个动作的偏好值（经过Softmax得到各个动作的概率），经过另一个全连接层得到当前状态的V-值。

**实验结果**：图中横轴是与游戏环境交互的总次数（即训练数据），纵轴是最近5个episode的平均episode奖励，可以看到大约在5e6次时，A3C的表现到达最好（episode奖励大于20），这也是Pong游戏能达到的最高分。所用时间只有不到3个小时。

**DQN：**

![img](https://pic3.zhimg.com/80/v2-8f3bda84cc2f12e0150880e469a7fd06_hd.jpg)

实验配置：GPU上使用一个agency，按照DQN的论文《[Playing Atari with Deep Reinforcement Learning](https://link.zhihu.com/?target=https%3A//www.cs.toronto.edu/%7Evmnih/docs/dqn.pdf)》，采用了replay memory和target net，使用仅卷积网络+全连接层，输出各动作Q-值。

实验结果：可以看出DQN也是大约在5e6次时，才能稳定在最优结果上，和A3C使用的总步数差不多，虽然DQN在开始提升地更快，在2e6次就到达了不错的水平。因此从与游戏环境交互的总次数来看，DQN效果更好，但是DQN运行5e6次的时间大约需要8个小时，而A3C只需要3个小时（充分利用4个Agency的平行处理），因此A3C总体优于DQN。

<div STYLE="page-break-after: always;"></div>

# **第十章：Advanced Policy Gradient - pathwise derivative**

这节接着第6节（策略梯度）的方向，探讨另一种策略梯度的方法（PD），以及提出、总结一些先进的策略梯度有关的思想，提出SVG和DDPG算法，应用到连续控制的环境中，得到了很好的表现，最后总结了policy gradient与Q-learning的优劣。

## **两种策略梯度的方法：得分函数（score function）和路径导数（pathwise derivative）**

首先来回顾一下之前讲的策略梯度方法：考虑期望函数：![E_{x\sim p(x|\theta)}(f(x))](https://www.zhihu.com/equation?tex=E_%7Bx%5Csim+p%28x%7C%5Ctheta%29%7D%28f%28x%29%29)，想要计算![\theta](https://www.zhihu.com/equation?tex=%5Ctheta)的梯度：

经过第6节的推导可知：![\nabla_{\theta}E_{x}[f(x)]=E_x[f(x)\nabla_{\theta}\log p(x|\theta)]](https://www.zhihu.com/equation?tex=%5Cnabla_%7B%5Ctheta%7DE_%7Bx%7D%5Bf%28x%29%5D%3DE_x%5Bf%28x%29%5Cnabla_%7B%5Ctheta%7D%5Clog+p%28x%7C%5Ctheta%29%5D)，在实际应用时，可以对x采样：![x_i\sim p(x|\theta)](https://www.zhihu.com/equation?tex=x_i%5Csim+p%28x%7C%5Ctheta%29)，来得到该期望梯度的无偏估计。现在把这种计算梯度的方法称为**score function（SF）**方法。这种方法的好处是无论![f(x)](https://www.zhihu.com/equation?tex=f%28x%29)是否连续，其导数是否可求，都能计算梯度。

## **pathwise derivative（PD）：**

另一方面（采用Reparameterized技巧），可以假定我们有一个随机变量![z](https://www.zhihu.com/equation?tex=z)服从某一固定分布，![x](https://www.zhihu.com/equation?tex=x)是![z](https://www.zhihu.com/equation?tex=z)的确定性（deterministic）函数，即![x=x(\theta,z)](https://www.zhihu.com/equation?tex=x%3Dx%28%5Ctheta%2Cz%29)，![\theta](https://www.zhihu.com/equation?tex=%5Ctheta)为确定性的参数，那么对x的采样就是，先对z的采样，再转换为x（实际上计算机中常用这种方法配合cdf来采样复杂分布）

因为对x的采样就是对z的采样，所以![\nabla_{\theta}E_{x}[f(x)]](https://www.zhihu.com/equation?tex=%5Cnabla_%7B%5Ctheta%7DE_%7Bx%7D%5Bf%28x%29%5D)等价于![\nabla_{\theta}E_{z}[f(x(\theta,z))]=E_{z}[\nabla_{\theta}f(x(\theta,z))]\approx \frac{1}{M}\sum_{i=1}^M f'\frac{\partial x}{\partial \theta}|_{z=z_i}](https://www.zhihu.com/equation?tex=%5Cnabla_%7B%5Ctheta%7DE_%7Bz%7D%5Bf%28x%28%5Ctheta%2Cz%29%29%5D%3DE_%7Bz%7D%5B%5Cnabla_%7B%5Ctheta%7Df%28x%28%5Ctheta%2Cz%29%29%5D%5Capprox+%5Cfrac%7B1%7D%7BM%7D%5Csum_%7Bi%3D1%7D%5EM+f%27%5Cfrac%7B%5Cpartial+x%7D%7B%5Cpartial+%5Ctheta%7D%7C_%7Bz%3Dz_i%7D)，

然而求这个梯度的梯度就需要![f](https://www.zhihu.com/equation?tex=f)是连续的，而且导数是已知的。我们称这种方法为**pathwise derivative（PD）**（称pathwise是因为当随机量z固定时，改变![\theta](https://www.zhihu.com/equation?tex=%5Ctheta)就是在改变整条路径x，然后根据路径的局部变动导致的奖励变动，来确定路径的改进方向。而SF则是固定了采样路径x，变动概率）

（如果记性好的话，会发现在第3节中“基于模型的强化学习的版本2.0“其实就是PD的思想，而当时是对于确定性策略而言的）

这样同样的期望梯度![\nabla_{\theta}E_{x}[f(x)]](https://www.zhihu.com/equation?tex=%5Cnabla_%7B%5Ctheta%7DE_%7Bx%7D%5Bf%28x%29%5D)有两种求解方法，对于连续已知的模型可用pathwise derivative，而对于不连续或未知的模型只能用score function。实际上，虽然两者都是对同一期望的估计，但PD的采样方差会比SF的方差小。

## **第三种：数值导数PD：**

很自然的想法，我们可以对PD中的f导数用数值导数来代替：（为方便起见，假设是一维情形）

![\qquad\nabla_{\theta}f(x(\theta,z)) \approx \frac{f(x(\theta+\sigma,z))-f(x(\theta-\sigma,z))}{2\sigma}](https://www.zhihu.com/equation?tex=%5Cqquad%5Cnabla_%7B%5Ctheta%7Df%28x%28%5Ctheta%2Cz%29%29+%5Capprox+%5Cfrac%7Bf%28x%28%5Ctheta%2B%5Csigma%2Cz%29%29-f%28x%28%5Ctheta-%5Csigma%2Cz%29%29%7D%7B2%5Csigma%7D)

实际上，这种方法和**score function（SF）**方法在有些情况下有相近的地方。

特别地当我们取![z \sim N(0,1), x=\theta+\sigma z](https://www.zhihu.com/equation?tex=z+%5Csim+N%280%2C1%29%2C+x%3D%5Ctheta%2B%5Csigma+z)时，通过简单的计算可以证明![f(x)\nabla_{\theta}\log p(x|\theta)=\frac{f(\theta+\sigma z))}{\sigma}z](https://www.zhihu.com/equation?tex=f%28x%29%5Cnabla_%7B%5Ctheta%7D%5Clog+p%28x%7C%5Ctheta%29%3D%5Cfrac%7Bf%28%5Ctheta%2B%5Csigma+z%29%29%7D%7B%5Csigma%7Dz)，因此在这种情况下SF做的是随机取动作空间中的一个方向z，考察这一方向的奖励大小。

## **SVG：将pathwise derivative（PD）应用到MDP中（需要连续情形）**

对于RL中的常见MDP情形：

![img](https://pic4.zhimg.com/80/v2-3fa074fd8e48970fc1931cbb670d99b4_hd.jpg)

所有动作服从带![\theta](https://www.zhihu.com/equation?tex=%5Ctheta)参数的策略分布，想要求![\nabla_{\theta}E_{\tau}[R(\tau)]](https://www.zhihu.com/equation?tex=%5Cnabla_%7B%5Ctheta%7DE_%7B%5Ctau%7D%5BR%28%5Ctau%29%5D)，可以用之前的SF：![E_\tau[R(\tau)\nabla_{\theta}\log p(\tau|\theta)]](https://www.zhihu.com/equation?tex=E_%5Ctau%5BR%28%5Ctau%29%5Cnabla_%7B%5Ctheta%7D%5Clog+p%28%5Ctau%7C%5Ctheta%29%5D)。

另一方面我们也可以用Reparameterized技巧：假设![a_t=\pi(s_t,z_t;\theta)](https://www.zhihu.com/equation?tex=a_t%3D%5Cpi%28s_t%2Cz_t%3B%5Ctheta%29)，![z_t](https://www.zhihu.com/equation?tex=z_t)服从某一固定分布，那么MDP就变为了：

![img](https://pic2.zhimg.com/80/v2-3f1bad2d7a58fd74d2c795041ac534e8_hd.jpg)

利用PD：![\nabla_{\theta}E_{z}[R_T(a(s_t,z_t;\theta))]=E_{z}[\nabla_{\theta}R_T(a(s_t,z_t;\theta))]](https://www.zhihu.com/equation?tex=%5Cnabla_%7B%5Ctheta%7DE_%7Bz%7D%5BR_T%28a%28s_t%2Cz_t%3B%5Ctheta%29%29%5D%3DE_%7Bz%7D%5B%5Cnabla_%7B%5Ctheta%7DR_T%28a%28s_t%2Cz_t%3B%5Ctheta%29%29%5D)，然而需要知道状态转移概率![P(s_2|s_1,a_1)](https://www.zhihu.com/equation?tex=P%28s_2%7Cs_1%2Ca_1%29)。而有模型的RL就是解决这个问题的，但是我们可以利用Q-function来代替解决：（**假定都是连续的**）

![img](https://pic1.zhimg.com/80/v2-a03e4e0c1cc28f448742b662dfc3b018_hd.jpg)

将外界的奖励用Q-function来代替，而Q-function可以用模型（nn）来近似。

这就是SVG(0)算法的思想，在15年paper《[Learning Continuous Control Policies by Stochastic Value Gradients](https://link.zhihu.com/?target=http%3A//papers.nips.cc/paper/5796-learning-continuous-control-policies-by-stochastic-value-gradients.pdf)》中被提出。

**SVG(0)算法**：利用![Q_{\phi}](https://www.zhihu.com/equation?tex=Q_%7B%5Cphi%7D)来近似![Q^{\pi,\gamma}](https://www.zhihu.com/equation?tex=Q%5E%7B%5Cpi%2C%5Cgamma%7D)，并以此进行策略梯度：

![img](https://pic1.zhimg.com/80/v2-e4fa45d5f02875e3f45d62a6f1ff0091_hd.jpg)

其中![TD(\lambda)](https://www.zhihu.com/equation?tex=TD%28%5Clambda%29)之后会说，可以认为是泛化的Advantage Estimation。可以说SVG算法和DQN有很大相似之处，都是根据当前Q-值来选择或优化策略，然后从采样路径中得到Bellman Backup或类似的target来更新Q。

这篇paper在这个思想上还做了很多变种算法：

**SVG(1)算法**：可以学习V-function：![V_{\phi}=V^{\pi,\gamma}](https://www.zhihu.com/equation?tex=V_%7B%5Cphi%7D%3DV%5E%7B%5Cpi%2C%5Cgamma%7D)和动态模型f：![s_{t+1}=f(s_t,a_t)+\zeta_t](https://www.zhihu.com/equation?tex=s_%7Bt%2B1%7D%3Df%28s_t%2Ca_t%29%2B%5Czeta_t)，然后根据这两者来定义Q-function。这里有个小trick，因为直接用近似的动态模型f，与真实情形会有很大的偏差不利于正确地确定Q。为此假设近似模型f与真实模型之前的差异是由较大的噪声![\zeta_t](https://www.zhihu.com/equation?tex=%5Czeta_t)引起的，给定采样路径![(s_t,a_t,s_{t+1})](https://www.zhihu.com/equation?tex=%28s_t%2Ca_t%2Cs_%7Bt%2B1%7D%29)噪声可以用![\zeta_t= s_{t+1}-f(s_t,a_t)](https://www.zhihu.com/equation?tex=%5Czeta_t%3D+s_%7Bt%2B1%7D-f%28s_t%2Ca_t%29)来求出。那么在计算Q值时：![Q(s_t,a_t)=E(r_t+\gamma V(s_{t+1}))=E(r_t+\gamma V(f(s_t,a_t)+\zeta_t)),a_t=\pi(s_t,z_t)](https://www.zhihu.com/equation?tex=Q%28s_t%2Ca_t%29%3DE%28r_t%2B%5Cgamma+V%28s_%7Bt%2B1%7D%29%29%3DE%28r_t%2B%5Cgamma+V%28f%28s_t%2Ca_t%29%2B%5Czeta_t%29%29%2Ca_t%3D%5Cpi%28s_t%2Cz_t%29)，在对其求导时，根据采样路径来固定所有的随机变量/噪声![\zeta_t,z_t](https://www.zhihu.com/equation?tex=%5Czeta_t%2Cz_t)，使用这个trick后，V的值就不会受动态模型f的较大偏差影响。

**SVG(** ![\infty](https://www.zhihu.com/equation?tex=%5Cinfty) **)算法**：SVG(1)是向后走了一步来确定Q-值，进一步，可以向后走无穷步来确定Q-值，那么只需要学习动态模型f：![s_{t+1}=f(s_t,a_t)+\zeta_t](https://www.zhihu.com/equation?tex=s_%7Bt%2B1%7D%3Df%28s_t%2Ca_t%29%2B%5Czeta_t)。那么给定一条路径，需要确定路径上所有的噪声，然后沿着路径反向传播梯度。在原paper中，采用了另一种推导方法，采用V-function的Bellman equation来推导V的梯度，根据导数公式来反向传播：（和基于模型的RL的梯度直接传入策略十分相似）

![img](https://pic4.zhimg.com/80/v2-c891dc4072bb0e1967478fdcdcff75e0_hd.jpg)

model-free的SVG(0)算法和model-based的SVG(1)、SVG(![\infty](https://www.zhihu.com/equation?tex=%5Cinfty))算法的优劣取决于外界的环境模型十分容易学到，而且与目标紧密相关。一般来说，对于固定的环境模型，而reward函数改变（如2D的连续控制任务），model-based的算法更好一些。在实验中，对于一些2D的连续控制任务，SVG(1)、SVG(![\infty](https://www.zhihu.com/equation?tex=%5Cinfty))能胜过A3C算法。而且加入Experience Replay之后，成果有显著提升。

> SVG算法与基于模型的RL的梯度直接传入策略思想相似且直接，区别是：梯度直接传入策略的return是有限T步return，只需要学习actor网络，而SVG是无限有折扣return，用值函数来代替无限return，因此除了要学习actor，还要学习更新critic网络。

## **确定性策略梯度Deterministic Policy Gradient（DPG）**

能观察到一下事实：对于服从Gauss分布的动作，采用score function策略梯度来估计，那么当Gauss分布的方差趋于0（确定性动作）时，SF估计的方差会趋于![\infty](https://www.zhihu.com/equation?tex=%5Cinfty)。这是因为在之前有证明，![z \sim N(0,1), x=\theta+\sigma z](https://www.zhihu.com/equation?tex=z+%5Csim+N%280%2C1%29%2C+x%3D%5Ctheta%2B%5Csigma+z)时，![f(x)\nabla_{\theta}\log p(x|\theta)=\frac{f(\theta+\sigma z))}{\sigma}z](https://www.zhihu.com/equation?tex=f%28x%29%5Cnabla_%7B%5Ctheta%7D%5Clog+p%28x%7C%5Ctheta%29%3D%5Cfrac%7Bf%28%5Ctheta%2B%5Csigma+z%29%29%7D%7B%5Csigma%7Dz)，当![\sigma\to0](https://www.zhihu.com/equation?tex=%5Csigma%5Cto0)，策略梯度的方差![\approx\frac{1}{\sigma^2}\to\infty](https://www.zhihu.com/equation?tex=%5Capprox%5Cfrac%7B1%7D%7B%5Csigma%5E2%7D%5Cto%5Cinfty)，因此SF方法对于确定性动作会有很大方差。

但对于SVG(0)梯度方差则是不会有很大影响：![\nabla_{\theta}\sum_tQ(s_t,\pi(s_t,\theta,z_t))](https://www.zhihu.com/equation?tex=%5Cnabla_%7B%5Ctheta%7D%5Csum_tQ%28s_t%2C%5Cpi%28s_t%2C%5Ctheta%2Cz_t%29%29)。

但确定性的问题是探索会不够，那探索时可以加点噪声（如 Ornstein-Uhlenbeck process作为噪声），然后用TD(0)来作为目标Q。只是这样的策略梯度有一些偏差，由于state的分布是由有噪声的策略产生的（当然如果Q和![\pi](https://www.zhihu.com/equation?tex=%5Cpi)有无限的表示能力，每个状态独立学习，那么就没有偏差了）。

在对Q网络更新时，值得注意的是TD(0)： ![Q_{\theta}=\delta,\delta=r+\gamma Q'-Q](https://www.zhihu.com/equation?tex=Q_%7B%5Ctheta%7D%3D%5Cdelta%2C%5Cdelta%3Dr%2B%5Cgamma+Q%27-Q) 为TD-error来作为Q的loss function时，这样的Q Bellman backups是off-policy的，与探索策略无关。这是因为目标Q= ![r+Q(s',\pi(s'))=r+E_{a'}(Q(s',\pi(a'|s'))](https://www.zhihu.com/equation?tex=r%2BQ%28s%27%2C%5Cpi%28s%27%29%29%3Dr%2BE_%7Ba%27%7D%28Q%28s%27%2C%5Cpi%28a%27%7Cs%27%29%29) ，后一个等式是由于 ![\pi](https://www.zhihu.com/equation?tex=%5Cpi) 是确定性策略，而最右边的式子就是expected Sarsa的目标Q，而expected Sarsa是off-policy，所以DPG也是off-policy的。

> 至于expected Sarsa是off-policy的，可以看看Sutton书的第六节。

## **DDPG（Deep Deterministic Policy Gradient）：**

在15年paper《[Continuous control with deep reinforcement learning](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/1509.02971.pdf)》中就将DPG与DQN结合起来，使用了DQN中的replay buffer和target network来增加稳定性。

因为之前讲了DPG是off-policy的，因此可以用replay memory来加快学习速度。

而DDPG中采用的target network与DQN的有所不同：DQN中的target network是1000个回合后才更新一次，而DDPG的target network每步都更新，但都只更新一部分：![（1-TAU）*target+TAU*origin](https://www.zhihu.com/equation?tex=%EF%BC%881-TAU%EF%BC%89%2Atarget%2BTAU%2Aorigin) ，TAU是个很小的数。

**DDPG算法：**

![img](https://pic2.zhimg.com/80/v2-d1bca3bf9b69a5a3ece06bf404fbfa05_hd.jpg)

其中目标Q用TD(0)：![\hat Q_t = r_t + \gamma Q_{\phi'}(s_{t+1},\pi(s_{t+1},\theta'))](https://www.zhihu.com/equation?tex=%5Chat+Q_t+%3D+r_t+%2B+%5Cgamma+Q_%7B%5Cphi%27%7D%28s_%7Bt%2B1%7D%2C%5Cpi%28s_%7Bt%2B1%7D%2C%5Ctheta%27%29%29)，其中 ![\phi',\theta'](https://www.zhihu.com/equation?tex=%5Cphi%27%2C%5Ctheta%27) 代表了使用target net来计算目标Q。

原论文中详细的算法，所有细节一览无余：其中在更新actor网络，计算 ![\frac{d}{da}Q(s,a)](https://www.zhihu.com/equation?tex=%5Cfrac%7Bd%7D%7Bda%7DQ%28s%2Ca%29) 时，采用了链式法则。

![img](https://pic4.zhimg.com/80/v2-c195c337e2cc26a3ebd5ca855405c5ef_hd.jpg)

SVG算法是DPG算法的随机版本，在实验中，对于简单2D连续控制任务，SVG的效果比DPG好一些。

## **Policy Gradient Methods vs Q-Function Regression Methods**

- Q-值迭代类型的算法更加sample-efficient，但是泛化性能较弱。（在实验中（如Atari），Q-learning能做好的，Policy Gradient也能做好）
- Policy Gradient容易Debug和理解（在hw4中可以看到）：不存在真空期，训练时学习曲线一般单调上升，可以用KL，entropy来对比新旧策略
- Q-值迭代能与探索策略很好的相容，因为它是off-policy的，replay memory能大大加快速度。
- Policy Gradient能与循环策略（使用RNN）很好相容
- Q-learning虽然可以用在连续动作中，但效果会很差。
- Policy Gradient能产生随机性的动作，并自动调节各个动作的概率。而Q-learning只能产生最greedy的动作。因此在状态空间有遮蔽的环境中有随机性的Policy Gradient会更好。

## **总结：**

这节主要总结了两种策略梯度的方法：

- REINFORCE / score function estimator：![\nabla_{\theta}\pi(s_t,\theta)\hat A_t](https://www.zhihu.com/equation?tex=%5Cnabla_%7B%5Ctheta%7D%5Cpi%28s_t%2C%5Ctheta%29%5Chat+A_t)

其中![\hat A_t](https://www.zhihu.com/equation?tex=%5Chat+A_t)为优势估计，用来减小方差，可以通过学习Q或V得到

- Pathwise derivative estimators：![E_{z}[\nabla_{\theta}R(a(s,z;\theta))]](https://www.zhihu.com/equation?tex=E_%7Bz%7D%5B%5Cnabla_%7B%5Ctheta%7DR%28a%28s%2Cz%3B%5Ctheta%29%29%5D)
- SVG(0)/DPG:![\frac{d}{da}Q(s,a)](https://www.zhihu.com/equation?tex=%5Cfrac%7Bd%7D%7Bda%7DQ%28s%2Ca%29)（学习Q, model-free, off-policy）
- SVG(1):![\frac{d}{da}(r+V(f(s,a)))](https://www.zhihu.com/equation?tex=%5Cfrac%7Bd%7D%7Bda%7D%28r%2BV%28f%28s%2Ca%29%29%29)（学习V和f, model-based, off-policy）
- SVG(![\infty](https://www.zhihu.com/equation?tex=%5Cinfty)):![\frac{d}{da_t}(r_t+\gamma r_{t+1}+\gamma^2r_{t+2}+…), where \quad r_{t}=r(s_t,a_t)](https://www.zhihu.com/equation?tex=%5Cfrac%7Bd%7D%7Bda_t%7D%28r_t%2B%5Cgamma+r_%7Bt%2B1%7D%2B%5Cgamma%5E2r_%7Bt%2B2%7D%2B%E2%80%A6%29%2C+where+%5Cquad+r_%7Bt%7D%3Dr%28s_t%2Ca_t%29)（学习f, model-based, on-policy）

总的来说，PD方法比SF更加sample-efficient和低方差，但是有很大bias，因此在很多情形下效果不太好。

## **DDPG实验：**

根据DDPG论文的算法以及其附录中的网络配置。应用于gym中的"MountainCarContinuous-v0"环境。

网络配置（论文）：

![img](https://pic4.zhimg.com/80/v2-a1a6334bec2bd58294d1c24ea8c53427_hd.jpg)

实验代码：[https://github.com/futurebelongtoML/RL_experiment/blob/master/ref_DDPG.py](https://link.zhihu.com/?target=https%3A//github.com/futurebelongtoML/RL_experiment/blob/master/ref_DDPG.py)，

关于代码部分，要感谢 

[@tx fan](https://www.zhihu.com/people/6eea0739ee059db5753d753b5c0e90d3)

实验结果：

![img](https://pic2.zhimg.com/80/v2-861f8fd8d3be105037f06a8f557ed010_hd.jpg)

可见在10个episode后，agency以及基本掌握了如何爬坡

100个episode时的结果：

​            

可以看出DDPG算法在适当调节参数后，在连续运动空间的环境中有十分出色的表现。

<div STYLE="page-break-after: always;"></div>

# **第十一章：Advanced Policy Gradient：TRPO, PPO**

这节也会讨论与第六节：策略梯度（Policy Gradient）相似的想法，但会用不同的推导思路，来推导出更加efficient，表现稳定提升的算法：NPG，TRPO以及PPO。

作为最后一节笔记，在最后会总结一下之前学过的4种RL思想。

## **损失函数和Improvement Theory：**

RL的目标是最大化策略得到的奖励总和，即策略的期望return（带有discount![\gamma](https://www.zhihu.com/equation?tex=%5Cgamma)）：

![img](https://pic2.zhimg.com/80/v2-deadeff4bce1b05e22c97ae6bfa19e22_hd.jpg)

常用的Monte Carlo思想和Policy Gradient思想是：根据当前的策略![\pi_{old}](https://www.zhihu.com/equation?tex=%5Cpi_%7Bold%7D)进行sample，根据sample到的路径，通过优化某一目标函数，来提升策略，即提升![\eta(\pi)](https://www.zhihu.com/equation?tex=%5Ceta%28%5Cpi%29)。（在之前的Policy Gradient算法中，以![\eta(\pi)](https://www.zhihu.com/equation?tex=%5Ceta%28%5Cpi%29)为目标，直接计算其梯度，采用梯度下降法）

首先我们可以证明一个**有用的等式**：

![img](https://pic2.zhimg.com/80/v2-1f4b3c8f6300d15466d2b37dd325236f_hd.jpg)

利用![A^{\pi_{old}}(s,a)=E_{s'\sim P(s'|s,a)}[r(s)+\gamma V^{\pi_{old}}(s')-V^{\pi_{old}}(s)]](https://www.zhihu.com/equation?tex=A%5E%7B%5Cpi_%7Bold%7D%7D%28s%2Ca%29%3DE_%7Bs%27%5Csim+P%28s%27%7Cs%2Ca%29%7D%5Br%28s%29%2B%5Cgamma+V%5E%7B%5Cpi_%7Bold%7D%7D%28s%27%29-V%5E%7B%5Cpi_%7Bold%7D%7D%28s%29%5D)即可证明。

如果令![\rho_{\pi}(s)=(P(s_0=s)+\gamma P(s_1=s)+…)](https://www.zhihu.com/equation?tex=%5Crho_%7B%5Cpi%7D%28s%29%3D%28P%28s_0%3Ds%29%2B%5Cgamma+P%28s_1%3Ds%29%2B%E2%80%A6%29)，可得到更简便的表达：

![\qquad\large \eta(\pi)=\eta(\pi_{old})+E_{s \sim \pi,a\sim \pi}[A^{\pi_{old}}]](https://www.zhihu.com/equation?tex=%5Cqquad%5Clarge+%5Ceta%28%5Cpi%29%3D%5Ceta%28%5Cpi_%7Bold%7D%29%2BE_%7Bs+%5Csim+%5Cpi%2Ca%5Csim+%5Cpi%7D%5BA%5E%7B%5Cpi_%7Bold%7D%7D%5D)

现在要利用这个等式来找到一个目标函数，作为优化目标，能够通过![\pi_{old}](https://www.zhihu.com/equation?tex=%5Cpi_%7Bold%7D)进行sample来提升策略（其实就是采用重要性采样importance sampling，off-policy RL中常用）：

![\qquad\large \eta(\pi)=\eta(\pi_{old})+E_{s \sim \pi,a\sim \pi_{old}}[\frac{\pi(a|s)}{\pi_{old}(a|s)}A^{\pi_{old}}]](https://www.zhihu.com/equation?tex=%5Cqquad%5Clarge+%5Ceta%28%5Cpi%29%3D%5Ceta%28%5Cpi_%7Bold%7D%29%2BE_%7Bs+%5Csim+%5Cpi%2Ca%5Csim+%5Cpi_%7Bold%7D%7D%5B%5Cfrac%7B%5Cpi%28a%7Cs%29%7D%7B%5Cpi_%7Bold%7D%28a%7Cs%29%7DA%5E%7B%5Cpi_%7Bold%7D%7D%5D)

但还需要改变s的分布，才能用![\pi_{old}](https://www.zhihu.com/equation?tex=%5Cpi_%7Bold%7D)进行sample得到上式的无偏估计。

为此不如直接定义一个**代替的目标函数![L_{\pi_{old}}(\pi)](https://www.zhihu.com/equation?tex=L_%7B%5Cpi_%7Bold%7D%7D%28%5Cpi%29)**，来代替![\eta(\pi)](https://www.zhihu.com/equation?tex=%5Ceta%28%5Cpi%29)（忽略s的分布改变）：（下式少加了常数![\eta(\pi_{old})](https://www.zhihu.com/equation?tex=%5Ceta%28%5Cpi_%7Bold%7D%29)，但对优化无影响）

![img](https://pic3.zhimg.com/80/v2-618b58bb5764ad67dc7afc8ef00a4ae9_hd.jpg)

这样的![L_{\pi_{old}}(\pi)](https://www.zhihu.com/equation?tex=L_%7B%5Cpi_%7Bold%7D%7D%28%5Cpi%29)是实际估计的（用![\pi_{old}](https://www.zhihu.com/equation?tex=%5Cpi_%7Bold%7D)进行sample即可），而且可以看到L的性质：![L_{\pi_{\theta_{old}}}(\pi_{\theta_{old}})=\eta(\pi_{\theta_{old}})](https://www.zhihu.com/equation?tex=L_%7B%5Cpi_%7B%5Ctheta_%7Bold%7D%7D%7D%28%5Cpi_%7B%5Ctheta_%7Bold%7D%7D%29%3D%5Ceta%28%5Cpi_%7B%5Ctheta_%7Bold%7D%7D%29)，以及

![img](https://pic1.zhimg.com/80/v2-35318b7ca2ff0f692c5998e23e29a531_hd.jpg)

也就是说在![\pi_{old}](https://www.zhihu.com/equation?tex=%5Cpi_%7Bold%7D)局部![L_{\pi_{\theta_{old}}}(\pi_{\theta})](https://www.zhihu.com/equation?tex=L_%7B%5Cpi_%7B%5Ctheta_%7Bold%7D%7D%7D%28%5Cpi_%7B%5Ctheta%7D%29)和![\eta(\pi_{\theta})](https://www.zhihu.com/equation?tex=%5Ceta%28%5Cpi_%7B%5Ctheta%7D%29)行为是一样的，但是步长增大的话就会有差别（即两者的一阶泰勒展开一样，高阶就有差别）。之前的Policy Gradient就是对![L_{\pi_{\theta_{old}}}(\pi_{\theta})](https://www.zhihu.com/equation?tex=L_%7B%5Cpi_%7B%5Ctheta_%7Bold%7D%7D%7D%28%5Cpi_%7B%5Ctheta%7D%29)一阶的梯度下降，也就是说只能在局部是保持![\eta](https://www.zhihu.com/equation?tex=%5Ceta)增长的，步长稍大一些，就不能保证了。

（步长大时，用![L_{\pi_{\theta_{old}}}(\pi_{\theta})](https://www.zhihu.com/equation?tex=L_%7B%5Cpi_%7B%5Ctheta_%7Bold%7D%7D%7D%28%5Cpi_%7B%5Ctheta%7D%29)代替![\eta(\pi_{\theta})](https://www.zhihu.com/equation?tex=%5Ceta%28%5Cpi_%7B%5Ctheta%7D%29)变得不准确，这是由忽略s的分布改变导致的）

**Improvement Theory：**为了解决这个问题，有theory能给出![L_{\pi_{\theta_{old}}}(\pi_{\theta})](https://www.zhihu.com/equation?tex=L_%7B%5Cpi_%7B%5Ctheta_%7Bold%7D%7D%7D%28%5Cpi_%7B%5Ctheta%7D%29)和![\eta(\pi_{\theta})](https://www.zhihu.com/equation?tex=%5Ceta%28%5Cpi_%7B%5Ctheta%7D%29)差异的界限：

![img](https://pic3.zhimg.com/80/v2-5527bb96afb18ad930d06216a0ed6194_hd.jpg)

(![\epsilon=\max A(s,a)](https://www.zhihu.com/equation?tex=%5Cepsilon%3D%5Cmax+A%28s%2Ca%29))这样给出了![\eta(\pi_{\theta})](https://www.zhihu.com/equation?tex=%5Ceta%28%5Cpi_%7B%5Ctheta%7D%29)的一个下界，同时可以看到当![\pi](https://www.zhihu.com/equation?tex=%5Cpi)取![\pi_{old}](https://www.zhihu.com/equation?tex=%5Cpi_%7Bold%7D)时不等式右边为![\eta(\pi_{\theta_{old}})](https://www.zhihu.com/equation?tex=%5Ceta%28%5Cpi_%7B%5Ctheta_%7Bold%7D%7D%29)，因此如果不等式右边能保持增加的话，就能保证![\eta(\pi_{\theta})](https://www.zhihu.com/equation?tex=%5Ceta%28%5Cpi_%7B%5Ctheta%7D%29)的单调提升。

## **实际优化：**

上述theory已经给出理论保障，只需优化：![\large L_{\pi_{old}}(\pi)-C\max_sKL[\pi_{old}(\cdot|s),\pi(\cdot|s)]](https://www.zhihu.com/equation?tex=%5Clarge+L_%7B%5Cpi_%7Bold%7D%7D%28%5Cpi%29-C%5Cmax_sKL%5B%5Cpi_%7Bold%7D%28%5Ccdot%7Cs%29%2C%5Cpi%28%5Ccdot%7Cs%29%5D)

在实际操作过程中：

- 用![\pi_{old}](https://www.zhihu.com/equation?tex=%5Cpi_%7Bold%7D)进行sample得到的路径来估计![L_{\pi_{old}}(\pi)](https://www.zhihu.com/equation?tex=L_%7B%5Cpi_%7Bold%7D%7D%28%5Cpi%29)：

![\hat L_{\pi_{old}}(\pi)=\sum_n\frac{\pi(a_n|s_n)}{\pi_{old}(a_n|s_n)}\hat A^{\pi_{old}}_n](https://www.zhihu.com/equation?tex=%5Chat+L_%7B%5Cpi_%7Bold%7D%7D%28%5Cpi%29%3D%5Csum_n%5Cfrac%7B%5Cpi%28a_n%7Cs_n%29%7D%7B%5Cpi_%7Bold%7D%28a_n%7Cs_n%29%7D%5Chat+A%5E%7B%5Cpi_%7Bold%7D%7D_n)如果只需要计算在![\pi_{old}](https://www.zhihu.com/equation?tex=%5Cpi_%7Bold%7D)处的梯度的话，

​        可以简单写成：![\Large \hat L_{\pi_{old}}(\pi)=\sum_n\log\pi(a_n|s_n)\hat A_n](https://www.zhihu.com/equation?tex=%5CLarge+%5Chat+L_%7B%5Cpi_%7Bold%7D%7D%28%5Cpi%29%3D%5Csum_n%5Clog%5Cpi%28a_n%7Cs_n%29%5Chat+A_n)

- 使用平均KL散度更佳：![\overline{KL}_{\pi_{old}}(\pi)= E_{s\sim\pi_{old}}[KL[\pi_{old}(\cdot|s),\pi(\cdot|s)]]](https://www.zhihu.com/equation?tex=%5Coverline%7BKL%7D_%7B%5Cpi_%7Bold%7D%7D%28%5Cpi%29%3D+E_%7Bs%5Csim%5Cpi_%7Bold%7D%7D%5BKL%5B%5Cpi_%7Bold%7D%28%5Ccdot%7Cs%29%2C%5Cpi%28%5Ccdot%7Cs%29%5D%5D)

​         实际用sample来估计：![\sum_nKL[\pi_{old}(\cdot|s_n),\pi(\cdot|s_n)]](https://www.zhihu.com/equation?tex=%5Csum_nKL%5B%5Cpi_%7Bold%7D%28%5Ccdot%7Cs_n%29%2C%5Cpi%28%5Ccdot%7Cs_n%29%5D)

- 对于常数C，如果用理论中的C的话，那么步长太小，太保守了，常用的处理方法有两种：

1. Natural policy gradient and PPO：使用定长或可调节的C
2. TRPO：将优化问题转化为在KL限制下的有限制的优化问题

接下来会对上述两种方法进行深入讨论

## **自然策略梯度 Natural Policy Gradient：**

要解决之前的优化问题：![\max_{\theta}L_{\pi_{\theta_{old}}}(\pi_{\theta})-C\overline{KL}_{\pi_{\theta_{old}}}(\pi_{\theta})](https://www.zhihu.com/equation?tex=%5Cmax_%7B%5Ctheta%7DL_%7B%5Cpi_%7B%5Ctheta_%7Bold%7D%7D%7D%28%5Cpi_%7B%5Ctheta%7D%29-C%5Coverline%7BKL%7D_%7B%5Cpi_%7B%5Ctheta_%7Bold%7D%7D%7D%28%5Cpi_%7B%5Ctheta%7D%29)，

对![L_{\pi_{\theta_{old}}}](https://www.zhihu.com/equation?tex=L_%7B%5Cpi_%7B%5Ctheta_%7Bold%7D%7D%7D)在![\pi_{old}](https://www.zhihu.com/equation?tex=%5Cpi_%7Bold%7D)处做一阶展开，对![\overline{KL}_{\pi_{\theta_{old}}}](https://www.zhihu.com/equation?tex=%5Coverline%7BKL%7D_%7B%5Cpi_%7B%5Ctheta_%7Bold%7D%7D%7D)做二阶展开（因为![L_{\pi_{\theta_{old}}}](https://www.zhihu.com/equation?tex=L_%7B%5Cpi_%7B%5Ctheta_%7Bold%7D%7D%7D)的二阶导相比![\overline{KL}_{\pi_{\theta_{old}}}](https://www.zhihu.com/equation?tex=%5Coverline%7BKL%7D_%7B%5Cpi_%7B%5Ctheta_%7Bold%7D%7D%7D)很小，而![\overline{KL}_{\pi_{\theta_{old}}}](https://www.zhihu.com/equation?tex=%5Coverline%7BKL%7D_%7B%5Cpi_%7B%5Ctheta_%7Bold%7D%7D%7D)的一阶导为0）：

![img](https://pic2.zhimg.com/80/v2-3cebd9ade4d6b8d2e6d50a9585f609db_hd.jpg)

对于这个二次优化问题，是有唯一的最优点：![\theta-\theta_{old}=\frac{1}{C}F^{-1}g](https://www.zhihu.com/equation?tex=%5Ctheta-%5Ctheta_%7Bold%7D%3D%5Cfrac%7B1%7D%7BC%7DF%5E%7B-1%7Dg)

出于计算复杂度的考虑，不能直接求Hessian矩阵F的逆。但是如果上过数值代数之类课程的，应该知道很多能解这个线性方程组（二次优化与线性方程组自然等价）的数值方法，其中最为流行的就是**共轭梯度法conjugate gradient（CG）**了。CG详细算法可以在任何一本数值代数或优化算法上找到，其介于梯度下降和牛顿法之间，不需要直接计算Hessian矩阵，只需要优化函数的一阶导信息。

- 详细地说，CG算法能近似解决![x=A^{-1}b](https://www.zhihu.com/equation?tex=x%3DA%5E%7B-1%7Db)（牛顿法结果）：在k步内，CG能在![b,Ab,A^2b,...,A^{k-1}b](https://www.zhihu.com/equation?tex=b%2CAb%2CA%5E2b%2C...%2CA%5E%7Bk-1%7Db)张成的子空间中找到![\frac{1}{2}xAx^T-xb](https://www.zhihu.com/equation?tex=%5Cfrac%7B1%7D%7B2%7DxAx%5ET-xb)在子空间中的最小值。
- 而且CG算法不需要实际生成完整的A矩阵，只需要能形成矩阵A与向量v相乘的函数：![v\to Av](https://www.zhihu.com/equation?tex=v%5Cto+Av)即可。
- 在这个具体应用中，也就是不需要计算出![H=F=\frac{\partial^2}{\partial^2\theta}\overline{KL}_{\pi_{\theta_{old}}}|_{\theta=\theta_{old}}](https://www.zhihu.com/equation?tex=H%3DF%3D%5Cfrac%7B%5Cpartial%5E2%7D%7B%5Cpartial%5E2%5Ctheta%7D%5Coverline%7BKL%7D_%7B%5Cpi_%7B%5Ctheta_%7Bold%7D%7D%7D%7C_%7B%5Ctheta%3D%5Ctheta_%7Bold%7D%7D)，只需要给任意一个向量v（和![\theta](https://www.zhihu.com/equation?tex=%5Ctheta)维数一样），能计算![v\to Hv](https://www.zhihu.com/equation?tex=v%5Cto+Hv)即可
- 具体实现就是（tensorflow）：分两次求导：

```
vector = tf.placeholder([dim_theta]) 
gradient = tf.grad(kl, theta) 
gradient_vector_product = tf.sum( gradient * vector ) 
hessian_vector_product = tf.grad(gradient_vector_product, theta)
```

其计算复杂度只有一阶导的2倍左右，但如果直接计算H，需要d倍复杂度

将上述过程总结可以得到

**Natural Policy Gradient算法**：

![img](https://pic1.zhimg.com/80/v2-f5ccb0bb80e1370cb95cd96e7bab62eb_hd.jpg)

其中常数C可以设置为常数，或者根据KL调节，而估计Advantage函数![\hat A_n](https://www.zhihu.com/equation?tex=%5Chat+A_n)，可以用之前A3C的方法（第六节），也可以用GAE（generalized advantage estimation）（第九节最后）

## **TRPO：Trust Region Policy Optimization**

对于之前的无限制优化问题：![\max_{\theta}L_{\pi_{\theta_{old}}}(\pi_{\theta})-C\overline{KL}_{\pi_{\theta_{old}}}(\pi_{\theta})](https://www.zhihu.com/equation?tex=%5Cmax_%7B%5Ctheta%7DL_%7B%5Cpi_%7B%5Ctheta_%7Bold%7D%7D%7D%28%5Cpi_%7B%5Ctheta%7D%29-C%5Coverline%7BKL%7D_%7B%5Cpi_%7B%5Ctheta_%7Bold%7D%7D%7D%28%5Cpi_%7B%5Ctheta%7D%29)，

**可以考虑相关的有限制优化问题**：![\max_{\theta}L_{\pi_{\theta_{old}}}(\pi_{\theta})](https://www.zhihu.com/equation?tex=%5Cmax_%7B%5Ctheta%7DL_%7B%5Cpi_%7B%5Ctheta_%7Bold%7D%7D%7D%28%5Cpi_%7B%5Ctheta%7D%29)服从![\overline{KL}_{\pi_{\theta_{old}}}(\pi_{\theta})\le\delta](https://www.zhihu.com/equation?tex=%5Coverline%7BKL%7D_%7B%5Cpi_%7B%5Ctheta_%7Bold%7D%7D%7D%28%5Cpi_%7B%5Ctheta%7D%29%5Cle%5Cdelta)，

那么解决有限制优化问题就近似解决了无限制优化问题，（这个也是合理的，因为KL散度是大于0的，那么最大化上式就需要KL尽可能小（比如小于能容忍的较小的常数![\delta](https://www.zhihu.com/equation?tex=%5Cdelta)），然后最大化![L_{\pi_{\theta_{old}}}(\pi_{\theta})](https://www.zhihu.com/equation?tex=L_%7B%5Cpi_%7B%5Ctheta_%7Bold%7D%7D%7D%28%5Cpi_%7B%5Ctheta%7D%29)，即转化为了有限制优化问题）

变为有限制优化问题的好处是：超参数![\delta](https://www.zhihu.com/equation?tex=%5Cdelta)比C要好确定，实际只需要取一个比较小的常数即可。而且这个有限制优化问题，能采取更大的步长，更快速训练。

对于上述的有限制优化问题可以用Lagrange Multiplier方法按照以下步骤求解：

 \1. 对![L_{\pi_{\theta_{old}}}](https://www.zhihu.com/equation?tex=L_%7B%5Cpi_%7B%5Ctheta_%7Bold%7D%7D%7D)做一阶展开，对![\overline{KL}_{\pi_{\theta_{old}}}](https://www.zhihu.com/equation?tex=%5Coverline%7BKL%7D_%7B%5Cpi_%7B%5Ctheta_%7Bold%7D%7D%7D)做二阶展开：

![\begin{equation} \max_{\theta}g(\theta-\theta_{old}), \\ subject\ to \frac{1}{2}(\theta-\theta_{old})^TF(\theta-\theta_{old})\le\delta \end{equation}](https://www.zhihu.com/equation?tex=%5Cbegin%7Bequation%7D+%5Cmax_%7B%5Ctheta%7Dg%28%5Ctheta-%5Ctheta_%7Bold%7D%29%2C+%5C%5C+subject%5C+to+%5Cfrac%7B1%7D%7B2%7D%28%5Ctheta-%5Ctheta_%7Bold%7D%29%5ETF%28%5Ctheta-%5Ctheta_%7Bold%7D%29%5Cle%5Cdelta+%5Cend%7Bequation%7D)

 \2. 做有限制优化问题的Lagrangian函数：![\qquad L(\theta,\lambda)=g(\theta-\theta_{old})-\frac{\lambda}{2}((\theta-\theta_{old})^TF(\theta-\theta_{old})-\delta)](https://www.zhihu.com/equation?tex=%5Cqquad+L%28%5Ctheta%2C%5Clambda%29%3Dg%28%5Ctheta-%5Ctheta_%7Bold%7D%29-%5Cfrac%7B%5Clambda%7D%7B2%7D%28%28%5Ctheta-%5Ctheta_%7Bold%7D%29%5ETF%28%5Ctheta-%5Ctheta_%7Bold%7D%29-%5Cdelta%29)

 \3. 用之前说的CG算法得到二次逼近的最优点：![\theta-\theta_{old}=\frac{1}{\lambda}F^{-1}g](https://www.zhihu.com/equation?tex=%5Ctheta-%5Ctheta_%7Bold%7D%3D%5Cfrac%7B1%7D%7B%5Clambda%7DF%5E%7B-1%7Dg)

 \4. 为了满足限制条件（KKT条件），需要更新方向s步长满足：![\frac{1}{2}s^TFs=\delta](https://www.zhihu.com/equation?tex=%5Cfrac%7B1%7D%7B2%7Ds%5ETFs%3D%5Cdelta)

因此对于之前得到的更新方向![s_{unscaled}=F^{-1}g](https://www.zhihu.com/equation?tex=s_%7Bunscaled%7D%3DF%5E%7B-1%7Dg)，需要rescale它的步长：

![\qquad\large s=\sqrt{\frac{2\delta}{s_{unscaled}^TFs_{unscaled}}}s_{unscaled}](https://www.zhihu.com/equation?tex=%5Cqquad%5Clarge+s%3D%5Csqrt%7B%5Cfrac%7B2%5Cdelta%7D%7Bs_%7Bunscaled%7D%5ETFs_%7Bunscaled%7D%7D%7Ds_%7Bunscaled%7D)

 \5. 线性搜索：上一步得到了等式约束的更新方向s，对于不等式约束（而且是凸函数），那么   只需要在方向s内进行线性搜索即可：使用步长![s,s/2,s/4,...](https://www.zhihu.com/equation?tex=s%2Cs%2F2%2Cs%2F4%2C...)直到优化目标![\max_{\theta}L_{\pi_{\theta_{old}}}(\pi_{\theta})](https://www.zhihu.com/equation?tex=%5Cmax_%7B%5Ctheta%7DL_%7B%5Cpi_%7B%5Ctheta_%7Bold%7D%7D%7D%28%5Cpi_%7B%5Ctheta%7D%29)有提升

将上述步骤用算法表示就是：

**TRPO算法：**

![img](https://pic2.zhimg.com/80/v2-7b830d836bd1cb88a3a52890850fd3d0_hd.jpg)

更多细节在15年paper《[Trust Region Policy Optimization](https://link.zhihu.com/?target=http%3A//proceedings.mlr.press/v37/schulman15.pdf)》中。

## **PPO：“Proximal” Policy Optimization**

在17年的paper：《[Proximal Policy Optimization Algorithms](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/1707.06347.pdf)》中：

实际上可以对Natural Policy Gradient算法做点改进和近似，得到更简单点的算法：

优化目标还是：![\max_{\theta}\hat L_{\pi_{\theta_{old}}}(\pi_{\theta})-\beta \overline{KL}_{\pi_{\theta_{old}}}(\pi_{\theta})](https://www.zhihu.com/equation?tex=%5Cmax_%7B%5Ctheta%7D%5Chat+L_%7B%5Cpi_%7B%5Ctheta_%7Bold%7D%7D%7D%28%5Cpi_%7B%5Ctheta%7D%29-%5Cbeta+%5Coverline%7BKL%7D_%7B%5Cpi_%7B%5Ctheta_%7Bold%7D%7D%7D%28%5Cpi_%7B%5Ctheta%7D%29)，

但是只进行梯度下降优化，而非牛顿法或CG的类似二阶的方法，即会求解![\hat L_{\pi_{\theta_{old}}}(\pi_{\theta})-\beta\overline{KL}_{\pi_{\theta_{old}}}(\pi_{\theta})](https://www.zhihu.com/equation?tex=%5Chat+L_%7B%5Cpi_%7B%5Ctheta_%7Bold%7D%7D%7D%28%5Cpi_%7B%5Ctheta%7D%29-%5Cbeta%5Coverline%7BKL%7D_%7B%5Cpi_%7B%5Ctheta_%7Bold%7D%7D%7D%28%5Cpi_%7B%5Ctheta%7D%29)的一阶导数

**PPO算法：**

![img](https://pic4.zhimg.com/80/v2-efaf666c841858dfbf9d2d96fbc485b2_hd.jpg)

与Natural Policy Gradient不同的是其可以根据KL来调节KL散度在Loss中的权重![\beta](https://www.zhihu.com/equation?tex=%5Cbeta)，而且只需要求一阶梯度即可。在实验中，PPO的表现能和TRPO差不多。（实际上，这种根据KL来调节的思想在homework4中也体现了：根据KL来调节学习率）

> 有一个小细节：之前说对KL散度在 ![\theta_{old}](https://www.zhihu.com/equation?tex=%5Ctheta_%7Bold%7D) 处的一阶导数为0，而PPO只做一阶导数的话，那KL项不就不起作用了？实际上，PPO算法中在收集到一个batch的data后会对 ![\hat L_{\pi_{\theta_{old}}}(\pi_{\theta})-\beta\overline{KL}_{\pi_{\theta_{old}}}(\pi_{\theta})](https://www.zhihu.com/equation?tex=%5Chat+L_%7B%5Cpi_%7B%5Ctheta_%7Bold%7D%7D%7D%28%5Cpi_%7B%5Ctheta%7D%29-%5Cbeta%5Coverline%7BKL%7D_%7B%5Cpi_%7B%5Ctheta_%7Bold%7D%7D%7D%28%5Cpi_%7B%5Ctheta%7D%29) 做多步的SGD，而且在这过程中， ![\theta_{old}](https://www.zhihu.com/equation?tex=%5Ctheta_%7Bold%7D) 保持不变，因此除了第一步的SGD，之后的KL梯度都不为0。

在论文中还提出了另一个Loss function：clipping loss

![img](https://pic4.zhimg.com/80/v2-7f196ec131725455dff9f2feccac91d8_hd.jpg)

容易看出这是policy gradient的一个下界，也能一定程度地保证更新后的 ![\theta](https://www.zhihu.com/equation?tex=%5Ctheta) 与 ![\theta_{old}](https://www.zhihu.com/equation?tex=%5Ctheta_%7Bold%7D) 相差不大，而且在连续控制实验中，取得最好效果（ ![\epsilon=0.2](https://www.zhihu.com/equation?tex=%5Cepsilon%3D0.2) ）

## **实验结果：**

![img](https://pic4.zhimg.com/80/v2-f890ed507d4e0650ee14fa690456045e_hd.jpg)

总体来说，TRPO或PPO算法比A2C算法在连续控制（较简单的输入）上能表现得更好，但是在高维的图像输入的Atari Game上，TRPO或PPO算法并没有显示出优势。

我自己做了一个简单的试验：[https://github.com/futurebelongtoML/RL_experiment/blob/master/TRPO_cartpole.py](https://link.zhihu.com/?target=https%3A//github.com/futurebelongtoML/RL_experiment/blob/master/TRPO_cartpole.py)（基于homework4）

![img](https://pic2.zhimg.com/80/v2-61bc77e4f7ad2c938a7729a413a97bb1_hd.jpg)

在完全一样的设置下，TRPO（黄色）能比A2C（蓝色）更快学到成果。

## **总结：**

这一节从等式：![\eta(\pi)=\eta(\pi_{old})+E_{s \sim \pi,a\sim \pi}[A^{\pi_{old}}]](https://www.zhihu.com/equation?tex=%5Ceta%28%5Cpi%29%3D%5Ceta%28%5Cpi_%7Bold%7D%29%2BE_%7Bs+%5Csim+%5Cpi%2Ca%5Csim+%5Cpi%7D%5BA%5E%7B%5Cpi_%7Bold%7D%7D%5D)出发，以policy gradient的思路，想要在原有策略![\pi_{old}](https://www.zhihu.com/equation?tex=%5Cpi_%7Bold%7D)基础上提升策略。为了能用![\pi_{old}](https://www.zhihu.com/equation?tex=%5Cpi_%7Bold%7D)进行sample，引入替代目标函数![L_{\pi_{old}}(\pi)=E_{s \sim \pi_{old},a\sim \pi_{old}}[\frac{\pi(a_n|s_n)}{\pi_{old}(a_n|s_n)}\hat A^{\pi_{old}}_n]](https://www.zhihu.com/equation?tex=L_%7B%5Cpi_%7Bold%7D%7D%28%5Cpi%29%3DE_%7Bs+%5Csim+%5Cpi_%7Bold%7D%2Ca%5Csim+%5Cpi_%7Bold%7D%7D%5B%5Cfrac%7B%5Cpi%28a_n%7Cs_n%29%7D%7B%5Cpi_%7Bold%7D%28a_n%7Cs_n%29%7D%5Chat+A%5E%7B%5Cpi_%7Bold%7D%7D_n%5D)，但是![L_{\pi_{\theta_{old}}}](https://www.zhihu.com/equation?tex=L_%7B%5Cpi_%7B%5Ctheta_%7Bold%7D%7D%7D)只和真正目标![\eta(\pi)](https://www.zhihu.com/equation?tex=%5Ceta%28%5Cpi%29)在局部一致，当步长增大，就无法保证![L_{\pi_{\theta_{old}}}](https://www.zhihu.com/equation?tex=L_%7B%5Cpi_%7B%5Ctheta_%7Bold%7D%7D%7D)是一个有效目标。

为了解决这个问题，提出了improvement theory：![\large \eta(\pi) \ge L_{\pi_{old}}(\pi)-C\max_sKL[\pi_{old}(\cdot|s),\pi(\cdot|s)]](https://www.zhihu.com/equation?tex=%5Clarge+%5Ceta%28%5Cpi%29+%5Cge+L_%7B%5Cpi_%7Bold%7D%7D%28%5Cpi%29-C%5Cmax_sKL%5B%5Cpi_%7Bold%7D%28%5Ccdot%7Cs%29%2C%5Cpi%28%5Ccdot%7Cs%29%5D)，

有了这个下界之后，就只需要优化不等式右边就能保证![\eta(\pi)](https://www.zhihu.com/equation?tex=%5Ceta%28%5Cpi%29)的提升。

在实际优化时，可以分为两种方法：

- Natural Policy Gradient：优化![\max_{\theta}L_{\pi_{\theta_{old}}}(\pi_{\theta})-C\overline{KL}_{\pi_{\theta_{old}}}(\pi_{\theta})](https://www.zhihu.com/equation?tex=%5Cmax_%7B%5Ctheta%7DL_%7B%5Cpi_%7B%5Ctheta_%7Bold%7D%7D%7D%28%5Cpi_%7B%5Ctheta%7D%29-C%5Coverline%7BKL%7D_%7B%5Cpi_%7B%5Ctheta_%7Bold%7D%7D%7D%28%5Cpi_%7B%5Ctheta%7D%29)，用对优化目标做二次逼近，得到牛顿法更新步：![\theta-\theta_{old}=\frac{1}{C}F^{-1}g](https://www.zhihu.com/equation?tex=%5Ctheta-%5Ctheta_%7Bold%7D%3D%5Cfrac%7B1%7D%7BC%7DF%5E%7B-1%7Dg)，考虑到计算复杂性，采用近似的二阶优化方法：共轭梯度法（CG），而且使用了小trick，不需要直接求出Hessian矩阵，只需要能计算![v\to Hv](https://www.zhihu.com/equation?tex=v%5Cto+Hv)即可
- PPO：“Proximal” Policy Optimization：优化![\max_{\theta}\hat L_{\pi_{\theta_{old}}}(\pi_{\theta})-C\overline{KL}_{\pi_{\theta_{old}}}(\pi_{\theta})](https://www.zhihu.com/equation?tex=%5Cmax_%7B%5Ctheta%7D%5Chat+L_%7B%5Cpi_%7B%5Ctheta_%7Bold%7D%7D%7D%28%5Cpi_%7B%5Ctheta%7D%29-C%5Coverline%7BKL%7D_%7B%5Cpi_%7B%5Ctheta_%7Bold%7D%7D%7D%28%5Cpi_%7B%5Ctheta%7D%29)，只进行一阶的SGD，但会根据KL大小来改变KL项权重C。
- TRPO：优化![\max_{\theta}L_{\pi_{\theta_{old}}}(\pi_{\theta})](https://www.zhihu.com/equation?tex=%5Cmax_%7B%5Ctheta%7DL_%7B%5Cpi_%7B%5Ctheta_%7Bold%7D%7D%7D%28%5Cpi_%7B%5Ctheta%7D%29)服从![\overline{KL}_{\pi_{\theta_{old}}}(\pi_{\theta})\le\delta](https://www.zhihu.com/equation?tex=%5Coverline%7BKL%7D_%7B%5Cpi_%7B%5Ctheta_%7Bold%7D%7D%7D%28%5Cpi_%7B%5Ctheta%7D%29%5Cle%5Cdelta)，转化为有限制优化问题，采用Lagrange Multiplier方法，在规划更新方向![s_{unscaled}=F^{-1}g](https://www.zhihu.com/equation?tex=s_%7Bunscaled%7D%3DF%5E%7B-1%7Dg)，并进行线性搜索。好处是超参数![\delta](https://www.zhihu.com/equation?tex=%5Cdelta)比C要好确定，且能采用较大步长。

Natural Policy Gradient这类方法比一般的Policy Gradient方法，能够保证![\eta(\pi)](https://www.zhihu.com/equation?tex=%5Ceta%28%5Cpi%29)单调增长，有更好的收敛性质，因此更加稳定，而且效率更高。

## **强化学习各种方法总结和一些Open Problem**

到目前为止，我们已经学习了很多的强化学习算法，从第一节的imitation learning到这节的TRPO，这些算法的总体思路可以分为四种：

1. 模仿学习imitation learning，将监督学习直接应用到RL环境中，根据有指导的事例进行学习，需要大量有标记的样本。
2. 有模型学习model-based RL，根据收集到的数据建立模型，然后根据model，可以用动态规划dynamic programming直接求解或指导策略，需要环境模型比较简单，而且要选好环境模型（NN，GP等），但比较data-efficient。
3. 无模型学习model-free RL中的Q-learning（以及Saras），思路是要学习到正确的值函数，然后根据值函数就能容易确定策略（greedy），由于Q-learning是off-policy的（用来sample的策略与bootstrap的策略可以不同），而且是on-line的（每行动一步都会更新值函数），因此可以用replay memory，大大提升了其效率。
4. 无模型学习model-free RL中的policy gradient，思路是直接学习策略函数，而不依赖值函数（虽然actor-critic会利用值函数），是on-policy和off-line的，gradient estimator的方差比较大，但能与异步asynchronous处理很好结合来加快速度，能处理连续动作的情况和部分观测情况，而且有时候策略本身会比环境模型和值函数更为简单，更容易建模些。同时与Q-learning相比本身带有exploration，不需要-greedy，因此能够产生更好的策略。

其中后三个方法能用一个high-level的示意图表示它们的三个主要过程：

![img](https://pic3.zhimg.com/80/v2-5d32dc68795e5d88858f7809c84f3809_hd.jpg)

根据当前策略在交互环境中sample路径 -> 调整环境模型/优化值函数 -> 提升策略，如此循环。

各种主要算法的效率比较（根据达到稳定的步数），可以从下图看出：

![img](https://pic1.zhimg.com/80/v2-5155aeaee62f353a2a427547a1846349_hd.jpg)

虽然图中有些算法应用在不同任务中，没有可比性，但是已经能大概说明各自效率。

现在已经有很多很好的算法了，它们主要要解决且还有待解决的是三个方面的problem：

1. 稳定性stability
2. 效率efficiency
3. 规模scale

稳定性不光指算法的收敛性，而且是算法关于超参数（比如学习率，网络模型）的稳定性，因为如果算法对超参数比较敏感的话，会需要很长时间进行调参。解决方法是希望算法有更好的收敛性质，比如DQN中的replay memory破坏数据间的联系和target net，以及TRPO算法能保证return单调上升。或者算法能够自动调参，如[Q-Prop算法](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/1611.02247.pdf)。

至于效率，上面那张图已经说明了各种算法的效率，可以根据实际情况进行选择各类算法，比如在现实环境中，一般用model-based的算法比较好，在复杂的模拟环境中可以用DQN或A3C。至于想要加快速率，DQN中的replay memory和A3C的异步处理起到重要作用。而Q-Prop将policy gradient与off-policy结合加快速率。此外可以利用之前的数据知识来加快学习过程：[RL2: Fast reinforcement learning via slow reinforcement learning](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/1611.02779.pdf)和[Learning to reinforcement learning](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/1611.05763.pdf)。（也称为meta learning，元学习）

要处理大规模的RL，比如AlphaGo，现实情形，那会遇到很多有变数的情形，就需要强大的泛化能力generalization，能在多任务中泛化策略，能用之前的经验和知识来提升当前策略。这方面有很多好的paper，列在Lecture的slide中。

<div STYLE="page-break-after: always;"></div>

# **专题补充：逆强化学习Inverse Reinforcement Learning**

在CS294的秋季课程中加入了一些新的内容，其中把逆强化学习课程内容完全翻新了，更为清晰，完整，连贯，因此值得系统学习一下。这一专题分成两个部分，分别对应两堂课的内容：1.Connections Between Inference and Control 2.Inverse Reinforcement Learning，前者用概率图的观点建立了优化控制与强化学习（Q-learning，policy gradient）之间联系，也可以解释人类动作，而它的推断过程又能引出第二部分逆强化学习。

## **Connections Between Inference and Control**

**问题的出发点是：我们用优化控制或强化学习得到的策略能用来解释人类的行为吗：**

![img](https://pic4.zhimg.com/80/v2-710203a7d67830538fab71faa2b2b4b0_hd.jpg)

我们之前使用优化控制或强化学习来解决上式，可以得到最优的策略。然而问题是人类或动物的动作并不是最优的，而是具有一些偏差的：比如让一个人去拿桌子上的一个橘子，那手的轨迹一定不是一条从起点到目标的直线，而是有一些弯曲的轨迹，也就是带有偏差的较优行为，但是这种偏差其实并不重要，只要最后拿到橘子就行了，也就是说1.有过程中一些偏差是不重要的，但另一些偏差就比较重要了（如最后没拿到）。而且每次拿橘子的动作也是不一样的，因此2.人的动作带有一定的随机性。由此我们可以认为3.人类行为轨迹分布是以最优策略为峰的随机分布。

## **决策过程的概率图模型**

为了解释人类的这种行为分布，如果再采用寻找最优的策略的思路就不太好了。为此我们引入概率图模型，虽然我们在第一节的时候就引入过这个概率图：

![img](https://pic3.zhimg.com/80/v2-d17d6d5173f20f1c72b7db7b6a60f0ae_hd.jpg)

但是这次要引入另一种概率图（最优模型，只用来模拟最优路径），其含义更为抽象，需要讲解一下：

![img](https://pic3.zhimg.com/80/v2-0ea76042890ab12e5e204e23bc182e6e_hd.jpg)

其中

1. 新节点![\mathcal{O_t}](https://www.zhihu.com/equation?tex=%5Cmathcal%7BO_t%7D)的含义比较抽象，引入它更多的是为数学上的解释（能让reward值以概率形式传播，而非只是Q-值），粗糙的含义是在![s_t,a_t](https://www.zhihu.com/equation?tex=s_t%2Ca_t)条件下人想要努力去获得当前奖励，称为optimality变量，只有![0,1](https://www.zhihu.com/equation?tex=0%2C1)两种取值，取1的概率（人想要达到最优）同比与当前奖励![p(\mathcal{O_t}|s_t,a_t)\varpropto exp(r(s_t,a_t))](https://www.zhihu.com/equation?tex=p%28%5Cmathcal%7BO_t%7D%7Cs_t%2Ca_t%29%5Cvarpropto+exp%28r%28s_t%2Ca_t%29%29)（奖励值高时人想要努力，低时人就不太想要努力），而且这个变量是可以被观测的。
2. ![s_t](https://www.zhihu.com/equation?tex=s_t)不再是![a_t](https://www.zhihu.com/equation?tex=a_t)的父节点了，**也就是说这里没有显式的策略![\pi(a|s)](https://www.zhihu.com/equation?tex=%5Cpi%28a%7Cs%29)**，那![s_t,a_t](https://www.zhihu.com/equation?tex=s_t%2Ca_t)的关系就要看![\mathcal{O_t}](https://www.zhihu.com/equation?tex=%5Cmathcal%7BO_t%7D)的取值了，从![\mathcal{O_t}](https://www.zhihu.com/equation?tex=%5Cmathcal%7BO_t%7D)取最优值可以反推出![s_t,a_t](https://www.zhihu.com/equation?tex=s_t%2Ca_t)的关系。作为父节点，我们需要给出![s_t,a_t](https://www.zhihu.com/equation?tex=s_t%2Ca_t)的先验分布![p(\tau)=p(s_{1:T},a_{1:T})](https://www.zhihu.com/equation?tex=p%28%5Ctau%29%3Dp%28s_%7B1%3AT%7D%2Ca_%7B1%3AT%7D%29)，这个代表了在物理环境允许的情况下可以做出的动作分布（比如在室内做出飞行的动作概率很小）

现在可以考察一下如果人每一步都很想达到最优，那人的轨迹分布是：

![img](https://pic4.zhimg.com/80/v2-c490cc87d9aaee8464def935c77c2582_hd.jpg)

最后的两项：![p(\tau)](https://www.zhihu.com/equation?tex=p%28%5Ctau%29)代表这条路径物理环境是否允许（即使很想要也总不能瞬间拿到橘子），![exp(\sum_tr(s_t,a_t))](https://www.zhihu.com/equation?tex=exp%28%5Csum_tr%28s_t%2Ca_t%29%29)代表这条路径的奖励值，因此人很想达到最优时，那奖励值越高的路径概率就越高。这两点都是十分合理的，而且最后得到的是一个路径分布，而非最优路径，因此这个分布能很好地解释人类的行为。

**既然这个模型能很好地解释人类的行为，那我们是否能充分地利用它呢？对它利用有以下三点：**

1. 这个模型能模拟suboptimal的动作轨迹（也就是给出一个在最优动作附近波动的动作分布），那么这种模型对inverse RL有很大意义，因为人类数据都可以说是suboptimal的，而inverse RL要据此找到optimal的。
2. 能用概率图的推断算法来解决控制和planning问题，联系了优化控制和强化学习
3. 解释了为什么带有随机性的策略更好，这对exploration和transfer问题很重要。

## **概率图Inference = 值迭代**

对于这个模型的利用采用概率图的推断算法，而有三种推断情形：

![img](https://pic4.zhimg.com/80/v2-9acb63429baa42a381be8166081f3323_hd.jpg)

> 其推导过程有些复杂，需要一些概率图知识，为了避免舍本逐末，减轻阅读压力，把这三种推断情形的详细解释放到附录中。

推导的结论是：

后向信息等价于值函数：

![img](https://pic1.zhimg.com/80/v2-3e42a9d45e3f639bb041cc854fb66c61_hd.jpg)

后向信息传播过程等价于值迭代过程。由后向信息传播导出的是soft max的值迭代：

![img](https://pic1.zhimg.com/80/v2-16490b0f04302b9f732f8a9e261530b3_hd.jpg)

![img](https://pic1.zhimg.com/80/v2-b1a2ab48a240f3f8cfa1684b220f8a3a_hd.jpg)

推断算法得出的policy等价于**Boltzmann exploration：**

![img](https://pic1.zhimg.com/80/v2-ea63fbb6787e744dcf0e515969d2e06f_hd.jpg)

**总结一下可以得到：**

## **Q-learning with soft optimality**

![img](https://pic3.zhimg.com/80/v2-e793015eff1df1bf897b9554beec5dac_hd.jpg)

只是改成Boltzmann exploration的策略，对应的Q-值更新使用soft max，而且这个更新也是off-policy的。

## **Policy gradient with soft optimality**

上面推导出的Boltzmann exploration的策略，与Policy gradient也有紧密联系：

![img](https://pic4.zhimg.com/80/v2-9056887d38359331e137299e65edb720_hd.jpg)

这是因为加入entropy项的Policy gradient目标等价于当前策略![\pi(a|s)](https://www.zhihu.com/equation?tex=%5Cpi%28a%7Cs%29)与最优策略![\frac{1}{Z}exp(Q(s,a))](https://www.zhihu.com/equation?tex=%5Cfrac%7B1%7D%7BZ%7Dexp%28Q%28s%2Ca%29%29)的KL散度：

![img](https://pic3.zhimg.com/80/v2-dbf6ad96d5082067112e7f9ca975bedc_hd.jpg)

那么由此加入entropy项的Policy gradient与使用duel net的soft Q-learning有联系，对![\pi](https://www.zhihu.com/equation?tex=%5Cpi)的更新与对![Q](https://www.zhihu.com/equation?tex=Q)的更新方式十分相似，由于篇幅有限，具体内容查看paper《Equivalence between policy gradients and soft Q- learning》和《 Bridging the gap between value and policy based reinforcement learning》

## **Soft Q-learning**

使用Q-learning with soft optimality时，在连续情况中有个小问题：Boltzmann exploration![\pi(a|s)\varpropto exp(Q(s,a))](https://www.zhihu.com/equation?tex=%5Cpi%28a%7Cs%29%5Cvarpropto+exp%28Q%28s%2Ca%29%29)无法直接采样，因为我们使用以下网络计算Q值，那么对于每一点我们只知道其概率值，而这个概率分布可能是十分复杂的，多峰的：

![img](https://pic3.zhimg.com/80/v2-46eb512d9f2ccf3f03310335b94bcae2_hd.jpg)

一种采样方法是利用SVGD，使用变分采样的方法：我们再训练一个采样网络，使其输出![\pi(a|s)\varpropto exp(Q(s,a))](https://www.zhihu.com/equation?tex=%5Cpi%28a%7Cs%29%5Cvarpropto+exp%28Q%28s%2Ca%29%29)

![img](https://pic1.zhimg.com/80/v2-537a63b2596a020e92c2911074dff800_hd.jpg)

说白了，这个和GAN十分相似，采样网络就是conditional GAN的generator，而原Q网络相当于一个discriminator。

## **使用soft optimality的好处**

1. 增加exploration，避免policy gradient坍缩成确定策略
2. （在一般情况中训练的soft optimality）更易于被finetune成更加specific的任务
3. 打破瓶颈，避免suboptimal。
4. 更好的robustness，因为允许最优策略以及最优路径有一定的随机性。训练Policy gradient时，不同的超参数选取很容易会落到不同的suboptimal动作中。
5. 能model人的动作（接下来的inverse RL）

## **逆强化学习Inverse RL**

之前我们的环境情形都是环境有一个客观的reward反应，或者可以根据人类的目的来设计一个reward函数，然而在很多的应用情形下这种客观的reward不存在，而且人类设计的reward也很难表示人类的目的，比如说想要机器学会倒水，甚至学会自然语言。但是我们有的是很多人类的数据，可以视为从一个最优策略分布中采样出的数据，那么我们是否可以根据这些数据来还原出reward函数，即知道人类的目的，然后使用这个reward来训练策略呢。

这个就是Inverse RL的思路，先根据数据得到reward函数，但是难点在于这些数据都是一个最优策略分布中采样出的，因此就像上一节所说的，这些样本是suboptimal，有随机性的。而且得到的reward函数也很难进行衡量。

传统上有对Inverse RL进行研究，但是由于传统研究方向和方法与当前的有很大的不同，所以将省略传统上对Inverse RL的研究。

既然上一节的概率图模型能很好地解释人类行为，那么是否这个模型能用来解决Inverse RL问题呢。

## **MaxEnt IRL算法**

由于环境的reward并不知道，所以我们希望用一个参数为![\psi](https://www.zhihu.com/equation?tex=%5Cpsi)神经网络来表示![r_{\psi}(s_t,a_t)](https://www.zhihu.com/equation?tex=r_%7B%5Cpsi%7D%28s_t%2Ca_t%29)，那么按照假设optimality变量![p(\mathcal{O_t}|s_t,a_t)\varpropto exp(r_{\psi}(s_t,a_t))](https://www.zhihu.com/equation?tex=p%28%5Cmathcal%7BO_t%7D%7Cs_t%2Ca_t%29%5Cvarpropto+exp%28r_%7B%5Cpsi%7D%28s_t%2Ca_t%29%29)，应用之前的概率图的结论可以得到**概率图模型下的最优路径分布**：

![img](https://pic3.zhimg.com/80/v2-4eb68c1cacac322a3cf5cbdb2fc13993_hd.jpg)

现在我们希望能根据数据来训练reward网络，方法就是最大化数据路径的概率图模型下的概率似然：

![img](https://pic2.zhimg.com/80/v2-2560a3d3b25e15045cc32b6a5bbf1958_hd.jpg)

等式右边前一项已经很清晰了，就是最大化数据的reward，但麻烦的部分是后面的归一化项![Z=\int p(\tau)exp(r_{\psi}(\tau))d\tau](https://www.zhihu.com/equation?tex=Z%3D%5Cint+p%28%5Ctau%29exp%28r_%7B%5Cpsi%7D%28%5Ctau%29%29d%5Ctau)，为此我们对其先求导：

![img](https://pic1.zhimg.com/80/v2-07d03a9d7f25a6529a797f37373f7f2e_hd.jpg)

可以看到经过求导，后一项可以化成期望形式，就简单许多：

![img](https://pic2.zhimg.com/80/v2-28d7e0f512ac9f39dfc5fee186fe69d9_hd.jpg)

**前一项是在增大数据分布的reward，后一项是在降低当前reward的最优模型给出的soft optimal策略。**

![img](https://pic2.zhimg.com/80/v2-673d4480641dacf11113370c728c5213_hd.jpg)

![img](https://pic1.zhimg.com/80/v2-39e033b4598efd1353c8121411e15b8d_hd.jpg)

因此从之前的概率图模型知识可以知道最优模型给出的soft optimal路径分布同比于后向信息![\beta_t](https://www.zhihu.com/equation?tex=%5Cbeta_t)与前向信息![\alpha_t](https://www.zhihu.com/equation?tex=%5Calpha_t)相乘。

这样一来Loss function导数中的两项我们都可以求了，那么就可以对reward网络进行更新：

**The MaxEnt IRL algorithm**：

![img](https://pic1.zhimg.com/80/v2-cbd73cf095120f6e4a4a28db7fc266c7_hd.jpg)

称为Max Entropy算法是因为（在线性情况）这等价于优化![max_{\psi}\mathcal{H}(\pi^{r_{\psi}}),s.t.E_{\pi^{r_{\psi}}}(r_{\psi})=E_{\pi^*}(r_{\psi})](https://www.zhihu.com/equation?tex=max_%7B%5Cpsi%7D%5Cmathcal%7BH%7D%28%5Cpi%5E%7Br_%7B%5Cpsi%7D%7D%29%2Cs.t.E_%7B%5Cpi%5E%7Br_%7B%5Cpsi%7D%7D%7D%28r_%7B%5Cpsi%7D%29%3DE_%7B%5Cpi%5E%2A%7D%28r_%7B%5Cpsi%7D%29)，即soft optimal策略与数据策略的平均reward相同时，熵最大的soft optimal模型。

## **sample-based MaxEnt IRL算法**

直接采用概率图模型推导出的MaxEnt IRL有个问题：需要计算后向信息![\beta_t](https://www.zhihu.com/equation?tex=%5Cbeta_t)与前向信息![\alpha_t](https://www.zhihu.com/equation?tex=%5Calpha_t)得到路径分布，并且做积分得到期望值，这对于大状态空间和连续空间是不可行的。因此一个想法是与其直接计算期望，不如用sample来估计。

因此改进方法为使用任一max-ent RL算法（如soft Q-learning，soft policy gradient）与环境互动，来得到当前reward下的路径样本，用这些路径来估计Loss function的后一项：

![img](https://pic2.zhimg.com/80/v2-d935638f1d505501b172435d3811424d_hd.jpg)

但这个方法也有一个显著问题：每一次更新reward网络，就要训练一遍max-ent RL算法，而RL算法训练常常需要成千上万步。因此与其每次reward更新时都训练一遍RL，不如每次reward更新时只改进一点原来的RL策略（由于reward每次更新很小，所以这个改进是合理的）。剩下的问题是有改进的RL路径样本对期望的估计成biased了，简单的方法是每次都用一个改进的RL路径样本batch来更新reward，但更data-efficient的方法是采用importance sampling（off-policy方法，在data-efficient专题再细讲）。

**与behavior cloning差异：**

Imitation learning中的behavior cloning也可以从数据中学习人类的行为，其直接对策略建模![\pi_{\theta}(a_t|s_t)](https://www.zhihu.com/equation?tex=%5Cpi_%7B%5Ctheta%7D%28a_t%7Cs_t%29)，最优化数据行为在策略模型中的概率似然，等价于求与数据行为分布KL散度最小的策略参数![\theta](https://www.zhihu.com/equation?tex=%5Ctheta)：

![\qquad L=-E_{(s,a)\sim p_{data}}(log(\pi_{\theta}(a|s)))](https://www.zhihu.com/equation?tex=%5Cqquad+L%3D-E_%7B%28s%2Ca%29%5Csim+p_%7Bdata%7D%7D%28log%28%5Cpi_%7B%5Ctheta%7D%28a%7Cs%29%29%29)

虽然behavior cloning和IRL都是目标为最大化数据行为的概率似然，然而它们建立的模型不同：BC直接对策略建模，而IRL利用soft optimal模型对reward进行建模，然后通过inference来得到策略。因此IRL的好处在于估计出了每步的reward，那么导出的策略会尽力最大化整条路径的reward和，能避免behavior cloning中的偏差问题![p_{data}(o_t)=p_{\pi_{\theta}}(o_t)](https://www.zhihu.com/equation?tex=p_%7Bdata%7D%28o_t%29%3Dp_%7B%5Cpi_%7B%5Ctheta%7D%7D%28o_t%29)。

而将会看到behavior cloning和IRL区别其实可以认为是普通auto-encoder和GAN的区别，GAN的D构建了能有复杂表示的目标（因此实验结果GAN优于其他方法），IRL的reward起到了一样的作用。

## **与GAN的联系**

对于接触过GAN的，上面过程十分像对抗学习过程，reward与policy的对抗：

![img](https://pic3.zhimg.com/80/v2-62e34068ec5a3ecfe6b49167108dbd87_hd.jpg)

reward网络就是discriminator，而policy网络就是generator。

**《Guided Cost Learning》 ICML 2016**

对于（标准GAN）discriminator的更新公式为：

![img](https://pic3.zhimg.com/80/v2-45fb62da244ade4bb9268ecfb0a237ed_hd.jpg)

因为最优分类器为：

![img](https://pic2.zhimg.com/80/v2-6b78381d615c4b62c1376f0cbafe9147_hd.jpg)

那么在IRL中我们假设D有如下参数化形式：

![img](https://pic2.zhimg.com/80/v2-969a6caecadfc2a1e007e9a2201612be_hd.jpg)

其中![\frac{1}{Z}exp(R_{\psi})](https://www.zhihu.com/equation?tex=%5Cfrac%7B1%7D%7BZ%7Dexp%28R_%7B%5Cpsi%7D%29)代表真实数据的概率分布，![q(\tau)](https://www.zhihu.com/equation?tex=q%28%5Ctau%29)为policy网络的分布。

将上式代入![L_{discriminator}](https://www.zhihu.com/equation?tex=L_%7Bdiscriminator%7D)就得到IRL中的reward更新公式。

接下来看generator/policy网络的更新过程：

![img](https://pic1.zhimg.com/80/v2-f53ce8664f53a85993689769604e2802_hd.jpg)

上一等式是generator网络的更新目标，而使用IRL的参数化形式后可以得到下一等式，就是policy网络目标。

**Generative Adversarial Imitation Learning Ho & Ermon, NIPS 2016**

但是在这篇paper中直接采用了更直接的GAN（而非之前较复杂的参数化形式）来训练IRL：discriminator就是使用标准GAN的D，即是个classifier对真实或生成样本输出（其是真实的）概率值，然后**把D的log值作为reward值**来训练policy：

![img](https://pic2.zhimg.com/80/v2-4bcd4fe30aa105974417fb2ef8b1d850_hd.jpg)

而这种更简单的形式在许多任务中取得了不错的结果：（然而原paper的论证过程绝不"简单"，我也还没完全读懂）

![img](https://pic2.zhimg.com/80/v2-757d33240f2efc8b5399b9e4f4be339b_hd.jpg)

## **总结：**

## **soft optimal model**

第一部分我们从解释人类的行为出发，人类行为suboptimal，有随机性，因此不能用之前的最优策略目标来解释。为此我们引入了一种特殊的概率图，称为soft optimal model，其中包含optimality变量![p(\mathcal{O_t}|s_t,a_t)\varpropto exp(r(s_t,a_t))](https://www.zhihu.com/equation?tex=p%28%5Cmathcal%7BO_t%7D%7Cs_t%2Ca_t%29%5Cvarpropto+exp%28r%28s_t%2Ca_t%29%29)携带了概率形式的reward信息。

之后采用了概率图的inference方法来推导

![img](https://pic4.zhimg.com/80/v2-9acb63429baa42a381be8166081f3323_hd.jpg)

其中发现后向信息等价于值函数：![Q(s_t,a_t)=\log\beta_t(s_t,a_t)](https://www.zhihu.com/equation?tex=Q%28s_t%2Ca_t%29%3D%5Clog%5Cbeta_t%28s_t%2Ca_t%29)，由于是soft optimal model，因此Q-value更新也是soft max的：![V(s_t)=\log E_{a_t}[exp(Q(s_t,a_t))]](https://www.zhihu.com/equation?tex=V%28s_t%29%3D%5Clog+E_%7Ba_t%7D%5Bexp%28Q%28s_t%2Ca_t%29%29%5D)。

第二点利用推断算法来计算策略![\pi(a|s)](https://www.zhihu.com/equation?tex=%5Cpi%28a%7Cs%29)，说明了inference=planning（在这个soft optimal model进行推断就等价于找到soft optimal的策略），而且发现了soft Q-learning算法，其策略为Boltzmann exploration![exp(Q(s,a)-V(s))](https://www.zhihu.com/equation?tex=exp%28Q%28s%2Ca%29-V%28s%29%29)。以及soft policy gradient算法，即目标函数加入entropy项。

这种soft optimal的算法有多种好处

1.增加exploration 2. 更易于被finetune成更加specific的任务 3.打破瓶颈，避免suboptimal。4.更好的robustness 5.能model人的动作

## **Inverse RL**

当只有人类的数据（suboptimal，有随机性）时，我们思路是用soft optimal model来学习出reward函数，那么根据soft optimal model来以最大数据的概率似然为目标，

![img](https://pic2.zhimg.com/80/v2-28d7e0f512ac9f39dfc5fee186fe69d9_hd.jpg)

由此直接推导出MaxEnt算法，但是为了应用到大状态空间中，将第二项用任一max-ent RL在当前reward下学到的策略样本路径来估计。

最后讨论了IRL与GAN的相似性：

![img](https://pic2.zhimg.com/80/v2-33eb2dcbea580b41994a759cf514db03_hd.jpg)

GAN的D可以被reward值参数化，来建立完全一致的联系。但是也可以更简单的，作为标准GAN来应用到IRL中，只是reward用log D替代。

## **附录：**

对于这个模型的利用采用概率图的推断算法，而有三种推断情形：

![img](https://pic4.zhimg.com/80/v2-9acb63429baa42a381be8166081f3323_hd.jpg)

接下来将仔细解释这三种推断情形。

## **后向信息Backward messages**

已知当前情形和动作![s_t,a_t](https://www.zhihu.com/equation?tex=s_t%2Ca_t)，那之后都想要达到最优的概率![\beta_t(s_t,a_t)=p(\mathcal{O_{t:T}}|s_t,a_t)](https://www.zhihu.com/equation?tex=%5Cbeta_t%28s_t%2Ca_t%29%3Dp%28%5Cmathcal%7BO_%7Bt%3AT%7D%7D%7Cs_t%2Ca_t%29)

![img](https://pic4.zhimg.com/80/v2-641fd09b254090b3659bce77ac8ed41c_hd.jpg)

其中离散情形时，积分换成求和即可。最后一项中![p(s_{t+1}|s_t,a_t)](https://www.zhihu.com/equation?tex=p%28s_%7Bt%2B1%7D%7Cs_t%2Ca_t%29)是环境转移概率，![p(\mathcal{O_t}|s_t,a_t)\varpropto exp(r(s_t,a_t))](https://www.zhihu.com/equation?tex=p%28%5Cmathcal%7BO_t%7D%7Cs_t%2Ca_t%29%5Cvarpropto+exp%28r%28s_t%2Ca_t%29%29)之前说了，现在要处理![p(\mathcal{O_{t+1:T}}|s_{t+1})](https://www.zhihu.com/equation?tex=p%28%5Cmathcal%7BO_%7Bt%2B1%3AT%7D%7D%7Cs_%7Bt%2B1%7D%29)这项，而且我们记它为![\beta_{t+1}(s_{t+1})](https://www.zhihu.com/equation?tex=%5Cbeta_%7Bt%2B1%7D%28s_%7Bt%2B1%7D%29)：

![img](https://pic4.zhimg.com/80/v2-bd241222ed2d6f720429543dc4b29035_hd.jpg)

我们把![p(a_{t+1}|s_{t+1})](https://www.zhihu.com/equation?tex=p%28a_%7Bt%2B1%7D%7Cs_%7Bt%2B1%7D%29)认为是先验分布，常为均匀分布（对于不均匀分布情况，![\log p(a_t|s_t)](https://www.zhihu.com/equation?tex=%5Clog+p%28a_t%7Cs_t%29)可以被添加到![r(s_t,a_t)](https://www.zhihu.com/equation?tex=r%28s_t%2Ca_t%29)中，因此均匀分布假设容易被推广，不细讲了）。

上述计算过程可以归结为下述循环，循环结束后每步的![\beta_t(s_t,a_t)](https://www.zhihu.com/equation?tex=%5Cbeta_t%28s_t%2Ca_t%29)就都知道了：

![img](https://pic3.zhimg.com/80/v2-8371e5db8deeb0415f7cb87233ebb540_hd.jpg)

由于这个循环过程从![T-1](https://www.zhihu.com/equation?tex=T-1)到1，所以信息是向后传播的。

## **后向信息与Q-learning的联系**

（由于![\mathcal{O}](https://www.zhihu.com/equation?tex=%5Cmathcal%7BO%7D)发生概率与reward成正比，那么想要之后的![\mathcal{O_{t:T}}](https://www.zhihu.com/equation?tex=%5Cmathcal%7BO_%7Bt%3AT%7D%7D)发生概率大的话，那就需要之后的reward大，因此![\beta](https://www.zhihu.com/equation?tex=%5Cbeta)可以接受到后面的reward信息来对当前![s_t,a_t](https://www.zhihu.com/equation?tex=s_t%2Ca_t)进行衡量，这点就自然和Q-value联系起来）

![img](https://pic1.zhimg.com/80/v2-3e42a9d45e3f639bb041cc854fb66c61_hd.jpg)

代入循环的第二步：![\large\beta_t(s_t)=E_{a_t\sim p(a_t|s_t)}[\beta_t(s_t,a_t)]](https://www.zhihu.com/equation?tex=%5Clarge%5Cbeta_t%28s_t%29%3DE_%7Ba_t%5Csim+p%28a_t%7Cs_t%29%7D%5B%5Cbeta_t%28s_t%2Ca_t%29%5D)，那么

![img](https://pic1.zhimg.com/80/v2-8f43288e5169d4545306315e421b847b_hd.jpg)

可以看到如果最大的![Q(s_t,a_t)](https://www.zhihu.com/equation?tex=Q%28s_t%2Ca_t%29)和其他的Q相比十分大时，那![V(s_t)=\max_{a_t}Q(s_t,a_t)](https://www.zhihu.com/equation?tex=V%28s_t%29%3D%5Cmax_%7Ba_t%7DQ%28s_t%2Ca_t%29)（Q-learning的第二步），因此![V(s_t)=\log E_{a_t}[exp(Q(s_t,a_t))]](https://www.zhihu.com/equation?tex=V%28s_t%29%3D%5Clog+E_%7Ba_t%7D%5Bexp%28Q%28s_t%2Ca_t%29%29%5D)是一种soft max（不要和softmax搞混了）。

与Q-learning比较：

![img](https://pic4.zhimg.com/80/v2-1a54b11f4f7eea0b0e13984460edcccd_hd.jpg)

**也就是把第二步的max改为一种soft max，而将会看到这个改动对应于soft optimality的策略。**

需要说明的是在循环的第一步中：

![img](https://pic2.zhimg.com/80/v2-ca08ae3f2d7ab6274928ca6e53902e5d_hd.jpg)

使用![Q(s_t,a_t)=r(s_t,a_t)+\log E_{s_{t+1}}[exp(V(s_{t+1}))]](https://www.zhihu.com/equation?tex=Q%28s_t%2Ca_t%29%3Dr%28s_t%2Ca_t%29%2B%5Clog+E_%7Bs_%7Bt%2B1%7D%7D%5Bexp%28V%28s_%7Bt%2B1%7D%29%29%5D)不好用，而用Q-learning对Q-Value更新更好，所以Q-learning第一步不需要改进。

因此后向信息其实可以理解为能量为值函数的能量函数，这点对应于![p(\mathcal{O_t}|s_t,a_t)](https://www.zhihu.com/equation?tex=p%28%5Cmathcal%7BO_t%7D%7Cs_t%2Ca_t%29)是以当前reward为能量的能量函数。

## **Policy computation**

我们最为关心的一点是根据这个模型导出的最优策略应该是怎样的，即计算![\pi(a_t|s_t)=p(a_t|s_t,\mathcal{O}_{1:T})](https://www.zhihu.com/equation?tex=%5Cpi%28a_t%7Cs_t%29%3Dp%28a_t%7Cs_t%2C%5Cmathcal%7BO%7D_%7B1%3AT%7D%29)：

![img](https://pic4.zhimg.com/80/v2-9fb810ea69b271e9e66cd9992c31dea6_hd.jpg)

因此这个模型导出的最优策略与Q值的exp成正比，是一种随机策略，而非Q-learning的greedy或![\epsilon](https://www.zhihu.com/equation?tex=%5Cepsilon)-greedy策略。

更进一步我们可以加上discount项![\gamma](https://www.zhihu.com/equation?tex=%5Cgamma)，以及temperature参数![\alpha](https://www.zhihu.com/equation?tex=%5Calpha)（加上这个参数，等价于把reward调小![\alpha](https://www.zhihu.com/equation?tex=%5Calpha)倍，而![\alpha\to0](https://www.zhihu.com/equation?tex=%5Calpha%5Cto0)时，那soft max退化为max，可以控制策略的exploration程度）：

![img](https://pic2.zhimg.com/80/v2-04d8d4f1a6f1b1470f6c8fd4f0a409b8_hd.jpg)

因此这个模型导出的最优策略与Q值的exp成正比，是一种随机策略，而非Q-learning的greedy或![\epsilon](https://www.zhihu.com/equation?tex=%5Cepsilon)-greedy策略。

更进一步我们可以加上discount项![\gamma](https://www.zhihu.com/equation?tex=%5Cgamma)，以及temperature参数![\alpha](https://www.zhihu.com/equation?tex=%5Calpha)（加上这个参数，等价于把reward调小![\alpha](https://www.zhihu.com/equation?tex=%5Calpha)倍，而![\alpha\to0](https://www.zhihu.com/equation?tex=%5Calpha%5Cto0)时，那soft max退化为max，可以控制策略的exploration程度）：

![img](https://pic1.zhimg.com/80/v2-c4f4617cb84dc0d3880265b3c54f18c8_hd.jpg)

![img](https://pic3.zhimg.com/80/v2-ea53f3a4ca8179326b48a9ee07c1aa8b_hd.jpg)

可以看出这个就是**Boltzmann exploration**。

## **Forward messages**

接下来想要知道![p(s_t|\mathcal{O}_{1:t-1})](https://www.zhihu.com/equation?tex=p%28s_t%7C%5Cmathcal%7BO%7D_%7B1%3At-1%7D%29)，即如果我每一步都是想要最优的，那么我t时刻会在哪里：

![img](https://pic3.zhimg.com/80/v2-07c3bfd1be5ececa053651d44b6dd016_hd.jpg)

这个计算过程没有什么好讲的地方。

而更重要的问题是![p(s_t|\mathcal{O}_{1:T})](https://www.zhihu.com/equation?tex=p%28s_t%7C%5Cmathcal%7BO%7D_%7B1%3AT%7D%29)也就是说我从头到尾每一步都要是最优的，那么我的状态分布应该是怎样的

![img](https://pic1.zhimg.com/80/v2-7ed242d1283f99ed6ff7b1e146087e0e_hd.jpg)

也就是前向信息与后向信息相乘，这个和概率图的置信信息校准一样，那么得到的就是状态的边缘分布：

![img](https://pic1.zhimg.com/80/v2-bd1699a2aafc6148203491b5a9e8526f_hd.jpg)

蓝色分布代表前向信息![p(s_t|\mathcal{O}_{1:t-1})](https://www.zhihu.com/equation?tex=p%28s_t%7C%5Cmathcal%7BO%7D_%7B1%3At-1%7D%29)，黄色代表后向信息![p(\mathcal{O_{t:T}}|s_{t})](https://www.zhihu.com/equation?tex=p%28%5Cmathcal%7BO_%7Bt%3AT%7D%7D%7Cs_%7Bt%7D%29)，而绿色区域就代表每一步都要是最优的轨迹分布。

