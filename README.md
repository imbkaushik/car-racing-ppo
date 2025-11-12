# PixelDrive: PPO Agent for Visual Car Racing

### Introduction

This project demonstrates a **continuous control task** in reinforcement learning, where an agent learns to control a car from **raw pixel inputs** using **Proximal Policy Optimization (PPO)**.

The environment is the **CarRacing-v0** environment from **OpenAI Gym**, a top-down racing simulator designed to test continuous control from high-dimensional visual inputs.

Each **state** is a `96×96` RGB image (the rendered frame).
Each **reward** is computed as follows:

* `+1000 / N` for every **track tile visited**, where `N` is the total number of tiles.
* `-0.1` per frame (to penalize unnecessary time).

For example, if the agent completes a track in **732 frames**,
its reward is:

```
Reward = 1000 - 0.1 * 732 = 926.8 points
```

To **solve** the environment, the agent must achieve an **average reward ≥ 900** over **100 consecutive episodes**.

<p align="center">
  <img src="images/plot_Reward_200-1000_0.8.png" width="600">
</p>

---

## Requirements

python==3.6  
numpy>=1.14  
gym==0.10.5  
torch==0.4.1  
torchvision==0.2.1  
matplotlib>=2.2  

---

## Environment Setup

The environment used for this project is provided by the **OpenAI Gym** package.
It can be instantiated as:

```python
import gym
env = gym.make('CarRacing-v0', verbose=0)
```

---

## Hyperparameters

The PPO agent is trained with the following hyperparameters:

| Parameter       | Value | Description                           |
| --------------- | ----- | ------------------------------------- |
| `GAMMA`         | 0.99  | Discount factor for future rewards    |
| `EPOCH`         | 8     | Number of PPO update epochs per batch |
| `MAX_SIZE`      | 2000  | Maximum size of replay buffer         |
| `BATCH`         | 128   | Batch size for optimizer updates      |
| `LEARNING_RATE` | 0.001 | Learning rate for optimizer           |
| `EPS`           | 0.1   | PPO clipping parameter                |

The **PPO clipped objective** is computed as:

```python
surr1 = ratio * advantage
surr2 = torch.clamp(ratio, 1.0 - EPS, 1.0 + EPS) * advantage
action_loss = -torch.min(surr1, surr2)
```

---

## Neural Network Architecture

<p align="center">
  <img src="images/network.png" width="500" height="500">
</p>

The policy and value networks are **CNN-based** models consisting of:

* **6 convolutional layers (`Conv2d`)**
* **6 ReLU activation functions**

The network estimates **two outputs**:

1. **Policy parameters** (`alpha`, `beta`) for the Beta distribution.
2. **Value function** for critic updates.

Reward, target, and advantage are calculated as:

```python
target = r + GAMMA * net(next_state)
advantage = target - net(state)[1]  # 1st return parameter from forward()
```

---

## Action Sampling — Beta Distribution

The agent’s policy outputs the parameters of a **Beta distribution**,
which models bounded continuous actions (a better alternative to Gaussian for `[0,1]` intervals).

```python
alpha, beta = net(state[index])[0]  # 0th return parameter
dist = Beta(alpha, beta)
```

**Reference Papers:**

* [Chou et al., 2017 – Improving Stochastic Policy Gradients in Continuous Control using the Beta Distribution](http://proceedings.mlr.press/v70/chou17a.html)
* [Fujita & Maeda, 2018 – Clipped Action Policy Gradient](https://arxiv.org/abs/1802.07564)

---

## PPO Update Mechanism

In standard policy gradients, a **single gradient update** is performed per data sample.
However, PPO introduces a **surrogate objective** that enables **multiple epochs** of updates per batch of collected data.

The PPO loss function ( L_t(\theta) ) is (approximately) maximized at each iteration:

<p align="center">
  <img src="images/objective_function_07.png" width="500">
</p>

Key hyperparameters in this process include:

* **c1**, **c2** (loss coefficients for value and entropy terms)
* **epoch** (number of optimization iterations per batch)

---

## Training the Agent

The agent learns to leverage **visual information** from surrounding frames to determine the **next optimal action**.

After training for approximately **6 hours and 53 minutes**, the agent achieved:

> **Average Reward: 901.81**
> **Episode: 2760**

<p align="center">
  <img src="images/plot_2760episodes.png" width="600">
</p>

### Last Recorded Episodes

| Episode | Timesteps | Score   | Avg. Score | Run Score | Time     |
| ------- | --------- | ------- | ---------- | --------- | -------- |
| 2750    | 94        | 1006.80 | 893.55     | 888.10    | 06:51:32 |
| 2751    | 100       | 992.34  | 893.43     | 889.14    | 06:51:41 |
| 2752    | 100       | 976.90  | 894.43     | 890.02    | 06:51:51 |
| 2753    | 100       | 871.87  | 894.01     | 889.84    | 06:52:01 |
| 2754    | 100       | 1000.60 | 894.20     | 890.95    | 06:52:10 |
| 2755    | 100       | 992.98  | 895.36     | 891.97    | 06:52:20 |
| 2756    | 100       | 941.98  | 895.57     | 892.47    | 06:52:30 |
| 2757    | 100       | 854.43  | 895.39     | 892.09    | 06:52:40 |
| 2758    | 100       | 989.55  | 895.27     | 893.06    | 06:52:49 |
| 2759    | 100       | 986.25  | 901.81     | 893.99    | 06:52:59 |

*Environment solved!*
**Running score:** 893.99
**Average score:** 901.81

---

## Learning from Raw Pixels

### 1. Moving Image to Dark Green

In RGB space, the **Dark Green vector** is represented as `[0.299, 0.587, 0.114]`.

Converting to integer gray levels:

```
[(int)(0.299*256), (int)(0.587*256), (int)(0.114*256)] = [76, 150, 29]
```

In hexadecimal:

```
[hex(76), hex(150), hex(29)] = ('0x4c', '0x96', '0x1d')
```

Thus, the **color code** is **#4d961d** → “Dark Green”.

---

### 2. Converting to Grayscale

Each pixel ( z = (a,b,c) ) is projected onto vector ( v = [0.299, 0.587, 0.114] ):

[
pr(z) = \frac{(z \cdot v)}{|v|}
]

Hence, pixel intensities become proportional to the Dark Green vector.

Example:

```python
image.shape = (433, 735, 3)
im_dot = np.dot(image, [0.299, 0.587, 0.114])
im_dot.shape = (433, 735)
```

The conversion function:

```python
img_gray = rgb2gray(img_rgb)
```

---

### 3. Stacking Frames for Temporal Context

To capture temporal information, **4 consecutive grayscale frames** are stacked to form a single **state** with shape `(4, 96, 96)`.

In `reset()` (within the `Wrapper` class):

```python
self.stack = [img_gray] * 4
```

At each timestep:

```python
stack.pop(0)
stack.append(img_gray)
```

This process ensures the model receives **spatial + temporal** input from recent frames.

---

## Watching the Trained Agent

Trained model weights are stored in the directory **`dir_chk/`**.
You can replay trajectories using the **`WatchAgent`** notebook.

---

### Evaluation Results

#### I. `model_weights_350-550.pth`

**Score range:** 350–550
Replay (5 episodes):

| Episode | Avg. Score | Score  | Time     |
| ------- | ---------- | ------ | -------- |
| 1       | 63.53      | 63.53  | 00:00:04 |
| 2       | 305.90     | 548.28 | 00:00:10 |
| 3       | 370.60     | 500.00 | 00:00:11 |
| 4       | 366.48     | 354.09 | 00:00:07 |
| 5       | 304.39     | 56.03  | 00:00:05 |

---

#### II. `model_weights_480-660.pth`

**Score range:** 480–660
Replay (5 episodes):

| Episode | Avg. Score | Score  | Time     |
| ------- | ---------- | ------ | -------- |
| 1       | 603.72     | 603.72 | 00:00:12 |
| 2       | 593.94     | 584.16 | 00:00:11 |
| 3       | 432.31     | 109.06 | 00:00:08 |
| 4       | 480.99     | 627.01 | 00:00:11 |
| 5       | 517.67     | 664.38 | 00:00:11 |

---

#### III. `model_weights_820-980.pth`

**Score range:** 820–980
Replay (5 episodes):

| Episode | Avg. Score | Score   | Time     |
| ------- | ---------- | ------- | -------- |
| 1       | 1003.80    | 1003.80 | 00:00:10 |
| 2       | 958.42     | 913.04  | 00:00:11 |
| 3       | 943.30     | 913.04  | 00:00:11 |
| 4       | 943.02     | 942.18  | 00:00:11 |
| 5       | 938.26     | 919.25  | 00:00:11 |

---

## Summary

- Implemented PPO with CNN and Beta-distribution policy    
- Trained directly from **raw pixels** using stacked frames  
- Solved environment with **>900 average reward**  
- Achieved stable and efficient learning over multiple epochs  

---

## References

* Po-Wei Chou, Daniel Maturana, Sebastian Scherer (2017).
  *Improving Stochastic Policy Gradients in Continuous Control with Deep Reinforcement Learning using the Beta Distribution.*
  Proceedings of the 34th International Conference on Machine Learning, PMLR 70:834-843.
  [Link](http://proceedings.mlr.press/v70/chou17a.html)

* Yasuhiro Fujita, Shin-ichi Maeda (2018).
  *Clipped Action Policy Gradient.*
  [arXiv:1802.07564](https://arxiv.org/abs/1802.07564)

