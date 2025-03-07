## The 7th Qiangwang International Elite Challenge on Cyber Mimic Defense: Implementation of Adversarial Attack

[The 7th Qiangwang International Elite Challenge on Cyber Mimic Defense](https://mp.weixin.qq.com/s/H8jUY1NjuefrD4VIsKZygg)

2nd-place implementation for the AI track.

**Team Name**: Hidden Face

The total competition score combines both black-box and white-box attack scores.

Scores are based on adversarial samples generated and submitted by participants, evaluated considering attack success rate, transferability (success rate of adversarial samples on unknown models), and stealthiness (perturbation magnitude).

(*Note*: Black-box attacks in this competition are not query-based.)

In our implementation, adversarial samples were generated using only the models provided by the competition and an open-source face recognition model. Unlike the first-place solution, our approach does not require retraining a new model specifically for generating adversarial samples.

## Proposed Solution

We apply a frequency domain transformation to the input images. Specifically, images are transformed from the spatial domain to the frequency domain using the Discrete Cosine Transform (DCT). Adversarial perturbations are generated directly in the frequency domain, significantly improving transferability and attack effectiveness against black-box models. This method achieves high success rates for both untargeted and targeted attacks.

### 1. Algorithm Design

**Adversarial Sample Generation:**

Reference Method: [Frequency-Domain Adversarial Attacks](https://arxiv.org/pdf/2305.16494)

(1) **Preprocessing:** Convert the input image from the spatial domain to the frequency domain using Discrete Cosine Transform (DCT).

(2) **Frequency Domain Transformation:** In the frequency domain, apply random transformations to the image spectrum, including adding Gaussian noise (ξ) and applying random matrices (M). These transformations simulate varied model responses to different spectral features.

(3) **Inverse Transformation:** Convert the modified spectrum back to the spatial domain using the Inverse Discrete Cosine Transform (IDCT), resulting in adversarially perturbed images.

(4) **Gradient Calculation:** Compute the gradient of the loss function on the perturbed images using pretrained models, capturing model sensitivity to input perturbations. Averaging gradients across multiple models yields a more robust update direction, simulating diversified training scenarios. Iteratively update the adversarial samples using these gradients to progressively enhance their attack capability.

(5) **Optimization:** Extend the original method by incorporating model aggregation, calculating a composite loss for gradient-based iteration, significantly enhancing robustness against mimic defense mechanisms.

**Attack Execution:** Apply generated adversarial samples to target models and evaluate attack success rates.

### 2. Code Reproduction

#### (1) Environment

- Operating System: Ubuntu 18.04
- CUDA Version: 11.6
- Python Version: 3.8

#### (2) External Open-Source Model

- **Download Link:** [https://share.weiyun.com/rID4J19c](https://share.weiyun.com/rID4J19c)
- **Basic Information:** IR152 is a deep neural network based on Inception-ResNet-152 architecture, commonly used as the backbone network in face recognition frameworks such as ArcFace. It integrates Inception modules with ResNet residual connections, consisting of 152 layers.
- **Loading Method:** Defined in `code/test/run.py`, function `load_surrogate_model(device)`:

```python
def load_surrogate_model(device):
    # Load pretrained white-box face recognition surrogate model
    fr_model = m.IR_152((112, 112))
    fr_model.load_state_dict(torch.load('model/ir152.pth'))
    fr_model.to(device)
    fr_model.eval()
    ...
```

#### (3) Reproduction Steps

Place the `no_target` and `target` data folders into the directory `code/data`.

**Install Dependencies:**

```
pip install -r requirements.txt
```

**Generate Adversarial Samples:**

- Untargeted attack, output path: `code/result_data/advimages/no_target_adv`

```
python test/run.py --mode 0 --saving_dir no_target_adv
```

- Targeted attack, output path: `code/result_data/advimages/target_adv`

```
python test/run.py --mode 1 --saving_dir target_adv
```



## 第七届强网拟态防御，对抗样本攻击实现（中文）
[第七届强网拟态防御](https://mp.weixin.qq.com/s/H8jUY1NjuefrD4VIsKZygg)
(THE 7th“QIANGWANG"INTERNATIONAL ELITE CHALLENGE ON CYBER MIMIC DEFENSE)
人工智能赛道第二名实现方案。

Team Name: Hidden Face

比赛总分累计黑盒攻击与白盒攻击两类攻击分数。

黑盒与白盒系统通过选手提交生成的对抗样本进行评分。
评分综合考虑攻击成功率，迁移性(对抗样本在未知模型上的攻击成功率)和隐蔽性(perturbation大小)。
(Note:比赛中的黑盒攻击不是查询式攻击)

在我们的实现方案中，仅通过比赛提供的模型与一个开源人脸识别模型进行生成对抗样本。
与第一名相比，我们的方案不需要攻击者重新训练新的模型用于生成对抗样本。

## 解决方案
我们对输入图像进行频谱变换，将输入的空间域图像经过离散余弦变换到频域，在频域生成对抗扰动，从而提高对抗性攻击的迁移性以及在黑盒模型下的攻击效果。该方法可同时实现较高的无目标和有目标攻击成功率。

### 1、算法设计：
**对抗样本生成：**

参考方法: [频域对抗攻击](https://arxiv.org/pdf/2305.16494)

（1）预处理：对输入图像应用离散余弦变换（DCT），将其从空间域转换到频域。

（2）频域变换：在频域中，对图像的频谱应用随机变换，包括添加高斯噪声（ξ）和应用随机矩阵（M）。这些变换旨在模拟不同的模型对频谱的不同响应。

（3）逆变换：使用逆离散余弦变换（IDCT）将变换后的频谱转换回空间域，得到对抗扰动后的图像。

（4）梯度计算：利用预训练模型对扰动后的图像计算损失函数的梯度，这些梯度反映了模型对输入图像的敏感性。通过平均多个模型的梯度，获得一个更稳定的更新方向，以此来模拟更多样化的训练模型。使用计算得到的梯度更新对抗样本，通过迭代过程逐步增强对抗样本的攻击能力。

（5）优化：在原方法的基础上，引入模型聚合方法，计算综合损失进行梯度迭代。大大增强了对抗攻击在拟态防御下的鲁棒性。

攻击执行：将生成的对抗样本应用于目标模型，评估其攻击成功率。

### 2、代码复现
#### （1）环境
操作系统：

CUDA Version: 11.6

Ubuntu18.04

Python版本：3.8

#### （2）外部的开源模型
开源地址：https://share.weiyun.com/rID4J19c

基本信息：IR152 是基于 Inception-ResNet-152 的深层神经网络架构。该模型结合了 Inception 模块 和 ResNet 残差连接，具有152 层深度，是 ArcFace 等人脸识别框架中常用的主干网络。

加载方式：code/test/run.py, 函数 `load_surrogate_model(device)`

```buildoutcfg
def load_surrogate_model(device):
    # Load pretrain white-box FR surrogate model
    fr_model = m.IR_152((112, 112))
    fr_model.load_state_dict(torch.load('model/ir152.pth'))
    fr_model.to(device)
    fr_model.eval()
    ...
```

#### （3）复现流程

将no_target，target数据文件夹放到`code/data`目录下。

**安装依赖：**
```
pip install -r requirements.txt
```

**生成对抗样本：**

- 无目标攻击，输出路径 `code/result_data/advimages/no_target_adv`

```
python test/run.py --mode 0 --saving_dir no_target_adv
```
- 有目标攻击，输出路径 `code/result_data/advimages/target_adv`：

```
python test/run.py --mode 1 --saving_dir target_adv
```
