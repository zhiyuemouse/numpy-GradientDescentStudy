# numpy-GradientDescentStudy
## 深度学习基础：用numpy更好的学习矩阵求导
### 介绍
Version1: 
用numpy实现梯度下降算法，采用3层的MLP网络(第一版只涉及矩阵乘法，没有引入偏置项、激活函数)先简单理解过程。


## 前向传播
$y_1 = W_1 \times x$ \
$y_2 = W_2 \times y_1$ \
$y_3 = W_3 \times y_2$

相当于3层无偏置项的全连接层
```python
import torch
from torch import nn

model = nn.Sequential(
    nn.Linear(3, 4, bias=False),
    nn.Linear(4, 5, bias=False),
    nn.Linear(5, 1, bias=False)
)
```

$L_{loss} = |y_3 - y_{true}|$



## 反向传播
$L_{loss}$对损失函数的输入(即第3层的输出)$y_3$的梯度为：\
当$y_3 > y_{true}$ 即 $y_3 - y_{true} > 0$时:
$\frac{\partial L_{loss}}{\partial y_3} = 1$ \
当$y_3 < y_{true}$ 即 $y_3 - y_{true} < 0$时:
$\frac{\partial L_{loss}}{\partial y_3} = -1$

$L_{loss}$对第3层的权重参数$W_3$的梯度为：
$\frac{\partial L_{loss}}{\partial W_3} = \frac{\partial L_{loss}}{\partial y_3} \frac{\partial y_3}{\partial W_3} = \pm1 \times y_2^T$

$L_{loss}$对第2层的权重参数$W_2$的梯度为：
$\frac{\partial L_{loss}}{\partial W_2} = \frac{\partial L_{loss}}{\partial y_3} \frac{\partial y_3}{\partial y_2} \frac{\partial y_2}{\partial W_2} = (\pm1 \times W_3)^T y_1^T$

$L_{loss}$对第2层的权重参数$W_2$的梯度为：
$\frac{\partial L_{loss}}{\partial W_1} = \frac{\partial L_{loss}}{\partial y_3} 
\frac{\partial y_3}{\partial y_2} \frac{\partial y_2}{\partial y_1} \frac{\partial y_1}{\partial W_1}
 = (\pm1 \times W_3 \times W_2)^T x^T$

 ## 梯度下降
$\alpha$ 即学习率 Learning Rate

### 更新第3层的权重
$
W_3 = W_3 - \alpha \times \frac{\partial L_{loss}}{\partial W_3}
$

### 更新第2层的权重
$
W_2 = W_2 - \alpha \times \frac{\partial L_{loss}}{\partial W_2}
$

### 更新后的第1层的权重
$
W_1 = W_1 - \alpha \times \frac{\partial L_{loss}}{\partial W_1}
$

## 整体更新过后再次前向传播查看新的Loss
$
y_{new} = W_3 \times W_2 \times W_1 \times x \\
L_{new} = |y_{new} - y_{true}|
$


**!以上为个人理解，如有错误，欢迎指正:smile:!**