---
layout:     post
title:      CNN python实现
subtitle:   
date:       2019-10-30
author:     danyang
catalog: true
tags:
    - 学习
    
---



### 反向传播核心公式

$$
\delta_j^L=\frac{\partial C}{\partial z_j^L}=\frac{\partial C}{\partial a_j^L}\frac{\partial a_j^L}{\partial z_j^L}=\frac{\partial C}{\partial a_j^L}\sigma'(z_j^L) \tag{BP1}
$$

$$
\delta^l=((w^{l+1})^T\delta^{l+1})\odot \sigma'(z^l) \tag{BP2}
$$

$$
\frac{\partial C}{\partial b_j^l}=\frac{\partial C}{\partial z_j^l}\frac{\partial z_j^l}{\partial b_j^l}=\frac{\partial C}{\partial z_j^l}=\delta_j^l \tag{BP3}
$$

$$
\frac{\partial C}{\partial w_{jk}^l}=\frac{\partial C}{\partial z_j^l}\frac{\partial z_j^L}{\partial w_{jk}^l}=\frac{\partial C}{\partial z_j^l}a_k^{l-1}=a_k^{l-1}\delta_j^l \tag{BP4}
$$

​       在CNN中，由于存在两个特殊的层——卷积层以及池化层，因此不能直接套用公式。这篇博客主要记录CNN中池化层和卷积层的反向传播以及Python实现。

### 池化层的反向传播

　　池化层没有激活函数可以直接看成用线性激活函数，即σ(z)=z，所以σ′(z)=1。在前向传播时，我们一般使用max或average对输入进行池化，而且池化区域大小已知。反向传播就是要从缩小后的误差$\delta^{l+1}$ ，还原池化前较大区域对应的误差$\delta^l$ 。根据（BP2），$\delta^l=((w^{l+1})^T\delta^{l+1})\odot \sigma'(z^l)$ ，在DNN中$w^{l+1}$是已知的，所以我们可以直接通过矩阵乘法将$l + 1$ 层的误差映射回ll层的误差，但对于池化层，要求$(w^{l+1})^T\delta^{l+1}$ 就需要一些特殊的操作了。

简单来说，就是需要将$\delta^{l+1}$ 上采样，扩展为原来的输入时的shape大小。代码如下：

```python
class MaxPoolingLayer(object):
    def __init__(self, input_width, input_height, channel_number, 
                 filter_width, filter_height, stride):
        self.input_width = input_width
        self.input_height = input_height
        self.channel_number = channel_number
        #滤波器参数
        self.filter_width = filter_width
        self.filter_height = filter_height
        #步长
        self.stride = stride 
        #经过卷积操作后的shape
        self.output_width = (input_width - filter_height) // stride + 1
        self.output_height = (input_height - filter_height) // self.stride + 1
        self.output_array = np.zeros((self.channel_number, self.output_height, self.output_width))
    
    def forward(self, input_array):
        for d in range(self.channel_number):#遍历channel
            for i in range(self.output_height):
                for j in range(self.output_width):#遍历卷积区域
                    self.output_array[d,i,j] = (
                        #get_patch函数用于寻找卷积区域，max()用于寻找区域内的最大值
                        get_patch(input_array[d], i, j,
                            self.filter_width, 
                            self.filter_height, 
                            self.stride).max())
    
    def backward(self, input_array, sensitivity_array):
        self.delta_array = np.zeros(input_array.shape)
        for d in range(self.channel_number):
            for i in range(self.output_height):
                for j in range(self.output_width):
                    #遍历每个池化区域
                    patch_array = get_patch(
                        input_array[d], i, j,
                        self.filter_width, 
                        self.filter_height, 
                        self.stride)
                    #get_max_index函数用于寻找池化前最大值的index
                    k, l = get_max_index(patch_array)
                    #进行上采样
                    self.delta_array[d, 
                        i * self.stride + k, 
                        j * self.stride + l] = \
                        sensitivity_array[d,i,j]
```

```python
def get_patch(input_array, i, j, filter_width, filter_height, stride):
    '''
    从输入数组中获取本次卷积的区域，
    自动适配输入为2D和3D的情况
    '''
    start_i = i * stride
    start_j = j * stride
    if input_array.ndim == 2:
        input_array_conv = input_array[start_i : start_i + filter_height, start_j : start_j + filter_width]
        return input_array_conv
    elif input_array.ndim == 3:
        input_array_conv = input_array[:,
            start_i : start_i + filter_height,
            start_j : start_j + filter_width]
        return input_array_conv
```

```python
def get_max_index(array):
    # 获取一个2D区域的最大值所在的索引
    max_i = 0
    max_j = 0
    max_value = array[0, 0]
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            if array[i, j] > max_value:
                max_value = array[i,j]
                max_i, max_j = i, j
    return max_i, max_j
```

### 卷积层的反向传播

假设$l$层的激活输出是一个3×3的矩阵，第l+1层卷积核$W^{l+1}$是一个2×2的矩阵，卷积步长为1，则输出$z^{l+1}$是一个2×2的矩阵。我们简化$b^l=0$，则有：
$$
z^{l+1}=a^l*W^{l+1} \tag{1}
$$
列出a，W，z的矩阵表达式如下：
$$
\left[ \begin{matrix} z_{11}&z_{12}\\z_{21}&z_{22}\end{matrix} \right]=\left[ \begin{matrix} a_{11}&a_{12}&a_{13}\\a_{21}&a_{22}&a_{23}\\a_{31}&a_{32}&a_{33}\end{matrix} \right] * \left[ \begin{matrix} w_{11}&w_{12}\\w_{21}&w_{22}\end{matrix} \right] \tag{2}
$$
可以得出：
$$
z_{11}=a_{11}w_{11}+a_{12}w_{12}+a_{21}w_{21}+a_{22}w_{22}\\z_{12}=a_{12}w_{11}+a_{13}w_{12}+a_{22}w_{21}+a_{23}w_{22}\\z_{21}=a_{21}w_{11}+a_{22}w_{12}+a_{31}w_{21}+a_{32}w_{22}\\z_{22}=a_{22}w_{11}+a_{23}w_{12}+a_{32}w_{21}+a_{33}w_{22} \tag{3}
$$
$\frac{\partial C}{\partial a^l}$为：
$$
\nabla a^l=\frac{\partial C}{\partial a^l}=\frac{\partial C}{\partial z^{l+1}}\frac{\partial z^{l+1}}{\partial a^l}=\delta^{l+1}\frac{\partial z^{l+1}}{\partial a^l} \tag{4}
$$
对于$\frac{\partial z^{l+1}}{\partial a^l}$：
$$
\nabla a_{11}=\delta_{11}^{l+1}\frac{\partial z_{11}^{l+1}}{\partial a_{11}^l}+\delta_{12}^{l+1}\frac{\partial z_{12}^{l+1}}{\partial a_{11}^l}+\delta_{21}^{l+1}\frac{\partial z_{21}^{l+1}}{\partial a_{11}^l}+\delta_{22}^{l+1}\frac{\partial z_{22}^{l+1}}{\partial a_{11}^l}=\delta_{11} w_{11}
$$

$$
\nabla a_{12}=\delta_{11}w_{12}+\delta_{12}w_{11}\\ \nabla a_{13}=\delta_{12}w_{12}\\ \nabla a_{21}=\delta_{11}w_{21}+\delta_{21}w_{11}\\ \nabla a_{22}=\delta_{11}w_{22}+\delta_{12}w_{21}+\delta_{21}w_{12}+\delta_{22}w_{11}\\ \nabla a_{23}=\delta_{12}w_{22}+\delta_{22}w_{12}\\ \nabla a_{31}=\delta_{21}w_{21}\\ \nabla a_{32}=\delta_{21}w_{22}+\delta_{22}w_{21}\\ \nabla a_{33}=\delta_{22}w_{22}
$$

归纳可得：
$$
\left[ \begin{matrix} \nabla a_{11}&\nabla a_{12}&\nabla a_{13}\\\nabla a_{21}&\nabla a_{22}&\nabla a_{23}\\\nabla a_{31}&\nabla a_{32}&\nabla a_{33}\end{matrix} \right]=\left[ \begin{matrix}0&0&0&0\\ 0&\delta_{11} & \delta_{12}&0\\ 0&\delta_{21} & \delta_{22}&0\\0&0&0&0\end{matrix} \right] * \left[ \begin{matrix} w_{22}&w_{21}\\w_{12}&w_{11}\end{matrix} \right] \tag{5}
$$
因此：
$$
\delta^l=(\delta^{l+1} * rot180(w^{l+1}))\odot \sigma'(z^l)
$$

$$
\frac{\partial C}{\partial w^l}=\frac{\partial C}{\partial z^l}\frac{\partial z^L}{\partial w^l}=\delta^l\frac{\partial z^L}{\partial w^l}=\delta^l*rot180(a^{l-1})
$$

#### 核心公式：


$$
\delta^l=(\delta^{l+1} * rot180(w^{l+1}))\odot \sigma'(z^l)
$$

$$
\frac{\partial C}{\partial w^l}=\frac{\partial C}{\partial z^l}\frac{\partial z^L}{\partial w^l}=\delta^l\frac{\partial z^L}{\partial w^l}=\delta^l*rot180(a^{l-1})
$$



```python
class ConvLayer(object):
    '''
    参数含义：
    input_width:输入图片尺寸——宽度
    input_height:输入图片尺寸——长度
    channel_number:通道数，彩色为3，灰色为1
    filter_width:卷积核的宽
    filter_height:卷积核的长
    filter_number:卷积核数量
    zero_padding：补零长度
    stride:步长
    activator:激活函数
    learning_rate:学习率
    '''
    def __init__(self, input_width, input_height,
                 channel_number, filter_width,
                 filter_height, filter_number,
                 zero_padding, stride, activator,
                 learning_rate):
        self.input_width = input_width
        self.input_height = input_height
        self.channel_number = channel_number
        self.filter_width = filter_width
        self.filter_height = filter_height
        self.filter_number = filter_number
        self.zero_padding = zero_padding
        self.stride = stride
        self.output_width = \
            ConvLayer.calculate_output_size(
            self.input_width, filter_width, zero_padding,
            stride)
        self.output_height = \
            ConvLayer.calculate_output_size(
            self.input_height, filter_height, zero_padding,
            stride)
        self.output_array = np.zeros((self.filter_number,
            self.output_height, self.output_width))
        self.filters = []
        for i in range(filter_number):
            self.filters.append(Filter(filter_width,
                filter_height, self.channel_number))
        self.activator = activator
        self.learning_rate = learning_rate
        
    def calculate_output_size(input_size, filter_size, zero_padding, stride):
        return (input_size + 2*zero_padding - filter_size) // stride + 1
    
    def forward(self, input_array):
        '''
        计算卷积层的输出
        输出结果保存在self.output_array
        '''
        self.input_array = input_array
        self.padded_input_array = padding(input_array,self.zero_padding)
        
        for f in range(self.filter_number):
            filter = self.filters[f]
            conv(self.padded_input_array, filter.get_weights(), self.output_array[f], self.stride, filter.get_bias())
        element_wise_op(self.output_array, 
                        self.activator.forward)
        
    def backward(self, input_array, sensitivity_array, activator):
        '''
        计算传递给前一层的误差项，以及计算每个权重的梯度
        前一层的误差项保存在self.delta_array
        梯度保存在Filter对象的weights_grad
        '''
        self.forward(input_array)
        self.bp_sensitivity_map(sensitivity_array, activator)
        self.bp_gradient(sensitivity_array)
        
    def update(self):
        '''
        按照梯度下降，更新权重
        '''
        for filter in self.filters:
            filter.update(self.learning_rate)

    def bp_sensitivity_map(self, sensitivity_array,
                           activator):
        '''
        计算传递到上一层的sensitivity map
        sensitivity_array: 本层的sensitivity map
        activator: 上一层的激活函数
        '''
        # 处理卷积步长，对原始sensitivity map进行扩展
        expanded_array = self.expand_sensitivity_map(
            sensitivity_array)
        # full卷积，对sensitivitiy map进行zero padding
        # 虽然原始输入的zero padding单元也会获得残差
        # 但这个残差不需要继续向上传递，因此就不计算了
        expanded_width = expanded_array.shape[2]
        zp = (self.input_width +  
              self.filter_width - 1 - expanded_width) // 2
        padded_array = padding(expanded_array, zp)
        print('=======================padded_array========================')
        print(padded_array)
        # 初始化delta_array，用于保存传递到上一层的
        # sensitivity map
        self.delta_array = self.create_delta_array()
        # 对于具有多个filter的卷积层来说，最终传递到上一层的
        # sensitivity map相当于所有的filter的
        # sensitivity map之和
        
        for f in range(self.filter_number):
            #print('f',f)
            filter = self.filters[f]
            # 将filter权重翻转180度
            flipped_weights = np.array(list(map(
                lambda i: np.rot90(i, 2), 
                filter.get_weights())))

#             flipped_weights = self.flip180(filter.get_weights())
            # 计算与一个filter对应的delta_array
            delta_array = self.create_delta_array()
            #print('padded_array[f]',padded_array[f])
            for d in range(delta_array.shape[0]):
                conv(padded_array[f], flipped_weights[d],
                    delta_array[d], 1, 0)
            self.delta_array += delta_array
        # 将计算结果与激活函数的偏导数做element-wise乘法操作
        derivative_array = np.array(self.input_array)
        element_wise_op(derivative_array, 
                        activator.backward)
        self.delta_array *= derivative_array
    
    def bp_gradient(self,sensitivity_array):
        # 处理卷积步长，对原始sensitivity map进行扩展
        expanded_array = self.expand_sensitivity_map(sensitivity_array)
        for f in range(self.filter_number):
            filter = self.filters[f]
            for d in range(filter.weights.shape[0]):
                conv(self.padded_input_array[d], expanded_array[f], filter.weights_grad[d], 1, 0)
            filter.bias_grad = expanded_array[f].sum()
    
    
    def expand_sensitivity_map(self, sensitivity_array):
        depth = sensitivity_array.shape[0]
        # 确定扩展后sensitivity map的大小
        # 计算stride为1时sensitivity map的大小
        expanded_width = (self.input_width - self.filter_width + 2 * self.zero_padding + 1)
        expanded_height = (self.input_height - self.filter_height + 2 * self.zero_padding + 1)
        # 构建新的sensitivity_map
        expand_array = np.zeros((depth, expanded_height, expanded_width))
        # 从原始sensitivity map拷贝误差值
        for i in range(self.output_height):
            for j in range(self.output_width):
                i_pos = i * self.stride
                j_pos = j * self.stride
                expand_array[:,i_pos,j_pos] = \
                    sensitivity_array[:,i,j]
        return expand_array
    
    def create_delta_array(self):
        return np.zeros((self.channel_number, self.input_height, self.input_width))
    
```

```python
class Filter(object):
    def __init__(self, width, height, depth):
        self.weights = np.random.uniform(-1e-4, 1e-4,(depth, height, width))
        self.bias = 0
        self.weights_grad = np.zeros(self.weights.shape)
        self.bias_grad = 0
    
    def __repr__(self):
        return 'filter weights:\n%s\nbias:\n%s' % (
            repr(self.weights), repr(self.bias))
    
    def get_weights(self):
        return self.weights
    
    def get_bias(self):
        return self.bias
    
    def update(self, learning_rate):
        self.weights -= learning_rate * self.weights_grad
        self.bias -= learning_rate * self.bias_grad
```

```python
def padding(input_array, zp):
    '''
    为数组增加Zero padding，自动适配输入为2D和3D的情况
    '''
    if zp == 0:
        return input_array
    else:
        if input_array.ndim == 3:
            input_width = input_array.shape[2]
            input_height = input_array.shape[1]
            input_depth = input_array.shape[0]
            padded_array = np.zeros((input_depth, input_height +2 * zp, input_width + 2 * zp))
            padded_array[:, zp : zp + input_height, zp : zp + input_width] = input_array
            return padded_array
        elif input_array.ndim == 2:
            input_width = input_array.shape[1]
            input_height = input_array.shape[0]
            padded_array = np.zeros((
                input_height + 2 * zp,
                input_width + 2 * zp))
            padded_array[zp : zp + input_height,
                zp : zp + input_width] = input_array
            return padded_array
```

```python
def conv(input_array, kernel_array, output_array, stride, bias):
    '''
    计算卷积，自动适配输入为2D和3D的情况
    '''
    channel_number = input_array.ndim
    output_width = output_array.shape[1]
    output_height = output_array.shape[0]
    kernel_width = kernel_array.shape[-1]
    kernel_height = kernel_array.shape[-2]
    for i in range(output_height):
        for j in range(output_width):
            output_array[i][j] = (    
                get_patch(input_array, i, j, kernel_width, 
                    kernel_height, stride) * kernel_array
                ).sum() + bias
```

代码连接：

[code](https://github.com/ZhengDanYang1/ZhengDanYang1.github.io/blob/master/学习/CNN.ipynb)

