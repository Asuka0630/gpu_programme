# Backpropgation in Deep Learning


## 1.backpropgation
反向传播是通过链式法则从尾层向前传播
+ 整个网络最后一层输出的结果通过一个Loss函数$\mathcal{L}:\mathbb{R}^n\mapsto \mathbb{R}$
+ 对于中间的某一层，该层的输入为$x$，输出为$y$，$\frac{\partial{\mathcal{L}}}{\mathbf{w}}$的形状与$\mathbf{w}$的形状相同
+ 反向传播时需要接收前一层传来$\frac{\partial L}{\partial y}$，通过该层的梯度公式计算出$\frac{\partial L}{\partial w}$以及$\frac{\partial L}{\partial x}$继续向前转递,每一层都迭代前向计算(注意：准确而言这里不能说通过$\frac{\partial{L}}{\partial y}\cdot \frac{\partial{y}}{\partial w}$来计算梯度，因此某些层中$\frac{\partial{y}}{\partial w}$不是显式计算的)
+ 并通过$w\leftarrow w-\eta\frac{\partial L}{\partial w}$更新参数

### 1.1.Linear layer
+ 输入：$\mathbf{x}.shape=[N,C_i]$分别表示特征，序列长度
+ 输入：$\mathbf{y}.shape=[N,C_o]$分别表示特征，序列长度
+ 参数：
   + weight：$\mathbf{w}.shape=[C_o, C_i]$
   + bias：$\mathbf{b}.shape=[C_o]$
+ 前向传播：
    $\mathbf{y} = \mathbf{xW}^T+\mathbf{b}$
+ 反向传播：
    + 公式：
    $$\begin{aligned}
        & \nabla \mathbf{b} = \nabla\mathbf{y} \\
        & \nabla \mathbf{W} = \nabla \mathbf{y}^T\times \mathbf{X}\\
        & \nabla\mathbf{X} = \nabla\mathbf{y}\times \mathbf{W}
    \end{aligned}$$
    + 推导：对于$\mathbf{Y=WX+B}$的梯度公式：
        1. 容易得到：
        $$\frac{\partial{\mathbf{y}}}{\partial{\mathbf{b}}} = 1 \Rightarrow \nabla \mathbf{b} = \nabla\mathbf{y}$$
        2. 对于$\mathbf{y}_{1\times n}=\mathbf{a}_{1\times k}\mathbf{X}_{k\times n}$的向量矩阵乘法，$\nabla \mathbf{a}=\mathbf{X}^T$，$\mathbf{a}$相当于将$\mathbf{X}$的行累加的系数$\mathbf{y} = \mathbf{a}_1\mathbf{x}_{1*}+\cdots+\mathbf{a}_k\mathbf{x}_{k*}$，$\frac{\partial{\mathbf{y}_i}}{\partial{\mathbf{a}_j}}=\sum_j^K\mathbf{x}_{ji}$
        也就是$\mathbf{y}_i$关于$\mathbf{j}$的偏导数为$\mathbf{X}$的第$i$列元素之和，从而：
        $$\nabla \mathbf{a} = \nabla \mathbf{y}\times \mathbf{X}^T$$
        3. 将上述结论扩展到$\mathbf{y}_{C_o,N}=\mathbf{W}_{C_o\times C_i}\mathbf{X}_{C_i,N}$
            矩阵$\mathbf{y}$可以看成是将$\mathbf{W}$的每一行作为系数得到$\mathbf{X}$的行向量的线性组合，因此对于$\mathbf{y}_{i*}$产生的梯度$\nabla_{\mathbf{W}}=\nabla \mathbf{y}_{i*}\mathbf{X}^T$，写成矩阵形式就是，对于$\mathbf{y=W\times X+b}$：
            $$\nabla \mathbf{W} = \nabla \mathbf{y\times X}^T$$
        4. 推导$\nabla \mathbf{X}$：
        $$\begin{aligned}
            & \mathbf{y}^T = (\mathbf{WX})^T+\mathbf{b}^T= \mathbf{X}^T\mathbf{W}^T+\mathbf{b}^T\\
        \end{aligned}$$
        根据结论2有：
        $$\begin{aligned}
            & \nabla \mathbf{X}^T = \nabla \mathbf{y}^T\times {\mathbf{W}^T}^T = \nabla \mathbf{y}^T\times \mathbf{W} \\
            & \Rightarrow \nabla\mathbf{X} = (\nabla \mathbf{y}^T\times \mathbf{W})^T = \mathbf{W}^T\nabla_{\mathbf{y}}
        \end{aligned}$$
    + 验证：
        ``` python
        import torch
        import torch.nn as nn

        linear = nn.Linear(4, 3).to("cuda")  # Y = XW^T+B
        inp = torch.tensor([[1, 2, 3, 4]], dtype=torch.float32, requires_grad=True).to("cuda")
        out: torch.Tensor = linear(inp)
        inp.retain_grad()
        out.retain_grad()

        loss = 0.5 * (out.norm()) * (out.norm())  # \sum y^2
        loss.backward()
        # print("out.grad:", out.grad)  # \nabla_y
        # print("weight.grad: ", linear.weight.grad)
        # print("input.grad: ", inp.grad)

        with torch.no_grad():
            weight_hand_grad = torch.matmul(out.T, inp) # \nabla_yX
            inp_hand_grad = torch.matmul(out, linear.weight) #\nabla_yW
        print("input.gradient check:", torch.allclose(inp.grad, inp_hand_grad))
        print("weight.gradient check:", torch.allclose(linear.weight.grad, weight_hand_grad))
        print("bias.gradient check:", torch.allclose(linear.bias.grad, out))
        ```

### 1.2.LayerNorm
LayerNorm的输入形状和Norm的维度可扩展[参考`反向传播->验证`中的代码]，这里仅对使用$[B,N,C]$形状的输入，并在$[C]$维度做归一化为例说明：
+ 输入：$\mathbf{x}.shape=[B,N,C]$分别表示batch_size，序列长度，元素维度
+ 输出：$\mathbf{y}.shape=[B,N,C]$分别表示batch_size，序列长度，元素维度
+ 参数：
   + weight：$\mathbf{w}.shape=[C]$
   + bias：$\mathbf{b}.shape=[C]$
+ 前向传播：
    $\odot$表示逐元素乘法，
    $$\mathbf{y}=\hat{\mathbf{x}}\odot\mathbf{w+b},\hat{\mathbf{x}}=\frac{\mathbf{x}-\mu(\mathbf{x})}{\sigma(\mathbf{x})+\epsilon}$$
+ 反向传播：
    + 公式：
        $$
            \nabla \mathbf{b} \xleftarrow[]{sum [0,1]}\nabla \mathbf{y} \\
            \nabla \mathbf{w} \xleftarrow[]{sum [0,1]}\nabla \mathbf{y} \odot \mathbf{\hat{x}}\\
            \nabla \mathbf{x} = {\sigma'}^{-1}(\nabla \hat{\mathbf{x}}-\overline{\nabla \hat{\mathbf{x}}}-\hat{\mathbf{x}}\cdot\overline{\hat{\mathbf{x}}\odot\nabla \hat{\mathbf{x}}}),\ \nabla \hat{\mathbf{x}} = \nabla \mathbf{y}\odot\mathbf{w},\sigma'=\sqrt{\sigma^2+\epsilon}
        $$
    + 推导：
        $\mathbf{\hat{x}}_{b,n,c}=\frac{x_{b,n,c}-\mu(\mathbf{x}_{b,n,*})}{\sigma{(\mathbf{x}_{b,n,*})+\epsilon}}$，有：
        $$\begin{aligned}
            & \frac{\partial{\mathbf{y}_{b,n,c}}}{\partial{\mathbf{w}_c}}=\mathbf{\hat{x}}_{b,n,c}\\
            & \Rightarrow \frac{\partial{\mathbf{y}_{*,*,c}}}{\partial{\mathbf{w}_c}}=\sum_b^B\sum_n^N\mathbf{\hat{x}}_{b,n,c},\\
            & \Rightarrow \frac{\partial{\mathcal{L}}}{\partial{\mathbf{w}}}\xleftarrow[]{sum [0,1]}\frac{\partial{\mathcal{L}}}{\partial{\mathbf{y}}}\odot \mathbf{\hat{x}} \\
            & \frac{\partial{\mathbf{y}_{b,n,c}}}{\partial{b}_c}=1 \\
            & \Rightarrow \frac{\partial{\mathcal{L}}}{\partial{\mathbf{b}}}\xleftarrow[]{sum [0,1]}\frac{\partial{\mathcal{L}}}{\partial{\mathbf{y}}}\\
        \end{aligned}$$
        还需要计算向前一层传递的梯度$\frac{\partial{\mathcal{L}}}{\partial{\mathbf{x}}}$，容易发现$\mathbf{y(x)}$中与$\mathbf{x}$有关的有$\mathbf{\mu}$和$\mathbf{\sigma}$，和$\mathbf{x}$自身，分别计算$\frac{\partial{\mathbf{\mu}}}{\partial{\mathbf{x}}}$和$\frac{\partial{\mathbf{\sigma}}}{\partial{\mathbf{x}}}$，由于LayerNorm层的计算方法，所以对于$B,N$维度是对称的，对于两个$N$维度的数据$\mathbf{x}_{n_1,*}$和$\mathbf{x}_{n_2,*}$，各自的梯度计算没有依赖关系且计算方法完全相同，因此只需要推导一维向量$\mathbf{x}\in \mathbb{R}^C$与及其对应的输出$\mathbf{y}$即可，令$\mathbf{\hat{x}}_{i}=\frac{\mathbf{x}_{i}-\mu(\mathbf{x})}{\sigma{(\mathbf{x})+\epsilon}}$，对于$\mu$和$\sigma$，他们是规约算子，因此对于任意的$i$的偏导数$\frac{\mu}{\partial{\mathbf{x}_i}}$都存在：
        $$\frac{\partial{\mu}}{\partial{\mathbf{x}_i}}=\frac{\partial{\frac{1}{C}\sum_i^C\mathbf{x}_i}}{\partial{\mathbf{x}_i}}=\frac{1}{C}$$
        对于$\sigma=\sqrt{\mu(\mathbf{x}^2)-\mu({\mathbf{x}})^2}$，有：
        $$\frac{\partial{\sigma}}{\partial{\mathbf{x}_i}}=\frac{1}{2}\cdot\frac{1}{\sigma}\cdot(\frac{1}{C}2\mathbf{x}_i-2\mu(\mathbf{x})\frac{1}{C})=\frac{1}{C}\cdot\sigma^{-1}(\mathbf{x}_i-\mu)\approx\frac{1}{C}\mathbf{\hat{x}}_i$$
        由于$\hat{\mathbf{x}_i}$包含了$\mu$和$\sigma$，因此对于$\hat{\mathbf{x}_i}$对于任意$\mathbf{x}_j$也都有偏导数存在：
        $$\begin{aligned}\frac{\partial{\mathbf{\hat{x}}}_j}{\partial{\mathbf{x}_i}} 
            &= (\frac{\partial{\mathbf{x}_j}}{\partial{\mathbf{x}_i}}-\frac{\partial{\mu}}{\partial{\mathbf{x}}}_i)\frac{1}{\sigma+\epsilon}+(\mathbf{x}_j-\mu)\frac{\partial{\frac{1}{\sigma}}}{\partial{\mathbf{x}_i}}\\
            &= (\delta_{ij}-\frac{1}{C})\frac{1}{\sigma+\epsilon}-(\mathbf{x}_j-\mu)\sigma^{-2}\frac{\partial{\sigma}}{\partial{\mathbf{x}}_i}\\
            &\approx \delta_{ij}\sigma^{-1}-\frac{1}{C}\sigma^{-1}-\frac{1}{C}\sigma^{-1}\mathbf{\hat{x}}_i\cdot\mathbf{\hat{x}}_j\\
            & [\delta_{ij}=(i==j\ ?\ 1:0)]
        \end{aligned}$$
        求梯度时，$\mathbf{y}_i$是关于$\mathbf{x}$的函数，即$y_i = f_i(x_1,x_2,\cdots,x_c)$，因此求梯度时需要累加$\mathbf{y}$的所有分量上的梯度：
        $$\begin{aligned}
            &\begin{aligned}\frac{\partial{\mathcal{L}}}{\partial{\mathbf{x}}_i}
                & =\sum_j^C\frac{\partial{\mathcal{L}}}{\partial{\mathbf{y}_j}}\cdot\frac{\partial{\mathbf{y}_j}}{\partial{\mathbf{x}_i}}\\
                &= \sum_j^C\frac{\partial{\mathcal{L}}}{\partial{\mathbf{y}_j}}\cdot\frac{\partial{\mathbf{y}_j}}{\partial{\mathbf{\hat{x}}}_i}\cdot\frac{\partial{\mathbf{\hat{x}}}_j}{\partial{\mathbf{\hat{x}}}_i}\\\
                &\approx \sum_j^C\frac{\partial{\mathcal{L}}}{\partial{\mathbf{y}_j}}\cdot\mathbf{w}_i[\delta_{ij}\sigma^{-1}-\frac{1}{C}\sigma^{-1}-\frac{1}{C}\sigma^{-1}\mathbf{\hat{x}}_i\cdot\mathbf{\hat{x}}_j]\\
                &= \sigma^{-1}[\nabla \mathbf{y}_i\mathbf{w}_i-\frac{1}{C}\sum_j^C\nabla \mathbf{y}_j\mathbf{w}_j- \frac{1}{C}\hat{\mathbf{x}_i}\sum_j^C\nabla \mathbf{y}_j\mathbf{w}_j\hat{\mathbf{x}_j}]\\
                &= \sigma^{-1}(\nabla \hat{\mathbf{x}_i}- \overline{\nabla \hat{\mathbf{x}}}-\hat{\mathbf{x}_i}\overline{\hat{\mathbf{x}}\odot\nabla \hat{\mathbf{x}}})\\
            \end{aligned}\\
            &\Rightarrow \frac{\partial{\mathcal{L}}}{\partial{\mathbf{x}}}\approx \sigma^{-1}(\nabla \hat{\mathbf{x}}-\overline{\nabla \hat{\mathbf{x}}}-\hat{\mathbf{x}}\cdot\overline{\hat{\mathbf{x}}\odot\nabla \hat{\mathbf{x}}}),\ \nabla \hat{\mathbf{x}} = \nabla \mathbf{y}\odot\mathbf{w}
        \end{aligned}$$
    + 验证：
        在torch.float32的类型下直接计算的梯度与pytorch存在差异，使用torch.float64精度可保持一致
        ``` python
        import torch
        import torch.nn as nn

        torch.manual_seed(0)
        device = "cuda"
        dtype = torch.float64
        eps = 1e-5
        criterion = lambda x: 0.5 * x.pow(2).sum()  # dL/dy = 0.5*2*y=y
        SHAPE = (2, 3, 4, 5)  # (B, C, H, W)
        NORM_SHAPE = SHAPE[2:]
        inp: torch.Tensor = torch.randn(SHAPE, dtype=dtype, requires_grad=True).to(device)
        # `normalized_shape` indicates which contiguous dimensions to take the mean over
        ln = nn.LayerNorm(NORM_SHAPE, eps=eps, dtype=dtype).to(device)
        inp.retain_grad()
        out: torch.Tensor = ln(inp)
        out.retain_grad()
        loss64: torch.Tensor = criterion(out)
        loss64.backward()
        DIM1 = tuple(range(len(SHAPE) - len(NORM_SHAPE), len(SHAPE)))
        DIM2 = tuple(range(0, len(SHAPE) - len(NORM_SHAPE)))
        with torch.no_grad():
            mean_ = inp.mean(dim=DIM1, keepdim=True)
            shift = inp - mean_
            var = torch.pow(shift, 2).mean(dim=DIM1, keepdim=True)
            rstd = torch.rsqrt(var + eps)
            x_hat = shift * rstd
            dy = out
            d_x_hat = dy * ln.weight
            dx = rstd * (
                d_x_hat
                - d_x_hat.mean(dim=DIM1, keepdim=True)
                - x_hat * (d_x_hat * x_hat).mean(dim=DIM1, keepdim=True)
            )
            db = dy.sum(dim=DIM2)
            dw = (dy * x_hat).sum(dim=DIM2)
        print("input.grad check: ", torch.allclose(dx, inp.grad))
        print("weight.grad check: ", torch.allclose(dw, ln.weight.grad))
        print("bias.grad check: ", torch.allclose(db, ln.bias.grad))
        ```

### 1.3.BatcNorm
BatchNorm的原理都类似，以BatchNorm1d为例进行说明：
+ 输入：$\mathbf{x}.shape=[B,C,N]$分别表示batch_size，元素维度，序列长度
+ 输出：$\mathbf{y}.shape=[B,C,N]$分别表示batch_size，元素维度，序列长度
+ 参数：
   + weight：$\mathbf{w}.shape=[C]$
   + bias：$\mathbf{b}.shape=[C]$
   + running_mean：$\mathbf{\mu}.shape=[C]$
   + running_var：$\mathbf{\sigma}.shape=[C]$
+ 前向传播：
    $$\mathbf{y}=\hat{\mathbf{x}}\odot\mathbf{w+b},\hat{\mathbf{x}}=\frac{\mathbf{x}-\mu_{batch}}{\sigma_{batch}+\epsilon}$$
+ 反向传播：
    + 公式：
        $\mu$和$\sigma$的更新由该层的超参数$momentum$控制当前批次的$\mu_{batch}$和$\sigma_{batch}$软更新：
        $$
            \mu = p\times \mu_{batch} + (1-p)\times\mu\\
            \sigma = p\times\sigma_{batch} + (1-p)\times\sigma\\
            p = momentum
        $$
        其他参数通过梯度更新：
         $$
            \nabla \mathbf{b} \xleftarrow[]{sum [0,2]}\nabla \mathbf{y} \\
            \nabla \mathbf{w} \xleftarrow[]{sum [0,2]}\nabla \mathbf{y} \odot \mathbf{\hat{x}}\\
            \nabla \mathbf{x} = {\sigma'}^{-1}(\nabla \hat{\mathbf{x}}-\overline{\nabla \hat{\mathbf{x}}}-\hat{\mathbf{x}}\cdot\overline{\hat{\mathbf{x}}\odot\nabla \hat{\mathbf{x}}}),\ \nabla \hat{\mathbf{x}} = \nabla \mathbf{y}\odot\mathbf{w},\sigma'=\sqrt{\sigma^2+\epsilon}
        $$
    + 验证：
        ```python
        import torch
        import torch.nn as nn

        torch.manual_seed(0)
        eps = 1e-5
        B, N, C = 2, 3, 4
        device = "cuda"
        DTYPE = torch.float32
        bn = nn.BatchNorm1d(C, eps=eps, dtype=DTYPE).to(device)
        bn.weight.data = torch.randn_like(bn.weight.data)
        bn.bias.data = torch.randn_like(bn.bias.data)

        criterion = lambda x: 0.5 * x.pow(2).sum()  # dL/dy = 0.5*2*y=y
        x = torch.randn((B, C, N), requires_grad=True, dtype=DTYPE).to(device)
        x.retain_grad()
        y: torch.Tensor = bn(x)
        y.retain_grad()
        loss: torch.Tensor = criterion(y)
        loss.backward()

        w = bn.weight.unsqueeze(0).unsqueeze(-1)
        b = bn.bias.unsqueeze(0).unsqueeze(-1)
        with torch.no_grad():
            mean_ = x.mean(dim=(0, 2), keepdim=True)  # shape=[1, C, 1]
            shift = x - mean_
            var_ = shift.pow(2).mean(dim=(0, 2), keepdim=True)  # shape=[1, C, 1]
            rstd_ = torch.rsqrt(var_ + eps)  # shape=[1, C, 1]
            x_hat = shift * rstd_  # shape=[B, C, N]
            manual_y = x_hat * w + b
            dy = manual_y  # shape=[B, C, N]
            dx_hat = dy * w  # shape=[B, C, N]
            dx = rstd_ * (
                dx_hat
                - dx_hat.mean(dim=(0, 2), keepdim=True)
                - x_hat * (dx_hat * x_hat).mean(dim=(0, 2), keepdim=True)
            )
            dw = (dy * x_hat).sum(dim=(0, 2))  # shape=[C]
            db = dy.sum(dim=(0, 2))  # shape=[C]
        # dx的误差比较大,不知道pytorch的底层做了什么处理,
        # 手动用float64计算转为32位仍与pytorch的32位结果有差异
        print("input.gradient check:", torch.allclose(dx, x.grad, atol=1e-5))
        print("weight.gradient check:", torch.allclose(dw, bn.weight.grad))
        print("bias.gradient check:", torch.allclose(db, bn.bias.grad))

        # pytorch在float64和float32位下依然有误差
        # bn64 = nn.BatchNorm1d(C, eps=eps, dtype=torch.float64).to(device)
        # bn64.weight.data = bn.weight.data.clone().to(torch.float64)
        # bn64.bias.data = bn.bias.data.clone().to(torch.float64)
        # x_64 = x.clone().to(torch.float64)
        # x_64.retain_grad()
        # y_64: torch.Tensor = bn64(x_64)
        # y_64.retain_grad()
        # loss_64: torch.Tensor = criterion(y_64)
        # loss_64.backward()
        # print("input.gradient check:", torch.allclose(x_64.grad, x.grad.to(torch.float64)))
        ```

### 1.4.Softmax/Logsoftmax
Logsoftmax可以避免Softmax的上溢($x_i$过大$e^{x_i}$溢出)和下溢($x_i$非常小,分母趋近于$0$)问题
+ 输入：$\mathbf{x}.shape=[N,C]$分别表示元素数量，元素维度
    对于高维的输入，全部都可以转换为对形如$[B,C,N]$的张量在`dim=1`维度上求解的情况，也就是相当于对每个batch中的每一列做Logsoftmax，这里仅以二维张量的情况做说明
+ 输出：$\mathbf{y}.shape=[N,C]$分别表示元素数量，元素维度
+ 参数：
+ 前向传播
    $$\begin{aligned}ln(\frac{e^{x_i}}{\sum^{N}_{j=0}e^{x_j}}) 
        & = x_i-ln(\sum^{N}_{j=0}e^{x_j})\\
        & = ln(\frac{e^{x_i}/e_{x_m}}{\sum^{N}_{j=0}e^{x_j}/e_{x_m}})\\
        & = (x_{i}-x_m)-ln(\sum_{j=0}^{N}e^{x_j-x_m})\\
    x_m = max\{\mathbf{x}\}\end{aligned}
    $$
+ 反向传播
    + 公式：
        令$\mathbf{g} = softmax(\mathbf{x})=e^{log\_softmax(\mathbf{x})}$，
        $$
            \mathbf{s}\xleftarrow[]{sum[1]}\nabla\mathbf{y}\\
            \nabla \mathbf{y} - \mathbf{s}\mathbf{g}_i
        $$
    + 推导：
    对于一个一维的输入$\mathbf{x}.shape=[C]$进行推导，二维情况是一维的并行扩展：
        $\begin{aligned}\frac{\partial{\mathbf{y}_k}}{\partial{\mathbf{x}_i}}
            & = \frac{\partial{(\mathbf{x}_k-ln(\sum^{N}_{j=0}e^{\mathbf{x}_j}))}}{\partial{\mathbf{x}_i}} \\
            & = \frac{\partial{\mathbf{x}_k}}{\partial{\mathbf{x_i}}} - \frac{\partial{ln(\sum^{N}_{j=0}e^{\mathbf{x}_j})}}{\partial{\mathbf{x_i}}}\\
            & = \delta_{ki} - \frac{1}{\sum_{j=0}^{N}e^{\mathbf{x}_j}}e^{\mathbf{x}_i},\delta_{ij}=(i==j\ ?\ 1:0)\\
            & = \delta_{ki} - softmax(\mathbf{x}_i)
        \end{aligned}$
        $\begin{aligned}\frac{\partial{\mathbf{L}}}{\partial{\mathbf{x}_i}}
            & =\sum^{C}_{k=1}\frac{\partial{\mathbf{L}}}{\partial\mathbf{y}_k}\frac{\partial{\mathbf{y}_k}}{\partial{\mathbf{x}_i}}\\
            & = \frac{\partial{\mathbf{L}}}{\partial\mathbf{y}_i}\frac{\partial{\mathbf{y}_i}}{\partial{\mathbf{x}_i}}+\sum^{C}_{k=1,k\neq i}\frac{\partial{\mathbf{L}}}{\partial\mathbf{y}_k}\frac{\partial{\mathbf{y}_k}}{\partial{\mathbf{x}_i}}\\
            & = \frac{\partial{\mathbf{L}}}{\partial\mathbf{y}_i}(1-softmax(\mathbf{x}_i))-\sum^{C}_{k=1,k\neq i}\frac{\partial{\mathbf{L}}}{\partial\mathbf{y}_k}softmax(\mathbf{x}_k)\\
            & = \frac{\partial{\mathbf{L}}}{\partial\mathbf{y}_i} - softmax(\mathbf{x}_i)\sum^{C}_{k=1}\frac{\partial{\mathbf{L}}}{\partial\mathbf{y}_i}
        \end{aligned}$
        这里分$i\neq j$和$i=j$两种情况讨论，但最终得到的结果相同，这里推导$i\neq j$的情况：

    + 验证：
    ```python
    import torch
    import torch.nn.functional as F
    import numpy as np


    def softmax(x: torch.Tensor, log: bool = True):
        shift = x - x.max(dim=-1, keepdim=True).values
        exp_shift = shift.exp()
        denorminator = exp_shift.sum(dim=-1, keepdim=True)
        if log:
            y = shift - denorminator.log()
        else:
            y = exp_shift / denorminator
        return y


    def backward_softmax(dy: torch.Tensor, y: torch.Tensor, log: bool):
        # log_softmax: dx = dy - dy.sum(-1) * exp(y)
        if log:
            x_grad = dy - (dy.sum(-1, keepdim=True) * torch.exp(y))
        # softmax: dx = y * (dy - (dy*y).sum(-1))
        else:
            x_grad = y * (dy - (dy * y).sum(-1, keepdim=True))
        return x_grad


    # torch.manual_seed(0)
    # np.random.seed(0)
    b = np.random.randint(1, 513)
    c = np.random.randint(1, 1025)
    log = False
    x = torch.randn((b, c), dtype=torch.float32, device="cuda", requires_grad=True)
    x2 = x.detach().clone().requires_grad_()
    # forward
    out = softmax(x, log=log)
    out2 = F.log_softmax(x2, dim=-1) if log else F.softmax(x2, dim=-1)
    # print(torch.allclose(out, out2, atol=1e-5))
    dy = torch.randn_like(out, dtype=torch.float32, device="cuda")
    dy2 = dy.detach().clone()
    # backward
    x_grad = backward_softmax(dy, out, log)
    out2.backward(dy2)
    print(torch.allclose(x2.grad, x_grad, atol=1e-5))
    ```

## 2.基于PyTorch的torch.autograd.Function实现自动梯度

### 2.1.实现Custom Forward/Backward
这是一个可以对张量实现自动反向传播**并累加**梯度的类。要是实现与其他类一样的自动反向传播和梯度累加，只要继承`torch.autograd.Function`类,并重写`forward`和`backward`函数即可

### 2.2.实现细节
在自定`forward`与`backward`函数时需要注意：
+ `forward`的输入参数数量必须要与`backward`的输出参数**数量相等且一一对应**，如果`forward`的输入参数包含不需要梯度的参数，在`backward`中返回`None`来填充
+ `forward`的输出参数数量与`backward`的输入参数数量相等，可以实现同时累加来自`forward`的多个可微输出的梯度，或逐个累加`forward`可微输出的梯度

### 2.3.实例代码
实例代码实现了max函数和一个双头输出的函数的自定义前向和反向过程，与其他的PyTorch原生算子没有区别，可以交叉使用。
``` python
import torch


class MaxWithIndices(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor):
        # with torch.no_grad():
        values, indices = input_tensor.max(dim=1)
        ctx.save_for_backward(indices)
        ctx.shape = input_tensor.shape
        return values, indices  # int tensor do not have gradient

    @staticmethod
    def backward(ctx, grad_output, grad_output_indices):
        # numbers of parameters should be the same as the output of forward
        # even if the other output do not need gradient
        # and it will be zero graidient
        (indices,) = ctx.saved_tensors
        grad_input = torch.zeros(
            ctx.shape, dtype=grad_output.dtype, device=grad_output.device
        )
        grad_input.scatter_(1, indices.unsqueeze(1), grad_output.unsqueeze(1))

        return grad_input  # output numebr should be the same as the input of forward, for no-gradient input, padding with None


class DoubleOutputFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor):
        output1 = input_tensor * 2  # dy/dx=2
        output2 = input_tensor + 3  # dy/dx=1
        return output1, output2

    @staticmethod
    def backward(
        ctx, grad_output1, grad_output2
    ):  # numbers of parameters should be the same as the output of forward
        grad_input = grad_output1 * 2 + grad_output2 * 1  # 计算梯度
        return grad_input


if __name__ == "__main__":
    shape = (1, 2, 2)
    device = "cuda"
    torch.manual_seed(0)
    ## only one output has gradient
    x = torch.randn(shape, device=device, requires_grad=True)
    y, _ = MaxWithIndices.apply(x)
    dy = torch.randn_like(y, device=device)
    y.backward(dy)
    print(x.grad)

    x1 = torch.tensor(1.0, requires_grad=True)
    y1, y2 = DoubleOutputFunction.apply(x1)
    ## compute gradient from y1 and y2 at the same time
    # torch.autograd.backward([y1, y2], [torch.tensor(1.0), torch.tensor(1.0)])
    # print(x1.grad)

    ## compute gradient from y1 and y2 separately,
    ## the other gradient will be zero
    y1.backward(torch.tensor(1.0))
    y2.backward(torch.tensor(1.0))
    # the result should be the same as the above
    print(x1.grad)
```