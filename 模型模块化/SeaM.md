我打算先复现一下论文中的代码。
`https://github.com/qibinhang/SeaM.git`
要用cuda，试一下之前的act gpu服务器还能不能用。。。

```
ssh suyunhe25@219.224.171.201 -A
Qoed78fPPHUlMnl
```
还能用。

在内网可以登录`ssh suyunhe25@192.168.5.201 -A`，密码同上。



还得学习集群的使用方法。

详尽内容请参阅 GPU 集群操作手册 https:/gitlab.act.buaa.edu.cn/ACT/gpu-cluster/-/wikis/home
用户名：`suyunhe25`
密码：`Qoed78fPPHUlMnl`


clone的时候报错
```
error: RPC failed; curl 16 Error in the HTTP2 framing layer fatal: expected flush after ref listing
```

可能是由于服务器不支持HTTP/2，或网络连接不稳定。解决办法是禁用`HTTP/2`
```
git config --global http.version HTTP/1.1
```


交互式命令：
```bash
srun --pty bash
```

查看conda环境
```shell
conda env list
```

查看一个conda环境中安装的包
```shell
conda list
```

提交任务
```shell
srun --gres=gpu:V100:1 test.sh
```
### 1. 数据集

`CIFAR-10`、`CIFAR-100`和 `ImageNet`.
CIFAR-10是一个更接近普适物体的彩色图像数据集。CIFAR-10 是由Hinton 的学生Alex Krizhevsky 和Ilya Sutskever 整理的一个用于识别普适物体的小型数据集。一共包含10 个类别的RGB 彩色图片：飞机（ airplane ）、汽车（ automobile ）、鸟类（ bird ）、猫（ cat ）、鹿（ deer ）、狗（ dog ）、蛙类（ frog ）、马（ horse ）、船（ ship ）和卡车（ truck ）。  
每个图片的尺寸为32 × 32 ，每个类别有6000个图像，数据集中一共有50000 张训练图片和10000 张测试图片。

![[Pasted image 20240723214146.png]]

github上的链接失效了，考虑手动下载。

#### CIFAR-10
一个更接近普适物体的彩色图像数据集。包含10 个类别的RGB 彩色图片：飞机（ airplane ）、汽车（ automobile ）、鸟类（ bird ）、猫（ cat ）、鹿（ deer ）、狗（ dog ）、蛙类（ frog ）、马（ horse ）、船（ ship ）和卡车（ truck ）。
每个图片的尺寸为32 × 32 ，每个类别有6000个图像，数据集中一共有50000 张训练图片和10000 张测试图片。

下载地址：[CIFAR-10 和 CIFAR-100 数据集 (toronto.edu)](https://www.cs.toronto.edu/~kriz/cifar.html)


### 2. 模型下载
自动下载太慢，考虑手动下载。
从`https://github.com/chenyaofo/pytorch-cifar-models/releases/download/vgg/cifar10_vgg16_bn-6ee7ea24.pt`
下载模型，放到`/home/LAB/suyunhe25/.cache/torch/hub/checkpoints/cifar10_vgg16_bn-6ee7ea24.pt`


### 3. 使用SeaM

```
  |--- README.md                        :  user guidance
  |--- data/                            :  experimental data
  |--- src/                             :  source code of our work
  |------ global_config.py              :  setting the path
  |------ binary_class/                 :  direct reuse on binary classification problems
  |--------- model_reengineering.py     :  re-engineering a trained model and then reuse the re-engineered model
  |--------- calculate_flop.py          :  calculating the number of FLOPs required by reusing the re-engineered and original models
  |--------- calculate_time_cost.py     :  calculating the inference time required by reusing the re-engineered and original models
  |--------- ......                 
  |------ multi_class/                  :  direct reuse on multi-class classification problems
  |--------- ...... 
  |------ defect_inherit/               :  indirect reuse 
  |--------- reengineering_finetune.py  :  re-engineering a trained model and then fine-tuning the re-engineered model
  |--------- standard_finetune.py       :  using standard fine-tuning approach to fine-tune a trained model
  |--------- eval_robustness.py         :  calculating the defect inheritance rate
  |--------- ......
```

SeaM有2个功能，直接复用和间接复用。
![[Pasted image 20240723195557.png]]

#### 3. 1. 直接复用
例如，在 ImageNet 数据集上构建的已训练模型能够识别 19 种昆虫。针对 一个昆虫十分类的目标任务，开发者能够直接复用从该已训练模型中分解得到的昆虫十分类模块。在部署运行所复用的模块时，由于 SeaM 通过将无关权重的值置零来实现 移除无关权重的效果，模块的实际权重数量仍然与已训练模型相同。但是，大量无关权 重的值被置零，使得模块的权重矩阵是稀疏的。因此，当在目标任务上部署和运行模块 时，开发者可使用稀疏模型运行引擎，例如 DeepSparse[144]，来编译并部署模块，从而 提高模块的推理速度。

代码中，直接复用分为 **二分类** 和 **多分类** 问题的重新设计。

#### 3.1.1. 二分类

##### 实验目录

```
cd src/binary_class
```

##### 代码分析

**args:**
```python
def get_args():
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, choices=['vgg16', 'resnet20', ], required=True)

parser.add_argument('--dataset', type=str, choices=['cifar10', 'cifar100'], required=True)

parser.add_argument('--target_class', type=int, required=True)

parser.add_argument('--shots', default=-1, type=int, help='how many samples for each classes.')

parser.add_argument('--seed', type=int, default=0, help='the random seed for sampling ``num_classes'' classes as target classes.')

parser.add_argument('--n_epochs', type=int, default=300)

parser.add_argument('--lr_head', type=float, default=0.1)

parser.add_argument('--lr_mask', type=float, default=0.1)

parser.add_argument('--alpha', type=float, default=1,

help='the weight for the weighted sum of two losses in re-engineering.')

parser.add_argument('--early_stop', type=int, default=-1)

parser.add_argument('--tuning_param', action='store_true')

args = parser.parse_args()

return args
```

默认状态：
```
args:  Namespace(alpha=1.0, dataset='cifar10', early_stop=-1, lr_head=0.1, lr_mask=0.01, model='vgg16', n_epochs=300, seed=0, shots=-1, target_class=0, tuning_param=False)
```

**model:**
通过这句加载：
```python
model = eval(f'{args.dataset}_{args.model}')(pretrained=True).to('cuda')
```

以VGG16替换：
```
VGG(
  (features): Sequential(
    (0): MaskConv(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU(inplace=True)
    (3): MaskConv(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (4): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (5): ReLU(inplace=True)
    (6): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (7): MaskConv(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (8): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (9): ReLU(inplace=True)
    (10): MaskConv(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (11): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (12): ReLU(inplace=True)
    (13): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (14): MaskConv(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (15): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (16): ReLU(inplace=True)
    (17): MaskConv(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (18): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (19): ReLU(inplace=True)
    (20): MaskConv(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (21): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (22): ReLU(inplace=True)
    (23): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (24): MaskConv(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (25): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (26): ReLU(inplace=True)
    (27): MaskConv(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (28): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (29): ReLU(inplace=True)
    (30): MaskConv(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (31): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (32): ReLU(inplace=True)
    (33): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (34): MaskConv(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (35): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (36): ReLU(inplace=True)
    (37): MaskConv(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (38): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (39): ReLU(inplace=True)
    (40): MaskConv(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (41): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (42): ReLU(inplace=True)
    (43): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (classifier): Sequential(
    (0): MaskLinear(in_features=512, out_features=512, bias=True)
    (1): ReLU(inplace=True)
    (2): Dropout(p=0.5, inplace=False)
    (3): MaskLinear(in_features=512, out_features=512, bias=True)
    (4): ReLU(inplace=True)
    (5): Dropout(p=0.5, inplace=False)
    (6): MaskLinear(in_features=512, out_features=10, bias=True)
  )
)
```





在`main()`中加载模型和数据集。然后调用`reengineering()`函数进行模型的重新设计。


weight_mask? bias_mask

**优化器设置：**

```python
# just optimize reengineered_model head
optim_h = torch.optim.Adam(
module_head.parameters(),
lr_head
)

# jointly optimize masks and reengineered_model head

optim_hm = torch.optim.Adam(
[{'params': mask_params, 'lr': lr_mask},
{'params': module_head.parameters(), 'lr': lr_head}]
)
```

- `optim_h` 只优化 `module_head` 的参数，学习率为 `lr_head`。
- `optim_hm` 同时优化 `mask_params` 和 `module_head` 的参数，分别使用 `lr_mask` 和 `lr_head`

优化策略：
```python
strategy = [0, 1, 1, 1, 1] * (int(15 / 5) + 1)

```

每 5 个 epochs 中有 1 个 epoch 只优化 `head`，其余 4 个 epochs 同时优化 `head` 和 `mask`。


训练和验证循环:
```python
for epoch in range(n_epochs):
    if strategy[epoch] == 0:
        print(f'\nEpoch {epoch}: Just   Head')
        optim = optim_h
    else:
        print(f'\nEpoch {epoch}: Head & Mask')
        optim = optim_hm

    reengineered_model = self._train(reengineered_model, optim, alpha)
    acc, loss, loss_pred, loss_weight_ratio = self._test(reengineered_model, alpha)

    # 更新最优模型
    if acc_with_ratio > best_acc_with_ratio:
        best_acc_with_ratio = acc_with_ratio
        bak_loss_weight_ratio = loss_weight_ratio
        bak_acc = acc
        bak_epoch = epoch
        bak_reengineered_model = copy.deepcopy(reengineered_model)

    if acc >= self.acc_pre_model:
        enable_early_stop = True

    if acc >= self.acc_pre_model:
        if int(loss_weight_ratio * 1000) < int(best_loss_weight_ratio * 1000):
            early_stop_epochs = 0
        elif loss_weight_ratio > best_loss_weight_ratio:
            continue
        elif acc <= best_acc:
            continue
        best_loss_weight_ratio = loss_weight_ratio
        best_acc = acc
        best_epoch = epoch
        best_reengineered_model = copy.deepcopy(reengineered_model)
    elif enable_early_stop:
        early_stop_epochs += 1
        if early_stop_epochs == early_stop:
            print(f'Early Stop.\n\n')
            break

if best_reengineered_model is None:
    best_reengineered_model = bak_reengineered_model
    best_acc = bak_acc
    best_loss_weight_ratio = bak_loss_weight_ratio
    best_epoch = bak_epoch

print(f"\n{'Epoch':^6}  {'Acc':^8}  {'Weight_ratio':^12}")
print(f'{best_epoch:^6}  {best_acc:^8.2%}  {best_loss_weight_ratio:^12.2%}')
return best_reengineered_model

```

- 遍历每个 epoch，根据 `strategy` 决定使用哪个优化器。
- 调用 `_train` 方法对 `reengineered_model` 进行训练。
- 调用 `_test` 方法对 `reengineered_model` 进行验证。
- 更新最优模型 `best_reengineered_model`，并根据早停策略决定是否提前停止训练。

训练方法`_train`
- 设置模型为训练模式，并固定 BatchNorm 层。
- 初始化存储损失和准确率的变量。
- 循环遍历训练数据集：
    - 将数据加载到设备（GPU/CPU）。
    - 清零优化器的梯度。
    - 前向传播计算输出。
    - 计算损失，包括预测损失和权重比率损失。
    - 反向传播计算梯度并更新参数。
    - 统计和累加损失值和准确率。
- 计算并打印训练阶段的总体准确率和平均损失。
- 返回训练后的模型。

其中，损失函数为：
```python
batch_pred_loss = F.cross_entropy(batch_outputs, batch_labels)
batch_weight_ratio_loss = reengineered_model.count_weight_ratio()
# batch_loss = (1 - alpha) * batch_pred_loss + alpha * batch_weight_ratio_loss
batch_loss = batch_pred_loss + alpha * batch_weight_ratio_loss
```
分别为**在目标数据集上的预测与真实标签之间的交叉熵损失**和**权重保留率**。

`count_weight_ratio()`疑似是库函数。


测试方法`_test`
- **`@torch.no_grad()`**：
    - 禁用梯度计算，用于测试和推理阶段，可以减少内存消耗，加快运算速度。
- **`reengineered_model.eval()`**：
    - 将模型设置为评估模式。在评估模式下，模型的 dropout 和 batch normalization 行为会有所不同。
- **初始化累积变量**：
    - `loss_pred, loss_weight_ratio, loss = 0.0, 0.0, 0.0`
        - 分别用于累积预测损失、权重比例损失和总损失。
    - `n_corrects = 0, n_samples = 0`
        - 用于累积正确预测的数量和总样本数。
- **遍历验证集**：
    - 使用 `tqdm` 库展示进度条，便于实时查看验证进度。
- **前向传播和损失计算**：
    - 将输入和标签加载到 GPU。
    - 计算模型的输出。
    - 使用交叉熵损失函数计算预测损失。
    - 计算权重比例损失。
    - 总损失为预测损失和权重比例损失的加权和。
- **累积结果**：
    - 累积样本数。
    - 计算预测正确的数量，并累积。
    - 将损失乘以样本数后累积，以便计算平均损失。
- **计算和打印结果**：
    - 计算准确率。
    - 计算平均损失。
    - 打印验证结果，包括准确率、总损失、预测损失和权重比例损失。
- **返回结果**：
    - 返回准确率和损失，便于后续分析和比较。

### 运行脚本

#### bug1：找不到模块
```
Traceback (most recent call last):
  File "/home/LAB/suyunhe25/SeaM/src/binary_class/model_reengineering.py", line 8, in <module>
    from binary_class.reengineer import Reengineer
ModuleNotFoundError: No module named 'binary_class'
```
原因是模块没有在Python路径里。
源码中添加了 `sys.path.append('../')` 和 `sys.path.append('../..')` 试图将上一级和上上一级目录添加到 Python 路径中，但是好像没生效，推测是用脚本提交作业产生的。

将项目根路径的绝对路径添加进去就解决了。

```python
sys.path.append('/home/LAB/suyunhe25/SeaM/src/')
```

```python
python model_reengineering.py --model vgg16 --dataset cifar10 --target_class 0 --lr_mask 0.01 --alpha 1
```


#### bug2: 安装支持cuda的pytorch时报错
```
ClobberError: This transaction has incompatible packages due to a shared path.   packages: defaults/linux-64::intel-openmp-2023.1.0-hdb19cb5_46306, defaults/linux-64::llvm-openmp-14.0.6-h9e868ea_0   path: 'lib/libomptarget.so'
```
解决方案：
```shell
conda clean --all
```

#### bug3:没有数据集

错误信息
```
srun: job 136795 queued and waiting for resources
srun: job 136795 has been allocated resources
in test.sh
Namespace(alpha=1.0, dataset='cifar10', early_stop=-1, lr_head=0.1, lr_mask=0.01, model='vgg16', n_epochs=300, seed=0, shots=-1, target_class=0, tuning_param=False)
Traceback (most recent call last):
  File "/home/LAB/suyunhe25/SeaM/src/binary_class/model_reengineering.py", line 121, in <module>
    main()
  File "/home/LAB/suyunhe25/SeaM/src/binary_class/model_reengineering.py", line 97, in main
    acc_pre_model = eval_pretrained_model()
  File "/home/LAB/suyunhe25/SeaM/src/binary_class/model_reengineering.py", line 90, in eval_pretrained_model
    dataset_test = load_dataset(args.dataset, is_train=False, target_class=args.target_class, reorganize=False)
  File "/home/LAB/suyunhe25/SeaM/src/binary_class/datasets/dataset_loader.py", line 10, in load_dataset
    dataset = load_cifar10(is_train, shots=shots, target_class=target_class, reorganize=reorganize)
  File "/home/LAB/suyunhe25/SeaM/src/binary_class/datasets/load_cifar10.py", line 22, in load_cifar10
    dataset =  datasets.CIFAR10(f'{config.dataset_dir}', train=is_train, transform=transform)
  File "/home/LAB/suyunhe25/anaconda3/envs/SeaM2/lib/python3.8/site-packages/torchvision/datasets/cifar.py", line 68, in __init__
    raise RuntimeError('Dataset not found or corrupted.' +
RuntimeError: Dataset not found or corrupted. You can use download=True to download it
srun: error: dell-gpu-15: task 0: Exited with exit code 1
```


下载后解决。


#### 训练结果：
![[Pasted image 20240726221255.png]]

在前 50 轮搜索中，权重保留率迅速下降，然后逐渐收敛。
准确率则维持了高标准。

![[Pasted image 20240727204146.png]]


### 问题与思考

#### 为什么模块化后准确率提升？
是因为任务变简单了，从10分类或100分类变成2分类或5分类。

#### 加入全链接层的影响

可以忽略不计。


![[Pasted image 20240727212841.png]]

![[Pasted image 20240727213108.png]]


#### vgg -> vgg for cifar
删除了`self.avgpool = nn.AdaptiveAvgPool2d((7, 7))`,用于将输入特征图的大小调整为固定尺寸。由于`cifar`数据集本身尺寸是固定的，所以可能不需要这个层，就删除了，从而减小计算量并简化模型结构。

重写了构造函数


#### vgg for cifar -> new vgg(reengineering use)

添加字段`is_reengineering`，初值为`false`。

**classifier** 中重写了linear。
```python
self.classifier = nn.Sequential(
	MaskLinear(512, 512, is_reengineering=is_reengineering), # new
	nn.ReLU(True),
	nn.Dropout(),
	MaskLinear(512, 512, is_reengineering=is_reengineering),
	nn.ReLU(True),
	nn.Dropout(),
	MaskLinear(512, num_classes, is_reengineering=is_reengineering),
)
```

与`nn.Linear`相比，添加了掩码，并根据掩码去计算`weight`和`bias`。


```python
```



添加模块头。
```python
if is_reengineering:
	self.module_head = nn.Sequential(
	nn.ReLU(inplace=True),
	nn.Linear(num_classes, 2)
)
```


