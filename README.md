# GGADFormer

## Requirements

- CUDA 11.8

To install requirements:

```bash
conda create -n GGADFormer python=3.8 -y
conda activate GGADFormer

# Install Pytorch (Depend on your CUDA version, see https://pytorch.org/get-started/previous-versions/)
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia

pip install -r requirements.txt
```

## 模型架构

在半监督图异常检测任务中，模型在训练阶段只能接触到少量的正常节点样本，无法获取任何异常节点信息。这使得模型难以学习“异常”模式，是该任务的核心挑战。GGADFormer 是一个生成式的、基于 Transformer 的框架，通过以下几个核心模块解决了这一挑战：1) 一种图结构感知的输入编码策略，防止过度平滑；2) 一种新颖的生成式策略，在缺乏真实异常样本的情况下，**人工合成高质量的伪异常样本**；3) 多重对比学习机制，确保模型学习到鲁棒且具有区分性的节点表征。

### 图结构感知的 Tokenization

传统的图神经网络（如 GCN）通过邻居聚合来学习节点表示，但这通常会导致\*\*过度平滑（over-smoothing）\*\*问题，即随着传播层数增加，节点的表示趋于同质化，难以区分，这对需要捕捉细微差异的异常检测任务是致命的。

为解决此问题并让 Transformer 感知到图结构信息，我们采用一种基于传播机制来编码节点的拓扑特征。

$$\text{features} = (1-\alpha) \cdot \mathbf{A} \cdot X_p + \alpha \cdot \mathbf{X}_0$$

其中，$\mathbf{A}$ 是归一化后的邻接矩阵，$\mathbf{X}_0$ 是节点的原始特征。经过 $k$ 次传播后，得到的 `feature` 融入了节点邻居的结构信息，同时超参数 $\alpha$ 保证了节点自身的原始特征不会被完全稀释，从而有效缓解了过度平滑问题，保留了节点的独特性。

我们将节点的原始特征 (`x_0`) 和传播后的结构特征 (`features`) 分别通过线性层和共享 MLP 进行编码，得到 `h_raw` 和 `h_prop`，然后将它们拼接作为 Transformer 的输入 Token。

### 生成式伪异常样本策略

由于训练时无法获取异常样本，GGADFormer 引入了一种生成式策略来模拟异常模式。我们认为，一个节点的“正常”模式可以由其邻居节点的嵌入所代表。因此，通过聚合训练集中部分正常节点的邻居嵌入，我们可以人工生成**伪异常样本**。

具体而言，我们从正常节点中随机选择一个子集，将其邻居节点的嵌入通过一个线性层并求平均，以此作为该节点的伪异常表示。这种\*\*“模式不匹配（pattern mismatch）”\*\*策略的直觉在于：当一个节点的邻居嵌入（代表其正常模式）被赋予另一个节点时，便人为地制造了一种特征或结构上的不一致性，从而模拟了真实异常点与其邻居的差异。

为了保证伪异常样本的质量，防止其过于偏离正常模式而变得“过于异常”，我们引入了重建损失 $L\_{rec}$：

$$L_{rec}=\frac{1}{|V_O|}\sum_{v_i\in V_O}||\hat {h}_i-(h_i+\epsilon)||^2$$

其中，$V\_O$ 为用于生成伪异常的节点集，$h\_i$ 为正常节点的嵌入，$\\hat{h}\_i$ 为其对应的伪异常嵌入，$\\epsilon$ 为高斯噪声。该损失约束了伪异常样本与原始正常样本的嵌入距离，确保生成的伪异常具有\*\*“可信性”\*\*。

### 图正常性对齐（Graph Normality Alignment）

在无监督/半监督学习中，\*\*表征坍缩（representation collapse）\*\*是一个常见问题，即所有节点最终被映射到嵌入空间的一个狭窄区域，导致区分性丧失。此外，即使是正常节点，其本身也具有异质性（heterogeneity）。

为解决这些问题，我们设计了\*\*“图正常性对齐”\*\*机制。我们用两个独立的 MLP 分别将原始特征编码 `h_raw` 和传播特征编码 `h_prop` 映射到另一个表示空间，得到 `z_raw` 和 `z_prop`。

在此空间中，我们利用 **InfoNCE 损失**进行双重对齐：

1.  **节点内对齐**：鼓励同一节点的 `z_raw` 和 `z_prop` 相互靠近。这确保了节点的原始特征信息和结构信息是互补且一致的，使节点表征更加鲁棒。
2.  **节点间对齐**：鼓励不同节点的 `z_raw` 和 `z_prop` 分别相互远离。这直接解决了表征坍缩问题，强制模型学习一个更具区分性的嵌入空间，即使是正常的节点，其独特性也能被充分保留。

### 中心点对齐（Central Point Alignment）

如果生成的伪异常样本被允许在嵌入空间中无限远离正常点簇，模型可能无法学习到“正常”与“异常”之间真正的边界。我们引入了\*\*“中心点对齐”\*\*机制，将伪异常样本约束在一个以正常点簇中心为中心的超平面球中，以强迫模型学习这种边界。

首先，我们计算所有正常节点的嵌入均值 `h_mean`，将其视为正常点簇的中心。然后，通过一个对比学习损失，鼓励伪异常样本的嵌入 `outlier_emb` 处于以 `h_mean` 为中心的超平面球内。

```python
outlier_to_center_dist = torch.norm(outlier_emb - h_mean.squeeze(0), p=2, dim=1)
margin_excess = outlier_to_center_dist - args.confidence_margin
con_loss = torch.mean(torch.relu(margin_excess))
```

此损失函数惩罚那些距离 `h_mean` 超过 `confidence_margin` 的伪异常样本。这确保了生成的伪异常样本位于一个\*\*“可信异常”\*\*的区域内，它们既不完全坍缩到正常簇中，也不被推到无限远处，从而为模型提供了更具挑战性和有效性的学习信号。

### 基于特征重构的表征正则化

在训练样本，特别是已标记的正常节点，极为稀疏的极端场景下，模型单靠对比学习机制可能难以充分、全面地学习到“正常”模式的内在分布。这可能导致学习到的表征空间存在“捷径解”，即模型仅仅学会了区分训练集里有限的几个样本，而没有真正泛化到对“正常性”这一抽象概念的理解。

为了解决这个问题，并从根本上提升节点表征的质量，我们引入了一个基于自编码器 (Autoencoder) 思想的重构任务。此任务在这里并非直接用于异常判断，而是作为一种强大的表征正则化 (Representation Regularization) 手段。

核心思想在于：一个高质量、信息丰富的节点表征 `emb`，理应蕴含足够的信息来还原其自身的原始构成。我们的 Transformer 编码器负责将输入 token（包含原始特征和结构信息）压缩成一个低维、稠密的表征 `emb`；我们则额外设计一个解码器 (Decoder)，它的唯一任务就是接收 `emb` 并尝试将其“还原”回原始的输入令牌。

由于训练数据绝大多数是正常节点，解码器为了成功完成重构任务（即最小化重构误差），会“迫使”上游的 Transformer 编码器去学习和提炼正常节点最本质、最关键的特征模式。那些随机的、非结构性的噪声会在这个压缩-解压过程中被自然地过滤掉。因此，最终得到的表征 emb 会是对“正常性”的一个高度浓缩和纯化的表达。

## 实验结果

我们在 `Amazon`, `photo` 和 `reddit` 三个不同的数据集上，采用仅有 5% 的训练集划分，GGADFormer 在 AUC 和 AP 指标上均展现出优异性能，且优于现有 SOTA 方法 GGAD。

AUC:


|Dataset|Amazon|Reddit|photo|
|-|-|-|-|
|GGAD|0.7514±0.0410|0.5274±0.0052|0.6114±0.0219|
|GGADFormer|0.8722±0.0279|0.5537±0.0165|0.6922±0.0457

AP:

|Dataset|Amazon|Reddit|photo|
|-|-|-|-|
|GGAD|0.3755±0.0749|0.0360±0.0003|0.1269±0.0091|
|GGADFormer|0.6395±0.1185|0.0409±0.0034|0.21355±0.0534


以下为复现实验所使用的超参数配置：

### GGAD

```bash
# Amazon
python run.py --embedding_dim=300 --model_type=GGAD --margin_loss_weight=1 --warmup_updates=0 --num_epoch=200 --peak_lr=1e-3 --end_lr=1e-3 --train_rate 0.05  --dataset=reddit 
```

```bash
# reddit
python run.py --embedding_dim=300 --model_type=GGAD --margin_loss_weight=1 --warmup_updates=0 --num_epoch=50 --peak_lr=1e-3 --end_lr=1e-3 --train_rate 0.05  --dataset=reddit 
```


```bash
# photo
python run.py --embedding_dim=300 --model_type=GGAD --margin_loss_weight=1 --warmup_updates=0 --num_epoch=50 --peak_lr=1e-3 --end_lr=1e-3 --train_rate 0.05  --dataset=photo 
```

### GGADFormer

```bash
# Amazon
# sweep bso1m8ot
python run.py --dataset=Amazon --GT_ffn_dim=256 --GT_num_layers=3 --embedding_dim=256 --peak_lr=1e-4 --end_lr=5e-5 --num_epoch=120 --warmup_epoch=30 --warmup_updates=30 --progregate_alpha=0.2 
```

```bash
# reddit
python run.py --GT_attention_dropout=0.4 --GT_dropout=0.4 --GT_ffn_dim=128 --GT_num_heads=2 --GT_num_layers=2 --con_loss_weight=10 --dataset=reddit --embedding_dim=128 --model_type=GGADFormer --num_epoch=400 --warmup_updates=50 --peak_lr=3e-4 --end_lr=1e-4 --pp_k=1 --progregate_alpha=0.1 --proj_dim=64 --seed=0 --train_rate=0.05 --warmup_epoch=50 --confidence_margin=2 --sample_rate=0.3
```

```bash
# photo
python run.py --GT_attention_dropout=0.4 --GT_dropout=0.4 --GT_ffn_dim=128 --GT_num_heads=2 --GT_num_layers=2 --con_loss_weight=10 --confidence_margin=2 --data_split_seed=42 --dataset=photo --embedding_dim=128 --end_lr=2e-4 --model_type=GGADFormer --num_epoch=150 --peak_lr=3e-4 --pp_k=3 --progregate_alpha=0.05 --proj_dim=64 --seed=4 --train_rate=0.05 --warmup_epoch=50 --warmup_updates=50
```