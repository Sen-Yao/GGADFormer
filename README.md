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

在半监督图异常检测任务中，模型在训练阶段只能接触到少量的正常节点样本，无法获取任何异常节点信息。这使得模型难以学习“异常”模式，是该任务的核心挑战。

GGADFormer 是一个生成式的、基于 Transformer 的图异常检测框架，通过以下几个核心模块解决了这一挑战：1) 图结构感知的输入编码策略，防止过度平滑；2) 一种新颖的生成式策略，在缺乏真实异常样本的情况下，**人工合成高质量的伪异常样本**；3) 多重学习机制，确保模型学习到鲁棒且具有区分性的节点表征。

### 图结构感知的 Tokenization

传统的图神经网络（如 GCN）通过邻居聚合来学习节点表示，但这通常会导致\*\*过度平滑（over-smoothing）\*\*问题，即随着传播层数增加，节点的表示趋于同质化，难以区分，这对需要捕捉细微差异的异常检测任务是致命的。

为解决此问题并让 Transformer 感知到图结构信息，我们采用一种基于传播机制来编码节点的拓扑特征。

$$X_p^{k+1} = (1-\alpha) \cdot \mathbf{A} \cdot X_p^{k} + \alpha \cdot \mathbf{X}_0$$

其中，$\mathbf{A}$ 是归一化后的邻接矩阵，$\mathbf{X}_0$ 是节点的原始特征。经过 $k$ 次传播后，得到了一系列的 token sequence $X_0, X_p^1, X_p^2,\cdots, X_p^k$，此序列融入了节点邻居的结构信息，同时超参数 $\alpha$ 保证了节点自身的原始特征不会被完全稀释，从而有效缓解了过度平滑问题，保留了节点的独特性。

token sequence 保留了用于对应节点用于异常检测所需要的所有图上下文信息，因此 Transformer 仅需要处理此序列，不需要计算全图中的节点信息，大大降低了计算开销。

### 基于跨空间重构误差引导的伪异常生成

在半监督学习中，尤其当已标记的正常节点极为稀疏时，单纯依赖对比学习机制的模型面临着学习“捷径解”的风险。模型可能仅学会区分有限的训练样本，而未能泛化至对“正常性”这一抽象概念的全面理解。为了应对这一挑战并从根本上提升节点表征的质量，我们设计了一个基于自编码器思想的协同框架，它不仅作为一种强大的**表征正则化 (Representation Regularization)** 手段，更创新性地扮演了**伪异常生成引导者 (Outlier Generation Guidance)** 的角色。

该框架的核心在于一个**编码器-解码器**结构。我们的Transformer编码器 $\mathcal{E}$ 负责将输入令牌 $\mathbf{t}_i$ (包含节点 $v_i$ 的原始及结构化特征) 压缩成一个低维、稠密的嵌入表征 $\mathbf{h}_i = \mathcal{E}(\mathbf{t}_i)$。我们额外设计了一个解码器 $\mathcal{D}_{tok}$，其任务是从嵌入 $\mathbf{h}_i$ “还原”回原始的输入令牌 $\hat{\mathbf{t}}_i = \mathcal{D}_{tok}(\mathbf{h}_i)$。我们认为，一个高质量、信息丰富的节点嵌入，理应蕴含足够的信息来还原其自身的全部构成。由于训练数据主要由正常节点构成，最小化重构误差的过程会“迫使”编码器 $\mathcal{E}$ 去学习和提炼正常模式最本质、最关键的特征，自然地过滤掉随机噪声，从而使最终的嵌入 $\mathbf{h}_i$ 成为对“正常性”的高度浓缩和纯化的表达。

然而，此重构机制的价值远不止于表征正则化。其产生的**重构误差向量**，通常被视为一个待优化的标量损失，实则蕴藏了用以生成高质量伪异常的关键信息。我们理论的出发点是**流形假设**，即正常节点的数据分布在嵌入空间中构成一个低维的“正常性流形”。解码器 $\mathcal{D}_{tok}$ 的存在隐式地定义了这个流形。对于一个正常节点 $v_i$，其在令牌空间的重构误差向量被定义为：
$$ \mathbf{e}^{(tok)}_i = \mathcal{D}_{tok}(\mathbf{h}_i) - \mathbf{t}_i $$
该向量 $\mathbf{e}^{(tok)}_i$ 精确地指明了在模型的当前认知下，样本 $v_i$ 的哪些方面最不符合其学到的“通用正常模式”。它是在高维、结构化的令牌空间中对“异常倾向”的直接量化。

为了在低维、稠密的嵌入空间中利用这一信息，我们引入了一个可学习的线性**投影层** $\mathcal{P}: \mathbb{R}^{d_{token}} \to \mathbb{R}^{d_{emb}}$，它负责将令牌空间中的结构化误差向量 $\mathbf{e}^{(tok)}_i$ **“语义提升”** (semantically lift) 为嵌入空间中的一个有意义的扰动方向 $\mathbf{e}^{(emb)}_i$：
$$ \mathbf{e}^{(emb)}_i = \mathcal{P}(\mathbf{e}^{(tok)}_i) $$
最终，我们通过将一个正常节点的原始嵌入 $\mathbf{h}_i$，沿着这个由模型自身指认的“最可疑”的异常方向 $\mathbf{e}^{(emb)}_i$ 进行扰动，来合成其对应的伪异常嵌入 $\tilde{\mathbf{h}}_i$：
$$ \tilde{\mathbf{h}}_i = \mathbf{h}_i + \alpha \cdot \mathbf{e}^{(emb)}_i $$
其中，$\alpha$ 是一个超参数，用于控制所生成异常的强度。这种生成方式生成的不是随机的或基于简单启发式规则的负样本，而是针对模型当前认知边界的**自适应硬负样本 (Adaptive Hard-Negative Samples)**。通过这种方式，自编码器框架形成了一个协同增强的闭环：解码器在作为正则化器的同时，主动地为对比学习模块提供了最具挑战性的学习信号，从而驱动整个模型学习一个更加鲁棒和紧凑的正常类别边界。

### 中心点对齐：构建可信异常边界

尽管我们设计的生成策略能够创造出与正常样本显著不同的伪异常点，但若不对其在嵌入空间中的位置加以约束，优化过程可能会陷入一个“捷径解”：模型仅仅学会将伪异常点的嵌入推向距正常点簇无穷远的位置，从而轻易地实现分离。这种无约束的分离虽然能降低损失，但会导致模型学习到一个松散、泛化能力差的决策边界，而未能精确刻画“正常”与“异常”之间那条微妙而关键的界线。

为了解决这一问题，我们引入了**中心点对齐 (Central Point Alignment)** 机制，其核心目标是强制所有生成的伪异常样本，落在一个以正常点簇中心为球心、预设半径为界的**可信异常超球 (Credible Anomaly Hypersphere)** 内。

首先，我们通过计算训练批次中所有正常节点嵌入的均值，来动态地确定正常点簇的**原型中心 (prototypical center)** $\mathbf{c}$：
$$ \mathbf{c} = \frac{1}{|V_{norm}|} \sum_{v_i \in V_{norm}} \mathbf{h}_i $$
其中，$V_{norm}$ 是当前批次中的正常节点集合，$\mathbf{h}_i$ 是其对应的嵌入。这个中心点 $\mathbf{c}$ 代表了模型在当前状态下对“普遍正常性”的凝聚表达。

接着，我们定义一个基于边距 (margin) 的对齐损失 $L_{CPA}$。对于每一个生成的伪异常嵌入 $\tilde{\mathbf{h}}_j$，我们计算其到中心点 $\mathbf{c}$ 的欧氏距离，并惩罚那些超出预设信心边距 $R$ (confidence margin) 的样本：
$$ L_{CPA} = \frac{1}{|\tilde{V}|} \sum_{\tilde{\mathbf{h}}_j \in \tilde{V}} \max(0, \ \|\tilde{\mathbf{h}}_j - \mathbf{c}\|_2 - R) $$
其中, $\tilde{V}$ 是生成的伪异常样本集合。此损失函数具有明确的几何意义：当一个伪异常点位于以 $\mathbf{c}$ 为中心、半径为 $R$ 的超球内部或表面时（$\|\tilde{\mathbf{h}}_j - \mathbf{c}\|_2 \le R$），不施加任何惩罚；一旦其越出该边界，损失将随其距离的增加而线性增长。

通过这种方式，“中心点对齐”机制确保了生成的伪异常样本始终处于一个“可信”的异常区域内。它们既不会因与正常点簇过于接近而失去作为负样本的价值，也不会因被推到无限远处而成为优化过程中的平凡样本。相反，它们被约束在决策边界附近，为模型提供了最具挑战性与信息量的学习信号，从而迫使模型学习到一个更加紧凑、精确且鲁棒的异常检测边界。


## 实验结果

我们在 `Amazon`, `photo` 和 `reddit` 三个不同的数据集上，采用仅有 5% 的训练集划分，GGADFormer 在 AUC 和 AP 指标上均展现出优异性能，且优于现有 SOTA 方法 GGAD。

AUC:


|Dataset|Amazon|Reddit|photo|
|-|-|-|-|
|GGAD|0.7514±0.0410|0.5274±0.0052|0.6114±0.0219|
|GGADFormer|0.9324±0.0189|0.5629±0.0161|0.8183±0.0202

AP:

|Dataset|Amazon|Reddit|photo|
|-|-|-|-|
|GGAD|0.3755±0.0749|0.0360±0.0003|0.1269±0.0091|
|GGADFormer|0.8080±0.0088|0.0418±0.0042|0.4756±0.0585


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
# sweep dl92w6w1
python run.py --dataset=Amazon --GT_ffn_dim=256 --GT_num_layers=3 --embedding_dim=256 --peak_lr=1e-3 --end_lr=4e-4 --num_epoch=700 --warmup_updates=30 --pp_k=2 --progregate_alpha=0.3  --con_loss_weight 20 --confidence_margin=0.3
```

```bash
# reddit
# sweep ecftoff3
python run.py --dataset=reddit --GT_ffn_dim=64 --GT_num_heads=4 --GT_num_layers=2 --embedding_dim=256 --peak_lr=1e-4  --end_lr=5e-5 --num_epoch=200 --warmup_updates=50 --pp_k=1 --progregate_alpha=0.2
```

```bash
# photo
# sweep 70v7achn
python run.py --dataset=photo --GT_ffn_dim=256 --GT_num_layers=3 --embedding_dim=256 --peak_lr=1e-3 --end_lr=8e-4 --num_epoch=150 --warmup_updates=50 --progregate_alpha=0.3 --con_loss_weight=20 --confidence_margin=0.3 --batch_size=128
```