import matplotlib.pyplot as plt
import numpy as np

# 设置全局字体和样式（KDD风格）
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 14,
    'axes.labelsize': 16,
    'axes.titlesize': 16,
    'xtick.labelsize': 13,
    'ytick.labelsize': 13,
    'legend.fontsize': 12,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'axes.linewidth': 1.2,
    'lines.linewidth': 2,
    'lines.markersize': 8,
})

# 数据
train_ratios = [1, 2, 5, 10, 15]
x_labels = ['1%', '2%', '5%', '10%', '15%']

data = {
    'Amazon': {
        'AUROC': [0.9162, 0.9396, 0.9341, 0.9388, 0.9323],
        'AUPRC': [0.7914, 0.8169, 0.8054, 0.8039, 0.7971]
    },
    'Elliptic': {
        'AUROC': [0.5567, 0.734, 0.7627, 0.7644, 0.7617],
        'AUPRC': [0.1135, 0.2203, 0.2813, 0.3114, 0.331]
    },
    'Tolokers': {
        'AUROC': [0.5876, 0.6258, 0.6496, 0.6714, 0.6702],
        'AUPRC': [0.2647, 0.294, 0.303, 0.3207, 0.3222]
    }
}

# KDD风格的颜色和标记
colors = ['#E24A33', '#348ABD', '#8EBA42']  # 红、蓝、绿
markers = ['o', 's', '^']  # 圆、方、三角
linestyles = ['-', '-', '-']

# 创建图形 (两个子图: AUROC 和 AUPRC)
fig, axes = plt.subplots(1, 2, figsize=(12, 3.5))

# 绘制 AUROC 子图
ax1 = axes[0]
ax1.set_box_aspect(0.4)
for idx, (dataset, values) in enumerate(data.items()):
    ax1.plot(range(len(train_ratios)), values['AUROC'], 
             color=colors[idx], marker=markers[idx], 
             linestyle=linestyles[idx], label=dataset,
             markeredgecolor='white', markeredgewidth=1.5)

ax1.set_xlabel('Training Ratio')
ax1.set_ylabel('AUROC')
ax1.set_xticks(range(len(train_ratios)))
ax1.set_xticklabels(x_labels)
ax1.set_ylim([0.3, 1.0])
ax1.legend(loc='lower right', frameon=True, edgecolor='black', fancybox=False)
ax1.grid(True, linestyle='--', alpha=0.4, linewidth=0.8)
fig.text(0.25, 0.02, '(a) AUROC', ha='center', fontsize=16)

# 绘制 AUPRC 子图
ax2 = axes[1]
ax2.set_box_aspect(0.4)
for idx, (dataset, values) in enumerate(data.items()):
    ax2.plot(range(len(train_ratios)), values['AUPRC'], 
             color=colors[idx], marker=markers[idx], 
             linestyle=linestyles[idx], label=dataset,
             markeredgecolor='white', markeredgewidth=1.5)

ax2.set_xlabel('Training Ratio')
ax2.set_ylabel('AUPRC')
ax2.set_xticks(range(len(train_ratios)))
ax2.set_xticklabels(x_labels)
ax2.set_ylim([0.0, 0.9])
ax2.legend(loc='lower right', frameon=True, edgecolor='black', fancybox=False)
ax2.grid(True, linestyle='--', alpha=0.4, linewidth=0.8)
fig.text(0.75, 0.02, '(b) AUPRC', ha='center', fontsize=16)

# 调整布局
plt.tight_layout()
plt.subplots_adjust(bottom=0.18)

# 保存为PDF
plt.savefig('training_ratio.pdf', format='pdf', bbox_inches='tight', pad_inches=0.05)
plt.savefig('training_ratio.png', format='png', bbox_inches='tight', pad_inches=0.05)

print("图表已保存为 training_ratio_analysis.pdf")
plt.show()