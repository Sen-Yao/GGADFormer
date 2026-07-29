import matplotlib.pyplot as plt
# Frozen five-seed means used in the AAAI 2027 label-ratio analysis.
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 8.5,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'axes.linewidth': 0.9,
    'lines.linewidth': 1.7,
    'lines.markersize': 5.5,
})

train_ratios = [1, 2, 3, 5, 7, 10, 15]
x_labels = [f'{ratio}%' for ratio in train_ratios]

auroc_ggad = [0.5125, 0.5133, 0.5137, 0.5098, 0.5374, 0.5306, 0.5411]
auroc_rho = [0.5137, 0.4823, 0.4885, 0.5098, 0.4879, 0.5763, 0.5849]
auroc_vecgad = [0.5876, 0.5992, 0.6505, 0.6496, 0.6637, 0.6714, 0.6702]

colors = ['#C23B32', '#2C6E9B', '#2F7D4A']
markers = ['o', 's', '^']
linestyles = ['-', '--', '-.']

fig, ax = plt.subplots(figsize=(4.8, 2.9))

for values, color, marker, linestyle, label in zip(
    [auroc_ggad, auroc_rho, auroc_vecgad],
    colors,
    markers,
    linestyles,
    ['GGAD', 'RHO', 'VecGAD'],
):
    ax.plot(
        range(len(train_ratios)),
        values,
        color=color,
        marker=marker,
        linestyle=linestyle,
        label=label,
        markeredgecolor='white',
        markeredgewidth=0.8,
    )

ax.set_xlabel('Training Ratio')
ax.set_ylabel('AUROC')
ax.set_xticks(range(len(train_ratios)))
ax.set_xticklabels(x_labels)
ax.set_ylim([0.46, 0.70])
ax.set_yticks([0.48, 0.54, 0.60, 0.66])
ax.legend(
    loc='lower center',
    bbox_to_anchor=(0.5, 1.01),
    ncol=3,
    frameon=False,
    columnspacing=1.4,
    handlelength=2.2,
)
ax.grid(True, linestyle='--', alpha=0.35, linewidth=0.6)

fig.tight_layout(pad=0.5)

plt.savefig('train_ratio_auroc.pdf', format='pdf', bbox_inches='tight', pad_inches=0.02)
plt.savefig('train_ratio_auroc.png', format='png', bbox_inches='tight', pad_inches=0.02)

print("图表已保存为 train_ratio_auroc.pdf")
