import numpy as np
import matplotlib
import matplotlib.patches as patches
import matplotlib.pyplot as plt

matplotlib.rcParams.update({'font.size': 28})

loss_fn_names = [
        'Stress',
        'SST (Low Temp, Incl KL)',
        'SST (High Temp, Incl KL)',
        'SNE (Low Temp, Incl KL)',
        'SNE (High Temp, Incl KL)',
        'SNE (Low Temp, Excl KL)',
        'SNE (High Temp, Excl KL)',
        'Distortion 1',
        'Distortion 2',
]
order_indices = [3, 4, 1, 2, 5, 6, 0, 7, 8]
metric_names = [
        'F1@1',
        'AUC F1@5',
        'AUC F1@10',
        'AUC F1@max',
        'Pearson R',
        'Distortion',
]
counts = [
        [0, 0, 2, 6, 1, 7],
        [16, 28, 19, 8, 0, 0],
        [0, 0, 11, 12, 24, 0],
        [62, 41, 21, 6, 0, 0],
        [0, 1, 10, 16, 24, 0],
        [9, 15, 12, 9, 0, 0],
        [0, 1, 9, 16, 39, 0],
        [7, 7, 3, 4, 1, 73],
        [2, 3, 9, 19, 7, 16],
]

width = 1
xs = np.arange(0, 2 * len(metric_names), 2)
heights = np.zeros(len(metric_names))

fig, ax = plt.subplots(figsize=(20, 10))
for i, idx in enumerate(order_indices):
    fn_counts = counts[idx]
    ax.bar(xs,
           fn_counts,
           bottom=heights,
           edgecolor='white',
           width=width,
           color=plt.cm.Set1.colors[idx],
           label=loss_fn_names[idx])
    heights += fn_counts

    if i == 5:
        for x, h in zip(xs, heights):
            rect = patches.Rectangle((x - width / 2, 0),
                                     width,
                                     h,
                                     linewidth=3,
                                     edgecolor='k',
                                     facecolor='none')
            ax.add_artist(rect)

ax.yaxis.grid(color='lightgray', lw=2, alpha=0.5)
ax.set_axisbelow(True)
ax.tick_params(axis='x', labelrotation=30)
ax.set_ylabel('Number of Experiments')
ax.set_title('Best Performing Loss Functions by Metric', y=1.01)
ax.set_ylim(top=100)
ax.set_xticks(xs)
ax.set_xticklabels(metric_names)
ax.legend(bbox_to_anchor=(1.05, 0.5), loc='center left', ncol=1)
plt.tight_layout()
fig.savefig('loss_fns.pdf', bbox_inches='tight')
