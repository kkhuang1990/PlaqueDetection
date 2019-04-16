# _*_ coding: utf-8 _*_

""" plot various metrics vs the number of images """
import matplotlib as mpl
mpl.use('Agg')

import matplotlib.pyplot as plt

x = [15, 23, 31, 39, 47, 55, 63]
hd95s = [3.5892, 3.3225, 3.4076, 4.5638, 3.4362, 3.3689, 3.379]
asds = [1.6966, 1.5622, 1.6021, 2.2233, 1.6539, 1.6295, 1.6107]
dscs = [0.8363, 0.8467, 0.8412, 0.8083, 0.8452, 0.8403, 0.8431]
vds = [0.7215, 0.5053, 0.6262, 0.592, 0.6854, 0.3322, 0.7321]

labels = ['HD95', 'ASD', 'DSC', 'RAVD']
colors = ['red', 'green', 'blue', 'saddlebrown']
plt.figure()
for label, metric, c in  zip(labels, [hd95s, asds, dscs, vds], colors):
    plt.plot(x, metric, c=c, label=label)

plt.legend()
plt.savefig("./metrics_vs_numimages.pdf")

