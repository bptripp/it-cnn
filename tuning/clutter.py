__author__ = 'bptripp'

import numpy as np
import matplotlib
matplotlib.rcParams['xtick.labelsize'] = 14
matplotlib.rcParams['ytick.labelsize'] = 14
import matplotlib.pyplot as plt
from cnn_stimuli import get_image_file_list
from alexnet import preprocess, load_net

model = load_net()

bottom_dir = '/Users/bptripp/code/salman-IT/salman/images/clutter/bottom/'
bottom_image_files = get_image_file_list(bottom_dir, 'png', with_path=True)
bottom_out = model.predict(preprocess(bottom_image_files))

top_dir = '/Users/bptripp/code/salman-IT/salman/images/clutter/top/'
top_image_files = get_image_file_list(top_dir, 'png', with_path=True)
top_out = model.predict(preprocess(top_image_files))

pair_dir = '/Users/bptripp/code/salman-IT/salman/images/clutter/pair/'
pair_image_files = get_image_file_list(pair_dir, 'png', with_path=True)
pair_out = model.predict(preprocess(pair_image_files))

print(pair_out.shape)

maxima = np.max(pair_out, axis=0)
# n = 100
# ind = (-maxima).argsort()[:n]
n = 500
ind = range(n)

sum_out = np.zeros_like(pair_out)
n_top = len(top_image_files)
n_bottom = len(bottom_image_files)
for i in range(n_top):
    for j in range(n_bottom):
        sum_out[i*n_bottom+j,:] = top_out[i,:] + bottom_out[j,:]

large_pair_out = pair_out[:,ind]
large_sum_out = sum_out[:,ind]
plt.figure(figsize=(6,4))
plt.scatter(large_sum_out, large_pair_out, marker='.', c='k')
plt.plot([0, 15], [0, 15], 'k')
plt.plot([0, 15], [0, 7.5], 'k--')
plt.xlim((0,16))
plt.ylim((0,16))
plt.xlabel('Sum of responses to single objects', fontsize=14)
plt.ylabel('Response to object pairs', fontsize=14)
plt.tight_layout()
plt.savefig('clutter.eps')
plt.show()
