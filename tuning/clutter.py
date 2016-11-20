__author__ = 'bptripp'

import numpy as np
import matplotlib
matplotlib.rcParams['xtick.labelsize'] = 14
matplotlib.rcParams['ytick.labelsize'] = 14
import matplotlib.pyplot as plt
from cnn_stimuli import get_image_file_list
from alexnet import preprocess, load_net, load_vgg

def get_clutter_responses(remove_level):
    # model = load_net(weights_path='../weights/alexnet_weights.h5', remove_level=remove_level)
    # use_vgg = False
    model = load_vgg(weights_path='../weights/vgg16_weights.h5', remove_level=remove_level)
    use_vgg = True

    bottom_dir = './images/clutter/bottom/'
    bottom_image_files = get_image_file_list(bottom_dir, 'png', with_path=True)
    bottom_out = model.predict(preprocess(bottom_image_files, use_vgg=use_vgg))

    top_dir = './images/clutter/top/'
    top_image_files = get_image_file_list(top_dir, 'png', with_path=True)
    top_out = model.predict(preprocess(top_image_files, use_vgg=use_vgg))

    pair_dir = './images/clutter/pair/'
    pair_image_files = get_image_file_list(pair_dir, 'png', with_path=True)
    pair_out = model.predict(preprocess(pair_image_files, use_vgg=use_vgg))

    maxima = np.max(pair_out, axis=0)
    n = 100
    ind = (-maxima).argsort()[:n]
    # n = 500
    # ind = range(n)

    sum_out = np.zeros_like(pair_out)
    n_top = len(top_image_files)
    n_bottom = len(bottom_image_files)
    for i in range(n_top):
        for j in range(n_bottom):
            sum_out[i*n_bottom+j,:] = top_out[i,:] + bottom_out[j,:]

    large_pair_out = pair_out[:,ind]
    large_sum_out = sum_out[:,ind]
    return large_pair_out, large_sum_out


if False:
    remove_level = 1
    large_pair_out, large_sum_out = get_clutter_responses(remove_level)
    plt.figure(figsize=(4.5,4))
    plt.scatter(large_sum_out, large_pair_out, marker='.', c='k')
    plt.plot([0, 15], [0, 15], 'k--')
    plt.plot([0, 15], [0, 7.5], 'k')
    plt.xlim((0,16))
    plt.ylim((0,16))
    plt.xlabel('Sum of responses to single objects', fontsize=14)
    plt.ylabel('Response to object pairs', fontsize=14)
    plt.tight_layout()
    plt.savefig('../figures/clutter-' + str(remove_level) + '.eps')
    plt.show()

if True:
    plt.figure(figsize=(6,2))
    edges = np.linspace(0, np.pi/2, 20)

    for remove_level in range(3):
        plt.subplot(1,3,remove_level+1)
        large_pair_out, large_sum_out = get_clutter_responses(remove_level)
        angle = np.arctan((large_pair_out.flatten() + 1e-6) / (large_sum_out.flatten() + 1e-6))
        plt.hist(angle, edges, color=[.5,.5,.5])
        # if remove_level == 1:
        #     plt.xlabel('Angle from horizontal (radians)', fontsize=14)
        plt.yticks([])
        plt.xticks([0, np.pi/4], ['0', 'pi/4'])
        plt.plot([np.pi/8, np.pi/8], plt.gca().get_ylim(), 'r')

    plt.tight_layout()
    plt.savefig('../figures/clutter-angles.eps')
    plt.show()