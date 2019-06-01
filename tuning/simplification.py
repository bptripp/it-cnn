__author__ = 'bptripp'

import numpy as np
import matplotlib
matplotlib.rcParams['xtick.labelsize'] = 14
matplotlib.rcParams['ytick.labelsize'] = 14
import matplotlib.pyplot as plt
from cnn_stimuli import get_image_file_list
from alexnet import load_vgg, load_net, preprocess

# remove_level = 2
# use_vgg = False
#
# if use_vgg:
#     model = load_vgg(weights_path='../weights/vgg16_weights.h5', remove_level=remove_level)
# else:
#     model = load_net(weights_path='../weights/alexnet_weights.h5', remove_level=remove_level)
#
#
# from_dir = './images/simplification/from/'
# from_image_files = get_image_file_list(from_dir, 'png', with_path=True)
# from_out = model.predict(preprocess(from_image_files, use_vgg=use_vgg))
#
# to_dir = './images/simplification/to/'
# to_image_files = get_image_file_list(to_dir, 'png', with_path=True)
# to_out = model.predict(preprocess(to_image_files, use_vgg=use_vgg))
#
# other_dir = './images/simplification/other/'
# other_image_files = get_image_file_list(other_dir, 'png', with_path=True)
# other_out = model.predict(preprocess(other_image_files, use_vgg=use_vgg))
#
# #TODO: get null distribution by shuffling froms instead of using tos
#
# # baseline = np.min(np.concatenate((from_out, to_out), axis=0), axis=0)
# baseline = np.min(from_out, axis=0)
# from_out = from_out - baseline
# to_out = to_out - baseline
#
# n = 50
# n_shapes = from_out.shape[0]
# # ratios = []
# # nulls = []
# plt.figure(figsize=(6,6))
# edges = np.linspace(0, 2.0, 11)
# labels = ['plant', 'hind leg', 'cube', 'person', 'ball', 'haystack', 'pom-pom', 'apple', 'hand', 'skewer', 'bread', 'cat']
# position = 1
# ratio_fractions_over_one = np.zeros(n_shapes)
# null_fractions_over_one = np.zeros(n_shapes)
#
# unit_maxima = np.max(np.concatenate((from_out, other_out), axis=0), axis=0)
# tolerance = np.mean(unit_maxima) * .05
# print(tolerance)
#
# for i in range(n_shapes):
#     # find units that respond most strongly to this shape, among "from" and "other" groups
#     this_shape_preferred = np.logical_and(from_out[i,:] >= unit_maxima-tolerance, from_out[i,:] > 1)
#     # print(sum(this_shape_preferred))
#     ind = np.where(this_shape_preferred)
#     # print(ind)
#
#     # from_out[i,:]
#
#     # response = from_out[i,:]
#     # ind = (-from_out[i,:]).argsort()[:n]
#     ratios = to_out[i,ind] / from_out[i,ind]
#     nulls = []
#     for j in range(1,n_shapes):
#         nulls.append(to_out[np.mod(i+j, n_shapes),ind] / from_out[i,ind])
#     nulls = np.array(nulls)
#
#     #extra fussing to make order of panels like Tanaka
#     plt.subplot(4,3,position)
#     position = position + 3
#     if position > 12:
#         position = position - 11
#
#     # print(ratios.flatten())
#     ratio_counts, e = np.histogram(ratios.flatten(), edges)
#     ratio_fractions = ratio_counts.astype(float) / np.sum(ratio_counts)
#     null_counts, e = np.histogram(nulls.flatten(), edges)
#     null_fractions = null_counts.astype(float) / np.sum(null_counts)
#
#     plt.bar(edges[:-1], ratio_fractions, width=.2, color=[0,0,1])
#     plt.bar(edges[:-1], -null_fractions, width=.2, color=[0.7,0.7,0.7])
#     plt.xlim((0,2))
#     plt.ylim((-.8,.8))
#     plt.xticks([0,1,2])
#     plt.yticks([-.8,-.4,0,.4,.8])
#     plt.title(labels[i])
#
#     if len(ratios.flatten()) == 0:
#         ratio_fractions_over_one[i] = np.nan
#         null_fractions_over_one[i] = np.nan
#     else:
#         ratio_fractions_over_one[i] = float(np.sum(ratios.flatten() > 1)) / len(ratios.flatten())
#         null_fractions_over_one[i] = float(np.sum(nulls.flatten() > 1)) / len(nulls.flatten())
#
# plt.tight_layout()
#
# print('>1 ' + str(np.nanmean(ratio_fractions_over_one)) + ' ' + str(np.nanmean(null_fractions_over_one)))
#
# if use_vgg:
#     plt.savefig('../figures/simplification-vgg-' + str(remove_level) + '.eps')
# else:
#     plt.savefig('../figures/simplification-alexnet-' + str(remove_level) + '.eps')
#
# plt.show()

# plot fractions over one (from repeated runs)
remove_level = np.array([0,1,2])

#values from all above threshold ...
vgg_ratio_over_one = [0.117499415481, 0.0418854340439, 0.0531346631493]
vgg_null_over_one = [0.0523417204617, 0.0151203682413, 0.0246139848455]
alexnet_ratio_over_one = [0.149668638138, 0.07128067702, .0675081863766]
alexnet_null_over_one = [0.0663521248382, 0.0251136422064, 0.0279159685923]

#values from top 50 ...
# vgg_ratio_over_one = [0.146666666667, 0.123333333333, 0.115]
# vgg_null_over_one = [0.0984848484848, 0.0943939393939, 0.0916666666667]
# alexnet_ratio_over_one = [0.166666666667, 0.116666666667, 0.106666666667]
# alexnet_null_over_one = [0.0763636363636, 0.050303030303, 0.0604545454545]

plt.figure(figsize=(6,2))
plt.subplot(1,2,1)
plt.plot(-remove_level, vgg_ratio_over_one, color='b')
plt.plot(-remove_level, vgg_null_over_one, color='k')
plt.ylim([0,.2])
plt.title('VGG-16')
plt.subplot(1,2,2)
plt.plot(-remove_level, alexnet_ratio_over_one, color='b')
plt.plot(-remove_level, alexnet_null_over_one, color='k')
plt.ylim([0,.2])
plt.title('Alexnet')
# plt.xlabel('Distance from output (layers)')
plt.tight_layout()
plt.savefig('../figures/simplification-over-one.eps')
plt.show()
