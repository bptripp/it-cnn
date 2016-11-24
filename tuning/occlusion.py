__author__ = 'bptripp'

from os.path import join
import numpy as np
import matplotlib
matplotlib.rcParams['xtick.labelsize'] = 14
matplotlib.rcParams['ytick.labelsize'] = 14
import matplotlib.pyplot as plt
from cnn_stimuli import get_image_file_list
from alexnet import preprocess, load_net, load_vgg

use_vgg = False
remove_level = 1
if use_vgg:
    model = load_vgg(weights_path='../weights/vgg16_weights.h5', remove_level=remove_level)
else:
    model = load_net(weights_path='../weights/alexnet_weights.h5', remove_level=remove_level)

def average_over_reps(responses, reps):
    n_occ = responses.shape[0]/reps
    averages = np.zeros((n_occ, responses.shape[1]))
    for i in range(n_occ):
        averages[i,:] = np.mean(responses[i*reps:(i+1)*reps,:], axis=0)
    # plt.subplot(2,1,1)
    # plt.plot(responses)
    # plt.subplot(2,1,2)
    # plt.plot(averages)
    # plt.show()
    return averages

reps = 10
out = []

# occlusion_type = 'black'
occlusion_type = 'red'
# occlusion_type = 'moving'
base_directory = './images/occlusions-' + occlusion_type

image_files = get_image_file_list(join(base_directory, 'circle'), 'png', with_path=True)
im = preprocess(image_files, use_vgg=use_vgg)
out.append(average_over_reps(model.predict(im), reps))
image_files = get_image_file_list(join(base_directory, 'square'), 'png', with_path=True)
im = preprocess(image_files, use_vgg=use_vgg)
# plt.plot(model.predict(im))
# plt.show()
out.append(average_over_reps(model.predict(im), reps))
image_files = get_image_file_list(join(base_directory, 'star'), 'png', with_path=True)
im = preprocess(image_files, use_vgg=use_vgg)
out.append(average_over_reps(model.predict(im), reps))
image_files = get_image_file_list(join(base_directory, 'triangle'), 'png', with_path=True)
im = preprocess(image_files, use_vgg=use_vgg)
out.append(average_over_reps(model.predict(im), reps))
image_files = get_image_file_list(join(base_directory, 'strange'), 'png', with_path=True)
im = preprocess(image_files, use_vgg=use_vgg)
out.append(average_over_reps(model.predict(im), reps))
out = np.array(out)
print(out.shape)

unoccluded_responses = out[:,0,:]
print(unoccluded_responses.shape)
var = np.var(unoccluded_responses, axis=0)
n = 500
ind = (-var).argsort()[:n]
# ind = range(n)

# most_selective_unoccluded = unoccluded_responses[:,ind]
# print(most_selective_unoccluded.shape)


# highest = np.argmax(most_selective_unoccluded, axis=0)
# circle_ind = [i for i in range(n) if highest[i] == 0]

most_selective = out[:,:,ind]
most_sorted = np.zeros_like(most_selective)

for i in range(n):
    order = np.argsort(most_selective[:,0,i])[::-1]
    most_sorted[:,:,i] = most_selective[order,:,i]

# out: shape, occlusion, neuron
print(out[:,1,0])
print(most_sorted[:,1,0])

means = np.mean(most_sorted, axis=2)
plt.figure(figsize=(6,4))
plt.plot(range(1,6), means[:,0]-means[:,4], 'kd-', label='NO')
plt.plot(range(1,6), means[:,1]-means[:,4], 'ko--', label='20% SV')
plt.plot(range(1,6), means[:,2]-means[:,4], 'k^-.', label='50% SV')
plt.plot(range(1,6), means[:,3]-means[:,4], 'ks:', label='90% SV')
legend = plt.legend(loc='upper right', shadow=False, frameon=False)

print(means)

print('preferred responses: ')
print(means[0,:]-means[0,4])

plt.xlabel('Shape rank', fontsize=16)
plt.ylabel('Mean Response', fontsize=16)
plt.tight_layout()
if use_vgg:
    plt.savefig('../figures/occlusions-' + occlusion_type + '-vgg-' + str(remove_level) + '.eps')
else:
    plt.savefig('../figures/occlusions-' + occlusion_type + '-alexnet-' + str(remove_level) + '.eps')

plt.show()
# print(highest)

# plt.plot(most_selective_unoccluded)
# plt.show()
# print(circle_ind)

# selective_responses = out[:,:,ind]
# circle_cell_responses = selective_responses[:,:,circle_ind] #dims: shapes, occlusions, neurons
#
# for i in range(len(circle_ind)):
#     plt.plot(circle_cell_responses[:,:,i])
#     plt.show()

