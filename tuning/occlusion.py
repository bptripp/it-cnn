__author__ = 'bptripp'

from os.path import join
import numpy as np
import matplotlib
matplotlib.rcParams['xtick.labelsize'] = 14
matplotlib.rcParams['ytick.labelsize'] = 14
import matplotlib.pyplot as plt
from cnn_stimuli import get_image_file_list
from alexnet import preprocess, load_net, load_vgg

# load IT neuron data from Kovacs figure 8 ...
kovacs = np.zeros((8,4))
NO = np.loadtxt(open('../data/NO.csv', 'rb'), delimiter=',')
kovacs[:,0] = NO[:,1]
MV20 = np.loadtxt(open('../data/MV20.csv', 'rb'), delimiter=',')
kovacs[:,1] = MV20[:,1]
MV50 = np.loadtxt(open('../data/MV50.csv', 'rb'), delimiter=',')
kovacs[:,2] = MV50[:,1]
MV90 = np.loadtxt(open('../data/MV90.csv', 'rb'), delimiter=',')
kovacs[:,3] = MV90[:,1]

def average_over_reps(responses, reps):
    n_occ = responses.shape[0]/reps
    averages = np.zeros((n_occ, responses.shape[1]))
    for i in range(n_occ):
        averages[i,:] = np.mean(responses[i*reps:(i+1)*reps,:], axis=0)
    return averages

def plot(data, plot_colour):
    assert data.shape[1] == 4
    plt.figure(figsize=(3.5,3.5))
    # max = data[0,0]
    shape_index = range(1,data.shape[0]+1)
    plt.plot(shape_index, data[:,0], plot_colour+'d-', label='NO')
    plt.plot(shape_index, data[:,1], plot_colour+'o--', label='20%')
    plt.plot(shape_index, data[:,2], plot_colour+'^-.', label='50%')
    plt.plot(shape_index, data[:,3], plot_colour+'s:', label='90%')
    plt.legend(loc='upper right', shadow=False, frameon=False)
    plt.xlabel('Shape rank', fontsize=16)
    plt.ylabel('Mean Response', fontsize=16)
    plt.tight_layout()

def get_mean_responses(use_vgg, remove_level):
    if use_vgg:
        model = load_vgg(weights_path='../weights/vgg16_weights.h5', remove_level=remove_level)
    else:
        model = load_net(weights_path='../weights/alexnet_weights.h5', remove_level=remove_level)

    reps = 10
    out = []
    shapes = ['circle', 'square', 'star', 'triangle', 'strange', 'h', 'arrow', 'stop']
    for shape in shapes:
        image_files = get_image_file_list(join(base_directory, shape), 'png', with_path=True)
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

    most_selective = out[:,:,ind]
    most_sorted = np.zeros_like(most_selective)

    for i in range(n):
        order = np.argsort(most_selective[:,0,i])[::-1]
        most_sorted[:,:,i] = most_selective[order,:,i]

    # out: shape, occlusion, neuron

    means = np.mean(most_sorted, axis=2)
    return means[:,:4]-means[:,4,None]


use_vgg = True

if use_vgg:
    plot_colour = 'g'
else:
    plot_colour = 'b'

# occlusion_type = 'black'
# occlusion_type = 'red'
occlusion_type = 'moving'
base_directory = './images/occlusions-' + occlusion_type

# # plot figure like Kovacs ...
# remove_level = 1
# means = get_mean_responses(use_vgg, remove_level)
# plot(means, plot_colour)
# if use_vgg:
#     plt.savefig('../figures/occlusions-' + occlusion_type + '-vgg-' + str(remove_level) + '.eps')
# else:
#     plt.savefig('../figures/occlusions-' + occlusion_type + '-alexnet-' + str(remove_level) + '.eps')
# plt.show()

# # replot Kovacs data ...
# plot(kovacs, 'k')
# plt.savefig('../figures/occlusions-kovacs.eps')
# plt.show()

# plot functions of occlusion level ...
visibility = [100, 80, 50, 10]
plt.figure(figsize=(3.5,3.5))

kovacs_relative = kovacs[0,:] - kovacs[-1,:]
plt.plot(visibility, kovacs_relative/kovacs_relative[0], 'k', label='IT')

remove_levels = [0, 1, 2]
labels = ['last layer', '2nd last', '3rd last']
line_formats = ['-', '--', '-.']
for i in range(3):
    means = get_mean_responses(use_vgg, remove_levels[i])
    relative = means[0,:] - means[-1,:]
    plt.plot(visibility, relative/relative[0], plot_colour+line_formats[i], label=labels[i])

plt.legend(loc='upper left', shadow=False, frameon=False)
plt.xlabel('Percent Visible', fontsize=16)
plt.ylabel('Mean Normalized Response', fontsize=16)
plt.xticks([20, 40, 60, 80, 100])
plt.ylim([-.1, 1.1])
plt.tight_layout()
network_name = 'vgg' if use_vgg else 'alexnet'
plt.savefig('../figures/occlusions-sigmoid-' + occlusion_type + '-' + network_name + '.eps')
plt.show()

