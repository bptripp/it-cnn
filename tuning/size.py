__author__ = 'bptripp'

import time
import numpy as np
import matplotlib
matplotlib.rcParams['xtick.labelsize'] = 18
matplotlib.rcParams['ytick.labelsize'] = 18
import matplotlib.pyplot as plt
from cnn_stimuli import get_image_file_list
from alexnet import preprocess, load_net, load_vgg

scales = np.logspace(np.log10(.05), np.log10(1.2), 45) #copied from cnn_stimuli (bad idea!)
model = load_net(weights_path='../weights/alexnet_weights.h5')
use_vgg = False
# model = load_vgg(weights_path='../weights/vgg16_weights.h5')
# use_vgg=True

# if use_vgg:
#     model = load_vgg(weights_path='../weights/vgg16_weights.h5', remove_level=remove_levels[i])
# else:
#     model = load_net(weights_path='../weights/alexnet_weights.h5', remove_level=remove_levels[i])

def correlations(out):
    cc = np.corrcoef(out.T)
    result = []
    for i in range(cc.shape[0]):
        for j in range(i+1,cc.shape[1]):
            result.append(cc[i][j])
    return result


def invariant(out):
    #dims: object, size
    # return np.mean([cc[0,1], cc[0,2], cc[1,2]]) > .7
    # return np.mean([cc[0,1], cc[0,2], cc[1,2]]) > .5
    return np.mean(correlations(out)) > .75


def clear_preference(out):
    # one size is clearly preferred in that it elicits a stronger response for each shape
    max_ind = np.argmax(out, axis=1)
    # print(max_ind)
    return np.max(np.abs(np.diff(max_ind))) == 0


def bandwidth(scales, responses):
    peak_ind = np.argmax(responses)
    peak = responses[peak_ind]

    top = peak_ind
    for i in range(peak_ind, len(responses)):
        if responses[i] > peak/2:
            top = i
        else:
            break

    bottom = peak_ind
    for i in range(peak_ind, -1, -1):
        if responses[i] > peak/2:
            bottom = i
        else:
            break

    result = np.log2(scales[top] / scales[bottom])
    # print('bandwidth')
    # print(peak_ind)
    # print(bottom)
    # print(top)
    # print(result)
    return result

if True:
    # plot Schwartz et al. (1983) example
    deg13 = np.loadtxt(open('../data/deg13.csv', 'rb'), delimiter=',')
    deg28 = np.loadtxt(open('../data/deg28.csv', 'rb'), delimiter=',')
    deg50 = np.loadtxt(open('../data/deg50.csv', 'rb'), delimiter=',')
    o = np.zeros((6,3))
    o[:,0] = deg13[:,1]
    o[:,1] = deg28[:,1]
    o[:,2] = deg50[:,1]

    plt.figure(figsize=(4,3.5))
    plt.plot(range(1,7), o)
    plt.tight_layout()
    plt.xlabel('Stimulus #', fontsize=18)
    plt.ylabel('Response (spikes/s)', fontsize=18)
    plt.tight_layout()
    plt.savefig('../figures/size-schwartz-example.eps')
    plt.show()

    # plt.semilogx([2, 4, 8, 16, 32, 64], o)

    print('Schwartz correlation ' + str(np.mean(correlations(o))))


if False:
    # plot mean correlations and fraction with clear preference
    layers = [-2, -1, 0]

    plt.figure(figsize=(4,3.5))
    alexnet_correlations = [0.495586845637, 0.530755560076, 0.721706983642]
    vgg_correlations = [0.444102093071, 0.524417339524, 0.613046758844]
    schwartz_correlation = 0.958892832321
    plt.plot(layers, alexnet_correlations)
    plt.plot(layers, vgg_correlations)
    plt.plot(layers, schwartz_correlation*np.array([1, 1, 1]), 'k--')
    plt.xlabel('Distance from output (layers)', fontsize=16)
    plt.ylabel('Mean correlation', fontsize=16)
    plt.xticks([-2,-1,0])
    plt.ylim([0, 1])
    plt.tight_layout()
    plt.savefig('../figures/size-correlations.eps')
    plt.show()

    plt.figure(figsize=(4,3.5))
    alexnet_fractions = [0.03125, 0.046875, 0.03125]
    vgg_fractions = [0.0625, 0.046875, 0.015625]
    plt.plot(layers, alexnet_fractions)
    plt.plot(layers, vgg_fractions)
    plt.xlabel('Distance from output (layers)', fontsize=16)
    plt.ylabel('Fraction with preference', fontsize=16)
    plt.xticks([-2,-1,0])
    plt.ylim([0, 1])
    plt.tight_layout()
    plt.savefig('../figures/size-preferences.eps')
    plt.show()


if False:
    use_vgg = True
    remove_level = 0
    if use_vgg:
        model = load_vgg(weights_path='../weights/vgg16_weights.h5', remove_level=remove_level)
    else:
        model = load_net(weights_path='../weights/alexnet_weights.h5', remove_level=remove_level)

    out = []
    stimuli = ['f1', 'f2', 'f3', 'f4', 'f5', 'f6']
    for stimulus in stimuli:
        image_files = get_image_file_list('./images/scales/' + stimulus, 'png', with_path=True)
        im = preprocess(image_files, use_vgg=use_vgg)
        out.append(model.predict(im))
    out = np.array(out)
    print(out.shape)

    # plot invariance with Schwartz stimuli
    i = 0
    c = 0
    n_invariant = 0
    n_clear = 0
    mean_correlations = []
    while c < 64:
        plt.subplot(8,8,c+1)
        o = out[:,:,i]
        # if np.max(o) > .1:
        if np.max(o) > 1:
            plt.plot(o)
            yl = plt.gca().get_ylim()

            if invariant(o):
                n_invariant = n_invariant + 1
                plt.text(4.3, yl[0] + (yl[1]-yl[0])*.8, 'c', fontsize=14)

            if clear_preference(o):
                n_clear = n_clear + 1
                plt.text(.1, yl[0] + (yl[1]-yl[0])*.8, 'p', fontsize=14)

            mean_correlations.append(np.mean(correlations(o)))

            plt.xticks([])
            plt.yticks([])
            c = c + 1
        i = i + 1

    print(mean_correlations)
    print('fraction with preference ' + str(float(n_clear)/64.))
    print('mean correlation ' + str(np.nanmean(mean_correlations)))

    plt.tight_layout()
    net = 'vgg16' if use_vgg else 'alexnet'
    plt.savefig('../figures/size-invariance-' + net + '-' + str(remove_level) + '.eps')
    plt.show()


if False: #plot invariance with naturalistic stimuli
    size_ind = (25,30,34) # roughly matched with Schwartz et al. (see cnn_stimuli)
    i = 0
    c = 0
    n_invariant = 0
    # for i in range(9):
    while c < 64:
        plt.subplot(8,8,c+1)
        o = out[:,size_ind,i]
        if np.max(o) > .1:
            plt.plot(o)
            # print(out[:,size_ind,i])
            # print(invariant(o))
            if invariant(o):
                n_invariant = n_invariant + 1
                plt.text(1.8, np.max(o)*.7, '*', fontsize=16)

            plt.xticks([])
            plt.yticks([])
            # plt.title(str(i) + ' ' + str(invariant(o)))
            c = c + 1
        i = i + 1
    print(n_invariant)
    plt.tight_layout()
    plt.savefig('../figures/size-invariance.eps')
    plt.show()

if False: # plot example tuning curves
    # image_files = get_image_file_list('./images/scales/banana', 'png', with_path=True)
    # image_files = get_image_file_list('./images/scales/shoe', 'png', with_path=True)
    image_files = get_image_file_list('./images/scales/corolla', 'png', with_path=True)
    im = preprocess(image_files, use_vgg=use_vgg)
    object_responses = model.predict(im)

    # object_responses = np.squeeze(out[0,:,:])
    maxima = np.max(object_responses, axis=0)
    n = 50
    ind = (-maxima).argsort()[:n]
    plt.plot(scales, object_responses[:,ind])

    prefs = np.zeros(n)
    for i in range(n):
        bandwidth(scales, object_responses[:,ind[i]])
        prefs[i] = scales[np.argmax(object_responses[:,ind[i]])]
    plt.show()

    plt.hist(prefs)
    plt.show()


def get_max_indices(object_responses, n=500):
    maxima = np.max(object_responses, axis=0)
    return (-maxima).argsort()[:n]


def get_bandwidth(object_responses, ind):
    n = len(ind)
    bandwidths = np.zeros(n)
    prefs = np.zeros(n)
    for i in range(n):
        size_tuning_curve = object_responses[:,ind[i]]
        bandwidths[i] = bandwidth(scales, size_tuning_curve)
        prefs[i] = scales[np.argmax(size_tuning_curve)]
    return bandwidths, prefs

if False:
    data = np.loadtxt(open('../data/ito.csv', 'rb'), delimiter=',')
    bandwidths = np.squeeze(data[:,1])
    bandwidths[bandwidths>3.95] = 4.01 #these are on the line
    print(bandwidths)

    fs = 18
    hist_edges = [x * .5 for x in range(13)]
    plt.figure(figsize=(3,3))
    plt.hist(bandwidths[bandwidths<4], hist_edges)
    plt.hist(bandwidths[bandwidths>=4], hist_edges, color=[1,0,0])
    plt.xlabel('Size BW (Octaves)', fontsize=fs)
    plt.ylabel('Frequency', fontsize=fs)
    plt.tight_layout()
    plt.savefig('../figures/bandwidth-ito.eps')
    plt.show()


if False: # plot histograms of bandwidth and preferred size
    out = []
    image_files = get_image_file_list('./images/scales/banana', 'png', with_path=True)
    im = preprocess(image_files, use_vgg=use_vgg)
    out.append(model.predict(im))
    image_files = get_image_file_list('./images/scales/shoe', 'png', with_path=True)
    im = preprocess(image_files, use_vgg=use_vgg)
    out.append(model.predict(im))
    image_files = get_image_file_list('./images/scales/corolla', 'png', with_path=True)
    im = preprocess(image_files, use_vgg=use_vgg)
    out.append(model.predict(im))
    out = np.array(out)

    fs = 18
    hist_edges = [x * .5 for x in range(13)]

    plt.figure(figsize=(3,3))
    n = 200
    bandwidths = np.zeros((3*n))
    for i in range(3):
        # plt.subplot(1,3,i+1)
        object_responses = np.squeeze(out[i,:,:])
        ind = get_max_indices(object_responses, n=200)
        bandwidths[n*i:n*(i+1)], prefs = get_bandwidth(object_responses, ind)
    plt.hist(bandwidths, hist_edges)
    plt.xlabel('Size BW (Octaves)', fontsize=fs)
    plt.ylabel('Frequency', fontsize=fs)


    plt.tight_layout()
    if use_vgg:
        plt.savefig('../figures/bandwidth-vgg16.eps')
    else:
        plt.savefig('../figures/bandwidth-alexnet.eps')

    plt.show()


    # print('preferences mean: ' + str(np.mean(prefs)))
    # print('preferences min: ' + str(np.min(prefs)))
    # print('preferences max: ' + str(np.max(prefs)))
    # print('bandwidth mean: ' + str(np.mean(bandwidths)))
    # print('bandwidth max: ' + str(np.max(bandwidths)))

    # plt.figure(figsize=(10,3))
    # plt.subplot(1,2,1)
    # plt.hist(prefs, 8)
    # plt.xlabel('Preferred Scale', fontsize=fs)
    # plt.ylabel('Frequency', fontsize=fs)
    # plt.subplot(1,2,2)
    # plt.hist(bandwidths, 10)
    # plt.xlabel('Size BW (Octaves)', fontsize=fs)
    # plt.ylabel('Frequency', fontsize=fs)
    # plt.tight_layout()
    # plt.savefig('../figures/bandwidth.eps')
    # plt.show()


