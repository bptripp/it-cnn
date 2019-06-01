__author__ = 'bptripp'

import cPickle as pickle
from scipy.optimize import curve_fit
import numpy as np
import matplotlib
matplotlib.rcParams['xtick.labelsize'] = 16
matplotlib.rcParams['ytick.labelsize'] = 16
import matplotlib.pyplot as plt
from cnn_stimuli import get_image_file_list
from alexnet import preprocess, load_net, load_vgg
from orientation import smooth

offsets = np.linspace(-75, 75, 150/5+1, dtype=int)


def get_centre_of_mass(y):
    """
    Centre of mass of a tuning function as in Op De Beeck et al. (2000), including
    only points >50% of peak.
    """
    ind = y > .5*np.max(y)
    masked = np.zeros(y.shape)
    masked[ind] = y[ind]
    return np.sum(offsets*masked) / np.sum(masked)


def get_width(y):
    """
    Width of a tuning function as in Op De Beeck et al. (2000); distance between where it falls
    to 50% of peak on each side.
    """
    max_ind = np.argmax(y)
    below_threshold = y < .5*np.max(y)
    # print(below_threshold)
    # low_and_left = np.logical_and(range(len(y)) < max_ind, below_threshold)
    # low_and_right = np.logical_and(range(len(y)) > max_ind, below_threshold)
    #
    # if max(low_and_left):
    #     low_ind = np.max(np.where(low_and_left))
    # else:
    #     low_ind = 0
    #
    # if max(low_and_right):
    #     high_ind = np.min(np.where(low_and_right))
    # else:
    #     high_ind = len(y)-1

    for i in range(max_ind, -1, -1):
        if below_threshold[i]:
            break
    if i == 0:
        low_ind = i
    else:
        low_ind = i + 1

    for i in range(max_ind, len(y)):
        if below_threshold[i]:
            break
    if i == len(y) - 1:
        high_ind = i
    else:
        high_ind = i - 1

    # print(y)
    # print('threshold: ' + str(.5*np.max(y)))
    # print(max_ind)
    # print(low_ind)
    # print(high_ind)
    # print('*******')
    #
    return offsets[high_ind] - offsets[low_ind]


if False:
    # plot selectivity vs. position tolerance
    model = load_net()
    image_files = get_image_file_list('./images/positions/banana', 'png', with_path=True)
    im = preprocess(image_files)
    out = model.predict(im)


    with open('activity-fraction.pkl', 'rb') as file:
        (ind, selectivity) = pickle.load(file)

    # n = 674
    n = len(ind)
    print(n)

    object_responses = out[:,ind]
    plt.plot(offsets, object_responses)
    plt.show()

    # maxima = np.max(out, axis=0)
    # ind = (-maxima).argsort()[:n]
    # smoothed = smooth(object_responses, ind)

    def get_sd(x, y):
        x = np.array(x, dtype=float)
        mean = sum(x*y)/n
        sigma = (sum(y*(x-mean)**2)/n)**.5
        return sigma


    widths = np.zeros(n)
    good_selectivity = []
    good_widths = []
    for i in range(n):
        widths[i] = get_sd(offsets, object_responses[:,i])
        if np.mean(object_responses[:,i]) > .001:
            good_selectivity.append(selectivity[i])
            good_widths.append(widths[i])
        # if widths[i] < 1:
        #     print(object_responses[:,i])

    with open('selectivity-vs-pos-tolerance.pkl', 'wb') as file:
        pickle.dump((good_selectivity, good_widths), file)

    print(np.corrcoef(good_selectivity, good_widths))
    plt.scatter(good_selectivity, good_widths)
    plt.show()


if True:
    # alexnet 0: mean width: 146.208333333 std centres: 3.49089969478
    # alexnet 1: mean width: 138.875 std centres: 5.96841285709
    # alexnet 2: mean width: 112.583333333 std centres: 23.4025005388

    # vgg 0: mean width: 150.0 std centres: 1.12932654355
    # vgg 1: mean width: 150.0 std centres: 1.422815326
    # vgg 2: mean width: 141.916666667 std centres: 11.2126510706

    data = np.loadtxt(open("../data/op-de-beeck-6.csv","rb"),delimiter=",")
    x = data[:,0]

    it_std_rf_centre = np.std(x)
    it_mean_rf_size = 10.3 # from Op De Beeck

    # # strongest 30 responses ...
    # alexnet_std_rf_centre = np.array([23.4025005388, 5.96841285709, 3.49089969478])
    # alexnet_mean_rf_size = np.array([112.583333333, 138.875, 146.208333333])
    # vgg_std_rf_centre = np.array([11.2126510706, 1.422815326, 1.12932654355])
    # vgg_mean_rf_size = np.array([141.916666667, 150.0, 150.0])

    # first 30 strong responses ...
    alexnet_std_rf_centre = np.array([30.2666393886, 14.4771892017, 5.46262378919])
    alexnet_mean_rf_size = np.array([62.5833333333, 108.625, 140.333333333])
    vgg_std_rf_centre = np.array([22.2973512678, 3.0048651618, 2.26856624582])
    vgg_mean_rf_size = np.array([97.5, 147.916666667, 149.625])


    layers = [-2, -1, 0]
    plt.figure(figsize=(5,3.5))
    plt.plot(layers, alexnet_std_rf_centre / alexnet_mean_rf_size, 'o-')
    plt.plot(layers, vgg_std_rf_centre / vgg_mean_rf_size, 's-')
    plt.plot(layers, it_std_rf_centre/it_mean_rf_size*np.array([1, 1, 1]), 'k--')
    plt.xlabel('Distance from output (layers)', fontsize=16)
    plt.ylabel('STD of centers / mean width', fontsize=16)
    plt.xticks([-2,-1,0])
    plt.tight_layout()
    plt.savefig('../figures/position-variability.eps')
    plt.show()


if False:
    # plot tuning curve examples

    remove_level = 0
    # model = load_net(weights_path='../weights/alexnet_weights.h5', remove_level=remove_level)
    # use_vgg = False
    model = load_vgg(weights_path='../weights/vgg16_weights.h5', remove_level=remove_level)
    use_vgg = True

    out = []
    image_files = get_image_file_list('./images/positions/staple', 'png', with_path=True)
    im = preprocess(image_files, use_vgg=use_vgg)
    out.append(model.predict(im))
    image_files = get_image_file_list('./images/positions/shoe', 'png', with_path=True)
    im = preprocess(image_files, use_vgg=use_vgg)
    out.append(model.predict(im))
    image_files = get_image_file_list('./images/positions/corolla', 'png', with_path=True)
    im = preprocess(image_files, use_vgg=use_vgg)
    out.append(model.predict(im))
    image_files = get_image_file_list('./images/positions/banana', 'png', with_path=True)
    im = preprocess(image_files, use_vgg=use_vgg)
    out.append(model.predict(im))
    out = np.array(out)
    # print(out.shape)

    # plot example tuning curves
    n = 30
    labels = ('staple', 'shoe', 'car', 'banana')
    plt.figure(figsize=(6,6))
    centres = []
    widths = []
    for i in range(4):
        object_responses = np.squeeze(out[i,:,:])
        # print(object_responses.shape)
        maxima = np.max(object_responses, axis=0)

        # ind = (-maxima).argsort()[:n]

        print('using first large n')
        ind = []
        j = 0
        while len(ind) < n:
            if maxima[j] > 2:
                ind.append(j)
            j = j + 1

        smoothed = smooth(object_responses, ind)

        for j in range(smoothed.shape[1]):
            centres.append(get_centre_of_mass(smoothed[:,j]))
            widths.append(get_width(smoothed[:,j]))

        # plt.plot(offsets, object_responses[:,ind])
        plt.subplot(2,2,i+1)

        if i >= 2:
            plt.xlabel('Offset (pixels)', fontsize=16)
        if i == 0 | i == 3:
            plt.ylabel('Response', fontsize=16)

        plt.title(labels[i], fontsize=16)

        plt.plot(offsets, smoothed)
        plt.xticks([-75,-25,25,75])
    plt.tight_layout()

    print(centres)
    print(widths)
    print('mean width: ' + str(np.mean(widths)) + ' std centres: ' + str(np.std(centres)))

    net = 'vgg16' if use_vgg else 'alexnet'
    plt.savefig('../figures/position-tuning-' + net + '-' + str(remove_level) + '.eps')

    plt.show()


def correlations(out):
    cc = np.corrcoef(out.T)
    result = []
    for i in range(cc.shape[0]):
        for j in range(i+1,cc.shape[1]):
            result.append(cc[i][j])
    return result


def invariant(out):
    return np.mean(correlations(out)) > .5


def clear_preference(out):
    # one size is clearly preferred in that it elicits a stronger response for each shape
    max_ind = np.argmax(out, axis=1)
    votes = np.zeros((5))
    for m in max_ind:
        votes[m] = votes[m] + 1

    return np.max(votes) >= 5
    # return np.max(np.abs(np.diff(max_ind))) == 0


if False:
    # plot Schwartz et al. (1983) example
    U = np.loadtxt(open('../data/U.csv', 'rb'), delimiter=',')
    L = np.loadtxt(open('../data/L.csv', 'rb'), delimiter=',')
    C = np.loadtxt(open('../data/C.csv', 'rb'), delimiter=',')
    I = np.loadtxt(open('../data/I.csv', 'rb'), delimiter=',')
    F = np.loadtxt(open('../data/F.csv', 'rb'), delimiter=',')
    o = np.zeros((6,5))
    o[:,0] = U[:,1]
    o[:,1] = L[:,1]
    o[:,2] = C[:,1]
    o[:,3] = I[:,1]
    o[:,4] = F[:,1]

    plt.figure(figsize=(4,3.5))
    plt.plot(range(1,7), o)
    plt.tight_layout()
    plt.xlabel('Stimulus #', fontsize=18)
    plt.ylabel('Response (spikes/s)', fontsize=18)
    plt.tight_layout()
    plt.savefig('../figures/position-schwartz-example.eps')
    plt.show()

    print('Schwartz correlation ' + str(np.mean(correlations(o))))


if False:
    # plot mean correlations and fraction with clear preference
    layers = [-2, -1, 0]

    plt.figure(figsize=(4,3.5))
    alexnet_correlations = [0.350468586188, 0.603050738337, 0.813774571373]
    vgg_correlations = [0.578857429221, 0.8000155323, 0.928289856194]
    schwartz_correlation = 0.791299618127
    plt.plot(layers, alexnet_correlations)
    plt.plot(layers, vgg_correlations)
    plt.plot(layers, schwartz_correlation*np.array([1, 1, 1]), 'k--')
    plt.xlabel('Distance from output (layers)', fontsize=16)
    plt.ylabel('Mean correlation', fontsize=16)
    plt.xticks([-2,-1,0])
    plt.ylim([0, 1])
    plt.tight_layout()
    plt.savefig('../figures/position-correlations.eps')
    plt.show()

    plt.figure(figsize=(4,3.5))
    alexnet_fractions = [0.21875, 0.125, 0.046875]
    vgg_fractions = [0.078125, 0.109375, 0.0]
    plt.plot(layers, alexnet_fractions)
    plt.plot(layers, vgg_fractions)
    plt.xlabel('Distance from output (layers)', fontsize=16)
    plt.ylabel('Fraction with preference', fontsize=16)
    plt.xticks([-2,-1,0])
    plt.ylim([0, 1])
    plt.tight_layout()
    plt.savefig('../figures/position-preferences.eps')
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
        image_files = get_image_file_list('./images/positions/'+stimulus, 'png', with_path=True)
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
    network_name = 'vgg' if use_vgg else 'alexnet'
    plt.savefig('../figures/position-invariance-schwartz-' + network_name + '-' + str(remove_level) + '.eps')
    plt.show()


