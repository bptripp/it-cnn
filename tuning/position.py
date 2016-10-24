__author__ = 'bptripp'

import cPickle as pickle
from scipy.optimize import curve_fit
import numpy as np
import matplotlib
matplotlib.rcParams['xtick.labelsize'] = 16
matplotlib.rcParams['ytick.labelsize'] = 16
import matplotlib.pyplot as plt
from cnn_stimuli import get_image_file_list
from alexnet import preprocess, load_net
from orientation import smooth

offsets = np.linspace(-75, 75, 150/5+1, dtype=int)

model = load_net()


def get_centre_of_mass(x, y):
    pass


if False:
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
    out = []
    image_files = get_image_file_list('./images/positions/banana', 'png', with_path=True)
    im = preprocess(image_files)
    out.append(model.predict(im))
    image_files = get_image_file_list('./images/positions/shoe', 'png', with_path=True)
    im = preprocess(image_files)
    out.append(model.predict(im))
    image_files = get_image_file_list('./images/positions/corolla', 'png', with_path=True)
    im = preprocess(image_files)
    out.append(model.predict(im))
    out = np.array(out)
    print(out.shape)

    # plot example tuning curves
    n = 30
    labels = ('banana', 'shoe', 'car')
    plt.figure(figsize=(10,3))
    for i in range(3):
        object_responses = np.squeeze(out[i,:,:])
        print(object_responses.shape)
        maxima = np.max(object_responses, axis=0)
        ind = (-maxima).argsort()[:n]
        smoothed = smooth(object_responses, ind)
        # plt.plot(offsets, object_responses[:,ind])
        plt.subplot(1,3,i+1)

        plt.xlabel('Offset (pixels)', fontsize=16)
        if i == 0:
            plt.ylabel('Response', fontsize=16)
        plt.title(labels[i], fontsize=16)

        plt.plot(offsets, smoothed)
        plt.xticks([-75,-50,-25,0,25,50,75])
    plt.tight_layout()
    plt.savefig('../figures/position-tuning.eps')
    plt.show()


if True:
    out = []
    image_files = get_image_file_list('./images/positions/f1', 'png', with_path=True)
    im = preprocess(image_files)
    out.append(model.predict(im))
    image_files = get_image_file_list('./images/positions/f2', 'png', with_path=True)
    im = preprocess(image_files)
    out.append(model.predict(im))
    image_files = get_image_file_list('./images/positions/f3', 'png', with_path=True)
    im = preprocess(image_files)
    out.append(model.predict(im))
    image_files = get_image_file_list('./images/positions/f4', 'png', with_path=True)
    im = preprocess(image_files)
    out.append(model.predict(im))
    image_files = get_image_file_list('./images/positions/f5', 'png', with_path=True)
    im = preprocess(image_files)
    out.append(model.predict(im))
    image_files = get_image_file_list('./images/positions/f6', 'png', with_path=True)
    im = preprocess(image_files)
    out.append(model.predict(im))
    out = np.array(out)
    print(out.shape)


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
    return np.max(np.abs(np.diff(max_ind))) == 0


if True: # plot invariance with Schwartz stimuli
    i = 0
    c = 0
    n_invariant = 0
    n_clear = 0
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

            plt.xticks([])
            plt.yticks([])
            c = c + 1
        i = i + 1
    print(n_invariant)
    plt.tight_layout()
    plt.savefig('../figures/position-invariance-schwartz.eps')
    plt.show()


