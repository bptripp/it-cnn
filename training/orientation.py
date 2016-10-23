__author__ = 'bptripp'

from os import listdir
from os.path import isfile, join, isdir, basename, split
import cPickle as pickle
import numpy as np
import matplotlib.pyplot as plt
import time
from salman.alexnet import preprocess, load_net
from bertsekas.auction import auction

"""
TODO:
- DONE load minibatch images
- DONE run network to get existing tuning
- DONE generate orientation tuning samples
- DONE choose Bertsekas integer resolution
- dicard all but ~200 output units
- normalize and use Bertsekas to pair up with empirical model
- scale empirical responses and make labels
- generator
- LATER optionally resample orientation tuning curves each minibatch to reduce sampling bias (although there are many samples)
- LATER recharacterize all tuning
- DONE load existing network & strip last layer
- DONE code to generate and save minibatch list
- DONE organize files better to reduce looping to find orientations
"""


def make_ideal_tuning_curves(angles, n):
    # from Salman's paper

    curves = np.zeros((len(angles), n))
    prefs = 360.*np.random.rand(n)
    i = 0
    widths = np.zeros(n)
    while i < n:
        width = 30. + 15. * np.random.randn()
        if width > 1:
            widths[i] = width
            i = i + 1

    for i in range(n):
        da = angles - prefs[i]
        da[da > 180] = da[da > 180] - 360
        da[da < -180] = da[da < -180] + 360
        curves[:,i] = np.exp(-(da)**2 / 2 / widths[i]**2)

    # plt.plot(angles, curves)
    # plt.title('Empirical')
    # plt.ylim([0,1])
    # plt.show()

    return curves


def find_stimuli(image_path, extension):
    stimuli = set()

    for d in listdir(image_path):
        if isdir(join(image_path, d)):
            for f in listdir(join(image_path, d)):
                if isfile(join(image_path, d, f)) and f.endswith(extension):
                    parts = f.split('-')
                    stim_name = parts[0] + '-' + parts[1]
                    stimuli.add(join(image_path, d, stim_name))

    return list(stimuli)


def get_images(stimulus, extension):
    return preprocess(get_image_paths(stimulus, extension))


def get_image_paths(stimulus, extension):
    directory, partial_name = split(stimulus)
    return [join(directory, f) for f in listdir(directory) if f.startswith(partial_name) and f.endswith(extension)]


def get_similarities(actual_curves, ideal_curves):
    actual_normalized = normalize_curves(actual_curves)
    ideal_normalized = normalize_curves(ideal_curves)
    return np.dot(actual_normalized.T, ideal_normalized)


def normalize_curves(tuning_curves):
    #one tuning curve per column
    nm = np.linalg.norm(tuning_curves, axis=0, ord=2)
    return tuning_curves / nm[None,:]


if __name__ == '__main__':
    angles = range(0, 361, 10)

    model = load_net(weights_path='../salman/weights/alexnet_weights.h5')

    image_path = '/Users/bptripp/data/orientations/'
    extension = '.png'
    stimuli = find_stimuli(image_path, extension)


    start_time = time.time()
    images = get_images(stimuli[0], extension)
    print('image time: ' + str(time.time() - start_time))
    actual_curves = model.predict(images)
    print('model time: ' + str(time.time() - start_time))

    n = 200
    # n = actual_curves.shape[1]
    ideal_curves = make_ideal_tuning_curves(angles, n)

    # print(ideal_curves.shape)
    # print(actual_curves.shape)

    similarities = get_similarities(actual_curves[:,:n], ideal_curves)
    # print(actual_curves[:,:n])
    # print(ideal_curves)
    A = ((1+similarities)*50).astype(int)
    print(A)



    # A = np.array([[5, 9, 2], [10, 3, 2], [8, 7, 4]])
    # print(A)
    start_time = time.time()
    assignments, prices = auction(A)
    print(assignments)
    print('auction time: ' + str(time.time() - start_time))
