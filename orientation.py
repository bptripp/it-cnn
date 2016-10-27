__author__ = 'bptripp'

import argparse
from os import listdir, rename
from os.path import isfile, join, isdir, basename, split
import cPickle as pickle
import numpy as np
import matplotlib.pyplot as plt
import time
from alexnet import preprocess, load_net
from auction import auction

"""
TODO:
- DONE load minibatch images
- DONE run network to get existing tuning
- DONE generate orientation tuning samples
- DONE choose Bertsekas integer resolution
- DONE dicard all but ~200 output units
- DONE normalize and use Bertsekas to pair up with empirical model
- DONE scale empirical responses and make labels
- DONE generator
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
    nm = np.linalg.norm(tuning_curves, axis=0, ord=2) + 1e-6
    return tuning_curves / nm[None,:]


def get_XY(model, stimulus):
    images = get_images(stimulus, extension)
    actual_curves = model.predict(images)

    n = 200
    ideal_curves = make_ideal_tuning_curves(angles, n)

    similarities = get_similarities(actual_curves[:,:n], ideal_curves)
    A = ((1+similarities)*50).astype(int)
    assignments, prices = auction(A)

    target_curves = get_targets(actual_curves, ideal_curves, assignments)

    return images, target_curves


def get_targets(actual_curves, ideal_curves, assignments):
    n = ideal_curves.shape[1]
    target_curves = actual_curves.copy()
    for i in range(n):
        target_curves[:,i] = ideal_curves[:,int(assignments[i])]
        scale_factor = np.abs(np.mean(actual_curves[:,i]) / np.mean(target_curves[:,i]))
        target_curves[:,i] = target_curves[:,i] * scale_factor
    return target_curves


def fix_file_names(image_path):
    for file_name in listdir(image_path):
        if isdir(join(image_path, file_name)):
            new_name = file_name.replace('-', '_')
            if not new_name == file_name:
                print('renaming ' + join(image_path, file_name) + ' to ' + join(image_path, new_name))
                rename(join(image_path, file_name), join(image_path, new_name))
                for image_file_name in listdir(join(image_path, new_name)):
                    image_new_name = image_file_name.replace(file_name, new_name)
                    print('renaming ' + join(image_path, new_name, image_file_name)
                          + ' to ' + join(image_path, new_name, image_new_name))
                    rename(join(image_path, new_name, image_file_name),
                           join(image_path, new_name, image_new_name))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('train_image_path', help='path to stimulus images for training')
    parser.add_argument('valid_image_path', help='path to stimulus images for validation')
    train_image_path = parser.parse_args().train_image_path
    valid_image_path = parser.parse_args().valid_image_path

    fix_file_names(valid_image_path)
    fix_file_names(train_image_path)

    extension = '.png'
    train_stimuli = find_stimuli(train_image_path, extension)
    valid_stimuli = find_stimuli(valid_image_path, extension)
    print(len(listdir(train_image_path)))
    print(len(listdir(valid_image_path)))
    print('Processing ' + str(len(train_stimuli)) + ' training inputs from ' + train_image_path)
    print('Processing ' + str(len(valid_stimuli)) + ' validation inputs from ' + valid_image_path)

    # for fn in listdir(train_image_path):
    #     if isdir(join(train_image_path, fn)):
    #         l = len(listdir(join(train_image_path, fn)))
    #         print(str(l) + ' in ' + fn)


    angles = range(0, 361, 10)

    model = load_net()

    # plt.figure(figsize=(8,8))
    #
    # for i in range(8):
    #     for j in range(8):
    #         ind = 8*i+j
    #         plt.subplot(8,8,ind+1)
    #         plt.plot(actual_curves[:,ind])
    #         # plt.plot(ideal_curves[:,assignments[ind]])
    #         plt.plot(target_curves[:,ind])
    #
    # plt.show()

    def generate_training():
        while 1:
            ind = np.random.randint(0, len(train_stimuli))
            X, Y = get_XY(model, train_stimuli[ind])
            yield X, Y

    def generate_validation():
        while 1:
            ind = np.random.randint(0, len(valid_stimuli))
            X, Y = get_XY(model, valid_stimuli[ind])
            yield X, Y

    h = model.fit_generator(generate_training(),
        samples_per_epoch=len(train_stimuli)*len(angles), nb_epoch=50,
        validation_data=generate_validation(), nb_val_samples=len(valid_stimuli)*len(angles))

    model.save_weights('orientation_weights.h5')

    f = open('orientation_history.pkl', 'wb')
    pickle.dump(h, f)
    f.close()
