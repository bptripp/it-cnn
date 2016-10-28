__author__ = 'bptripp'

import argparse
import numpy as np
import cPickle as pickle
import matplotlib.pyplot as plt
from alexnet import preprocess, load_net
from orientation import find_stimuli, get_images

parser = argparse.ArgumentParser()
parser.add_argument('action', help='either save (evaluate and save tuning curves) or plot (plot examples)')
parser.add_argument('valid_image_path', help='path to stimulus images for validation')
valid_image_path = parser.parse_args().valid_image_path
action = parser.parse_args().action

curves_file = 'orientation_curves.pkl'
n = 200

if action == 'save':
    model = load_net()
    model.load_weights('orientation_weights.h5')

    extension = '.png'
    valid_stimuli = find_stimuli(valid_image_path, extension)

    curves = []
    for stimulus in valid_stimuli:
        print(stimulus)
        images = get_images(stimulus, extension)
        curves.append(model.predict(images)[:,:n])
    curves = np.array(curves, dtype=np.float16)
    print(curves.shape)

    f = open(curves_file, 'wb')
    pickle.dump((valid_stimuli, curves), f)
    f.close()

if action == 'plot':
    f = open(curves_file, 'rb')
    valid_stimuli, curves = pickle.load(f)
    f.close()

    plt.figure(figsize=(8,8))

    for i in range(8):
        for j in range(8):
            ind = 8*i+j
            plt.subplot(8,8,ind+1)
            plt.plot(curves[0,:,ind])

    plt.show()
