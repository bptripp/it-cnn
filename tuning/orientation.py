__author__ = 'bptripp'

# TODO: artefact every 11 degrees except also 290,291,295

import time
import numpy as np
import matplotlib.pyplot as plt
from cnn_stimuli import get_image_file_list
from alexnet import preprocess, load_net
from scipy.signal import fftconvolve


def mean_corr(out):
    cc = np.corrcoef(out.T)
    n = cc.shape[0]
    print('n: ' + str(n))
    print(out.shape)
    cc_list = []
    for i in range(n):
        for j in range(i+1,n):
            cc_list.append(cc[i,j])
    print(cc_list)
    mean_cc = np.mean(cc_list)
    print('mean corr: ' + str(mean_cc))


def smooth(out, ind, n=5):
    l = out.shape[0]
    wrapped = np.zeros((out.shape[0]*3, len(ind)))
    wrapped[0:l,:] = out[:,ind]
    wrapped[l:2*l,:] = out[:,ind]
    wrapped[2*l:3*l,:] = out[:,ind]

    filter = 1./n*np.ones(n)
    print('filter: ' + str(filter))
    # filter = [.25, .25, .25]
    for i in range(wrapped.shape[1]):
        # smooth[:,i] = np.convolve(smooth[:,i], filter, 'same')
        wrapped[:,i] = fftconvolve(wrapped[:,i], filter, 'same')

    return wrapped[l:2*l,:]


def plot_curves(out, n, kernel_len=5, angles=None, normalize=False):
    print('plotting')
    if angles is None:
        angles = np.linspace(0, 360, 91)

    maxima = np.max(out, axis=0)
    # n = 10
    ind = (-maxima).argsort()[:n]

    smoothed = smooth(out, ind, n=kernel_len)
    # if smooth:
    #     smoothed = smooth(out, ind)
    # else:
    #     smoothed = out[:,ind]

    mean_corr(smoothed)
    # print(len(angles))
    # print(smoothed.shape)

    if normalize:
        smoothed = smoothed / np.max(smoothed, axis=0)

    plt.plot(angles, smoothed)
    plt.xlabel('Angle (degrees)')
    plt.xlim([0, 360])


if __name__ == '__main__':
    model = load_net()

    # plt.figure(figsize=(10,3))
    # image_files = get_image_file_list('/Users/bptripp/code/salman-IT/salman/images/banana-rotations/', 'png', with_path=True)
    # im = preprocess(image_files)
    # out = model.predict(im)
    # plt.subplot(1,3,1)
    # plot_curves(out, 10)
    # plt.ylabel('Response')
    # image_files = get_image_file_list('/Users/bptripp/code/salman-IT/salman/images/shoe-rotations/', 'png', with_path=True)
    # im = preprocess(image_files)
    # out = model.predict(im)
    # plt.subplot(1,3,2)
    # plot_curves(out, 10)
    # image_files = get_image_file_list('/Users/bptripp/code/salman-IT/salman/images/corolla-rotations/', 'png', with_path=True)
    # im = preprocess(image_files)
    # out = model.predict(im)
    # plt.subplot(1,3,3)
    # plot_curves(out, 10)
    # plt.tight_layout()
    # plt.savefig('orientation.eps')
    # plt.show()

    # plt.figure(figsize=(7,3))
    # image_files = get_image_file_list('/Users/bptripp/code/salman-IT/salman/images/staple-cropped/', 'jpg', with_path=True)
    # im = preprocess(image_files)
    # out = model.predict(im)
    # plt.subplot(1,2,1)
    # plot_curves(out, 10, kernel_len=3, angles=range(0,361,10))
    # plt.ylabel('Response')
    # plt.title('Staple')
    # image_files = get_image_file_list('/Users/bptripp/code/salman-IT/salman/images/scooter-cropped/', 'jpg', with_path=True)
    # im = preprocess(image_files)
    # out = model.predict(im)
    # plt.subplot(1,2,2)
    # plot_curves(out, 10, kernel_len=3, angles=range(0,361,15))
    # plt.title('Scooter')
    # plt.tight_layout()
    # plt.savefig('orientation3d.eps')
    # plt.show()

    plt.figure(figsize=(7,3))
    n = 15
    kernel_len = 3
    angles = range(0,361,10)
    image_files = get_image_file_list('/Users/bptripp/code/salman-IT/salman/images/staple-cropped/', 'jpg', with_path=True)
    im = preprocess(image_files)
    out = model.predict(im)
    plt.subplot(1,2,1)
    plot_curves(out, n, kernel_len=kernel_len, angles=angles, normalize=True)
    plt.ylabel('Response')
    plt.ylim([0,1])
    plt.title('CNN')

    fake = np.zeros((out.shape[0], n))
    np.random.seed(1)
    prefs = 360.*np.random.rand(n)
    i = 0
    widths = np.zeros(n)
    while i < n:
        width = 30. + 15. * np.random.randn()
        if width > 1:
            widths[i] = width
            i = i + 1
    print(widths)

    for i in range(n):
        fake[:,i] = np.exp(-(angles-prefs[i])**2 / 2 / widths[i]**2)


    # image_files = get_image_file_list('/Users/bptripp/code/salman-IT/salman/images/scooter-cropped/', 'jpg', with_path=True)
    # im = preprocess(image_files)
    # out = model.predict(im)
    plt.subplot(1,2,2)
    plot_curves(fake, n, kernel_len=kernel_len, angles=angles, normalize=True)
    plt.title('Empirical')
    plt.ylim([0,1])

    plt.tight_layout()
    plt.savefig('orientation-salman.eps')
    plt.show()
