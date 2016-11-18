__author__ = 'bptripp'

import csv
import time
import numpy as np
import matplotlib.pyplot as plt
from cnn_stimuli import get_image_file_list
from alexnet import preprocess, load_net, load_vgg
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
    for i in range(wrapped.shape[1]):
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
    plt.xlim([np.min(angles), np.max(angles)])


def freiwald_depths(out):
    """
    From Freiwald & Tsao (2010) Science, supplementary material, page 6 ...

    Head orientation tuning depth was computed using the mean
    response to frontal faces (Rfrontal), and the mean response to full profile faces in the preferred
    direction (Rprofile) as follows:

    Tuning Depth = (Rfrontal - Rprofile) / (Rfrontal + Rprofile)
    """
    assert(out.shape[0] == 25)

    frontal = np.maximum(0, out[0,:])
    profile = np.maximum(0, np.maximum(out[6,:], out[19,:]))

    m = np.max(out, axis=0)

    frontal = frontal[m >= 1]
    profile = profile[m >= 1]

    return (frontal - profile) / (frontal + profile + 1e-3)


def plot_freiwald_histograms():
    fractions = []
    fractions.append(np.loadtxt(open("../data/freiwald-4h-am.csv","rb"),delimiter=","))
    fractions.append(np.loadtxt(open("../data/freiwald-4h-al.csv","rb"),delimiter=","))
    fractions.append(np.loadtxt(open("../data/freiwald-4h-mlmf.csv","rb"),delimiter=","))

    plt.figure(figsize=(3,2*3))
    for i in range(3):
        plt.subplot(3,1,i+1)

        f = fractions[i]
        hist_edges = np.linspace(-1, 1, len(f)+1)
        # hist_edges = hist_edges[:-1]
        print(f.shape)
        print(hist_edges.shape)

        plt.bar(hist_edges[:-1]+.02, f, width=hist_edges[1]-hist_edges[0]-.04, color=[.5,.5,.5])
        plt.xlim([-1, 1])
        plt.ylim([0, .55])
        plt.ylabel('Fraction of cells')

    plt.xlabel('Head orientation tuning depth')
    plt.tight_layout()
    plt.savefig('../figures/orientation-freiwald.eps')
    plt.show()


def plot_logothetis_and_freiwald_tuning_data():
    plt.figure(figsize=(9,2.5))

    plt.subplot(1,3,1)
    plot_csv_tuning_curve('../data/logothetis-a.csv')
    plot_csv_tuning_curve('../data/logothetis-b.csv')
    plot_csv_tuning_curve('../data/logothetis-c.csv')
    plot_csv_tuning_curve('../data/logothetis-d.csv')
    plot_csv_tuning_curve('../data/logothetis-e.csv')
    plt.ylabel('Response')
    plt.xlabel('Angle (degrees)')
    plt.xlim([-180, 180])
    plt.xticks([-180, -60, 60, 180])

    plt.subplot(1,3,3)
    plot_csv_tuning_curve('../data/freiwald-1.csv')
    plot_csv_tuning_curve('../data/freiwald-2.csv')
    plot_csv_tuning_curve('../data/freiwald-3.csv')
    plot_csv_tuning_curve('../data/freiwald-4.csv')
    plt.xlabel('Angle (degrees)')
    plt.xlim([-180, 180])
    plt.xticks([-180, -60, 60, 180])

    plt.tight_layout()
    plt.savefig('../figures/tuning-data.eps')
    plt.show()


def plot_csv_tuning_curve(filename):
    data = np.loadtxt(open(filename, 'rb'), delimiter=',')
    ind = np.argsort(data[:,0])
    plt.plot(data[ind,0], data[ind,1])


if __name__ == '__main__':
    # model = load_vgg(weights_path='../weights/vgg16_weights.h5', remove_level=2)
    # use_vgg = True
    model = load_net(weights_path='../weights/alexnet_weights.h5', remove_level=1)
    use_vgg = False

    plt.figure(figsize=(6,6))
    # image_files = get_image_file_list('./images/swiss-knife-rotations/', 'png', with_path=True)
    image_files = get_image_file_list('./images/staple-rotations/', 'png', with_path=True)
    im = preprocess(image_files, use_vgg=use_vgg)
    out = model.predict(im)
    # plt.subplot(2,2,1)
    plt.subplot2grid((5,2), (0,0), rowspan=2)
    plot_curves(out, 10)
    # plt.ylim([0,12])
    plt.ylabel('Response')
    image_files = get_image_file_list('./images/shoe-rotations/', 'png', with_path=True)
    im = preprocess(image_files, use_vgg=use_vgg)
    out = model.predict(im)
    # plt.subplot(2,2,2)
    plt.subplot2grid((5,2), (0,1), rowspan=2)
    plot_curves(out, 10)
    # plt.ylim([0,12])
    image_files = get_image_file_list('./images/corolla-rotations/', 'png', with_path=True)
    im = preprocess(image_files, use_vgg=use_vgg)
    out = model.predict(im)
    # plt.subplot(2,2,3)
    plt.subplot2grid((5,2), (3,0), rowspan=2)
    plot_curves(out, 10)
    # plt.ylim([0,12])
    plt.xlabel('Angle (degrees)')
    plt.ylabel('Response')
    image_files = get_image_file_list('./images/banana-rotations/', 'png', with_path=True)
    im = preprocess(image_files, use_vgg=use_vgg)
    out = model.predict(im)
    # plt.subplot(2,2,4)
    plt.subplot2grid((5,2), (3,1), rowspan=2)
    plot_curves(out, 10)
    # plt.ylim([0,12])
    plt.xlabel('Angle (degrees)')
    plt.tight_layout()
    plt.savefig('../figures/orientation.eps')
    plt.show()

    # plot_freiwald_histograms()
    # # plot_logothetis_and_freiwald_tuning_data()
    #
    # remove_levels = [0,1,2]
    # use_vgg = True
    #
    # # remove_levels = [0,1,2]
    # # use_vgg = False
    #
    # plt.figure(figsize=(9,2*len(remove_levels)))
    #
    # hist_edges = np.linspace(-1, 1, 10)
    # freiwalds = []
    #
    # for i in range(len(remove_levels)):
    #     if use_vgg:
    #         model = load_vgg(weights_path='../weights/vgg16_weights.h5', remove_level=remove_levels[i])
    #     else:
    #         model = load_net(weights_path='../weights/alexnet_weights.h5', remove_level=remove_levels[i])
    #
    #     plt.subplot(len(remove_levels),3,(3*i)+1)
    #     image_files = get_image_file_list('./source-images/staple/', 'jpg', with_path=True)
    #     im = preprocess(image_files, use_vgg=use_vgg)
    #     out = model.predict(im)
    #     # angles = np.array(range(0,361,10))
    #     wrap_indices = range(18, 36) + range(19)
    #     plot_curves(out[wrap_indices,:], 10, kernel_len=3, angles=range(-180,181,10))
    #     plt.xticks([-180, -60, 60, 180])
    #     plt.ylabel('Response')
    #     # plt.title('Staple')
    #     if i == len(remove_levels)-1:
    #         plt.xlabel('Angle (degrees)')
    #
    #
    #     plt.subplot(len(remove_levels),3,(3*i)+2)
    #     image_files = get_image_file_list('./source-images/scooter/', 'jpg', with_path=True)
    #     im = preprocess(image_files, use_vgg=use_vgg)
    #     out = model.predict(im)
    #     wrap_indices = range(12,24) + range(13)
    #     plot_curves(out[wrap_indices,:], 10, kernel_len=3, angles=range(-180,181,15))
    #     plt.xticks([-180, -60, 60, 180])
    #     # plt.title('Scooter')
    #     if i == len(remove_levels)-1:
    #         plt.xlabel('Angle (degrees)')
    #
    #     plt.subplot(len(remove_levels),3,(3*i)+3)
    #     image_files = get_image_file_list('./source-images/head/', 'jpg', with_path=True)
    #     im = preprocess(image_files, use_vgg=use_vgg)
    #     out = model.predict(im)
    #     wrap_indices = range(12,24) + range(13)
    #     plot_curves(out[wrap_indices,:], 10, kernel_len=3, angles=range(-180,181,15))
    #     plt.xticks([-180, -60, 60, 180])
    #     # plt.title('Head')
    #     if i == len(remove_levels)-1:
    #         plt.xlabel('Angle (degrees)')
    #
    #     fd = freiwald_depths(out)
    #     h, e = np.histogram(fd, hist_edges)
    #     h = h.astype(float)
    #     h = h / np.sum(h)
    #     freiwalds.append(h)
    #
    # plt.tight_layout()
    # plt.savefig('../figures/orientation3d.eps')
    # plt.show()
    #
    # plt.figure(figsize=(3,2*len(remove_levels)))
    # for i in range(len(remove_levels)):
    #     plt.subplot(len(remove_levels),1,i+1)
    #     plt.bar(hist_edges[:-1]+.02, freiwalds[i], width=.2-.04, color=[.5,.5,.5])
    #     plt.xlim([-1, 1])
    #     plt.ylim([0, .55])
    #     plt.ylabel('Fraction of units')
    #
    # plt.xlabel('Head orientation tuning depth')
    # plt.tight_layout()
    # plt.savefig('../figures/orientation-freiwald-net.eps')
    # plt.show()


    # plt.figure(figsize=(7,3))
    # n = 15
    # kernel_len = 3
    # angles = range(0,361,10)
    # image_files = get_image_file_list('./source-images/staple/', 'jpg', with_path=True)
    # im = preprocess(image_files)
    # out = model.predict(im)
    # plt.subplot(1,2,1)
    # plot_curves(out, n, kernel_len=kernel_len, angles=angles, normalize=True)
    # plt.ylabel('Response')
    # plt.ylim([0,1])
    # plt.title('CNN')
    #
    # fake = np.zeros((out.shape[0], n))
    # np.random.seed(1)
    # prefs = 360.*np.random.rand(n)
    # i = 0
    # widths = np.zeros(n)
    # while i < n:
    #     width = 30. + 15. * np.random.randn()
    #     if width > 1:
    #         widths[i] = width
    #         i = i + 1
    # print(widths)
    #
    # for i in range(n):
    #     fake[:,i] = np.exp(-(angles-prefs[i])**2 / 2 / widths[i]**2)
    #
    #
    # plt.subplot(1,2,2)
    # plot_curves(fake, n, kernel_len=kernel_len, angles=angles, normalize=True)
    # plt.title('Empirical')
    # plt.ylim([0,1])
    #
    # plt.tight_layout()
    # plt.savefig('../figures/orientation-salman.eps')
    # plt.show()
