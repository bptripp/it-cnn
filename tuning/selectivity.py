__author__ = 'bptripp'

from cnn_stimuli import get_image_file_list
import cPickle as pickle
import time
import numpy as np
import matplotlib.pyplot as plt
from alexnet import preprocess, load_net, load_vgg


def excess_kurtosis(columns):
    m = np.mean(columns, axis=0)
    sd = np.std(columns, axis=0)
    result = np.zeros(columns.shape[1])

    for i in range(columns.shape[1]):
        column = columns[:,i]
        d = column - m[i]
        result[i] = np.sum(d**4) / columns.shape[0] / sd[i]**4 - 3

    return result


# Copying these two functions from Salman's repo,
# ObjectSelectivity/sparseness.py and ObjectSelectivity/kurtosis_selectivity_profile.py

def calculate_kurtosis(rates_per_object):
    """
    Given an array of firing rates of the neuron to objects, return the sparseness metric
    Kurtosis (actually excess kurtosis) of the neuron as defined in:
    [1]  Lehky, S. R., Kiani, R., Esteky, H., & Tanaka, K. (2011). Statistics of
         visual responses in primate inferotemporal cortex to object stimuli.
         Journal of Neurophysiology, 106(3), 1097-117.
    Kurtosis  =  (sum (Ri - Rmean)**4 / (n*sigma**4)) - 3
    :param rates_per_object: array of firing rates of the neuron to multiple objects.
    :return: kurtosis sparseness.
    This is defined outside the class as it is used by other selectivity profiles.
    """
    n = np.float(rates_per_object.shape[0])

    rates_mean = np.mean(rates_per_object)
    rates_sigma = np.std(rates_per_object)

    kurtosis = np.sum((rates_per_object - rates_mean)**4) / (n * rates_sigma**4) - 3

    # kurtosis2= np.sum((rates_per_object - rates_mean)**4) / n \
    #            / (np.sum((rates_per_object - rates_mean)**2) / n)** 2 - 3

    return kurtosis


def activity_fraction(rates_per_object):
    R = rates_per_object
    n = len(rates_per_object)

    return n/(n-1) * ( 1 - np.sum(R/n)**2 / np.sum(R**2/n) )

    # num = 1 - (np.sum(R)/n)**2 / np.sum(R**2)/n
    # den = 1 - 1/n
    # return num / den




def plot_selectivity_and_sparseness(r_mat, font_size=10):

    # plt.figure(figsize=)
    # fig = plt.figure()
    # print(fig.get_size_inches())

    # f, ax_arr = plt.subplots(2, 1, sharex=True, figsize=(3.5,5))
    f, ax_arr = plt.subplots(2, 1, sharex=False, figsize=(3,5))

    # Single Neuron selectivities
    n_neurons = r_mat.shape[0]
    n_objs = r_mat.shape[1]

    selectivities = np.zeros(n_neurons)
    sparsenesses = np.zeros(n_objs)

    for n_idx in np.arange(n_neurons):
        rates = r_mat[n_idx, :]
        selectivities[n_idx] = calculate_kurtosis(rates)

    for o_idx in np.arange(n_objs):
        rates = r_mat[:, o_idx]
        sparsenesses[o_idx] = calculate_kurtosis(rates)

    print(np.mean(selectivities))
    print(np.mean(sparsenesses))
    print('min selectivity: ' + str(np.min(selectivities)))
    print('max selectivity: ' + str(np.max(selectivities)))

    # Plot selectivities ------------------------------------------------
    ax_arr[0].hist(np.clip(selectivities, -10, 25), bins=np.arange(-5, 850, step=1), color='red')

    ax_arr[0].set_ylabel('frequency', fontsize=font_size)
    ax_arr[0].set_xlabel('kurtosis', fontsize=font_size)
    ax_arr[0].tick_params(axis='x', labelsize=font_size)
    ax_arr[0].tick_params(axis='y', labelsize=font_size)
    # ax_arr[0].set_xlim([0.1, 850])

    ax_arr[0].annotate('mean=%0.2f' % np.mean(selectivities),
                       xy=(0.55, 0.98),
                       xycoords='axes fraction',
                       fontsize=font_size,
                       horizontalalignment='left',
                       verticalalignment='top')

    ax_arr[0].annotate('med.=%0.2f' % np.median(selectivities),
                       xy=(0.55, 0.88),
                       xycoords='axes fraction',
                       fontsize=font_size,
                       horizontalalignment='left',
                       verticalalignment='top')

    ax_arr[0].annotate('n=%d' % len(selectivities),
                       xy=(0.55, 0.78),
                       xycoords='axes fraction',
                       fontsize=font_size,
                       horizontalalignment='left',
                       verticalalignment='top')

    ax_arr[0].annotate('single-neuron',
                       xy=(0.01, 0.98),
                       xycoords='axes fraction',
                       fontsize=font_size,
                       horizontalalignment='left',
                       verticalalignment='top')
    # ax_arr[0].set_ylim([0, 40])
    # ax_arr[0].set_xlim([0, 200])
    # ax_arr[0].set_ylim([0, 130])

    # ax_arr[0].set_xscale('log')

    # Plot sparsenesses ------------------------------------------------
    ax_arr[1].hist(np.clip(sparsenesses, -10, 60), bins=np.arange(-5, 850, step=3))

    ax_arr[1].set_ylabel('frequency', fontsize=font_size)
    ax_arr[1].set_xlabel('kurtosis', fontsize=font_size)
    ax_arr[1].tick_params(axis='x', labelsize=font_size)
    ax_arr[1].tick_params(axis='y', labelsize=font_size)

    ax_arr[1].annotate('mean=%0.2f' % np.mean(sparsenesses),
                       xy=(0.55, 0.98),
                       xycoords='axes fraction',
                       fontsize=font_size,
                       horizontalalignment='left',
                       verticalalignment='top')

    ax_arr[1].annotate('med.=%0.2f' % np.median(sparsenesses),
                       xy=(0.55, 0.88),
                       xycoords='axes fraction',
                       fontsize=font_size,
                       horizontalalignment='left',
                       verticalalignment='top')

    ax_arr[1].annotate('n=%d' % len(sparsenesses),
                       xy=(0.55, 0.78),
                       xycoords='axes fraction',
                       fontsize=font_size,
                       horizontalalignment='left',
                       verticalalignment='top')

    ax_arr[1].annotate('population',
                       xy=(0.01, 0.98),
                       xycoords='axes fraction',
                       fontsize=font_size,
                       horizontalalignment='left',
                       verticalalignment='top')

    ax_arr[0].set_xlim([-2, 26])
    ax_arr[1].set_xlim([-2, 62])
    # ax_arr[1].set_ylim([0, 300])

    plt.tight_layout()

    # ax_arr[1].set_xscale('log')

if False:
    with open('face-preference-alexnet-0.pkl', 'rb') as file:
        alexnet0 = pickle.load(file)
    with open('face-preference-alexnet-1.pkl', 'rb') as file:
        alexnet1 = pickle.load(file)
    with open('face-preference-alexnet-2.pkl', 'rb') as file:
        alexnet2 = pickle.load(file)
    with open('face-preference-vgg-0.pkl', 'rb') as file:
        vgg0 = pickle.load(file)
    with open('face-preference-vgg-1.pkl', 'rb') as file:
        vgg1 = pickle.load(file)
    with open('face-preference-vgg-2.pkl', 'rb') as file:
        vgg2 = pickle.load(file)

    edges = np.linspace(-5, 5, 21)
    plt.figure(figsize=(8,4.5))
    plt.subplot(2,3,1)
    plt.hist(alexnet2, edges)
    plt.ylabel('AlexNet Unit Count', fontsize=16)
    plt.title('output-2', fontsize=16)
    plt.subplot(2,3,2)
    plt.hist(alexnet1, edges)
    plt.title('output-1', fontsize=16)
    plt.subplot(2,3,3)
    plt.hist(alexnet0, edges)
    plt.title('output', fontsize=16)
    plt.subplot(2,3,4)
    plt.hist(vgg2, edges, color='g')
    plt.ylabel('VGG Unit Count', fontsize=16)
    plt.subplot(2,3,5)
    plt.hist(vgg1, edges, color='g')
    plt.xlabel('Preference for Face Images', fontsize=16)
    plt.subplot(2,3,6)
    plt.hist(vgg0, edges, color='g')
    plt.tight_layout(pad=0.05)
    plt.savefig('../figures/selectivity-faces.eps')
    plt.show()



if False:
    use_vgg = True
    remove_level = 2
    if use_vgg:
        model = load_vgg(weights_path='../weights/vgg16_weights.h5', remove_level=remove_level)
    else:
        model = load_net(weights_path='../weights/alexnet_weights.h5', remove_level=remove_level)

    image_files = get_image_file_list('./images/lehky-processed/', 'png', with_path=True)
    im = preprocess(image_files, use_vgg=use_vgg)
    print(image_files)

    mainly_faces = [197]
    mainly_faces.extend(range(170, 178))
    mainly_faces.extend(range(181, 196))
    mainly_faces.extend(range(203, 214))
    mainly_faces.extend(range(216, 224))

    faces_major = [141, 142, 165, 169, 179, 196, 214, 215, 271]
    faces_major.extend(range(144, 147))
    faces_major.extend(range(157, 159))

    faces_present = [131, 143, 178, 180, 198, 230, 233, 234, 305, 306, 316, 372, 470]
    faces_present.extend(range(134, 141))
    faces_present.extend(range(147, 150))
    faces_present.extend(range(155, 157))
    faces_present.extend(range(161, 165))
    faces_present.extend(range(365, 369))
    faces_present.extend(faces_major)
    faces_present.extend(mainly_faces)

    faces_ind = []
    for i in range(len(image_files)):
        for j in range(len(mainly_faces)):
            if str(mainly_faces[j]) + '.' in image_files[i]:
                faces_ind.append(i)

    no_faces_ind = []
    for i in range(len(image_files)):
        has_face = False
        for j in range(len(faces_present)):
            if str(faces_present[j]) + '.' in image_files[i]:
                has_face = True
        if not has_face:
            no_faces_ind.append(i)

    # print(faces_ind)
    # print(no_faces_ind)

    start_time = time.time()
    out = model.predict(im)
    print(out.shape)

    f = out[faces_ind,:]
    nf = out[no_faces_ind,:]
    print(f.shape)
    print(nf.shape)

    face_preference = np.mean(f, axis=0) - np.mean(nf, axis=0)
    vf = np.var(f, axis=0) + 1e-3 # small constant in case zero variance due to lack of response
    vnf = np.var(nf, axis=0) + 1e-3
    d_prime = face_preference / np.sqrt((vf + vnf)/2)

    network_name = 'vgg' if use_vgg else 'alexnet'
    with open('face-preference-' + network_name + '-' + str(remove_level) + '.pkl', 'wb') as file:
        pickle.dump(d_prime, file)

    print(d_prime)
    plt.hist(d_prime)
    plt.show()


if True:
    use_vgg = False
    remove_level = 1
    if use_vgg:
        model = load_vgg(weights_path='../weights/vgg16_weights.h5', remove_level=remove_level)
    else:
        model = load_net(weights_path='../weights/alexnet_weights.h5', remove_level=remove_level)

    # model = load_net(weights_path='../weights/alexnet_weights.h5')

    image_files = get_image_file_list('./images/lehky-processed/', 'png', with_path=True)
    im = preprocess(image_files, use_vgg=use_vgg)


    start_time = time.time()
    out = model.predict(im)
    print('prediction time: ' + str(time.time() - start_time))

    # with open('lehky.pkl', 'wb') as file:
    #     pickle.dump(out, file)

    # with open('lehky.pkl', 'rb') as file:
    #     out = pickle.load(file)


    n = 674
    # use first n or n with greatest responses
    if False:
        rect = np.maximum(0, out[:,:n])
    else:
        maxima = np.max(out, axis=0)

        ind = np.zeros(n, dtype=int)
        c = 0
        i = 0
        while c < n:
            if maxima[i] > 2:
                ind[c] = i
                c = c + 1
            i = i + 1

        # ind = (-maxima).argsort()[:n]

        rect = np.maximum(0, out[:,ind])


    selectivity = excess_kurtosis(rect)
    sparseness = excess_kurtosis(rect.T)

    print(np.mean(selectivity))
    print(np.mean(sparseness))
    print(np.max(selectivity))
    print(np.max(sparseness))

    plot_selectivity_and_sparseness(rect.T, 11)
    network_name = 'vgg' if use_vgg else 'alexnet'
    plt.savefig('../figures/selectivity-' + network_name + '-' + str(remove_level) + '-talk.eps')
    plt.show()


if False:
    plt.figure(figsize=(4,3.8))
    plt.scatter(3.5, 12.51, c='k', marker='x', s=40, label='IT') # from Lehky et al. Fig 4A and 4B
    selectivity_alexnet = [10.53, 28.59, 31.44]
    sparseness_alexnet = [4.04, 8.85, 6.61]
    selectivity_vgg = [26.79, 14.44, 34.65]
    sparseness_vgg = [6.59, 3.40, 3.54]
    plt.scatter([10.53, 28.59, 31.44], [4.04, 8.85, 6.61], c='b', marker='o', s=30, label='Alexnet')
    plt.scatter([26.79, 14.44, 34.65], [6.59, 3.40, 3.54], c='g', marker='s', s=45, label='VGG-16')
    plt.plot([0, 40], [0, 40], 'k')
    plt.xlim([0,38])
    plt.ylim([0,38])
    gap = 0.4
    plt.text(3.5+gap, 9.61+gap+.05, 'IT')
    plt.text(selectivity_alexnet[0]+gap, sparseness_alexnet[0]+gap, 'out')
    plt.text(selectivity_alexnet[1]+gap, sparseness_alexnet[1]+gap, 'out-1')
    plt.text(selectivity_alexnet[2]+gap, sparseness_alexnet[2]+gap, 'out-2')
    plt.text(selectivity_vgg[0]+gap, sparseness_vgg[0]+gap, 'out')
    plt.text(selectivity_vgg[1]+gap, sparseness_vgg[1]+gap, 'out-1')
    plt.text(selectivity_vgg[2]+gap, sparseness_vgg[2]+gap, 'out-2')
    plt.xlabel('Selectivity')
    plt.ylabel('Sparseness')
    plt.tight_layout()
    plt.savefig('../figures/cnn-selectivity.eps')
    plt.show()


if False:
    r_mat = rect.T
    n_neurons = r_mat.shape[0]
    activity_fractions = np.zeros(n_neurons)
    for n_idx in np.arange(n_neurons):
        rates = r_mat[n_idx, :]
        activity_fractions[n_idx] = activity_fraction(rates)

    print(activity_fractions)
    plt.plot(activity_fractions)
    plt.show()

    rate = np.mean(rect,0)
    # with open('activity-fraction.pkl', 'wb') as file:
    #     pickle.dump((ind, activity_fractions), file)


# bins = np.linspace(0, 1000, 501)
# plt.figure()
# plt.subplot(2,1,1)
# plt.hist(selectivity, bins)
# # plt.xlim([0, 100])
# # plt.ylim([0, 100])
# plt.subplot(2,1,2)
# plt.hist(sparseness, bins)
# # plt.xlim([0, 100])
# # plt.ylim([0, 100])
# plt.show()
#
# note: there is a NaN due to single kurtosis much less than gaussian
# print(np.corrcoef(np.mean(rect,0), np.log(selectivity+1)))
# plt.figure()
# plt.scatter(np.mean(rect,0), np.log(selectivity+1))
# plt.gca().set_xscale('log')
# plt.gca().set_yscale('log')
# plt.show()

#
# rate = np.mean(rect,0)
# with open('rate-vs-selectivity.pkl', 'wb') as file:
#     pickle.dump((ind, rate, selectivity), file)
