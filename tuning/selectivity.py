__author__ = 'bptripp'

from cnn_stimuli import get_image_file_list
import cPickle as pickle
import time
import numpy as np
import matplotlib.pyplot as plt
from alexnet import preprocess, load_net


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




def plot_selectivity_and_sparseness(r_mat, font_size=50):

    # plt.figure(figsize=)
    # fig = plt.figure()
    # print(fig.get_size_inches())
    f, ax_arr = plt.subplots(2, 1, sharex=True, figsize=(6,5))

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
    ax_arr[0].hist(selectivities, bins=np.arange(-5, 850, step=1))

    ax_arr[0].set_ylabel('Frequency', fontsize=font_size)
    # ax_arr[0].set_xlabel('Kurtosis', fontsize=font_size)
    ax_arr[0].tick_params(axis='x', labelsize=font_size)
    ax_arr[0].tick_params(axis='y', labelsize=font_size)
    # ax_arr[0].set_xlim([0.1, 850])

    ax_arr[0].annotate('Mean=%0.2f' % np.mean(selectivities),
                       xy=(0.99, 0.9),
                       xycoords='axes fraction',
                       fontsize=font_size,
                       horizontalalignment='right',
                       verticalalignment='top')

    ax_arr[0].annotate('N=%d' % len(selectivities),
                       xy=(0.99, 0.75),
                       xycoords='axes fraction',
                       fontsize=font_size,
                       horizontalalignment='right',
                       verticalalignment='top')

    ax_arr[0].annotate('Single Neuron Selectivity',
                       xy=(0.7, 0.95),
                       xycoords='axes fraction',
                       fontsize=font_size,
                       horizontalalignment='right',
                       verticalalignment='top')
    # ax_arr[0].set_ylim([0, 40])
    # ax_arr[0].set_xlim([0, 200])
    ax_arr[0].set_ylim([0, 130])

    # ax_arr[0].set_xscale('log')

    # Plot sparsenesses ------------------------------------------------
    ax_arr[1].hist(sparsenesses, bins=np.arange(-5, 850, step=1))

    ax_arr[1].set_ylabel('Frequency', fontsize=font_size)
    ax_arr[1].set_xlabel('Kurtosis', fontsize=font_size)
    ax_arr[1].tick_params(axis='x', labelsize=font_size)
    ax_arr[1].tick_params(axis='y', labelsize=font_size)

    ax_arr[1].annotate('Mean=%0.2f' % np.mean(sparsenesses),
                       xy=(0.99, 0.9),
                       xycoords='axes fraction',
                       fontsize=font_size,
                       horizontalalignment='right',
                       verticalalignment='top')

    ax_arr[1].annotate('N=%d' % len(sparsenesses),
                       xy=(0.99, 0.75),
                       xycoords='axes fraction',
                       fontsize=font_size,
                       horizontalalignment='right',
                       verticalalignment='top')

    ax_arr[1].annotate('Population Sparseness',
                       xy=(0.7, 0.95),
                       xycoords='axes fraction',
                       fontsize=font_size,
                       horizontalalignment='right',
                       verticalalignment='top')

    ax_arr[1].set_xlim([-3, 70])
    ax_arr[1].set_ylim([0, 180])

    plt.tight_layout()

    # ax_arr[1].set_xscale('log')


# image_files = get_image_file_list('/Users/bptripp/code/salman-IT/salman/images/lehky-processed/', 'png', with_path=True)
# im = preprocess(image_files)
#
# model = load_net()
#
# start_time = time.time()
# out = model.predict(im)
# print('prediction time: ' + str(time.time() - start_time))
#
# with open('lehky.pkl', 'wb') as file:
#     pickle.dump(out, file)

with open('lehky.pkl', 'rb') as file:
    out = pickle.load(file)


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

if True:
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
    with open('activity-fraction.pkl', 'wb') as file:
        pickle.dump((ind, activity_fractions), file)



plot_selectivity_and_sparseness(rect.T, 16)
plt.savefig('../figures/selectivity-salman.eps')
plt.show()



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


rate = np.mean(rect,0)
with open('rate-vs-selectivity.pkl', 'wb') as file:
    pickle.dump((ind, rate, selectivity), file)
