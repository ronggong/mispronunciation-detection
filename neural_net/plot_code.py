import numpy as np
import matplotlib.style
import matplotlib as mpl
mpl.style.use('classic')
import matplotlib.pyplot as plt

fontsize = 15


def plot_spectro_att(mfcc0,
                     att_vector,
                     hopsize_t,
                     filename_save):

    plt.figure(figsize=(16, 6))

    ax1 = plt.subplot(2, 1, 1)
    y = np.arange(0, 80)
    x = np.arange(0, mfcc0.shape[0]) * hopsize_t
    plt.pcolormesh(x, y, np.transpose(mfcc0))

    ax1.set_ylabel('Syllable\nlog-mel spectro', fontsize=fontsize)
    ax1.axis('tight')

    ax2 = plt.subplot(2, 1, 2)
    x = np.arange(0, len(att_vector)) * hopsize_t
    plt.plot(x, att_vector)

    ax2.set_ylabel('Attention\nvector', fontsize=fontsize)
    ax2.axis('tight')
    plt.xlabel('time (s)')

    plt.savefig(filename_save, bbox_inches='tight')

    # plt.show()