import matplotlib, os
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from flarestack.analyses.ccsn.necker_2019.ccsn_helpers \
    import updated_sn_catalogue_name, sn_cats, sn_times, pdf_names, raw_output_dir
import logging
from flarestack.shared import plot_output_dir


plot_dir = plot_output_dir(raw_output_dir+'/catalogue_visualization/difference_stasik/')

if not os.path.isdir(plot_dir):
    os.makedirs(plot_dir)


def autolabel(rects, axis):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        axis.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 2),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


def plot_difference_tot(filename):

    N = {}
    for flagged in [True, False]:

        N['flagged' if flagged else 'unflagged'] = []
        Nnew = []

        for i, cat in enumerate(sn_cats):
            catarr = np.load(updated_sn_catalogue_name(cat, flagged=flagged))
            N['flagged' if flagged else 'unflagged'].append(len(catarr))

            newcat = np.load(updated_sn_catalogue_name(cat, pdf_name='missed_objects'))
            Nnew.append(len(newcat))

    fig, ax = plt.subplots()

    x = np.arange(len(sn_cats))
    width = 0.4

    r1 = ax.bar(x + width/2, N['unflagged'], width, label='new catalogue')
    r2 = ax.bar(x - width/2, N['flagged'], width, label='old catalogue')
    r1new = ax.bar(x + width / 2, Nnew, width,
                   bottom=np.array(N['unflagged']) - np.array(Nnew),
                   hatch = '//', color = r1[0].get_facecolor(),
                   label='previously not included'
                   )

    ax.set_ylabel('number of objects')
    ax.set_xlabel('SN types')
    ax.set_xticks(x)
    ax.set_xticklabels(sn_cats)
    ax.set_ylim(np.array(ax.get_ylim()) * [1, 1.1])
    ax.set_title('number of objects in catalogues')
    ax.legend()

    autolabel(r1, ax)
    autolabel(r2, ax)

    fig.savefig(filename)
    plt.close()


def plot_difference_individual(sn_types, filename):

    fig, axs = plt.subplots(len(sn_types))

    for i, sn_type in enumerate(sn_types):
        ax = axs[i]
        N = {}

        cats = []
        for pdf_type in sn_times[sn_type]:
            for pdf_time in sn_times[sn_type][pdf_type]:
                cats.append(pdf_names(pdf_type, pdf_time))

        for flagged in [True, False]:
            N['flagged' if flagged else 'unflagged'] = []

            for cat in cats:
                catarr = np.load(updated_sn_catalogue_name(sn_type, pdf_name=cat, flagged=flagged))
                N['flagged' if flagged else 'unflagged'].append(len(catarr))

        x = np.arange(len(cats))
        width = 0.4

        r1 = ax.bar(x + width / 2, N['unflagged'], width, label='new')
        r2 = ax.bar(x - width / 2, N['flagged'], width, label='old')

        ax.set_ylabel(f'SN type {sn_type}')
        ax.set_xticks(x)
        ax.set_xticklabels(cats)
        ax.set_ylim(np.array(ax.get_ylim()) * [1,1.1])

        autolabel(r1, ax)
        autolabel(r2, ax)

    axs[-1].set_xlabel('PDF type')
    axs[0].set_title('catalogues with wrong classifications')
    axs[-1].legend()

    fig.savefig(filename)
    plt.close()

plot_difference_tot(plot_dir + 'total.pdf')
plot_difference_individual(['Ibc', 'IIn'], plot_dir + 'individual.pdf')
