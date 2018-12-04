import numpy as np
import csv
import seaborn as sns
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt


def main():
    label_matrix = []
    with open('train.csv', 'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        next(csv_reader)
        for row in csv_reader:
            labels = np.zeros(28)
            for label in row[1].split():
                labels[int(label)] = 1
            label_matrix.append(labels)
    label_matrix = np.asmatrix(label_matrix)

    labels = ['Nucleoplasm',
              'Nuclear membrane',
              'Nucleoli',
              'Nucleoli fibrillar center',
              'Nuclear speckles',
              'Nuclear bodies',
              'Endoplasmic reticulum',
              'Golgi apparatus',
              'Peroxisomes',
              'Endosomes',
              'Lysosomes',
              'Intermediate filaments',
              'Actin filaments',
              'Focal adhesion sites',
              'Microtubules',
              'Microtubule ends',
              'Cytokinetic bridge',
              'Mitotic spindle',
              'Microtubule organizing center',
              'Centrosome',
              'Lipid droplets',
              'Plasma membrane',
              'Cell junctions',
              'Mitochondria',
              'Aggresome',
              'Cytosol',
              'Cytoplasmic bodies',
              'Rods & rings']


    sns.set(style="white")

    d = pd.DataFrame(data=label_matrix, columns=labels)

    # Compute the correlation matrix
    corr = d.corr()

    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure
    font = {'family': 'serif',
            'color': 'k',
            'size': 24}
    f, ax = plt.subplots(figsize=(11, 9))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})

    # chance font size
    f.axes[0].tick_params(labelsize=14)
    f.axes[1].tick_params(labelsize=14)
    plt.title('Correlation between labels', font)

    plt.show()

if __name__ == '__main__':
    main()