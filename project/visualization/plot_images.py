import cv2
import matplotlib.pyplot as plt


#create custom color maps
cdict1 = {'red':   ((0.0,  0.0, 0.0),
                   (1.0,  0.0, 0.0)),

         'green': ((0.0,  0.0, 0.0),
                   (0.75, 1.0, 1.0),
                   (1.0,  1.0, 1.0)),

         'blue':  ((0.0,  0.0, 0.0),
                   (1.0,  0.0, 0.0))}

cdict2 = {'red':   ((0.0,  0.0, 0.0),
                   (0.75, 1.0, 1.0),
                   (1.0,  1.0, 1.0)),

         'green': ((0.0,  0.0, 0.0),
                   (1.0,  0.0, 0.0)),

         'blue':  ((0.0,  0.0, 0.0),
                   (1.0,  0.0, 0.0))}

cdict3 = {'red':   ((0.0,  0.0, 0.0),
                   (1.0,  0.0, 0.0)),

         'green': ((0.0,  0.0, 0.0),
                   (1.0,  0.0, 0.0)),

         'blue':  ((0.0,  0.0, 0.0),
                   (0.75, 1.0, 1.0),
                   (1.0,  1.0, 1.0))}

cdict4 = {'red': ((0.0,  0.0, 0.0),
                   (0.75, 1.0, 1.0),
                   (1.0,  1.0, 1.0)),

         'green': ((0.0,  0.0, 0.0),
                   (0.75, 1.0, 1.0),
                   (1.0,  1.0, 1.0)),

         'blue':  ((0.0,  0.0, 0.0),
                   (1.0,  0.0, 0.0))}

plt.register_cmap(name='greens', data=cdict1)
plt.register_cmap(name='reds', data=cdict2)
plt.register_cmap(name='blues', data=cdict3)
plt.register_cmap(name='yellows', data=cdict4)




def plot_images(blue, green, red, yellow):
    figure_size = 20
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(figure_size,figure_size))
    ax[0, 0].imshow(green, cmap="greens")
    ax[0, 0].set_title("Protein of interest", fontsize=18)
    ax[0, 1].imshow(red, cmap="reds")
    ax[0, 1].set_title("Microtubules", fontsize=18)
    ax[1, 0].imshow(blue, cmap="blues")
    ax[1, 0].set_title("Nucleus", fontsize=18)
    ax[1, 1].imshow(yellow, cmap="yellows")
    ax[1, 1].set_title("Endoplasmic reticulum", fontsize=18)
    for i in range(2):
        for j in range(2):
            ax[i, j].set_xticklabels([])
            ax[i, j].set_yticklabels([])
            ax[i, j].tick_params(left=False, bottom=False)
    plt.show()




def main():
    image_files = ['00008af0-bad0-11e8-b2b8-ac1f6b6435d0_blue.png',
                   '00008af0-bad0-11e8-b2b8-ac1f6b6435d0_green.png',
                   '00008af0-bad0-11e8-b2b8-ac1f6b6435d0_red.png',
                   '00008af0-bad0-11e8-b2b8-ac1f6b6435d0_yellow.png']
    images = list(map(lambda x: cv2.imread(x, 0), image_files))

    plot_images(images[0], images[1], images[2], images[3])



if __name__ == "__main__":
    main()