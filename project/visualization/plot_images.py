import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


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
    """
    Displays a 2x2 table of each of the channel images.
    """
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


def make_rgb_image_from_four_channels(channels: list, image_width=512, image_height=512) -> np.ndarray:
    """
    Creates an RGB image from the four channels.
    """
    rgb_image = np.zeros(shape=(image_height, image_width, 3), dtype=np.float)
    yellow = np.array(Image.open(channels[0]))
    # yellow is red + green
    rgb_image[:, :, 0] += yellow/2   
    rgb_image[:, :, 1] += yellow/2
    # loop for R,G and B channels
    for index, channel in enumerate(channels[1:]):
        current_image = Image.open(channel)
        rgb_image[:, :, index] += current_image
    # Normalize image
    rgb_image = rgb_image / rgb_image.max() * 255
    return rgb_image.astype(np.uint8)


def overlay_images(image_names, nrows=1, ncols=1):
    """
    Displays an image with the four channels overlaid on each other.
    """
    rgb_image = make_rgb_image_from_four_channels(image_names)

    # show image
    plt.imshow(rgb_image)
    plt.show()




def main():
    image_files = ['00008af0-bad0-11e8-b2b8-ac1f6b6435d0_blue.png',
                   '00008af0-bad0-11e8-b2b8-ac1f6b6435d0_green.png',
                   '00008af0-bad0-11e8-b2b8-ac1f6b6435d0_red.png',
                   '00008af0-bad0-11e8-b2b8-ac1f6b6435d0_yellow.png']
    images = list(map(lambda x: cv2.imread(x, 0), image_files))

    #plot_images(images[0], images[1], images[2], images[3])

    overlay_images(image_files)



if __name__ == "__main__":
    main()
