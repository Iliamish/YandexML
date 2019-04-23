import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import img_as_float
import skimage
from skimage.io import imread, imshow
from sklearn.cluster import KMeans
from typing import Tuple

image = img_as_float(imread("parrots.jpg"))

w, h, d = image.shape
pixels = pd.DataFrame(np.reshape(image, (w*h, d)), columns=["R", "G", "B"])
pixels.head()

def cluster_pixels(pixels, n_clusters):
    model = KMeans(n_clusters=n_clusters, init="k-means++", random_state=241)
    pixels["cluster"] = model.fit_predict(pixels)

    return pixels

def mean_median_image(pixels) :
    means = pixels.groupby("cluster").mean().values
    mean_pixels = np.array([means[c] for c in pixels["cluster"]])
    mean_image = np.reshape(mean_pixels, (w, h, d))

    medians = pixels.groupby("cluster").median().values
    median_pixels = np.array([medians[c] for c in pixels["cluster"]])
    median_image = np.reshape(median_pixels, (w, h, d))

    return mean_image, median_image

def psnr(image1, image2) :
    mse = np.mean((image1 - image2) ** 2)
    return 10.0 * np.log10(1.0 / mse)


def show_images(mean_image, median_image) :
    fig = plt.figure()

    ax = fig.add_subplot(1, 2, 1)
    ax.title.set_text("Mean image")
    imshow(mean_image)

    ax = fig.add_subplot(1, 2, 2)
    ax.title.set_text("Median image")
    imshow(median_image)

    plt.show()


for n in range(1, 21):
    print(f"Clustering: {n}")

    cpixels = cluster_pixels(pixels, n)
    mean_image, median_image = mean_median_image(cpixels)
    show_images(mean_image, median_image)

    psnr_mean, psnr_median = psnr(image, mean_image), psnr(image, median_image)
    print(f"PSNR (mean): {psnr_mean:.2f}\nPSNR (median): {psnr_median:.2f}\n\n")

    if psnr_mean > 20 or psnr_median > 20:
        print(str(n))
        break