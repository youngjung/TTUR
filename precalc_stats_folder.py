#!/usr/bin/env python3

import os
import glob
import numpy as np
from scipy.misc import imread

import tensorflow as tf

import fid


def run_for_folder(dir_images, fname_output):
    # if you have downloaded and extracted
    #   http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
    # set this path to the directory where the extracted files are, otherwise
    # just set it to None and the script will later download the files for you
    print("check for inception model..", end=" ", flush=True)
    inception_path = None
    inception_path = fid.check_or_download_inception(inception_path)  # download inception if necessary
    print("ok")
    print("create inception graph..", end=" ", flush=True)
    fid.create_inception_graph(inception_path)  # load the graph into the current TF graph
    print("ok")

    # crawl images
    image_list = []
    for ext in ['jpg', 'png']:
        image_list.extend(glob.glob(os.path.join(dir_images, '*.{}'.format(ext))))
    images = np.array([imread(str(fn)).astype(np.float32) for fn in image_list])
    assert len(images) > 0, "{} does not contain images (jpg, png)".format(dir_images)
    print(len(images), 'images are found')

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        mu, sigma = fid.calculate_activation_statistics(images, sess, batch_size=100)
        np.savez_compressed(fname_output, mu=mu, sigma=sigma)


if __name__ == '__main__':
    import fire
    fire.Fire(run_for_folder)
