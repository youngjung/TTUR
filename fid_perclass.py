#!/usr/bin/env python3

import os
import glob
import numpy as np
from tqdm import tqdm
from scipy.misc import imread

import tensorflow as tf

import fid


def perclass_fid_between_dataset_and_samples(dir_stats, root_samples):
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

    classnames = [os.path.splitext(fname)[0]
                  for fname in os.listdir(dir_stats)
                  if os.path.splitext(fname)[1] == '.npz']
    print('classnames: ', classnames)
    for classname in classnames:
        assert os.path.exists(os.path.join(root_samples, classname))

    fids = []
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for classname in tqdm(classnames):
            # load stats
            fname_stat = os.path.join(dir_stats, '{}.npz'.format(classname))
            f = np.load(fname_stat)
            m, s = f['mu'][:], f['sigma'][:]
            f.close()

            # calc stats
            dir_images = os.path.join(root_samples, classname)
            mu, sigma = run_perclass(dir_images, sess)

            # compute fid
            fid_value = fid.calculate_frechet_distance(mu, sigma, m, s)
            fids.append(fid_value)
            print(classname, fid_value)

    print()
    for classname, fid_value in zip(classnames, fids):
        print(classname, fid_value)
    fids = np.array(fids)
    print('fid: {} +_ {}'.format(fids.mean(), fids.std()))


def run_perclass(dir_images, sess):
    # loads all images into memory (this might require a lot of RAM!)
    image_list = []
    for ext in ['jpg', 'png']:
        image_list.extend(glob.glob(os.path.join(dir_images, '*.{}'.format(ext))))
    images = np.array([imread(str(fn)).astype(np.float32) for fn in image_list])
    assert len(images) > 0, "{} does not contain images (jpg, png)".format(dir_images)
    mu, sigma = fid.calculate_activation_statistics(images, sess, batch_size=100)
    return mu, sigma


if __name__ == '__main__':
    import fire
    fire.Fire(perclass_fid_between_dataset_and_samples)
