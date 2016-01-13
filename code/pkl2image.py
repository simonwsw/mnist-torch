import cPickle, gzip, os
import numpy as np
import scipy.misc

def save_image(image_dir, data_set):
    for i in range(len(data_set[0])):
        # data name with data id + label
        file_name = os.path.join(image_dir,
            'img' + str(i).zfill(5) + '_' + str(data_set[1][i]) + '.jpg')
        scipy.misc.imsave(file_name, np.reshape(data_set[0][i], (28, 28)))

def pkl2image(pkl_file, train_dir, test_dir):
    # load data from pickle
    with gzip.open(pkl_file, 'rb') as f:
        train_set, valid_set, test_set = cPickle.load(f)

    # save only train and test
    save_image(train_dir, train_set)
    save_image(test_dir, test_set)

if __name__ == '__main__':
    pkl2image(pkl_file='data/mnist.pkl.gz', train_dir='data/train',
        test_dir='data/test')
