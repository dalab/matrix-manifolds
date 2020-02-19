import sys
import time

import tensorflow as tf


def sym(a):
    return 0.5 * (a + tf.matrix_transpose(a))


def main():
    n = 100000
    a = sym(tf.random_uniform(shape=(n, 2, 2)))

    with tf.Session() as sess:
        start = time.time()
        sess.run(tf.linalg.eigh(a))
        end = time.time()
        print('Time: ', end - start)


if __name__ == '__main__':
    sys.exit(main())
