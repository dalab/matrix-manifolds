import sys
import time

import torch


def sym(a):
    return 0.5 * (a + a.transpose(-2, -1))


def main():
    n = 10000
    a = sym(torch.rand(n, 2, 2))

    start = time.time()
    _ = torch.symeig(a)
    end = time.time()
    print('Time cpu: ', end - start)

    a = a.cuda()
    start = time.time()
    _ = torch.symeig(a)
    end = time.time()
    print('Time gpu: ', end - start)


if __name__ == '__main__':
    sys.exit(main())
