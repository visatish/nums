import cupy as cp
import numpy as np

from utils import Timer, list_prod

MAT_SIZE = [
    10**5,
    10**6,
    10**7
]
NUM_ITERS = 10
if __name__ == '__main__':
    for mat_size in MAT_SIZE:
        print('')

        X_shape = (mat_size, 250)

        timings = []
        for i in range(NUM_ITERS+1):
            print_func = print if i != 0 else lambda _: None
            with Timer(
                'Size %d iter %d' % (mat_size, i),
                print_func=print_func
            ) as t:
                with Timer('Init', print_func=print_func):
                    X = cp.asarray(
                        np.random.uniform(
                            size=list_prod(X_shape)
                        ).astype(np.float32).reshape(X_shape)
                    )
                    cp.cuda.Device(0).synchronize()
                with Timer('Comp', print_func=print_func):
                    X_sqr = X.T @ X
                    cp.cuda.Device(0).synchronize()
            timings.append(t.time_elapsed)
        print('Mean time for size %d: %.3fs.' % (mat_size, np.mean(timings[1:])))
        print('')
