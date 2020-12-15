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
            ) as tt:
                with Timer('Init', print_func=print_func) as it:
                    X = cp.asarray(
                        np.random.uniform(
                            size=list_prod(X_shape)
                        ).astype(np.float32).reshape(X_shape)
                    )
                    cp.cuda.Device(0).synchronize()
                with Timer('Comp', print_func=print_func) as ct:
                    X_sqr = X.T @ X
                    cp.cuda.Device(0).synchronize()
            timings.append(
                (it.time_elapsed, ct.time_elapsed, tt.time_elapsed)
            )
        mean_times = np.mean(np.asarray(timings), axis=0)
        print('Mean time for size %d:' % mat_size)
        print('  Initialization time: %.3fs' % mean_times[0])
        print('  Computation time: %.3fs' % mean_times[1])
        print('  Total time: %.3fs' % mean_times[2])

        print('')
