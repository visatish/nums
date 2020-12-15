import numpy as np
import ray

from nums.core.systems import cupy_compute
from nums.core.systems.systems import RaySystem
from nums.core.systems.filesystem import FileSystem
from nums.core.systems.schedulers import BlockCyclicScheduler
from nums.core.array.application import ArrayApplication

from utils import Timer

NUM_CPUS = 4
NUM_GPUS = [1, 2, 4]
CLUSTER_SHAPE = (1, 1)
MAT_SIZE = [
    10**5,
    10**6,
    10**7
]
NUM_ITERS = 10
if __name__ == '__main__':
    for num_gpus in NUM_GPUS:
        ray.init(num_cpus=NUM_CPUS, num_gpus=num_gpus)
        try:
            for mat_size in MAT_SIZE:
                scheduler = BlockCyclicScheduler(
                    compute_module=cupy_compute,
                    cluster_shape=CLUSTER_SHAPE,
                    use_head=True,
#                    verbose=True
                    verbose=False
                )
                system = RaySystem(compute_module=cupy_compute, scheduler=scheduler)
                system.init()
                try:
                    print('')
                    app = ArrayApplication(system=system, filesystem=FileSystem(system))

                    X_shape = (mat_size, 250)

                    timings = []
                    for i in range(NUM_ITERS+1):
                        print_func = print if i != 0 else lambda _: None
                        with Timer(
                            '%d GPUs and size %d iter %d' % (num_gpus, mat_size, i),
                            print_func=print_func
                        ) as tt:
                            with Timer('Init', print_func=print_func) as it:
                                X = app.random_state().uniform(
                                    0.0,
                                    1.0,
                                    X_shape,
                                    (int(X_shape[0] / num_gpus), X_shape[1]),
                                     dtype=np.float32
                                )
                                X.touch()
                            with Timer('Comp', print_func=print_func) as ct:
                                X_sqr = X.T @ X
                                X_sqr.touch()
                        timings.append(
                            (it.time_elapsed, ct.time_elapsed, tt.time_elapsed)
                        )
                    mean_times = np.mean(np.asarray(timings), axis=0)
                    print(
                        'Mean time for %d GPUs and size %d:' %
                        (num_gpus, mat_size)
                    )
                    print('  Initialization time: %.3fs' % mean_times[0])
                    print('  Computation time: %.3fs' % mean_times[1])
                    print('  Total time: %.3fs' % mean_times[2])
                    print('')
                finally:
                    system.shutdown()
        finally:
            ray.shutdown()
