from mpi4py import MPI
import tensorflow as tf
import numpy as np

class RunningMeanStd(tf.Module):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-2, shape=()):
        super(RunningMeanStd, self).__init__()
        self.shape = shape

        # Initialize variables
        self._sum = tf.Variable(
            initial_value=tf.zeros(shape, dtype=tf.float64),
            trainable=False,
            name="runningsum"
        )
        self._sumsq = tf.Variable(
            initial_value=tf.constant(epsilon, shape=shape, dtype=tf.float64),
            trainable=False,
            name="runningsumsq"
        )
        self._count = tf.Variable(
            initial_value=tf.constant(epsilon, dtype=tf.float64),
            trainable=False,
            name="count"
        )

    @property
    def mean(self):
        return tf.cast(self._sum / self._count, tf.float32)

    @property
    def std(self):
        return tf.sqrt(tf.maximum(tf.cast(self._sumsq / self._count, tf.float32) - tf.square(self.mean), 1e-2))

    @tf.function
    def update(self, x):
        x = tf.cast(x, tf.float64)
        n_elements = tf.size(x).numpy()
        n = int(np.prod(self.shape))
        totalvec = np.zeros(n * 2 + 1, dtype='float64')
        addvec = np.concatenate([
            tf.reduce_sum(x, axis=0).numpy().ravel(),
            tf.reduce_sum(tf.square(x), axis=0).numpy().ravel(),
            np.array([x.shape[0]], dtype='float64')
        ])
        MPI.COMM_WORLD.Allreduce(addvec, totalvec, op=MPI.SUM)
        sum_new = tf.constant(totalvec[0:n].reshape(self.shape), dtype=tf.float64)
        sumsq_new = tf.constant(totalvec[n:2*n].reshape(self.shape), dtype=tf.float64)
        count_new = tf.constant(totalvec[2*n], dtype=tf.float64)
        self._sum.assign_add(sum_new)
        self._sumsq.assign_add(sumsq_new)
        self._count.assign_add(count_new)

@tf.function
def test_runningmeanstd():
    test_cases = [
        (np.random.randn(3), np.random.randn(4), np.random.randn(5)),
        (np.random.randn(3, 2), np.random.randn(4, 2), np.random.randn(5, 2)),
    ]

    for x1, x2, x3 in test_cases:
        rms = RunningMeanStd(epsilon=0.0, shape=x1.shape[1:])
        
        x = np.concatenate([x1, x2, x3], axis=0)
        ms1 = [x.mean(axis=0), x.std(axis=0)]
        rms.update(x1)
        rms.update(x2)
        rms.update(x3)
        ms2 = [rms.mean.numpy(), rms.std.numpy()]

        assert np.allclose(ms1, ms2), "Mean and std do not match!"

@tf.function
def test_dist():
    np.random.seed(0)
    p1, p2, p3 = (np.random.randn(3, 1), np.random.randn(4, 1), np.random.randn(5, 1))
    q1, q2, q3 = (np.random.randn(6, 1), np.random.randn(7, 1), np.random.randn(8, 1))

    comm = MPI.COMM_WORLD
    assert comm.Get_size() == 2, "MPI world size is not 2."
    
    if comm.Get_rank() == 0:
        x1, x2, x3 = p1, p2, p3
    elif comm.Get_rank() == 1:
        x1, x2, x3 = q1, q2, q3
    else:
        raise ValueError("Unexpected MPI rank.")

    rms = RunningMeanStd(epsilon=0.0, shape=(1,))

    rms.update(x1)
    rms.update(x2)
    rms.update(x3)

    bigvec = np.concatenate([p1, p2, p3, q1, q2, q3])

    def checkallclose(x, y):
        print("Computed:", x)
        print("Expected:", y)
        return np.allclose(x, y)

    assert checkallclose(
        bigvec.mean(axis=0),
        rms.mean.numpy(),
    ), "Mean check failed."
    assert checkallclose(
        bigvec.std(axis=0),
        rms.std.numpy(),
    ), "Std check failed."

if __name__ == "__main__":
    # Run with mpirun -np 2 python <filename>
    test_dist()
