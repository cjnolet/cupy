import unittest

import cupy
from cupy import testing

import pytest


@testing.parameterize(*testing.product({
    'format': ['csc', 'csr'],
    'density': [0.9],
    'dtype': ['float32', 'float64', 'complex64', 'complex128'],
    'n_rows': [100000],
    'n_cols': [1000]
}))
@testing.with_requires('scipy')
class TestIndexing(unittest.TestCase):

    def _run(self, maj, min=None):
        a = cupy.sparse.random(self.n_rows, self.n_cols,
                               format=self.format,
                               density=self.density)

        print(self.format)
        print(self.dtype)

        # sparse.random doesn't support complex types
        # so we need to cast
        a = a.astype(self.dtype)
        if min is not None:
            expected = a.get()

            with cupy.prof.time_range(message="cpu", color_id=1):
                import time
                start_cpu = time.time()
                expected = expected[maj, min]
                print("cpu_time: %s" % (time.time()-start_cpu))

            with cupy.prof.time_range(message="gpu", color_id=2, sync=True):
                start_gpu = time.time()
                actual = a[maj, min]
                print("gpu_time: %s" % (time.time() - start_gpu))
        else:
            expected = a.get()

            import time
            start_cpu = time.time()
            with cupy.prof.time_range(message="cpu", color_id=1):
                expected = expected[maj]
            print("cpu_time: %s" % (time.time()-start_cpu))

            start_gpu = time.time()
            with cupy.prof.time_range(message="gpu", sync=True, color_id=2):
                actual = a[maj]
            print("gpu_time: %s" % (time.time() - start_gpu))

        if cupy.sparse.isspmatrix(actual):
            actual.sort_indices()
            expected.sort_indices()

            cupy.testing.assert_array_equal(actual.indptr,
                                            cupy.array(expected.indptr))

            cupy.testing.assert_array_equal(actual.indices,
                                            cupy.array(expected.indices))

            cupy.testing.assert_array_equal(actual.data,
                                            cupy.array(expected.data))

        else:
            cupy.testing.assert_array_equal(actual.ravel(),
                                            cupy.array(expected).ravel())


    import cupy.prof

    def test_major_fancy(self):

        import numpy

        r = numpy.arange(self.n_cols)
        m = numpy.arange(self.n_rows)

        c = cupy.asarray(numpy.random.choice(r, size=100))
        n = cupy.asarray(numpy.random.choice(m, size=100)).astype("int32")

        with cupy.prof.time_range(message="test_major_fancy. format=%s, dtype=%s" % (self.format, self.dtype)):
            self._run(slice(50, 500))
