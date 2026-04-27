"""Zector - Ultra-fast vector database powered by Zig HNSW + IVF engine.

Zero-copy Python binding using ctypes + numpy. No serialization overhead.
"""

import ctypes
import numpy as np
import os
import sys

# Load shared library
_lib_ext = ".dylib" if sys.platform == "darwin" else ".so"
_lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"zig-out/lib/libzector{_lib_ext}")
if not os.path.exists(_lib_path):
    # Fallback: check project root
    _alt = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"libzector{_lib_ext}")
    if os.path.exists(_alt):
        _lib_path = _alt
    else:
        raise FileNotFoundError(
            f"libzector{_lib_ext} not found. Build it with: zig build shared -Doptimize=ReleaseFast"
        )

_lib = ctypes.CDLL(_lib_path)

# C-API signatures
_lib.zector_init.argtypes = [ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32]
_lib.zector_init.restype = ctypes.c_void_p

_lib.zector_free.argtypes = [ctypes.c_void_p]
_lib.zector_free.restype = None

_lib.zector_add.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float)]
_lib.zector_add.restype = ctypes.c_int64

_lib.zector_add_batch.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_uint32]
_lib.zector_add_batch.restype = ctypes.c_int32

_lib.zector_build_index.argtypes = [ctypes.c_void_p]
_lib.zector_build_index.restype = ctypes.c_int32

_lib.zector_set_search_ef.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
_lib.zector_set_search_ef.restype = None

_lib.zector_set_turbo_enabled.argtypes = [ctypes.c_void_p, ctypes.c_bool]
_lib.zector_set_turbo_enabled.restype = None

_lib.zector_search.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_uint32,
    ctypes.POINTER(ctypes.c_uint64),
    ctypes.POINTER(ctypes.c_float),
]
_lib.zector_search.restype = ctypes.c_int32

_lib.zector_count.argtypes = [ctypes.c_void_p]
_lib.zector_count.restype = ctypes.c_uint64

_lib.zector_dimension.argtypes = [ctypes.c_void_p]
_lib.zector_dimension.restype = ctypes.c_uint32


def _f32_ptr(arr):
    """Get a ctypes float pointer from a contiguous float32 numpy array."""
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


def _u64_ptr(arr):
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64))


class ZectorDB:
    """Ultra-fast vector database backed by Zig HNSW + IVF engine."""

    def __init__(self, dim, max_elements=100_000, m=32, ef_construction=400):
        self.dim = dim
        self._ptr = _lib.zector_init(dim, max_elements, m, ef_construction)
        if not self._ptr:
            raise RuntimeError("Failed to initialize ZectorDB")

    def add(self, vector):
        """Add a single vector. Returns its index."""
        vec = np.ascontiguousarray(vector, dtype=np.float32).ravel()
        if vec.shape[0] != self.dim:
            raise ValueError(f"Expected dimension {self.dim}, got {vec.shape[0]}")
        idx = _lib.zector_add(self._ptr, _f32_ptr(vec))
        if idx < 0:
            raise RuntimeError("Failed to add vector")
        return idx

    def add_batch(self, vectors):
        """Add multiple vectors at once (much faster than repeated add()).

        Args:
            vectors: numpy array of shape (n, dim), dtype float32
        """
        vecs = np.ascontiguousarray(vectors, dtype=np.float32)
        if vecs.ndim == 1:
            vecs = vecs.reshape(1, -1)
        if vecs.shape[1] != self.dim:
            raise ValueError(f"Expected dimension {self.dim}, got {vecs.shape[1]}")
        n = vecs.shape[0]
        rc = _lib.zector_add_batch(self._ptr, _f32_ptr(vecs), n)
        if rc < 0:
            raise RuntimeError("Failed to add batch")

    def build_index(self):
        """Rebuild the turbo IVF/PQ index. Call after bulk insertion."""
        if _lib.zector_build_index(self._ptr) < 0:
            raise RuntimeError("Failed to build index")

    def set_search_ef(self, ef):
        """Set search ef parameter. Higher = better recall, slower."""
        _lib.zector_set_search_ef(self._ptr, ef)

    def set_turbo_enabled(self, enabled):
        """Enable or disable the IVF turbo path."""
        _lib.zector_set_turbo_enabled(self._ptr, bool(enabled))

    def search(self, query, k=10):
        """Find k nearest neighbors.

        Returns:
            (ids, distances) - numpy arrays of shape (found,)
        """
        q = np.ascontiguousarray(query, dtype=np.float32).ravel()
        if q.shape[0] != self.dim:
            raise ValueError(f"Expected dimension {self.dim}, got {q.shape[0]}")
        out_ids = np.zeros(k, dtype=np.uint64)
        out_dists = np.zeros(k, dtype=np.float32)
        found = _lib.zector_search(self._ptr, _f32_ptr(q), k, _u64_ptr(out_ids), _f32_ptr(out_dists))
        if found < 0:
            raise RuntimeError("Search failed")
        return out_ids[:found], out_dists[:found]

    def batch_search(self, queries, k=10):
        """Search multiple queries. Returns lists of (ids, dists) per query."""
        qs = np.ascontiguousarray(queries, dtype=np.float32)
        if qs.ndim == 1:
            qs = qs.reshape(1, -1)
        results = []
        for i in range(qs.shape[0]):
            results.append(self.search(qs[i], k))
        return results

    @property
    def count(self):
        return _lib.zector_count(self._ptr)

    def __len__(self):
        return self.count

    def __del__(self):
        if hasattr(self, "_ptr") and self._ptr:
            _lib.zector_free(self._ptr)
            self._ptr = None

    def __repr__(self):
        return f"ZectorDB(dim={self.dim}, count={self.count})"


if __name__ == "__main__":
    import time

    print("=== Zector Python Binding Test ===\n")

    dim = 128
    n_vectors = 50_000
    k = 10

    # Initialize
    db = ZectorDB(dim=dim, max_elements=n_vectors, m=32, ef_construction=400)
    print(f"Initialized: {db}")

    # Generate clustered data
    print(f"Generating {n_vectors} vectors...")
    rng = np.random.default_rng(42)
    n_clusters = 100
    centroids = rng.standard_normal((n_clusters, dim)).astype(np.float32)
    labels = rng.integers(0, n_clusters, size=n_vectors)
    noise = rng.standard_normal((n_vectors, dim)).astype(np.float32) * 0.1
    vectors = centroids[labels] + noise
    # Normalize
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = (vectors / norms).astype(np.float32)

    # Batch insert
    print("Inserting vectors...")
    t0 = time.perf_counter()
    db.add_batch(vectors)
    t_insert = time.perf_counter() - t0
    print(f"Inserted {n_vectors} vectors in {t_insert:.2f}s ({n_vectors/t_insert:.0f} vectors/sec)")

    # Build turbo index
    print("Building turbo index...")
    t0 = time.perf_counter()
    db.build_index()
    t_build = time.perf_counter() - t0
    print(f"Index built in {t_build:.2f}s")
    print(f"Total: {db}")

    # Search
    db.set_search_ef(128)
    query = vectors[0]

    # Warmup
    db.search(query, k)

    # Benchmark
    n_queries = 1000
    queries = vectors[rng.integers(0, n_vectors, size=n_queries)]
    t0 = time.perf_counter()
    for i in range(n_queries):
        db.search(queries[i], k)
    t_search = time.perf_counter() - t0

    qps = n_queries / t_search
    avg_us = t_search / n_queries * 1e6
    print(f"\nSearch benchmark ({n_queries} queries, k={k}):")
    print(f"  {qps:.0f} QPS  ({avg_us:.0f} us/query)")

    # Verify results
    ids, dists = db.search(query, k=5)
    print(f"\nTop-5 results for query[0]:")
    for i, (idx, d) in enumerate(zip(ids, dists)):
        print(f"  #{i+1}  ID={idx}  dist={d:.4f}")
