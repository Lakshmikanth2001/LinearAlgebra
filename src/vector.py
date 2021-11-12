import numpy as np


def magnitude(v):
    mag = 0

    for x in v:
        mag = mag + x * x

    return np.sqrt(mag)


def vector_norm(v):
    mag = magnitude(v)

    return [x / mag for x in v]


def dot(va, vb):
    assert len(va) == len(vb)

    v_sum = 0

    for x, y in zip(va, vb):
        v_sum = v_sum + x * y

    return v_sum


def cross(va, vb):
    assert len(va) == len(vb)
