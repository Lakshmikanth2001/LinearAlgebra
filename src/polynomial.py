import numpy as np
from typing import Union

def horner_method(poly, index, x):
    if index == 0:
        return poly[index]
    else:
        return poly[index] + x * horner_method(poly, index - 1, x)


class Polynomial:
    def __init__(self, poly: list):
        self.poly = poly
        self.size = len(poly)

    @property
    def degree(self):
        return self.size-1

    def __len__(self):
        return self.size

    def __iter__(self):
        return iter(self.poly)

    def __add__(self, poly_1):
        assert poly_1.size == self.size
        return [x + y for x, y in zip(self.poly, poly_1)]

    def __sub__(self, poly_1):
        assert poly_1.size == self.size
        return [x - y for x, y in zip(self.poly, poly_1)]

    def __mul__(self, poly_1):

        deg_1 = self.size - 1
        deg_2 = self.size - 1

        deg_req = deg_1 + deg_2

        k = 1
        while k < deg_req:
            k = 2 * k

        theta = 2 * np.pi * 1j / k

        # polynomials in point value representation
        poly_1_c = []
        poly_2_c = []

        for i in range(k):
            w = np.exp(theta * i)
            poly_1_c.append(self(w))
            poly_2_c.append(poly_1(w))

        print(poly_1_c)
        print(poly_2_c)

    def __call__(self, x):
        return horner_method(self.poly, self.size - 1, x)

    def solve(self, iterations=100):

        degree = self.size - 1
        roots = [i for i in np.arange(-10, 10, 0.01)]
        diff_poly = Polynomial([self.poly[i]*(degree - i) for i in range(degree)])

        for i in range(len(roots)):
            for it in range(iterations):
                roots[i] = roots[i] - self(roots[i])/diff_poly(roots[i])

        return np.around(roots, 3)


if __name__ == "__main__":
    p1 = Polynomial([-2, 3, -4])
    p2 = Polynomial([4, 5, 6])
    p3 = Polynomial([1, -9, 26, -24])
    p4 = Polynomial([1, 0, 2])

    print(p1.solve())
