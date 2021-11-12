from src import fast_expo
import pytest
from src.det import det
from src.det import echlon_form
from src.main import solve
from src.inverse import inverse
import numpy as np
import numpy.testing as npt


class TestBasic:

    @pytest.mark.skip(reason="floating point arithmetic is not working")
    def test_fast_expo(self):
        from src import fast_expo
        from numpy.testing import assert_almost_equal
        j = 1
        while j < 100:
            for i in range(100):
                try:
                    assert_almost_equal(fast_expo(j, i), pow(j, i), decimal=6)
                except Exception as e:
                    print(j, i)
                    raise e
            j = j + 0.1


class TestPoly:

    def test_horner_method(self):
        from src.polynomial import Polynomial
        p1 = Polynomial([1, 5, 6])
        p2 = Polynomial([1, 0, -1])

        assert p1(-2) == 0 and p1(-3) == 0
        assert p2(1) == 0 and p2(-1) == 0


class TestDet:

    def test_0(self):
        g = [[1, 0, 2, -1],
            [3, 0, 0, 5],
            [2, 1, 4, -3],
            [1, 0, 5, 0]]
        assert det(g, 4) == 30

    def test_1(self):
        g1 = [[1, 2, 3], [4, 5, 6], [7, 10, 9]]
        assert det(g1, 3) == 12

    def test_2(self):
        g2 = [[21, 17, 7, 10], [24, 22, 6, 10], [6, 8, 2, 3], [6, 7, 1, 2]]
        assert det(g2, 4) == -24

    def test_echlon_form(self):
        g = [[1, 0, 2, -1],
            [3, 0, 0, 5],
            [2, 1, 4, -3],
            [1, 0, 5, 0]]

        size = len(g)
        e = echlon_form(g, len(g), len(g))

        for i in range(size - 1, -1, -1):
            for j in range(0, i - 1, 1):
                assert e[i][j] == 0

        e_det = 1
        for i in range(len(g)):
            e_det = e_det * e[i][i]
        assert det(g, len(g)) == e_det


class TestInv:

    def test_0(self):
        m = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        size = len(m)
        m_inv = inverse(m, size)
        assert np.array_equal(m, m_inv)

    def test_1(self):
        m = [[3, 0, 0], [0, 3, 0], [0, 0, 3]]
        m_det = det(m, len(m))
        size = len(m)
        m_inv = inverse(m, size)

        m = np.array(m)
        m_inv = np.array(m_inv)
        assert np.array_equal(m / 9, m_inv)


class TestSolve:

    @staticmethod
    def solve_test_helper(matrix, x, y):
        size = len(matrix)

        for i in range(size):
            row_sum = 0
            for j in range(size):
                row_sum = row_sum + matrix[i][j] * x[j]
            npt.assert_almost_equal(y[i], row_sum)

    @staticmethod
    def test_random():
        for _ in range(10000):
            matrix = np.random.random(size=(9, 9))
            y = np.random.random(9)
            x = solve(matrix, y)
            TestSolve.solve_test_helper(matrix, x, y)

    def test_0(self):
        matrix = [[1, 1, 1], [2, 5, 8], [6, -6, 7]]
        y = [5, 7, 10]
        x = solve(matrix, y)
        TestSolve.solve_test_helper(matrix, x, y)

    def test_1(self):
        matrix = [[1, 1, 1], [2, 5, 8], [6, 6, 7]]
        y = [5, 7, 10]
        x = solve(matrix, y)
        TestSolve.solve_test_helper(matrix, x, y)

    @pytest.mark.skip(reason="inconsistent system")
    def test_2(self):
        pass
        matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        y = [10, 11, 12]
        x = solve(matrix, y)
        TestSolve.solve_test_helper(matrix, x, y)

    @pytest.mark.skip(reason="inconsistent system")
    def test_3(self):
        for _ in range(10):
            matrix = np.random.randint(0, 100, size=(9, 9))
            y = np.random.randint(0, 100, size=9)
            x = solve(matrix, y)
            TestSolve.solve_test_helper(matrix, x, y)


def signal_test_helper(xn, nature):
    from src.digital_signal import dft, inverse_dft
    xk = dft(xn)
    xn_constructed = inverse_dft(xk, signal_nature=nature)
    npt.assert_almost_equal(xn, xn_constructed)


class TestDigitalSignal:

    def test_0(self):
        xn = [0, 0.756, 0.788, 0.256, 0.778]
        signal_test_helper(xn, nature='real')

    def test_1(self):
        xn = [-1j, 0.47j, 0.11j, 1.1j, -2j, -1.47j, 0.11j]
        signal_test_helper(xn, nature='imaginary')

    def test_2(self):
        theta = np.arange(0, 2*np.pi, 150)
        f1 = 10
        xn = np.sin(2*np.pi*f1*theta)
        f2 = 20
        xn += np.sin(2*np.pi*f2*theta)
        f3 = 30
        xn += np.sin(2*np.pi*f3*theta)
        signal_test_helper(xn, nature='real')
