################################################################################
"""
`electricpy.power_flow`  - power_flow.

>>> from electricpy import power_flow
"""
################################################################################

import numpy as np
from typing import Dict, Tuple


def compute_power(V_bus: np.ndarray, Y_bus: np.ndarray):
    r"""Compute Complex Power at each bus."""
    n = len(V_bus)

    assert V_bus.dtype == Y_bus.dtype
    assert np.shape(Y_bus) == (n, n)

    try:
        assert V_bus.dtype == np.complex
    except AssertionError:
        V_bus = V_bus.astype(np.complex)

    V_bus = V_bus.reshape(n, 1)

    I_bus = np.matmul(Y_bus, V_bus)

    S_bus = I_bus.conjugate() * V_bus

    return S_bus


def build_Y_bus(y: Dict[Tuple[str, str], complex]):
    """Build the Y bus matrix."""
    # build the Y bus matrix

    largest_bus = max(max(bus) for bus in y.keys())

    Y_bus = np.zeros((largest_bus, largest_bus), dtype=np.complex)

    y_sum = dict()

    for key in y.keys():
        bus_0, bus_1 = key
        bus_0 += -1
        bus_1 += -1

        Y_bus[bus_0, bus_1] = -y[key]
        Y_bus[bus_1, bus_0] = -y[key]

        y_sum[bus_0] = y_sum.get(bus_0, 0) + y[key]
        y_sum[bus_1] = y_sum.get(bus_1, 0) + y[key]

    for i in range(largest_bus):
        Y_bus[i, i] = y_sum.get(i, 0)

    return Y_bus


def compute_bus_power(V_bus: np.ndarray, Y_bus: np.ndarray, bus_index: int):
    """Compute Complex Power at a particular bus_index."""
    n = len(V_bus)

    assert V_bus.dtype == Y_bus.dtype
    assert np.shape(Y_bus) == (n, n)

    try:
        assert V_bus.dtype == np.complex
    except AssertionError:
        V_bus = V_bus.astype(np.complex)

    V_bus = V_bus.reshape(n, 1)

    Y = Y_bus[bus_index, :]

    Y = Y.reshape(1, n)

    I_bus = np.matmul(Y_bus, V_bus)

    S_bus = I_bus * V_bus.conjugate()

    return S_bus


def gauss_seidel(V_bus: np.ndarray, Y_bus: np.ndarray, S_bus: np.ndarray, bus_remarks: Dict[int, str],
                Q_limits: Dict[int , str] = {}, max_iter: int = 10, tol: float = 1e-6,
                power_base: float = 100, accelaration: float = 1.6):
    """Solve the power flow using the Gauss-Seidel method."""
    n = len(V_bus)

    assert V_bus.dtype == Y_bus.dtype
    assert V_bus.dtype == S_bus.dtype
    assert np.shape(Y_bus) == (n, n)

    try:
        assert V_bus.dtype == np.complex
    except AssertionError:
        V_bus = V_bus.astype(np.complex)

    V_bus = V_bus.reshape(n, 1)

    S_bus = S_bus.reshape(n, 1) / power_base

    V_bus_old = V_bus.copy()

    def compute_voltage_phasor(bus_index, V_bus_old):
        """Compute the voltage phasor at a particular bus_index."""
        k = np.matmul(Y_bus[bus_index, :], V_bus_old)

        v_bus = (S_bus[bus_index, 0] / V_bus_old[bus_index, 0].conjugate()) - \
            k.sum() + V_bus_old[bus_index, 0]*Y_bus[bus_index, bus_index]

        v_bus = v_bus / Y_bus[bus_index, bus_index]

        return v_bus

    for i in range(max_iter):

        for j in range(1, n):

            if bus_remarks[j] == 'Slack Bus':
                pass

            elif bus_remarks[j] == 'PQ Bus':

                V_bus[j, 0] = compute_voltage_phasor(j, V_bus_old)

            elif bus_remarks[j] == 'PV Bus':

                Q = np.imag(compute_bus_power(V_bus_old, Y_bus, j))

                if Q_limits.get(j, 0) != 0:
                    Q_max = abs(Q_limits[j])
                    Q_min = abs(Q_limits[j])

                    Q = abs(Q)

                    if Q < Q_max and Q > Q_min:
                        pass
                    elif Q > Q_max:
                        Q = Q_max
                        S_bus[j, 0] = np.real(S_bus[j, 0]) - Q_max
                    elif Q < Q_min:
                        Q = Q_min
                        S_bus[j, 0] = np.real(S_bus[j, 0]) - Q_min

            V_bus[j, 0] = V_bus_old[j, 0] + accelaration * \
                (V_bus[j, 0] - V_bus_old[j, 0])

            V_bus_old = V_bus.copy()

        yield V_bus


if __name__ == "__main__":

    S_bus = np.array([
        -50 + 30.99j,
        -170 + 105.35j,
        -200 + 123.94j,
        238 + 49.58j
    ])

    y = {
        (1, 2): 3.815629 - 19.078144j,
        (1, 3): 5.169561 - 25.847809j,
        (2, 4): 5.169561 - 25.847809j,
        (3, 4): 3.023705 - 15.118528j,
    }

    Y_bus = build_Y_bus(y)

    V_bus = np.array([
        1.0,
        1.0,
        1.0,
        1.02,
    ], dtype=np.complex)

    bus_remarks = {
        0: "Slack Bus",
        1: "PQ Bus",
        2: "PQ Bus",
        3: "PQ Bus",
    }

    # set numpy print precision
    np.set_printoptions(precision=3)

    stack = list()

    for voltages in gauss_seidel(V_bus, Y_bus, S_bus, bus_remarks, max_iter=100, tol=1e-6):

        data = voltages.copy()

        print(data)

        if not stack:
            stack.append(data)
        else:
            voltages_old = stack.pop()
            stack.append(voltages.copy())
