from typing import Callable, Match, Union
import numpy as np


def integral(function: Callable, a: Union[float, complex], b: Union[float, complex],
            step_size: float = 1e-6, method = "trapizodial") -> Union[float, complex]:
    """
    Calculates the integral of a function using the given method.

    :param method: The method to use for the integration.
    :param a: The lower bound of the integral.
    :param b: The upper bound of the integral.
    :param n: The number of subintervals to use.
    :return: The integral of the function.
    """

    start = a
    answer = 0

    while start < b:
        
        y0 = function(start)
        y1 = function(start + step_size)

        match method:

            case "trapizodial":
                answer += (y0 + y1) * step_size / 2

            case "simpson":
                y2 = function(start + step_size / 2)
                answer += (y0 + 4 * y2 + y1) * step_size / 6

            case "simpson3":
                y2 = function(start + step_size / 3)
                y3 = function(start + step_size / 3 * 2)
                answer += (y0 + 3 * y2 + 3 * y3 + y1) * step_size / 8

            case _:
                raise ValueError("Invalid method")

        start += step_size 

    return answer

if __name__ == "__main__":

    print(integral(lambda x: np.sqrt(1 + 4*x*x), 1, 2, method = "simpson3"))