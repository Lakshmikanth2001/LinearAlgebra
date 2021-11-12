from src.main import solve
from src.det import echlon_form

if __name__ == "__main__":
    A = [[1, 2, 9], [6, -6, 10], [1, 7, 8]]
    y = [8, 0, -1]
    x = solve(A, y)
    print(x)