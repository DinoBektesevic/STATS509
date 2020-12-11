import numpy as np
import matplotlib.pyplot as plt


def problem6c():
    x = np.arange(-1.5, 1.5, 0.01)
    f = lambda n: np.exp(-n/2.0 * x**2)
    for i in range(1, 10):
        plt.plot(x, f(i), label=f"n={i}")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    problem6c()
