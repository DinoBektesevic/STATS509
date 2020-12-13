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


def problem7a():
    theta = np.arange(0, 6, 1)
    t = np.arange(0, 8, 1)

    # f vs theta plot for different t
    for i in t:
        constT = np.exp(-(i -theta))
        mask = i >= theta
        plt.plot(theta[mask], constT[mask], label=f"t={i}")
    plt.xlabel(r"$\theta$")
    plt.ylabel(r"$f(t|\theta)$")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # f v t plot for different theta
    for i in theta:
        constTheta = np.exp(-(t - i))
        mask = t >= i
        plt.plot(t[mask], constTheta[mask], label=fr"$\theta={i}$")
    plt.xlabel(r"t")
    plt.ylabel(r"$f(t|\theta)$")
    plt.legend()
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    problem6c()
    problem7a()
