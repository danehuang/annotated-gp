import jax
from jax import random, Array
import jax.numpy as jnp
import matplotlib.pyplot as plt
from typing import *


# =================================================
# Toy datasets
# =================================================

def create_toy_dataset(f: Callable[[Array], Array]) -> tuple[Array, Array]:
    # Inputs
    xs = jnp.array([   # inputs
        [-2.],         # x_1
        [1],           # x_2
        [2.],          # x_3
    ])

    # Outputs
    ys = jax.vmap(f)(xs)

    return xs, ys


def create_toy_sparse_dataset(f: Callable, key) -> tuple[Array, Array, Array, Array]:
    # Data points
    N = 10
    xs = jnp.linspace(-3, 3, N).reshape(-1, 1)

    # Inducing points
    M = 4
    zs = jnp.linspace(-3, 3, M)

    # Function to approximate
    f = lambda x: jnp.sin(x)

    # Noise
    Sigma = jnp.diag(1e-2 * jnp.ones(N))
    es = random.multivariate_normal(key, mean=jnp.zeros(N), cov=Sigma, shape=(1,)).transpose()

    # Observations
    ys = jax.vmap(f)(xs) + es  # outputs
    return xs, zs, ys, Sigma


def create_toy_deriv_dataset(f: Callable, key):
    N = 7; D = 1
    xs = jnp.linspace(-3, 3, N).reshape(-1, 1)

    # 1a. Sample random function
    def f(x: Array) -> float:
        return jnp.sin(x)

    # 1b. Associated gradient
    grad_f = jax.jacrev(f)

    # 2. Sample noise
    Sigma = jnp.diag(jnp.array([1e-4, 1e-2]))
    es = random.multivariate_normal(key, mean=jnp.zeros(1 + D), cov=Sigma, shape=(N,))

    # 3. Produce observations
    ys = jnp.array([f(x) + e[0] for x, e in zip(xs, es)])                   # outputs
    gs = jnp.array([grad_f(x).reshape(-1) + e[1] for x, e in zip(xs, es)])
    
    return xs, ys, gs, Sigma


def plot_deriv_dataset(xs: Array, ys: Array, gs: Array, f: Callable) -> None:
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1); plt.axis("equal") 
    plt.plot(xs, ys, marker="o", linestyle="None", label="Dataset")
    plt.plot(jnp.linspace(-3, 3), f(jnp.linspace(-3, 3)), label="f")
    plt.title("Dataset"); plt.xlabel("X"); plt.ylabel("Y"); plt.legend();

    plt.subplot(1, 2, 2); plt.axis("equal")
    origin = jnp.array([xs, ys])
    plt.plot(xs, ys, marker="o", linestyle="None", label="Dataset")
    plt.plot(jnp.linspace(-3, 3), f(jnp.linspace(-3, 3)), label="f")
    plt.quiver(*origin, jnp.ones(len(xs)), gs, label="Gradient")
    plt.title("Dataset with Derivatives"); plt.xlabel("X"); plt.ylabel("Y"); plt.legend();

# =================================================
# Plotting
# =================================================

class PlotContext:
    def __init__(self, title="Plot Title", xlabel="X-axis", ylabel="Y-axis"):
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel

    def __enter__(self):
        self.fig, self.ax = plt.subplots(figsize=(8, 4))
        self.ax.axis("equal")
        self.ax.set_title(self.title)
        self.ax.set_xlabel(self.xlabel)
        self.ax.set_ylabel(self.ylabel)
        return self.ax

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is not None:
            print(f"An error occurred: {exc_value}")
        self.ax.legend()
        # self.ax.grid(True)
        plt.show()
        # Return False to propagate exceptions, True to suppress
        return False


# =================================================
# Mean
# =================================================

def mu(x: Array) -> float:
    """Constant mean function.
    """
    return 0


def mk_mean(xs: Array) -> Array:
    return jax.vmap(mu)(xs)


# =================================================
# Covariance 
# =================================================

def k(x: Array, y: Array, weight=1.0, length_scale=1.0) -> float:
    """The squared-exponential kernel function.
    """
    scaled_diff = (x - y) / length_scale
    radius2 = jnp.dot(scaled_diff, scaled_diff)
    c = (weight * weight)
    e = jnp.exp(-0.5 * radius2)
    return c * e


def mk_cov(k: Callable, xs1: Array, xs2: Array) -> Array:
    return jnp.stack([jnp.stack([k(x1, x2) for x2 in xs2]) for x1 in xs1])


# ------------------------------=
# Kernel Solving 
# ------------------------------=

def covariance_solve(K, b):
    L = jnp.linalg.cholesky(K)
    y = jax.scipy.linalg.solve_triangular(L, b, lower=True)
    return jax.scipy.linalg.solve_triangular(L.transpose(), y)


def cholesky_inv(K):
    L_K = jnp.linalg.cholesky(K)
    inv_L = jnp.linalg.inv(L_K)
    return inv_L.T @ inv_L


# ------------------------------=
# Kernel with Derivatives 
# ------------------------------=

def kern_blk(k: Callable, x1: Array, x2: Array) -> Array:
    kern = jnp.array([k(x1, x2)])
    jac2 = jax.jacrev(k, argnums=1)(x1, x2)
    f_jac1 = jax.jacrev(k, argnums=0)
    jac1 = f_jac1(x1, x2).reshape(-1, 1)
    # Using forward-mode AD with reverse-mode AD to get the second derivative
    hes = jax.jacfwd(f_jac1, argnums=1)(x1, x2)

    # Put everything together
    top = jnp.concatenate([kern, jac2]).reshape(1, -1)
    bot = jnp.concatenate([jac1, hes], axis=1)
    K = jnp.concatenate([top, bot])
    return K


def mk_cov_blk(k: Callable, xs1: Array, xs2: Array) -> Array:
    return jnp.concatenate([
        jnp.concatenate([
            kern_blk(k, x1, x2) for x2 in xs2
        ], axis=1) for x1 in xs1
    ])


# =================================================
# Utility 
# =================================================

def flatten(lists):
    flat_list = []
    for row in lists:
        flat_list += row
    return flat_list
