import numpy as np


def mod_x(x):
    return np.sum(np.abs(x), axis=1, keepdims=True)


def l1_norm(x):
    return np.sum(np.abs(x), axis=1, keepdims=True)


def l2_norm(x):
    return np.sqrt(np.sum(x, axis=1, keepdims=True)**2)


def mod_function(x):
    return np.sum((3 / 4) * np.abs(x + 5) + (1 / 2) * np.abs(x - 5) +
                  (3 / 2) * np.abs(x - 10),
                  keepdims=True,
                  axis=1)


def mod_function2(x):
    return np.sum(np.abs(x + 3) + np.abs(x + 9) + np.abs(x + 5),
                  keepdims=True,
                  axis=1)


def combination_function(x):
    return np.max(np.hstack(
        [mod_x(x), x**2, mod_function(x),
         mod_function2(x)]),
                  keepdims=True,
                  axis=1)


breakpoint()