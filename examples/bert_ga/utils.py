import numpy as np

DATASET_PATH = "datasets"


def linear_gen_weight(gen):
    # add 1 for smoothing
    return gen + 1


def square_gen_weight(gen):
    # add 1 for smoothing
    return (gen + 1) ** 2


def exp_gen_weight(gen):
    return np.e ** gen


def log_gen_weight(gen):
    # add 2 for smoothing
    return np.log(gen + 2)


def sqrt_gen_weight(gen):
    # add 1 for smoothing
    return (gen + 1) ** 0.5


def cosine_similarity(a, b):
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b)
