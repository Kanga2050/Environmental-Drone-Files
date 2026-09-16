import numpy as np


def fit(a, b):
    ma, mb = a.mean(0), b.mean(0)
    A, B = a - ma, b - mb
    U, S, Vt = np.linalg.svd(B.T @ A / len(a))
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(U @ Vt))
    R = U @ D @ Vt
    s = np.trace(np.diag(S) @ D) / (A ** 2).sum(1).mean()
    return s, R, mb - s * R @ ma


def robust(a, b, it=4, keep=0.8):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.ones(len(a), bool)
    for _ in range(it):
        s, R, t = fit(a[m], b[m])
        r = np.linalg.norm(a @ R.T * s + t - b, axis=1)
        m = r <= np.quantile(r, keep)
    s, R, t = fit(a[m], b[m])
    r = np.linalg.norm(a @ R.T * s + t - b, axis=1)
    return (s, R, t), float(np.median(r[m]))


def apply(T, p):
    s, R, t = T
    return p @ R.T * s + t
