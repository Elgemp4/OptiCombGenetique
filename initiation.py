import numpy as np
from scipy.linalg import lu, qr

from solution import Solution
from sklearn.decomposition import NMF, TruncatedSVD, PCA, FastICA

def initiate_algo(X, m, n, rank, lower_w, higher_w, lower_h, higher_h, method):
    if method == 'nmf':
        # If NMF is not available (because there is a negative value in X, then use SVD)
        if (X < 0).any():
            svd = TruncatedSVD(n_components=rank)
            W_float = svd.fit_transform(X)
            Sigma = np.diag(svd.singular_values_)
            H_float = svd.components_

            W_float = W_float @ np.sqrt(Sigma)
            H_float = np.sqrt(Sigma) @ H_float
        else:
            model = NMF(n_components=rank, init='random', max_iter=50000)
            W_float = model.fit_transform(X)
            H_float = model.components_

    elif method == 'svd':
        svd = TruncatedSVD(n_components=rank)
        W_float = svd.fit_transform(X)
        Sigma = np.diag(svd.singular_values_)
        H_float = svd.components_

        W_float = W_float @ np.sqrt(Sigma)
        H_float = np.sqrt(Sigma) @ H_float
    elif method == 'pca':
        pca = PCA(n_components=rank)
        W_float = pca.fit_transform(X)
        H_float = pca.components_
    elif method == 'ica':
        ica = FastICA(n_components=rank, max_iter=1000, random_state=42)
        W_float = ica.fit_transform(X)  # Forme (m, rank)
        H_float = ica.mixing_.T  # Forme (rank, n)
    elif method == "lu":
        P, L, U = lu(X)

        W_full = P @ L

        W_float = W_full[:, :rank]  # Truncation of the array
        H_float = U[:rank, :]  # Truncation of the array

    elif method == "qr":
        Q, R = qr(X, mode='economic')

        W_float = Q[:, :rank]  # Truncation of the array
        H_float = R[:rank, :]  # Trunction of the array
    else:
        W_float = np.random.randint(lower_w, higher_w + 1, (m, rank))
        H_float = np.random.randint(lower_h, higher_h + 1, (rank, n))

    W_entier = np.round(W_float).astype(int)
    H_entier = np.round(H_float).astype(int)

    W_clamped = np.clip(W_entier, lower_w, higher_w)
    H_clamped = np.clip(H_entier, lower_h, higher_h)

    sol = Solution(W_clamped, H_clamped)
    sol.compute_score(X)
    print("Method : ", method, "Score : ", sol.score)
    return sol