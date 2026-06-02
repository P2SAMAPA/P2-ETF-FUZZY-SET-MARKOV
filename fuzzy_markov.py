import numpy as np
from sklearn.preprocessing import StandardScaler

def fuzzy_cmeans(X, n_clusters=3, m=2.0, max_iter=50, tol=1e-4):
    """
    Fuzzy C-Means clustering.
    Returns:
        U: membership matrix (n_samples x n_clusters)
        centers: cluster centers (n_clusters x n_features)
    """
    n_samples, n_features = X.shape
    # Random initialisation of membership matrix
    U = np.random.rand(n_samples, n_clusters)
    U = U / U.sum(axis=1, keepdims=True)
    for _ in range(max_iter):
        # Compute cluster centers
        Um = U ** m
        centers = (Um.T @ X) / Um.sum(axis=0)[:, np.newaxis]
        # Compute distances
        dist = np.zeros((n_samples, n_clusters))
        for j in range(n_clusters):
            diff = X - centers[j]
            dist[:, j] = np.linalg.norm(diff, axis=1)
        # Update membership
        for i in range(n_samples):
            for j in range(n_clusters):
                if dist[i, j] == 0:
                    U[i, j] = 1.0
                else:
                    denom = np.sum((dist[i, j] / dist[i, :]) ** (2/(m-1)))
                    U[i, j] = 1.0 / denom
        # Check convergence
        if _ > 0 and np.linalg.norm(U - U_old) < tol:
            break
        U_old = U.copy()
    return U, centers

def fuzzy_markov_score(returns, n_clusters=3, fuzziness=2.0, max_iter=50):
    """
    For each ETF (univariate), compute the membership in the highest‑variance fuzzy regime.
    We first cluster the returns into n_clusters fuzzy regimes based on their values.
    Then for the last day's return, compute membership in the cluster with highest variance.
    Score = that membership probability.
    """
    # returns is a pandas Series
    X = returns.values.reshape(-1, 1)
    # Standardise (optional)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    U, centers = fuzzy_cmeans(X_scaled, n_clusters, fuzziness, max_iter)
    # Compute variance of each cluster (in original scale? we use scaled)
    cluster_var = np.zeros(n_clusters)
    for j in range(n_clusters):
        # Weighted variance: sum U_ij * (x_i - c_j)^2 / sum U_ij
        weights = U[:, j]
        weighted_var = np.average((X_scaled.flatten() - centers[j, 0])**2, weights=weights)
        cluster_var[j] = weighted_var
    # Index of highest variance cluster
    high_var_idx = np.argmax(cluster_var)
    # Membership of the last observation (today)
    last_membership = U[-1, high_var_idx]
    return float(last_membership)
