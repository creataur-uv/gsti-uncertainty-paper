import torch


class PCA_SVD:
    def __init__(self, n_components):
        self.n_components = n_components
        self.mean = None
        self.components = None
        self.singular_values = None
        self.explained_variance = None
        self.explained_variance_ratio = None

    def fit(self, X):
        """
        X: tensor of shape (n_samples, n_features)
        """

        # Center data
        self.mean = torch.mean(X, dim=0)
        X_centered = X - self.mean

        # Singular Value Decomposition
        U, S, Vt = torch.linalg.svd(X_centered, full_matrices=False)

        # Principal directions
        self.components = Vt[:self.n_components]

        # Singular values
        self.singular_values = S[:self.n_components]

        # Explained variance
        n_samples = X.shape[0]

        explained_variance = (S ** 2) / (n_samples - 1)

        self.explained_variance = explained_variance[:self.n_components]

        # Explained variance ratio
        total_variance = explained_variance.sum()

        self.explained_variance_ratio = (
            self.explained_variance / total_variance
        )

    def transform(self, X):
        """
        Project data into PCA space
        """

        X_centered = X - self.mean

        return torch.matmul(X_centered, self.components.T)

    def inverse_transform(self, Z):
        """
        Reconstruct original data
        """

        return torch.matmul(Z, self.components) + self.mean

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


# ---------------------------------------------------
# Example
# ---------------------------------------------------
## For testing only
if __name__ == "__main__":
    # Random dataset
    X = torch.randn(1000, 50)

    # Reduce to 5 dimensions
    pca = PCA_SVD(n_components=5)

    # Fit and transform
    Z = pca.fit_transform(X)

    print("Reduced shape:", Z.shape)

    print("\nExplained variance:")
    print(pca.explained_variance)

    print("\nExplained variance ratio:")
    print(pca.explained_variance_ratio)

    # Reconstruction
    X_reconstructed = pca.inverse_transform(Z)

    print("\nReconstructed shape:", X_reconstructed.shape)