import os
from itertools import combinations

import matplotlib.pyplot as plt
import openml
import pandas as pd
import seaborn as sns
from openml import OpenMLDataset
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from sklearn.decomposition import PCA
from config import Config


def create_assets_folder() -> None:
    """
    Create the assets folder if it doesn't exist.
    """

    if not os.path.exists(Config.DIR_ASSETS):
        os.makedirs(Config.DIR_ASSETS)

    if not os.path.exists(Config.DIR_EDA):
        os.makedirs(Config.DIR_EDA)

    if not os.path.exists(Config.DIR_PCA):
        os.makedirs(Config.DIR_PCA)


class EDA:
    """
    Class for Exploratory Data Analysis (EDA) on a dataset.
    """

    def __init__(
        self,
        dataset: OpenMLDataset | None = None,
        X: pd.DataFrame | None = None,
        y: pd.Series | None = None,
        X_train: pd.DataFrame | None = None,
        X_test: pd.DataFrame | None = None,
        y_train: pd.Series | None = None,
        y_test: pd.Series | None = None,
    ):
        """
        Initialize the EDA class.

        On initialization, assets folders are created if they don't exist, following Config class.
        :param dataset: OpenMLDataset object
        :param X: Features
        :param y: Target variable
        """
        self.dataset: OpenMLDataset = dataset
        self.X: pd.DataFrame = X
        self.y: pd.Series = y

        self.X_train: pd.DataFrame = pd.DataFrame()
        self.X_test: pd.DataFrame = pd.DataFrame()
        self.y_train: pd.Series = pd.Series()
        self.y_test: pd.Series = pd.Series()

        # Create the assets folder - if it doesn't exist
        create_assets_folder()

    def load_dataset(self) -> None:
        """
        Load a dataset from OpenML.
        Source of dataset: OpenML - https://www.openml.org/search?type=data&status=active&id=43582&sort=runs
        """
        # Load the dataset from local repository
        self.dataset = openml.datasets.get_dataset(Config.DATASET_ID)
        self.X, self.y, _, _ = self.dataset.get_data(target=self.dataset.default_target_attribute)

    def summary_table(self) -> pd.DataFrame:
        """
        Return a summary table of the dataset.
        We show the different features by - Feature, Description
        """
        # Get the feature names and types
        feature_names = self.X.columns
        feature_types = self.X.dtypes

        df = (
            pd.DataFrame(
                {
                    "Feature Name": feature_names,
                    "Type": feature_types,
                    "Distinct Values": [self.X[col].nunique() for col in feature_names],
                    "Missing Values": [self.X[col].isnull().sum() for col in feature_names],
                }
            )
            .sort_values(by="Feature Name")
            .reset_index(drop=True)
        )

        # Describe the dataset with describe() method
        print("----------------------------------")
        print(self.X.describe().to_markdown())
        print("----------------------------------")
        return df

    def plot_visualization(
        self, X: pd.DataFrame | None = None, y: pd.Series | None = None, file_name_suffix=""
    ) -> None:
        """
        Plot visualizations of the dataset.
        1. Histograms / KDE plots
        2. Correlation heatmap
        3. Class distribution (sns.countplot)

        :param file_name_suffix: Suffix for the file name to avoid overwriting
        """

        # Histograms
        X[sorted(X.columns)].hist(figsize=(20, 15), bins=100)
        plt.suptitle("Histograms of the dataset", fontsize=20)
        plt.tight_layout(pad=1)
        plt.savefig(Config.DIR_EDA + f"/histograms{file_name_suffix}.png")
        plt.close()

        # Correlation heatmap
        plt.figure(figsize=(20, 15))
        sns.heatmap(X[sorted(X.columns)].corr(), annot=True, fmt=".2f", cmap="coolwarm")
        plt.title("Correlation heatmap of the dataset", fontsize=20)
        plt.savefig(Config.DIR_EDA + f"/correlation_heatmap{file_name_suffix}.png")
        plt.close()

        # Class distribution
        plt.figure(figsize=(20, 15))
        sns.countplot(x=y, data=X)
        plt.title("Class distribution of the dataset", fontsize=20)
        plt.xticks(rotation=90)
        plt.savefig(Config.DIR_EDA + f"/class_distribution{file_name_suffix}.png")
        plt.close()

        # Boxplots
        # Check for outliers using boxplots
        n_cols = 3
        n_rows = (len(X.columns) + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
        axes = axes.flatten()

        for i, col in enumerate(X.columns):
            sns.boxplot(y=X[col], ax=axes[i])
            axes[i].set_title(f"Boxplot of {col}")

        # Remove empty subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.savefig(Config.DIR_EDA + f"/boxplots{file_name_suffix}.png")
        plt.close()

        # Check for outliers using IQR
        lower_bound, upper_bound = self.outliers_iqr(X)
        outliers = ((X < lower_bound) | (X > upper_bound)).sum()
        print("----------------------------------")
        print("Number of outliers in the dataset:")
        print(outliers)
        print("----------------------------------")

    @staticmethod
    def outliers_iqr(X: pd.DataFrame | None = None) -> (pd.Series, pd.Series):
        """
        Calculate the IQR and return the lower and upper bounds for outliers.

        :return: lower_bound, upper_bound
        """
        if X is None or X.empty:
            raise ValueError("Missing or empty DataFrame provided for outlier detection.")

        Q1 = X.quantile(0.25)
        Q3 = X.quantile(0.75)

        IQR: pd.Series = Q3 - Q1

        lower_bound: pd.Series = Q1 - 1.5 * IQR
        upper_bound: pd.Series = Q3 + 1.5 * IQR

        return lower_bound, upper_bound

    @staticmethod
    def data_check(X: pd.DataFrame | None = None) -> None:
        """
        Check for missing data or anomalous zeros (e.g., Insulin and SkinThickness have zeros which are unrealistic).
        """
        if X is None or X.empty:
            raise ValueError("Missing or empty DataFrame provided for data check.")

        # Check for missing values
        print("----------------------------------")
        print("Missing values in the dataset:")
        print(X.isnull().sum())
        print("----------------------------------")

        # Check for duplicates
        print("--------------------------------")
        print("Duplicate rows in the dataset:")
        print(X.duplicated().sum())
        print("----------------------------------")

        # Check for anomalous zeros in certain columns
        columns_with_invalid_zeros = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

        print("Anomalous zeros (likely invalid) in specific columns:")
        for col in columns_with_invalid_zeros:
            zero_count = (X[col] == 0).sum()
            print(f"{col}: {zero_count} zero(s)")
        print("----------------------------------")

    def split_data(
        self, test_size: float = Config.TEST_SIZE, random_state: int = Config.RANDOM_STATE
    ) -> None:
        """
        Split the dataset into training and testing sets.
        :param test_size: Proportion of the dataset to include in the test split
        :param random_state: Random seed for reproducibility
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state
        )

    def standardize_data(self) -> None:
        """
        Standardize the data (use StandardScaler). Pay attention to potential data leakage when standardizing features.
        """

        if self.X_train.empty or self.X_test.empty:
            raise ValueError(
                "Data must be split into training and testing sets before standardization."
            )

        scaler = StandardScaler()

        columns = self.X_train.columns
        train_idx = self.X_train.index
        test_idx = self.X_test.index

        self.X_train = pd.DataFrame(
            scaler.fit_transform(self.X_train), columns=columns, index=train_idx
        )
        self.X_test = pd.DataFrame(scaler.transform(self.X_test), columns=columns, index=test_idx)
        del self.X, self.y


class K_means:
    """
    Class for K-means clustering.
    """

    def __init__(
        self,
        X: pd.DataFrame | None = None,
    ):
        """
        Initialize the K-means class.
        :param X: Training features
        """
        self.X = X
        self.kmeans = None
        self.cluster_labels = None

    def k_means_clustering(self, k: int = 2) -> None:
        """
        Perform K-means clustering on the dataset.
        :param k: Number of clusters
        """
        self.kmeans = KMeans(n_clusters=k, random_state=Config.RANDOM_STATE)

        self.cluster_labels = self.kmeans.fit_predict(self.X)
        print(f"KMeans clustering applied with k={k}")
        print(pd.Series(self.cluster_labels).value_counts().sort_index())

    def plot_clusters(self, feature_a: str, feature_b: str) -> None:
        """
        Plot the clusters obtained from K-means clustering using two selected features.

        :param feature_a: Feature for the x-axis (e.g., 'Glucose')
        :param feature_b: Feature for the y-axis (e.g., 'BMI')
        """
        if self.kmeans is None or self.cluster_labels is None:
            raise ValueError(
                "KMeans clustering has not been applied yet. Please run k_means_clustering() first."
            )

        if feature_a not in self.X.columns or feature_b not in self.X.columns:
            raise ValueError(
                f"Features '{feature_a}' and '{feature_b}' must be in the dataset columns."
            )

        if self.X[feature_a].isnull().any() or self.X[feature_b].isnull().any():
            raise ValueError(
                f"Features '{feature_a}' and '{feature_b}' must not contain null values."
            )

        plt.figure(figsize=(10, 6))
        sns.scatterplot(
            x=self.X[feature_a],
            y=self.X[feature_b],
            hue=self.cluster_labels,
            palette="Set1",
            alpha=0.7,
        )
        plt.title(f"KMeans Clusters (k={self.kmeans.n_clusters}) on {feature_a} vs {feature_b}")
        plt.xlabel(feature_a)
        plt.ylabel(feature_b)
        plt.legend(title="Cluster")
        plt.tight_layout()
        plt.savefig(Config.DIR_EDA + f"/kmeans_clusters_{feature_a}_{feature_b}.png")
        plt.show()
        plt.close()

    def all_feature_plot(self):
        features = self.X.columns
        if "Cluster" in features:
            features = features.drop("Cluster")

        for feature_x, feature_y in combinations(features, 2):
            plt.figure(figsize=(8, 6))
            sns.scatterplot(
                x=self.X[feature_x],
                y=self.X[feature_y],
                hue=self.cluster_labels,
                palette="Set1",
                alpha=0.7,
            )

            # Plot Centroids
            centers = self.kmeans.cluster_centers_
            feature_idx_a = self.X.columns.get_loc(feature_x)
            feature_idx_b = self.X.columns.get_loc(feature_y)

            plt.scatter(
                centers[:, feature_idx_a],
                centers[:, feature_idx_b],
                c='black',
                s=100,
                marker='+',
                linewidths=2,
                label='Centroids',
            )

            # Add labels to centroids
            for idx, (x, y) in enumerate(zip(centers[:, feature_idx_a], centers[:, feature_idx_b])):
                plt.text(
                    x,
                    y,
                    f"C{idx}",
                    fontsize=12,
                    fontweight='bold',
                    color='black',
                )

            plt.title(f"KMeans Clusters on {feature_x} vs {feature_y}")
            plt.xlabel(feature_x)
            plt.ylabel(feature_y)
            plt.legend(title="Cluster")
            plt.tight_layout()

            plt.savefig(Config.DIR_EDA + f"/kmeans_clusters_{feature_x}_{feature_y}.png")
            plt.show()
            plt.close()

class PCA_lab():
    """
    Class for PCA and visualization.
    """

    def __init__(self, X: pd.DataFrame | None = None):
        """
        Initialize the PCA class.
        :param X: Training features
        """
        self.X = X
        self.pca = None
        self.X_pca = None

    def apply_pca(self, n_components: int = 2) -> None:
        """
        Apply PCA to the dataset.
        :param n_components: Number of principal components to keep
        """
        self.pca = PCA(n_components=n_components)
        self.X_pca = self.pca.fit_transform(self.X)

    def explained_variance_ratio(self) -> None:
        """
        Print the explained variance ratio of each principal component.
        """
        if self.pca is None:
            raise ValueError("PCA has not been applied yet. Please run apply_pca() first.")

        print("----------------------------------")
        print("Explained variance ratio of each principal component:")
        for i, var in enumerate(self.pca.explained_variance_ratio_):
            print(f"PC{i + 1}: {var:.4f}")
        print("----------------------------------")

    def plot_pca_clusters(self, cluster_labels, save_path=None):
        """
        Plot the 2D PCA-transformed data colored by KMeans cluster labels.
        """
        if self.X_pca is None:
            raise ValueError("PCA has not been applied yet.")

        plt.figure(figsize=(8, 6))
        sns.scatterplot(
            x=self.X_pca[:, 0],
            y=self.X_pca[:, 1],
            hue=cluster_labels,
            palette="Set1",
            alpha=0.7
        )
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.title("KMeans Clusters in PCA Space")
        plt.legend(title="Cluster")
        plt.tight_layout()
        plt.savefig(save_path or Config.DIR_PCA + "/pca_clusters.png")

    def pca_feature_contributions(self):
        """
        Show PCA loadings: feature contributions to PC1 and PC2.
        """
        if self.pca is None:
            raise ValueError("PCA has not been applied yet.")

        loadings = pd.DataFrame(
            self.pca.components_.T,
            columns=[f"PC{i + 1}" for i in range(self.pca.n_components_)],
            index=self.X.columns
        )
        print("\nFeature contributions (PCA loadings):")
        print(loadings)

        # Optionally: return sorted by magnitude for easier interpretation
        sorted_pc1 = loadings["PC1"].abs().sort_values(ascending=False)
        sorted_pc2 = loadings["PC2"].abs().sort_values(ascending=False)

        print("--------------------------------")
        print("\nTop features contributing to PC1:")
        print(sorted_pc1.head())

        print("\nTop features contributing to PC2:")
        print(sorted_pc2.head())
        print("--------------------------------")