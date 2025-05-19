import os

import matplotlib.pyplot as plt
import openml
import pandas as pd
import seaborn as sns
from openml import OpenMLDataset
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from config import Config
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def create_assets_folder() -> None:
    """
    Create the assets folder if it doesn't exist.
    """

    if not os.path.exists(Config.DIR_ASSETS):
        os.makedirs(Config.DIR_ASSETS)

    if not os.path.exists(Config.DIR_EDA):
        os.makedirs(Config.DIR_EDA)


class EDA:
    """
    Class for Exploratory Data Analysis (EDA) on a dataset.
    """

    def __init__(self, dataset: OpenMLDataset | None = None, X: pd.DataFrame | None = None, y: pd.Series | None = None):
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
        return df

    def plot_visualization(self, file_name_suffix="") -> None:
        """
        Plot visualizations of the dataset.
        1. Histograms / KDE plots
        2. Correlation heatmap
        3. Class distribution (sns.countplot)

        :param file_name_suffix: Suffix for the file name to avoid overwriting
        """

        # Histograms
        self.X[sorted(self.X.columns)].hist(figsize=(20, 15), bins=100)
        plt.suptitle("Histograms of the dataset", fontsize=20)
        plt.tight_layout(pad=1)
        plt.savefig(Config.DIR_EDA + f"/histograms{file_name_suffix}.png")
        plt.close()

        # Correlation heatmap
        plt.figure(figsize=(20, 15))
        sns.heatmap(self.X[sorted(self.X.columns)].corr(), annot=True, fmt=".2f", cmap="coolwarm")
        plt.title("Correlation heatmap of the dataset", fontsize=20)
        plt.savefig(Config.DIR_EDA + f"/correlation_heatmap{file_name_suffix}.png")
        plt.close()

        # Class distribution
        plt.figure(figsize=(20, 15))
        sns.countplot(x=self.y, data=self.X)
        plt.title("Class distribution of the dataset", fontsize=20)
        plt.xticks(rotation=90)
        plt.savefig(Config.DIR_EDA + f"/class_distribution{file_name_suffix}.png")
        plt.close()

        # Boxplots
        # Check for outliers using boxplots
        n_cols = 3
        n_rows = (len(self.X.columns) + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
        axes = axes.flatten()

        for i, col in enumerate(self.X.columns):
            sns.boxplot(y=self.X[col], ax=axes[i])
            axes[i].set_title(f"Boxplot of {col}")

        # Remove empty subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.savefig(Config.DIR_EDA + f"/boxplots{file_name_suffix}.png")
        plt.close()

        # Check for outliers using IQR
        lower_bound, upper_bound = self.outliers_iqr()
        outliers = ((self.X < lower_bound) | (self.X > upper_bound)).sum()
        print("----------------------------------")
        print("Number of outliers in the dataset:")
        print(outliers)
        print("----------------------------------")

    def outliers_iqr(self) -> (pd.Series, pd.Series):
        """
        Calculate the IQR and return the lower and upper bounds for outliers.

        :return: lower_bound, upper_bound
        """
        Q1 = self.X.quantile(0.25)
        Q3 = self.X.quantile(0.75)

        IQR: pd.Series = Q3 - Q1

        lower_bound: pd.Series = Q1 - 1.5 * IQR
        upper_bound: pd.Series = Q3 + 1.5 * IQR

        return lower_bound, upper_bound

    def data_check(self) -> None:
        """
        Check for missing data or anomalous zeros (e.g., Insulin and SkinThickness have zeros which are unrealistic).
        """
        # Check for missing values
        print("----------------------------------")
        print("Missing values in the dataset:")
        print(self.X.isnull().sum())
        print("----------------------------------")

        # Check for anomalous zeros in certain columns
        columns_with_invalid_zeros = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

        print("Anomalous zeros (likely invalid) in specific columns:")
        for col in columns_with_invalid_zeros:
            zero_count = (self.X[col] == 0).sum()
            print(f"{col}: {zero_count} zero(s)")
        print("----------------------------------")

    def standardize_data(self) -> None:
        """
        Standardize the data (use StandardScaler). Pay attention to potential data leakage when standardizing features.
        """
        scaler = StandardScaler()
        self.X = pd.DataFrame(scaler.fit_transform(self.X), columns=self.X.columns)


class K_means(EDA):
    """
    Class for K-means clustering.
    """
    def __init__(self, X: pd.DataFrame | None = None, y: pd.Series | None = None):
        """
        Initialize the K-means class.
        :param X: Features
        :param y: Target variable
        """
        super().__init__(X=X, y=y)
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
        plt.figure(figsize=(10, 6))
        sns.scatterplot(
            x=self.X[feature_a],
            y=self.X[feature_b],
            hue=self.cluster_labels,
            palette="Set1",
            alpha=0.7
        )
        plt.title(f"KMeans Clusters (k={self.kmeans.n_clusters}) on {feature_a} vs {feature_b}")
        plt.xlabel(feature_a)
        plt.ylabel(feature_b)
        plt.legend(title="Cluster")
        plt.tight_layout()
        plt.savefig(Config.DIR_EDA + f"/kmeans_clusters_{feature_a}_{feature_b}.png")
        plt.show()
        plt.close()
