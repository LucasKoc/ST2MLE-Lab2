from eda import EDA, K_means

if __name__ == '__main__':
    ####################################################################################
    # Part 1 - EDA, Clustering and Dimensionality reduction
    ####################################################################################
    eda = EDA()

    # Exercise 1: Load the dataset using pandas.
    eda.load_dataset()

    # Exercise 2: Display basic statistics (describe()) for all features.
    print(eda.summary_table().to_markdown())

    # Exercise 3: Visualize the data distribution of each feature (histograms, boxplots).
    eda.plot_visualization()

    # Exercise 4: Check for missing data or anomalous zeros (e.g., Insulin and SkinThickness have zeros which are unrealistic).
    eda.data_check()

    # Exercise 5: Identify potential outliers using boxplots

    # Exercise 6: Standardize the data (use StandardScaler). Pay attention to potential data leakage when standardizing features.
    eda.standardize_data()
    eda.plot_visualization(file_name_suffix="_standardized")

    ####################################################################################
    # Part 2 - K-Means Clustering
    ####################################################################################
    k_means = K_means(eda.X, eda.y)

    # Exercise 1 - Apply K-Means clustering (with k=2) to the standardized data
    k_means.k_means_clustering(k=2)

    # Exercise 2 - Visualize the resulting clusters using a scatter plot (select two features for visualization)
    k_means.plot_clusters(feature_a="Glucose", feature_b="BMI")
