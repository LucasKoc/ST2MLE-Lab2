from eda import EDA, K_means, PCA_lab
from text_classification import Text_Preprocessing

if __name__ == '__main__':
    """
    ####################################################################################
    # Part 1 - EDA, Clustering and Dimensionality reduction
    ####################################################################################
    eda = EDA()

    # Exercise 1: Load the dataset using pandas.
    eda.load_dataset()

    # Exercise 2: Display basic statistics (describe()) for all features.
    print(eda.summary_table().to_markdown())

    # Exercise 3: Visualize the data distribution of each feature (histograms, boxplots).
    eda.plot_visualization(X=eda.X, y=eda.y)

    # Exercise 4: Check for missing data or anomalous zeros (e.g., Insulin and SkinThickness have zeros which are unrealistic).
    eda.data_check(X=eda.X)

    # Exercise 5: Identify potential outliers using boxplots

    # Exercise 6: Standardize the data (use StandardScaler). Pay attention to potential data leakage when standardizing features.
    eda.split_data()
    eda.standardize_data()
    eda.plot_visualization(file_name_suffix="_standardized", X=eda.X_train, y=eda.y_train)

    ####################################################################################
    # Part 2 - K-Means Clustering
    ####################################################################################
    k_means = K_means(X=eda.X_train)

    # Exercise 1 - Apply K-Means clustering (with k=2) to the standardized data
    k_means.k_means_clustering(k=2)

    # Exercise 2 - Visualize the resulting clusters using a scatter plot (select two features for visualization)
    # Do all combinations of features
    k_means.all_feature_plot()

    ####################################################################################
    # Part 3 - PCA and Visualization
    ####################################################################################
    pca = PCA_lab(X=eda.X_train)

    # Exercise 1: Apply PCA to reduce the dataset to 2 components.
    pca.apply_pca(n_components=2)

    # Exercise 2: Display the explained variance ratio for each principal component
    pca.explained_variance_ratio()

    # Exercise 3: Plot the data points using the first 2 principal components.
    pca.plot_pca_clusters(
        cluster_labels=k_means.cluster_labels,
    )

    pca.pca_feature_contributions()

    ####################################################################################
    # Part 4 - K-Means on PCA-transformed data
    ####################################################################################

    # Exercise 1: Apply K-Means clustering again, but now on the 2D PCA-transformed data.
    k_means = K_means(X=pca.X_pca)
    k_means.k_means_clustering(k=2)
    pca_cluster_labels = k_means.cluster_labels

    # Exercise 2: Compare the clusters obtained in Exercise 2 vs. Exercise 4.
    pca.plot_pca_clusters(
        cluster_labels=pca_cluster_labels,
        file_name_suffix='_with_kmeans_clusters',
    )

    pca.pca_feature_contributions()
    """

    ####################################################################################
    # Part 2 - Text classification using Bag of Words, TF-IDF, Word2Vec and BERT embeddings.
    ####################################################################################
    # Preprocessing
    text_preprocessing = Text_Preprocessing()

    text_preprocessing.load_dataset()
    text_preprocessing.preprocess_texts()
    text_preprocessing.split_data()

    text_preprocessing.show_sample()

    # Bag of Words
    text_preprocessing.bag_of_words()
    text_preprocessing.naive_bayes_model()

    # TF-IDF
    text_preprocessing.tf_idf()
    text_preprocessing.naive_bayes_model()


