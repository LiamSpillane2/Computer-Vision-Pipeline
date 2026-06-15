"""---Functions in this file---
dataset_info: provides the details of a dataframe for a cursory glance at its contents
sort_classes: counts the number of instances of each class within a dataset
data_distribution: returns the skew, kurtosis, and other distribution information
PCA_analysis: performs PCA analysis on a dataset
cluster_kmeans: performs a cluster analysis using kmeans

plot_results: plot setup for basic plots
plot_kcluster: plots results from cluster analysis
plot_pca: plots results from pca analysis
"""

# import libraries
import os
import random
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
from scipy import stats
from pathlib import Path
from collections import Counter
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D  # Enables 3D projection
from sklearn.preprocessing import normalize

"""---Analysis Functions---
    Functions for use in exploratory data analysis.
    Image loading functions must be called first.
    """


# function that runs several basic commands to describe the dataset for a cursory glance
def dataset_info(df):
    print(f"Dataset shape: {df.shape}\n")  # shape of the dataset
    df.head()  # first five rows of the dataset
    df.info()  # column names, non-null count, and data types
    df.describe()  # basic statistics for each column
    num_df = df.select_dtypes(include="number")
    corr = num_df.corr(method="pearson")
    sns.heatmap(corr)
    return corr


# function to determine the number of instances of each class
def sort_classes(folder_path):
    counts = Counter()
    for txt_file in Path(folder_path).glob("*.txt"):
        with open(txt_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    class_id = int(line.split()[0])
                    counts[class_id] += 1
                except (ValueError, IndexError):
                    pass  # skip malformed rows

    return dict(sorted(counts.items()))


# function that evaluates the distribution of the data
def data_distribution(results):
    fig, ax = plt.subplots(1, figsize=(9, 7))
    ax.grid(linestyle="None")
    plt.hist(results, bins=50, range=[0, 1])
    plt.show()
    print("Skewness = %.3f" % np.mean(stats.skew(results)))
    print("Kurtosis = %.3f" % np.mean(stats.kurtosis(results)))
    print("Shapiro test = %.3f, P-value = %.3f" % stats.shapiro(results))


# run PCA analysis and plot
def PCA_analysis(num_components, img_arrays, num_clusters):

    # inputs
    # num_components = number of components to be used in PCA analysis (int)
    # img_dict = files to be used for analysis (dict)
    # num_clusters = number of clusters to be used in kmeans clustering (int)

    # separate dictionary into image names and histogram bin counts
    # flattened_histograms = list(img_arrays)
    # create 2D array of all histogram lists in list
    histograms_matrix = np.array(img_arrays)

    # normalize histogram values
    X_norm = normalize(histograms_matrix, norm="l1", axis=1)

    # PCA analysis
    pca = PCA(num_components)
    X_pca = pca.fit_transform(X_norm)

    # cluster_labels = cluster_kmeans(img_arrays, X_pca, num_clusters)
    # plot_pca(cluster_labels, X_pca, pca)
    # plt.show()
    return X_pca


# cluster using k means clustering
def cluster_kmeans(img_arrays, X_pca, num_clusters):
    sorted_image_names = list(img_arrays.keys())
    # define K-Means model
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init="auto")

    # run k-means on PCA array
    cluster_labels = kmeans.fit_predict(X_pca)

    # create dictionary with cluster groups keys and empty dictionary values
    cluster_groups = {i: [] for i in range(num_clusters)}

    # add images to dictionary based on their cluster label
    for img_name, label in zip(sorted_image_names, cluster_labels):
        cluster_groups[label].append(img_name)

    return cluster_labels, cluster_groups


"""---Plotting functions---
    Functions to be used for visualizing data with different analysis methods.
    Analysis functions must be called before plotting functions.
    Some analysis functions call plotting functions automatically. 
    """


# General plotting function for a single chart
def plot_results(result, plot_type, title, x_label, y_label):

    # inputs
    # result = results from analysis you wanted to be plotted (df)
    # plot_type = what plot you want to create (str)
    # title = title of the plot (str)
    # x_label = x-axis label (str)
    # y_label = y-axis label (str)

    # define x and y values
    num_cols = result.shape[1]
    x_vals = result.iloc[:, 0]
    y_vals = result[:, 1]

    # determine plot type and plot data
    if plot_type == "bar":
        plt.bar(x_vals, y_vals)
    if plot_type == "hist":
        plt.hist(result)
    if plot_type == "pie":
        result.iloc[num_cols].plot(kind="pie", autopct="%1.1f%%")
    if plot_type == "line":
        plt.plot(x_vals, y_vals)
    else:
        print(
            "Error: valid chart type not entered. Please enter the chart you want to use."
        )

    # plot details
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.tight_layout()

    return plt.show()


# plot kmeans clustering analysis results
def plot_kcluster(img_files, samples_per_cluster, num_clusters, cluster_groups):
    # define figure and axes: rows = num_clusters, cols = samples_per_cluster
    fig, axes = plt.subplots(
        num_clusters, samples_per_cluster, figsize=(12, 3 * num_clusters)
    )

    # define overall figure title
    fig.suptitle("Samples per Cluster Group", fontsize=16, fontweight="bold")

    # loop through cluster groupings
    for cluster_id in range(num_clusters):

        # get images in a cluster group
        images_in_cluster = cluster_groups[cluster_id]

        # get minimum between samples selected and images in cluster
        num_to_sample = min(samples_per_cluster, len(images_in_cluster))

        # pick random sample from each cluster group
        sampled_images = random.sample(images_in_cluster, num_to_sample)

        # for each sample image in each cluster...
        for i in range(samples_per_cluster):

            # define an axis
            ax = axes[cluster_id, i]

            if i < len(sampled_images):
                # define image path
                img_name = sampled_images[i]
                img_path = os.path.join(img_files, img_name)

                try:
                    # open image file
                    img = Image.open(img_path)

                    # show image on axis
                    ax.imshow(img)

                    # set title
                    ax.set_title(f"{img_name}\n(Cluster {cluster_id})", fontsize=8)
                except FileNotFoundError:
                    ax.text(
                        0.5,
                        0.5,
                        "File\nNot Found",
                        ha="center",
                        va="center",
                        color="red",
                        fontsize=9,
                    )
                except Exception as e:
                    ax.text(
                        0.5,
                        0.5,
                        "Error\nLoading",
                        ha="center",
                        va="center",
                        color="orange",
                        fontsize=9,
                    )
            else:
                # Placeholder text if a cluster has fewer images than samples_per_cluster
                ax.text(
                    0.5,
                    0.5,
                    "No More\nImages",
                    ha="center",
                    va="center",
                    color="gray",
                    fontsize=9,
                )

            ax.axis("off")

    plt.tight_layout()
    return plt.show()


# plot PCA analysis results
def plot_pca(cluster_labels, X_pca, pca):
    # Initialize a 3D figure
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Scatter plot using the 3 PCA dimensions
    # 'cluster_labels' maps to your KMeans or clustering outputs
    scatter = ax.scatter(
        X_pca[:, 0],  # Principal Component 1
        X_pca[:, 1],  # Principal Component 2
        X_pca[:, 2],  # Principal Component 3
        c=cluster_labels,  # Color points by cluster ID
        cmap="tab10",  # cmap to use
        s=50,  # Size of points
        alpha=0.8,  # Transparency to see overlapping points
    )

    # Label the 3 dimensional axes
    # Labels include the percentage of dataset variance explained by each axis
    ax.set_xlabel(f"PC 1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC 2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax.set_zlabel(f"PC 3 ({pca.explained_variance_ratio_[2]:.1%})")

    # Add decorative and analytical elements
    plt.title("3D PCA Space of Image Histograms", fontsize=14, fontweight="bold")
    fig.colorbar(scatter, ax=ax, label="Cluster Assignment", pad=0.1)

    plt.tight_layout()
    return plt.show()
