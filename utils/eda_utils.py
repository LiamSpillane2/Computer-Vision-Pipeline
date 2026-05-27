"""---Functions in this file---
    img_to_arry: converts image to an array
    hist_arr: loads image and returns a normalized histogram
    random_sample: grabs a random sample of files
    
    dataset_info: provides the details of a dataframe for a cursory glance at its contents
    PCA_analysis: performs PCA analysis on a dataset
    cluster_kmeans: performs a cluster analysis using kmeans
    
    plot_results: plot setup for basic plots
    plot_kcluster: plots results from cluster analysis
    plot_pca: plots results from pca analysis
    
    """

# import libraries
import pandas as pd
import os
import random
import seaborn as sns
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Enables 3D projection
import cv2
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans

"""---Image loading---
    Load images from a directory for use in feature extraction and exploratory data analysis
    """


# Convert an image to an array
def img_to_arry(direct_path, file_name):
    
    img_files = direct_path
    files = os.listdir(img_files)
    test_file = os.path.join(img_files, file_name)

    # open test file, convert to grayscale and convert to numpy array
    img = Image.open(test_file)
    gs_img = img.convert("L")
    image_arr = np.asarray(gs_img)
    return files, image_arr, test_file


#load an image and return normalized histogram of pixel values
def hist_arr(im_path):
    img = cv2.imread(im_path)
    
    if img is None:
        return FileNotFoundError("One or both image paths are invalid.")

    # convert to HSV color space (better for color/lighting variations)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # calculate histograms
    hist = cv2.calcHist([gray_img], [0], None, [256], [0, 256])
    cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

    return hist


#random sample of files
# must run img_to_arry before running
def random_sample(files, img_files):
    files = random.sample(files, k=1000)
    img_dict = {}
    for file in files:
        tf_path = os.path.join(img_files, file) 

        hist_output = hist_arr(tf_path)
        img_dict[file] = hist_output.flatten()
    return img_dict



"""---Analysis Functions---
    Functions for use in exploratory data analysis.
    Image loading functions must be called first.
    """

# function that runs several basic commands to describe the dataset for a cursory glance
def dataset_info(df):
    print(f"Dataset shape: {df.shape}\n") #shape of the dataset
    df.head() #first five rows of the dataset
    df.info() #column names, non-null count, and data types
    df.describe() #basic statistics for each column
    num_df = df.select_dtypes(include='number')
    corr = num_df.corr(method = 'pearson')
    sns.heatmap(corr)
    return corr

# run PCA analysis and plot
def PCA_analysis(num_components, img_dict, num_clusters):
    
    # inputs
        # num_components = number of components to be used in PCA analysis (int)
        # img_dict = files to be used for analysis (dict)
        # num_clusters = number of clusters to be used in kmeans clustering (int)
    
    # separate dictionary into image names and histogram bin counts
    sorted_image_names = list(img_dict.keys())
    flattened_histograms = list(img_dict.values())
    # create 2D array of all histogram lists in list
    histograms_matrix = np.array(flattened_histograms)

    # normalize histogram values
    X_norm = normalize(histograms_matrix, norm='l1', axis=1)

    # PCA analysis 
    pca = PCA(num_components)
    X_pca = pca.fit_transform(X_norm)

    cluster_labels = cluster_kmeans(img_dict, X_pca, num_clusters)
    plot_pca(cluster_labels, X_pca, pca)
    return plt.show(), X_pca
    

# cluster using k means clustering
def cluster_kmeans(img_dict, X_pca, num_clusters):
    sorted_image_names = list(img_dict.keys())
    # define K-Means model
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init='auto')

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
def plot_results (result, plot_type, title, x_label, y_label):
    
    # inputs
        #result = results from analysis you wanted to be plotted (df)
        #plot_type = what plot you want to create (str)
        #title = title of the plot (str)
        #x_label = x-axis label (str)
        #y_label = y-axis label (str)

    # define x and y values
    num_cols = result.shape[1]
    x_vals = result.iloc[:,0]
    y_vals = result[:,1]

    #determine plot type and plot data
    if plot_type == 'bar':
        plt.bar(x_vals, y_vals)
    if plot_type == 'hist':
        plt.hist(result)
    if plot_type == 'pie':
        result.iloc[num_cols].plot(kind="pie", autopct="%1.1f%%")
    if plot_type == 'line':
        plt.plot(x_vals, y_vals)
    else:
        print("Error: valid chart type not entered. Please enter the chart you want to use.")

    #plot details
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.tight_layout()

    return plt.show()

# plot kmeans clustering analysis results
def plot_kcluster(samples_per_cluster, num_clusters, cluster_groups):
    # define figure and axes: rows = num_clusters, cols = samples_per_cluster
    fig, axes = plt.subplots(num_clusters, samples_per_cluster, figsize=(12, 3 * num_clusters))

    # define overall figure title
    fig.suptitle("Samples per Cluster Group", fontsize=16, fontweight='bold')

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
                    ax.text(0.5, 0.5, "File\nNot Found", ha='center', va='center', color='red', fontsize=9)
                except Exception as e:
                    ax.text(0.5, 0.5, "Error\nLoading", ha='center', va='center', color='orange', fontsize=9)
            else:
                # Placeholder text if a cluster has fewer images than samples_per_cluster
                ax.text(0.5, 0.5, "No More\nImages", ha='center', va='center', color='gray', fontsize=9)
            
            ax.axis('off')

    plt.tight_layout()
    return plt.show()

# plot PCA analysis results
def plot_pca(cluster_labels, X_pca, pca):
    # Initialize a 3D figure
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot using the 3 PCA dimensions
    # 'cluster_labels' maps to your KMeans or clustering outputs
    scatter = ax.scatter(
        X_pca[:, 0],  # Principal Component 1
        X_pca[:, 1],  # Principal Component 2
        X_pca[:, 2],  # Principal Component 3
        c=cluster_labels,  # Color points by cluster ID
        cmap='tab10',      # cmap to use
        s=50,              # Size of points
        alpha=0.8          # Transparency to see overlapping points
    )

    # Label the 3 dimensional axes
    # Labels include the percentage of dataset variance explained by each axis
    ax.set_xlabel(f'PC 1 ({pca.explained_variance_ratio_[0]:.1%})')
    ax.set_ylabel(f'PC 2 ({pca.explained_variance_ratio_[1]:.1%})')
    ax.set_zlabel(f'PC 3 ({pca.explained_variance_ratio_[2]:.1%})')

    # Add decorative and analytical elements
    plt.title('3D PCA Space of Image Histograms', fontsize=14, fontweight='bold')
    fig.colorbar(scatter, ax=ax, label='Cluster Assignment', pad=0.1)

    plt.tight_layout()
    return plt.show()