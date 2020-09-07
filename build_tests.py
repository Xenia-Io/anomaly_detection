from feature_extractor import *
import matplotlib.pyplot as plt
from pre_processor import Preprocessor
from sklearn.metrics import roc_curve, precision_recall_curve, auc, silhouette_score, silhouette_samples
from sklearn.ensemble import IsolationForest
import matplotlib.cm as cm
from sklearn.cluster import KMeans
from sklearn_extensions.fuzzy_kmeans import KMedians


class Tester():

    def __init__(self, epochs, n_batch, filename, is_supervised=False):

        self.epochs = epochs
        self.n_batch = n_batch
        self.filename = filename
        self.is_supervised = is_supervised


    def run_isoForest(self):
        preprocessor = Preprocessor(self.filename, self.is_supervised)
        preprocessor.preprocessing()

        print("x_all shape passed in iForest model: ", preprocessor.x_all.shape)

        print('Starting fitting Isolation Forests')
        model = IsolationForest(contamination=0.03)
        cluster_labels = model.fit_predict(preprocessor.x_all)

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed clusters
        silhouette_avg = silhouette_score(preprocessor.x_all, cluster_labels)
        print("The average silhouette_score is :", silhouette_avg)

        # Compute the silhouette scores for each sample: shape=(350,)
        sample_silhouette_values = silhouette_samples(preprocessor.x_all, cluster_labels)
        print("sample_silhouette_values: " , sample_silhouette_values.shape)


    def run_kMedians(self):
        preprocessor = Preprocessor(self.filename, self.is_supervised)
        preprocessor.preprocessing()

        print('Starting fitting K-Medians model', preprocessor.x_all.shape)

        # Use silhouette score
        range_n_clusters = [2, 3, 4, 5, 6, 10, 18, 20, 30, 40, 50, 67]
        # range_n_clusters = [2,3,4,5,6]
        print("Number of clusters: ", range_n_clusters)
        print("Random sample 499 from x_all: ", preprocessor.x_all[499])
        print("Shape of x_all passed in the model: ", preprocessor.x_all.shape)

        for n_clusters in range_n_clusters:

            # Create a subplot with 1 row and 2 columns
            fig, (ax1, ax2) = plt.subplots(1, 2)
            fig.set_size_inches(18, 7)

            # The 1st subplot is the silhouette plot
            ax1.set_xlim([-0.1, 1])
            # Inserting blank space between silhouette plots of clusters to demarcate them
            ax1.set_ylim([0, len(preprocessor.x_all) + (n_clusters + 1) * 10])

            # Initialize the model
            clusterer = KMedians(k=int(n_clusters))
            clusterer.fit(preprocessor.x_all)
            cluster_labels = clusterer.labels_
            # print("cluster_labels : ", cluster_labels)

            # The silhouette_score gives the average value for all the samples.
            silhouette_avg = silhouette_score(preprocessor.x_all, cluster_labels)
            print("For n_clusters =", n_clusters,
                  "The average silhouette_score is :", silhouette_avg)

            # Compute the silhouette scores for each sample
            sample_silhouette_values = silhouette_samples(preprocessor.x_all, cluster_labels)

            y_lower = 10
            for i in range(n_clusters):
                # Aggregate silhouette scores for samples belonging to cluster i, and sort them
                ith_cluster_silhouette_values = \
                    sample_silhouette_values[cluster_labels == i]

                ith_cluster_silhouette_values.sort()

                size_cluster_i = ith_cluster_silhouette_values.shape[0]
                y_upper = y_lower + size_cluster_i

                color = cm.nipy_spectral(float(i) / n_clusters)
                ax1.fill_betweenx(np.arange(y_lower, y_upper),
                                  0, ith_cluster_silhouette_values,
                                  facecolor=color, edgecolor=color, alpha=0.7)

                # Label the silhouette plots with their cluster numbers at the middle
                ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

                # Compute the new y_lower for next plot
                y_lower = y_upper + 10  # 10 for the 0 samples

            ax1.set_title("The silhouette plot for the various clusters.")
            ax1.set_xlabel("The silhouette coefficient values")
            ax1.set_ylabel("Cluster label")

            # The vertical line for average silhouette score of all the values
            ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

            ax1.set_yticks([])  # Clear the yaxis labels / ticks
            ax1.set_xticks([-0.5, -0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

            # 2nd Plot showing the actual clusters formed
            colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
            ax2.scatter(preprocessor.x_all[:, 0], preprocessor.x_all[:, 1], marker='.', s=130, lw=0, alpha=0.9,
                        c=colors, edgecolor='k')

            # Labeling the clusters
            centers = clusterer.cluster_centers_
            # print(type(centers), len(centers), centers)
            # Draw white circles at cluster centers
            ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                        c="white", alpha=1, s=200, edgecolor='k')

            for i, c in enumerate(centers):
                ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                            s=50, edgecolor='k')

            ax2.set_title("The visualization of the clustered data.")
            ax2.set_xlabel("Feature space for the 1st feature")
            ax2.set_ylabel("Feature space for the 2nd feature")

            plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                          "with n_clusters = %d" % n_clusters),
                         fontsize=14, fontweight='bold')

        plt.show()


    def run_kMeans(self):
        preprocessor = Preprocessor(self.filename, self.is_supervised)
        preprocessor.preprocessing()

        print('Starting fitting K-Means model')
        # Use silhouette score
        range_n_clusters = [2, 3, 4, 5, 6, 10, 18, 20, 30, 40, 50, 67]
        # range_n_clusters = [2,3,4,5,6]
        print("Number of clusters: ", range_n_clusters)
        print("Random sample 499 from x_all: ", preprocessor.x_all[499])
        print("Shape of x_all passed in the model: ", preprocessor.x_all.shape)

        for n_clusters in range_n_clusters:

            # Create a subplot with 1 row and 2 columns
            fig, (ax1, ax2) = plt.subplots(1, 2)
            fig.set_size_inches(18, 7)

            # The 1st subplot is the silhouette plot
            ax1.set_xlim([-0.1, 1])
            # Inserting blank space between silhouette plots of clusters to demarcate them
            ax1.set_ylim([0, len(preprocessor.x_all) + (n_clusters + 1) * 10])

            # Initialize with n_clusters and a random generator seed for reproducibility
            clusterer = KMeans(n_clusters=n_clusters, random_state=10)
            cluster_labels = clusterer.fit_predict(preprocessor.x_all)
            # print("cluster_labels : ", cluster_labels)

            # The silhouette_score gives the average value for all the samples.
            silhouette_avg = silhouette_score(preprocessor.x_all, cluster_labels)
            print("For n_clusters =", n_clusters,
                  "The average silhouette_score is :", silhouette_avg)

            # Compute the silhouette scores for each sample
            sample_silhouette_values = silhouette_samples(preprocessor.x_all, cluster_labels)

            y_lower = 10
            for i in range(n_clusters):
                # Aggregate silhouette scores for samples belonging to cluster_i and sort them
                ith_cluster_silhouette_values = \
                    sample_silhouette_values[cluster_labels == i]

                ith_cluster_silhouette_values.sort()

                size_cluster_i = ith_cluster_silhouette_values.shape[0]
                y_upper = y_lower + size_cluster_i

                color = cm.nipy_spectral(float(i) / n_clusters)
                ax1.fill_betweenx(np.arange(y_lower, y_upper),
                                  0, ith_cluster_silhouette_values,
                                  facecolor=color, edgecolor=color, alpha=0.7)

                # Label the silhouette plots with their cluster numbers at the middle
                ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

                # Compute the new y_lower for next plot
                y_lower = y_upper + 10  # 10 for the 0 samples

            ax1.set_title("The silhouette plot for the various clusters.")
            ax1.set_xlabel("The silhouette coefficient values")
            ax1.set_ylabel("Cluster label")

            # The vertical line for average silhouette score of all the values
            ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

            ax1.set_yticks([])  # Clear the yaxis labels / ticks
            ax1.set_xticks([-0.5, -0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

            # 2nd Plot showing the actual clusters formed
            colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
            ax2.scatter(preprocessor.x_all[:, 0], preprocessor.x_all[:, 1], marker='.', s=130, lw=0, alpha=0.9,
                        c=colors, edgecolor='k')

            # Labeling the clusters
            centers = clusterer.cluster_centers_
            # print(type(centers), len(centers), centers)
            # Draw white circles at cluster centers
            ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                        c="white", alpha=1, s=200, edgecolor='k')

            for i, c in enumerate(centers):
                ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                            s=50, edgecolor='k')

            ax2.set_title("The visualization of the clustered data.")
            ax2.set_xlabel("Feature space for the 1st feature")
            ax2.set_ylabel("Feature space for the 2nd feature")

            plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                          "with n_clusters = %d" % n_clusters),
                         fontsize=14, fontweight='bold')

        plt.show()
