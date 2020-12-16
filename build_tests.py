from sklearn.metrics import calinski_harabasz_score, silhouette_score, silhouette_samples, davies_bouldin_score
from sklearn_extensions.fuzzy_kmeans import KMedians
from sklearn.datasets import make_moons, make_blobs
from mpl_toolkits.axes_grid1 import host_subplot
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from pre_processor import Preprocessor
import mpl_toolkits.axisartist as AA
from sklearn.cluster import KMeans
from feature_extractor import *
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn import svm
import time


class Tester():

    def __init__(self, epochs, n_batch, filename, is_supervised=False, visualize=False):

        self.epochs = epochs
        self.n_batch = n_batch
        self.filename = filename
        self.is_supervised = is_supervised
        self.visualize = visualize

    def comparisons(self):

        preprocessor = Preprocessor(self.filename, self.is_supervised, self.visualize)
        preprocessor.preprocessing(pca=True)
        print("x_all shape passed in: ", preprocessor.x_all.shape)

        # Settings
        n_samples = preprocessor.x_all.shape[0]
        outliers_fraction = 0.15
        n_outliers = int(outliers_fraction * n_samples)
        n_inliers = n_samples - n_outliers

        print('Starting fitting Models')

        anomaly_algorithms = [
            ("Robust covariance", EllipticEnvelope(contamination=outliers_fraction)),
            ("One-Class SVM", svm.OneClassSVM(nu=outliers_fraction, kernel="rbf",
                                              gamma=0.1)),
            ("Isolation Forest", IsolationForest(contamination=outliers_fraction,
                                                 random_state=42)),
            ("Local Outlier Factor", LocalOutlierFactor(
                n_neighbors=35, contamination=outliers_fraction))]

        # Compare given classifiers under given settings
        x_min, x_max = preprocessor.x_all[:, 1].min() - 1, preprocessor.x_all[:, 1].max() + 1
        y_min, y_max = preprocessor.x_all[:, 0].min() - 1, preprocessor.x_all[:, 0].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                             np.arange(y_min, y_max, 0.1))

        plt.figure(figsize=(len(anomaly_algorithms) * 2 + 3, 12.5))
        plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05,
                            hspace=.01)

        plot_num = 1
        rng = np.random.RandomState(42)


        # Add outliers
        # X = np.concatenate([preprocessor.x_all, rng.uniform(low=-6, high=6,
        #                                    size=(n_outliers, 2))], axis=0)
        X = preprocessor.x_all
        for name, model in anomaly_algorithms:
            t0 = time.time()
            model.fit(X)
            t1 = time.time()
            plt.subplot(1, len(anomaly_algorithms), plot_num)
            plt.title(name, size=14)

            # fit the data and tag outliers
            if name == "Local Outlier Factor":
                y_pred = model.fit_predict(X)
            else:
                y_pred = model.fit(X).predict(X)

            # plot the levels lines and the points
            if name != "Local Outlier Factor":  # LOF does not implement predict
                # Z = model.decision_function(xy).reshape(XX.shape)
                Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
                Z = Z.reshape(xx.shape)
                plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='black')

            colors = np.array(['#377eb8', '#ff7f00'])
            plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[(y_pred + 1) // 2])

            plt.xlim(-5, 5)
            plt.ylim(-5, 5)
            plt.xticks(())
            plt.yticks(())
            plt.text(.99, .01, ('%.2fs' % (t1 - t0)).lstrip('0'),
                     transform=plt.gca().transAxes, size=15,
                     horizontalalignment='right')
            plot_num += 1

        plt.show()


    def run_isoForest(self):
        """
        https://medium.com/@often_weird/isolation-forest-algorithm-for-anomaly-detection-f88af2d5518d
        :return:
        """
        preprocessor = Preprocessor(self.filename, self.is_supervised, self.visualize)
        preprocessor.preprocessing(pca=True)

        print("x_all shape passed in iForest model: ", preprocessor.x_all.shape)

        print('Starting fitting Isolation Forests')
        model = IsolationForest(contamination=0.03, n_estimators=10, warm_start=True, max_samples=100)
        cluster_labels = model.fit_predict(preprocessor.x_all)

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed clusters
        silhouette_avg = silhouette_score(preprocessor.x_all, cluster_labels)
        # CH criterion is most suitable in case when clusters are more
        # or less spherical and compact in their middle
        # the higher, the better
        ch_score = calinski_harabasz_score(preprocessor.x_all, cluster_labels)
        print("The average silhouette_score is :", silhouette_avg)
        print("The average calinski_harabasz_score is: ", ch_score)

        # Compute the silhouette scores for each sample: shape=(350,)
        sample_silhouette_values = silhouette_samples(preprocessor.x_all, cluster_labels)
        print("sample_silhouette_values: " , sample_silhouette_values.shape)

        # Plot decision lines
        model.fit(preprocessor.x_train)
        y_pred_train = model.predict(preprocessor.x_train)
        y_pred_test = model.predict(preprocessor.x_test)

        # plot the line, the samples, and the nearest vectors to the plane
        x_min, x_max = preprocessor.x_train[:, 1].min() - 1, preprocessor.x_train[:, 1].max() + 1
        y_min, y_max = preprocessor.x_test[:, 0].min() - 1, preprocessor.x_test[:, 0].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                             np.arange(y_min, y_max, 0.1))

        Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

        plt.title("IsolationForest on X features")
        plt.contourf(xx, yy, Z, cmap=plt.cm.Blues_r)
        b1 = plt.scatter(preprocessor.x_train[:, 0], preprocessor.x_train[:, 1], c='red')
        b2 = plt.scatter(preprocessor.x_test[:, 0], preprocessor.x_test[:, 1], c='green')

        plt.axis('tight')
        plt.xlim((-5, 5))
        plt.ylim((-5, 5))
        plt.legend([b1, b2],
                   ["X_training observations", "X_testing observations"],
                   loc="lower right")
        plt.show()


    def run_kMedians(self):
        print('Starting fitting K-Medians model')
        preprocessor = Preprocessor(self.filename, self.is_supervised, self.visualize)
        preprocessor.preprocessing(umap=True)
        print("preprocessor.x_all.shape : ", preprocessor.x_all.shape)

        # Use silhouette score
        range_n_clusters = [2, 3, 4, 5, 6, 10, 20, 30, 40, 50, 67]
        davies_bouldin_scores = list()
        silhouette_scores = list()
        ch_score = list()

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
            silhouette_scores.append(round(silhouette_score(preprocessor.x_all, cluster_labels), 2))
            ch_score.append(round(calinski_harabasz_score(preprocessor.x_all, cluster_labels), 2))
            davies_bouldin_scores.append(round(davies_bouldin_score(preprocessor.x_all, cluster_labels), 2))

            print("For n_clusters =", n_clusters,
                  "The average silhouette_score is :", silhouette_scores)
            print("For n_clusters =", n_clusters,
                  "The average calinski_harabasz score is :", ch_score)

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

        self.compare_scores(silhouette_scores, ch_score, davies_bouldin_scores, range_n_clusters)


    def run_kMeans(self):
        preprocessor = Preprocessor(self.filename, self.is_supervised, self.visualize)
        preprocessor.preprocessing()

        print('Starting fitting K-Means model')
        # Use silhouette score
        range_n_clusters = [2, 3, 4, 5, 6, 10, 20, 30, 40, 50, 67]

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
            ch_score = calinski_harabasz_score(preprocessor.x_all, cluster_labels)
            print("For n_clusters =", n_clusters,
                  "The average silhouette_score is :", silhouette_avg)
            print("For n_clusters =", n_clusters,
                  "The average calinski_harabasz score is :", ch_score)

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


    def compare_scores(self, silhouette_scores, ch_score, davies_bouldin_scores, range_n_clusters):
        host = host_subplot(111, axes_class=AA.Axes)
        plt.subplots_adjust(right=0.75)

        par1 = host.twinx()
        par2 = host.twinx()

        offset = 60
        new_fixed_axis = par2.get_grid_helper().new_fixed_axis
        par2.axis["right"] = new_fixed_axis(loc="right", axes=par2,
                                            offset=(offset, 0))

        par2.axis["right"].toggle(all=True)

        y_min, y_max = min(silhouette_scores) - 1, max(silhouette_scores) + 1
        host.set_xlim(range_n_clusters[0], range_n_clusters[-1])
        host.set_ylim(y_min, y_max)

        host.set_xlabel("Number of clusters")
        host.set_ylabel("Silhoutte scores")
        par1.set_ylabel("Calinski-Harabasz scores")
        par2.set_ylabel("Davies-Bouldin scores")

        p1, = host.plot(range_n_clusters, silhouette_scores, label="Silhoutte scores")
        p2, = par1.plot(range_n_clusters, ch_score, label="Calinski-Harabasz scores")
        p3, = par2.plot(range_n_clusters, davies_bouldin_scores, label="Davies-Bouldin scores")

        y0_min, y0_max = min(ch_score) - 1, max(ch_score) + 1
        y_min, y_max = min(davies_bouldin_scores) - 1, max(davies_bouldin_scores) + 1
        par1.set_ylim(y0_min, y0_max)
        par2.set_ylim(y_min, y_max)

        host.legend()

        host.axis["left"].label.set_color(p1.get_color())
        par1.axis["right"].label.set_color(p2.get_color())
        par2.axis["right"].label.set_color(p3.get_color())

        plt.draw()
        plt.show()


