from sklearn.metrics import calinski_harabasz_score, silhouette_score, silhouette_samples, davies_bouldin_score
from sklearn_extensions.fuzzy_kmeans import KMedians
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from pre_processor import Preprocessor
from sklearn.cluster import KMeans
from feature_extractor import *
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn import svm
import numpy as np
import time


class Tester():

    def __init__(self, epochs, n_batch, filename, is_supervised=False, visualize=False):

        self.epochs = epochs
        self.n_batch = n_batch
        self.filename = filename
        self.is_supervised = is_supervised
        self.visualize = visualize

    def comparisons(self):

        # Data preprocessing
        preprocessor = Preprocessor(self.filename, self.is_supervised, self.visualize)
        preprocessor.preprocessing(umap=True, pca=False)

        n_samples = preprocessor.x_all.shape[0]
        outliers_fraction = 0.15
        n_outliers = int(outliers_fraction * n_samples)

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
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.2),
                             np.arange(y_min, y_max, 0.2))

        plt.figure(figsize=(2, 2))
        plot_num = 1
        rng = np.random.RandomState(42)

        X = preprocessor.x_all
        for name, model in anomaly_algorithms:
            t0 = time.time()
            model.fit(X)
            t1 = time.time()
            plt.subplot(2, 2, plot_num)
            plt.title(name, size=14)

            # fit the data and tag outliers
            if name == "Local Outlier Factor":
                y_pred = model.fit_predict(X)
            else:
                y_pred = model.fit(X).predict(X)

            if name != "Local Outlier Factor":  # LOF does not implement predict
                Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
                Z = Z.reshape(xx.shape)
                plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='black')

            colors = np.array(['#377eb8', '#ff7f00'])
            plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[(y_pred + 1) // 2])

            plt.xlim(x_min, x_max)
            plt.ylim(y_min, y_max)
            plt.xticks(())
            plt.yticks(())
            plt.text(.99, .01, ('%.2fs' % (t1 - t0)).lstrip('0'),
                     transform=plt.gca().transAxes, size=15,
                     horizontalalignment='right')
            plot_num += 1

        plt.show()


    def run_isoForest(self, umap=False, tsne=False, pca=False):

        # Data preprocessing
        preprocessor = Preprocessor(self.filename, self.is_supervised, self.visualize)
        preprocessor.preprocessing(umap=umap, tsne=tsne, pca=pca)

        # Build dataframe from all data points in the given dataset
        x_all_array = np.asarray(preprocessor.x_all)
        df = pd.DataFrame({'x0': x_all_array[:, 0], 'x1': x_all_array[:, 1]})
        to_model_columns = df.columns[0:2]

        # Build, train and test Classifier
        clf = IsolationForest(contamination=0.05, n_estimators=100, warm_start=True, max_samples=100)
        clf.fit(df[to_model_columns])
        pred = clf.predict(df[to_model_columns])
        cluster_labels = clf.fit_predict(df[to_model_columns])

        # Results Evaluation
        silhouette_avg = silhouette_score(list(df[to_model_columns].values), cluster_labels)
        ch_score = calinski_harabasz_score(list(df[to_model_columns].values), cluster_labels)
        db_score = davies_bouldin_score(list(df[to_model_columns].values), cluster_labels)
        print("The average silhouette_score is :", silhouette_avg)
        print("The average calinski_harabasz_score is: ", ch_score)
        print("The average davies_bouldin_score is: ", db_score)

        # Find outliers
        df['anomaly'] = pred
        outliers = df.loc[df['anomaly'] == -1]
        outlier_index = list(outliers.index)

        # Find the number of anomalies and normal points (points classified -1 are anomalous)
        print(df['anomaly'].value_counts())

        # Plot predictions of the model
        fig, ax = plt.subplots(figsize=(8, 8))

        # storing it to be displayed later
        plt.legend(loc='best')

        # plot the line, the samples, and the nearest vectors to the plane
        x_min, x_max = preprocessor.x_train[:, 1].min() - 2, preprocessor.x_train[:, 1].max() + 5
        y_min, y_max = preprocessor.x_test[:, 0].min() - 3, preprocessor.x_test[:, 0].max() + 5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 1.0),
                             np.arange(y_min, y_max, 1.0))

        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        plt.contourf(xx, yy, Z, cmap=plt.cm.Blues_r)

        # Plotting input data and predictions
        b1 = plt.scatter(df.iloc[:, 0], df.iloc[:, 1], c='green',
                        s=20, label="normal points")
        b2 = plt.scatter(df.iloc[outlier_index, 0], df.iloc[outlier_index, 1], c='green', s=20, edgecolors='red',
                        label="predicted outliers")

        plt.axis('tight')
        plt.yticks(fontsize=20)
        plt.xticks(fontsize=20)
        plt.xlim((-2, 4))
        plt.ylim((-3, 17))
        plt.legend([b1, b2],
                   ["Clean  observations", "Fraud  observations"],
                   loc="lower right", fontsize=18)

        if pca:
            plt.title('Outliers prediction for clean and fraud instances using PCA')
        elif tsne:
            plt.title('Outliers prediction for clean and fraud instances using TSNE')
        else:
            plt.title('Outliers prediction for clean and fraud instances using UMAP')

        plt.show()


    def run_kMedians(self):
        print('Starting fitting K-Medians model')
        preprocessor = Preprocessor(self.filename, self.is_supervised, self.visualize)
        preprocessor.preprocessing(pca=True)

        # Use silhouette score
        range_n_clusters = [2, 3, 4, 5, 6, 10, 20, 30, 40, 50, 67]
        davies_bouldin_scores = list()
        silhouette_scores = list()
        ch_score = list()

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

            # The silhouette_score gives the average value for all the samples.
            silhouette_avg = silhouette_score(preprocessor.x_all, cluster_labels)
            silhouette_scores.append(round(silhouette_score(preprocessor.x_all, cluster_labels), 2))
            ch_score.append(round(calinski_harabasz_score(preprocessor.x_all, cluster_labels), 2))
            davies_bouldin_scores.append(round(davies_bouldin_score(preprocessor.x_all, cluster_labels), 2))

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
            ax1.set_xlabel("The silhouette coefficient values", fontsize=16)
            ax1.set_ylabel("Cluster label", fontsize=16)
            ax1.xaxis.set_tick_params(labelsize=18)
            ax1.yaxis.set_tick_params(labelsize=18)

            # The vertical line for average silhouette score of all the values
            ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

            ax1.set_yticks([])  # Clear the yaxis labels / ticks
            ax1.set_xticks([-0.5, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])

            # 2nd Plot showing the actual clusters formed
            colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
            ax2.scatter(preprocessor.x_all[:, 0], preprocessor.x_all[:, 1], marker='.', s=130, lw=0, alpha=0.9,
                        c=colors, edgecolor='k')

            # Labeling the clusters
            centers = clusterer.cluster_centers_

            # Draw white circles at cluster centers
            ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                        c="white", alpha=1, s=200, edgecolor='k')

            for i, c in enumerate(centers):
                ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                            s=50, edgecolor='k')

            ax2.set_title("The visualization of the clustered data.")
            ax2.set_xlabel("Feature space for the 1st feature", fontsize=16)
            ax2.set_ylabel("Feature space for the 2nd feature", fontsize=16)
            ax2.xaxis.set_tick_params(labelsize=18)
            ax2.yaxis.set_tick_params(labelsize=18)
            plt.suptitle(("Silhouette analysis for KMedians clustering on sample data "
                          "with n_clusters = %d" % n_clusters),
                         fontsize=14, fontweight='bold')

        print("The silhouette scores are :", silhouette_scores)
        print("The calinski_harabasz scores are :", ch_score)
        print("The davies_bouldin scores are :", davies_bouldin_scores)

        plt.show()

        self.compare_scores(silhouette_scores, ch_score, davies_bouldin_scores, range_n_clusters)


    def run_kMeans(self):
        preprocessor = Preprocessor(self.filename, self.is_supervised, self.visualize)
        preprocessor.preprocessing()

        print('Starting fitting K-Means model')
        range_n_clusters = [2, 3, 4, 5, 6, 10, 20, 30, 40, 50, 67]

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
            ax1.set_xlabel("The silhouette coefficient values", fontsize=16)
            ax1.set_ylabel("Cluster label", fontsize=16)

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

            # Draw white circles at cluster centers
            ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                        c="white", alpha=1, s=200, edgecolor='k')

            for i, c in enumerate(centers):
                ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                            s=50, edgecolor='k')

            ax2.set_title("The visualization of the clustered data.")
            ax2.set_xlabel("Feature space for the 1st feature", fontsize=16)
            ax2.set_ylabel("Feature space for the 2nd feature", fontsize=16)

            plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                          "with n_clusters = %d" % n_clusters),
                         fontsize=14, fontweight='bold')

        plt.show()


    def make_patch_spines_invisible(self, ax):
        ax.set_frame_on(True)
        ax.patch.set_visible(False)
        for sp in ax.spines.values():
            sp.set_visible(False)


    def compare_scores(self, silhouette_scores, ch_score, davies_bouldin_scores, range_n_clusters):
        fig, host = plt.subplots()
        fig.subplots_adjust(right=0.75)

        par1 = host.twinx()
        par2 = host.twinx()

        par2.spines["right"].set_position(("axes", 1.2))

        y_min, y_max = min(silhouette_scores) - 1, max(silhouette_scores) + 3
        host.set_xlim(range_n_clusters[0], range_n_clusters[-1])
        host.set_ylim(y_min, y_max)

        host.set_xlabel("Number of clusters")
        host.set_ylabel("Silhoutte scores")
        par1.set_ylabel("Calinski-Harabasz scores")
        par2.set_ylabel("Davies-Bouldin scores")

        p1, = host.plot(range_n_clusters, silhouette_scores, "b-", label="Silhoutte scores")
        p2, = par1.plot(range_n_clusters, ch_score, "r-", label="Calinski-Harabasz scores")
        p3, = par2.plot(range_n_clusters, davies_bouldin_scores,"g-", label="Davies-Bouldin scores")

        y0_min, y0_max = min(ch_score) - 1, max(ch_score) + 1
        y_min, y_max = min(davies_bouldin_scores) - 1, max(davies_bouldin_scores) + 1
        par1.set_ylim(y0_min, y0_max)
        par2.set_ylim(y_min, y_max)

        host.legend()

        host.yaxis.label.set_color(p1.get_color())
        par1.yaxis.label.set_color(p2.get_color())
        par2.yaxis.label.set_color(p3.get_color())

        tkw = dict(size=4, width=1.5)
        host.tick_params(axis='y', colors=p1.get_color(), **tkw)
        par1.tick_params(axis='y', colors=p2.get_color(), **tkw)
        par2.tick_params(axis='y', colors=p3.get_color(), **tkw)
        host.tick_params(axis='x', **tkw)
        lines = [p1, p2, p3]

        host.legend(lines, [l.get_label() for l in lines], loc="upper right")
        plt.yticks(fontsize=14)
        plt.xticks(fontsize=14)

        plt.draw()
        plt.show()
