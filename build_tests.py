from feature_extractor import *
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from preprocessor import Preprocessor
# from isolation_forest_model import IsolationForest
from sklearn.metrics import roc_curve, precision_recall_curve, auc, roc_auc_score, silhouette_score, silhouette_samples
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import matplotlib.cm as cm
from sklearn.cluster import KMeans

class Tester():

    def __init__(self, epochs, n_batch):

        self.epochs = epochs
        self.n_batch = n_batch


    def run_isoForest(self):
        preprocessor = Preprocessor()
        preprocessor.preprocessing('logs_lhcb.json')

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


    def run_kMeans(self):
        preprocessor = Preprocessor()
        preprocessor.preprocessing('logs_lhcb.json')

        print('Starting fitting KMeans model')
        # Use silhouette score
        # range_n_clusters = [2, 3, 4, 5, 6, 10, 20]
        range_n_clusters = [2,3,4,5,6]
        print("Number of clusters: \n", range_n_clusters)
        print(preprocessor.x_all[499])
        print("shape of x_all passed in the model: ", preprocessor.x_all.shape)

        # for i in range(preprocessor.x_train.shape[0]):
        #     for j in range(preprocessor.x_train.shape[1]):
        #         preprocessor.x_train[i][j] = round(preprocessor.x_train[i][j], 3)
        # print(preprocessor.x_train[300])

        for n_clusters in range_n_clusters:

            # Create a subplot with 1 row and 2 columns
            fig, (ax1, ax2) = plt.subplots(1, 2)
            fig.set_size_inches(18, 7)

            # The 1st subplot is the silhouette plot
            # The silhouette coefficient can range from -1, 1 but in this example all
            # lie within [-0.1, 1]
            ax1.set_xlim([-0.1, 1])
            # The (n_clusters+1)*10 is for inserting blank space between silhouette
            # plots of individual clusters, to demarcate them clearly.
            ax1.set_ylim([0, len(preprocessor.x_all) + (n_clusters + 1) * 10])

            # Initialize the clusterer with n_clusters value and a random generator
            # seed of 10 for reproducibility.
            clusterer = KMeans(n_clusters=n_clusters, random_state=10)
            cluster_labels = clusterer.fit_predict(preprocessor.x_all)
            print("cluster_labels : ", cluster_labels)
            # The silhouette_score gives the average value for all the samples.
            # This gives a perspective into the density and separation of the formed
            # clusters
            silhouette_avg = silhouette_score(preprocessor.x_all, cluster_labels)
            print("For n_clusters =", n_clusters,
                  "The average silhouette_score is :", silhouette_avg)

            # Compute the silhouette scores for each sample
            sample_silhouette_values = silhouette_samples(preprocessor.x_all, cluster_labels)

            y_lower = 10
            for i in range(n_clusters):
                # Aggregate the silhouette scores for samples belonging to
                # cluster i, and sort them
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


    def run(self):
        preprocessor = Preprocessor()
        preprocessor.preprocessing('logs_lhcb.json')

        print('Starting fitting Isolation Forests')

        model = IsolationForest(contamination=0.03)

        # scaler = StandardScaler()
        # # X_scaled = scaler.fit_transform(preprocessor.x_train)

        model.fit(preprocessor.x_train)
        # x_train_labels = model.fit_predict(preprocessor.x_train)
        # print("x_train_labels: ", x_train_labels)

        scoring = -model.decision_function(preprocessor.x_test)  # the lower,the more normal
        # print("scoring: ", scoring)
        y_pred_test = model.predict(preprocessor.x_test)

        #Calculate the ROC curve
        fpr, tpr, thresholds = roc_curve(y_pred_test, scoring)

        # Calculate the PR curve
        precision, recall = precision_recall_curve(y_pred_test, scoring)[:2]

        # Calculate the AUC curve
        # auc = roc_auc_score(y_pred_test, scoring)
        # print('AUC: %.6f' % auc)

        # print("precision : ", precision)
        # print("recall: ", recall)

        AUC = auc(fpr, tpr)
        AUPR = auc(recall, precision)
        print("AUPR: ", AUPR)

        plt.subplot(121)
        plt.plot(fpr, tpr, lw=1, label='%s (area = %0.3f)' % ("Isolation Forest", AUC))
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate', fontsize=15)
        plt.ylabel('True Positive Rate', fontsize=15)
        plt.title('ROC curve', fontsize=15)
        plt.legend(loc="lower right", prop={'size': 12})

        plt.subplot(122)
        plt.plot(recall, precision, lw=1, label='%s (area = %0.3f)'
                                                % ("Isolation Forest", AUPR))
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('Recall', fontsize=15)
        plt.ylabel('Precision', fontsize=15)
        plt.title('PR curve', fontsize=15)
        plt.legend(loc="lower right", prop={'size': 12})

        plt.show()


    def run_test(self):

        preprocessor = Preprocessor()
        preprocessor.preprocessing('logs_lhcb.json')

        print('Starting fitting Isolation Forests')

        model = IsolationForest(contamination=0.03)
        # model.fit(preprocessor.x_train)

        print("Shape of x_test: ", preprocessor.x_test.shape)
        print("Shape of x_train[0].shape: ", preprocessor.x_train.shape[0])  # 350
        print("x_train[300]: ", preprocessor.x_train[300])
        print("Shape of x_outliers: ", preprocessor.x_outliers.shape)

        # get dimensions - N is 350 messages
        N = preprocessor.x_train.shape[0]
        messages_per_batch = int(N / self.n_batch)  # 350:35 = 10 messages per batch

        train_accuracy = []
        for epoch in range(self.epochs):
            for batch in range(self.n_batch):
                j_start = (batch) * messages_per_batch
                j_end = (batch + 1) * messages_per_batch

                X_batch = preprocessor.x_train[j_start:j_end, :]
                # print("X_batch : ", X_batch.shape)

                model.fit(X_batch)  # fit 10 trees

            y_pred_train = model.predict(preprocessor.x_train)

            # TRAINING: Compute accuracy per epoch and save the results in list
            epoch_accuracy = list(y_pred_train).count(1) / y_pred_train.shape[0]
            with open('out.txt', 'a') as f:
                f.write("\n epoch accuracy %.19f: " % epoch_accuracy)

            f.close()
            train_accuracy.append(epoch_accuracy)

        plt.plot(np.array(train_accuracy))
        plt.xlabel('epoch')
        plt.ylabel('train accuracy')
        plt.show()

        print("train_accuracy : ", train_accuracy)
        # print("Shape of x_test: ", preprocessor.x_test.shape)
        # print("Shape of x_train: ", preprocessor.x_train.shape)
        # print("Shape of x_outliers: ", preprocessor.x_outliers.shape)


        # Make predictions
        y_pred_train = model.predict(preprocessor.x_train)
        y_predictions = model.predict(preprocessor.x_test)  # fit the added trees
        y_pred_outliers = model.predict(preprocessor.x_outliers)
        print("Y_predictions on x_test: ", y_predictions)
        print("y_pred_train on x_train: ", y_pred_train)
        print("Y outliers: ", y_pred_outliers)
        print("y_predictions.shape : ", y_predictions.shape)
        print("y_pred_outliers.shape : ", y_pred_outliers.shape)

        # Accuracy for predictions
        print("list(y_predictions).count(1) : ", list(y_predictions).count(1))
        print("y_predictions.shape[0] : ", y_predictions.shape[0])
        print("Accuracy for predictions:", list(y_predictions).count(1) / y_predictions.shape[0])

        # Accuracy for outliers
        print(list(y_pred_outliers).count(-1))
        print("y_pred_outliers.shape[0]: ", y_pred_outliers.shape[0])
        print("Accuracy for outliers:", list(y_pred_outliers).count(-1) / y_pred_outliers.shape[0])