from feature_extractor import *
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from preprocessor import Preprocessor
from isolation_forest_model import IsolationForest



class IsolationForest_tester():

    def __init__(self, epochs, n_batch):

        self.epochs = epochs
        self.n_batch = n_batch


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