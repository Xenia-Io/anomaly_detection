from isolation_forest_test import IsolationForest_tester


def main():
    tester = IsolationForest_tester(20, 35)
    # tester.run_test()
    # tester.run()
    # tester.run_test_isoForest()
    tester.run_kMeans()

if __name__ == "__main__":
    main()
    # testing()

