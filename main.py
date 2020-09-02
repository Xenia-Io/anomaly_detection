from build_tests import Tester


def main():
    tester = Tester(20, 35)
    # tester.run_test()
    # tester.run()
    tester.run_isoForest()
    # tester.run_kMeans()

if __name__ == "__main__":
    main()
    # testing()

