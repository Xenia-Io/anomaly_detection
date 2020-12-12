from build_tests import Tester


def main():

    tester = Tester(10, 500, './data/big_dataset.json',is_supervised=False, visualize=True)
    # tester.run_isoForest()
    # tester.run_kMeans()
    tester.run_kMedians()



if __name__ == "__main__":
    main()


