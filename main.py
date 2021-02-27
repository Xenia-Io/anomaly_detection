from build_tests import Tester


def main():
    tester = Tester(10, 500, './data/dataset2.json',is_supervised=False, visualize=False)
    # tester.run_isoForest_version2(umap=False, tsne=True, pca=False)
    # tester.run_isoForest()
    tester.comparisons()
    # tester.run_kMeans()
    # tester.run_kMedians()



if __name__ == "__main__":
    main()


