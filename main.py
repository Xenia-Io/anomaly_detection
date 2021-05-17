from build_tests import Tester
import numpy as np

def main():
    tester = Tester(10, 500, 'data/dataset_100k.json', is_supervised=False, visualize=False)
    tester.run_isoForest(umap=False, tsne=False, pca=True)
    # tester.comparisons()
    # tester.run_kMeans()
    # tester.run_kMedians()


if __name__ == "__main__":
    main()
