from build_tests import Tester
from pre_processor import Preprocessor
# from auto_encoder import Autoencoder
import pandas as pd

def main():

    tester = Tester(20, 35, 'data/logs_lhcb.json',is_supervised=False, visualize=False)
    tester.run_isoForest()
    # tester.run_kMeans()
    # tester.run_kMedians()

    # preprocessor = Preprocessor('logs_lhcb.json', False, visualize=True)
    # preprocessor = Preprocessor('logs_for_supervised.json', True, visualize=True)
    # preprocessor.preprocessing()

if __name__ == "__main__":
    main()
    # testing()

