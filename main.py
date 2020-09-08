from build_tests import Tester
from pre_processor import Preprocessor
from auto_encoder import Autoencoder

def main():
    tester = Tester(20, 35, 'logs_lhcb.json',is_supervised=False, visualize=True)
    # tester.run_isoForest()
    # tester.run_kMeans()
    # tester.run_kMedians()

    # preprocessor = Preprocessor('logs_lhcb.json', False, visualize=True)
    preprocessor = Preprocessor('logs_for_supervised.json', True, visualize=True)
    preprocessor.preprocessing()

if __name__ == "__main__":
    main()
    # testing()

