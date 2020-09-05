from build_tests import Tester
from pre_processor import Preprocessor

def main():
    tester = Tester(20, 35, 'logs_lhcb.json')
    tester.run_isoForest()
    # tester.run_kMeans()
    # tester.run_kMedians()


if __name__ == "__main__":
    main()
    # testing()

