import logging
import sys
import argparse


from matplotlib import pyplot as plt
import igp2 as ip

logger = logging.Logger(__name__)



def main(args):
    scenario_map = ip.Map.parse_from_opendrive("scenarios/maps/scenario1.xodr")
    ip.plot_map(scenario_map, midline=True)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process scenario parameters.")
    parser.add_argument('--sid', type=int, default=1, help='Scenario ID')
    parser.add_argument('--qid', type=int, default=0, help='Index of query to run')
    arguments = parser.parse_args()

    sys.exit(main(arguments))
