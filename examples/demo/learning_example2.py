import os
import sys

import specless as sl


# Add the AicraftTurnaroundA directory to sys.path for importing the data preparation script
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
EXAMPLE_DIR = os.path.dirname(CURRENT_DIR)
AIRCRAFT_TURNAROUND_DIR = os.path.join(EXAMPLE_DIR, "AircraftTurnaround")
print(CURRENT_DIR, EXAMPLE_DIR, AIRCRAFT_TURNAROUND_DIR)
sys.path.append(AIRCRAFT_TURNAROUND_DIR)
from prepare_aircraft_turnaround_data import load_aircraft_turnaround_data


def main():
    # Load a list of traces sampled from an aircraft turnaround task
    demonstrations = load_aircraft_turnaround_data()

    inference = sl.TPOInferenceAlgorithm()
    specification: sl.Specification = inference.infer(demonstrations)

    sl.draw_graph(specification, filepath=os.path.join(CURRENT_DIR, "tpo"))
    print(specification)


if __name__ == "__main__":
    main()
