import os
from pathlib import Path

from prepare_aircraft_turnaround_data import load_aircraft_turnaround_data

import specless as sl

CURRENT_DIR = Path(os.getcwd())
EXAMPLE_DIR = CURRENT_DIR.parent
HOME_DIR = EXAMPLE_DIR.parent
os.chdir(HOME_DIR)

CURRENT_DATA_DIR = os.path.join(CURRENT_DIR, "data")
CURRENT_OUTPUT_DIR = os.path.join(CURRENT_DIR, "output")
CURRENT_CONFIG_FILENAME = os.path.join(CURRENT_DIR, "config.json")

LOG_DIR: Path = Path.cwd().joinpath(".log")
print(str(LOG_DIR))


def main():
    # Load a list of traces sampled from an aircraft turnaround task
    demonstrations = load_aircraft_turnaround_data()

    inference = sl.TPOInferenceAlgorithm()
    specification: sl.Specification = inference.infer(demonstrations)

    filepath = os.path.join(LOG_DIR, "tpo.png")
    sl.draw_graph(specification, filepath)
    print(specification)


if __name__ == "__main__":
    main()
