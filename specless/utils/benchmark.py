import csv
import itertools
import logging
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

CURR_FILE_PATH = Path(__file__)


class BenchmarkLogger:
    def __init__(self, logfilename: str = "logs.log"):
        # Setup Logger
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setLevel(logging.DEBUG)
        stdout_handler.setFormatter(formatter)

        file_handler = logging.FileHandler(logfilename)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(stdout_handler)

    def start(
        self,
        experiment_func: Callable[[Any], Tuple],
        arg_dict: Dict[str, List],
        return_key_strs: List[str],
        csvfilepath: str,
    ):
        column_names: List[str] = list(arg_dict.keys()) + return_key_strs
        arglists: List[List[Any]] = list(arg_dict.values())

        logger = logging.getLogger()
        logger.info(f"Column Names: {column_names}")

        with open(csvfilepath, "w") as f:
            writer = csv.writer(f, delimiter=",")
            writer.writerow(column_names)

        args_product = itertools.product(*arglists)
        for args in args_product:
            result: Tuple = experiment_func(*args)

            row = (*args, *result)

            with open(csvfilepath, "a") as f:
                writer = csv.writer(f, delimiter=",")
                writer.writerow(row)
