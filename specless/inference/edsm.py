"""
Inference Algorithm
===================
Inference algorithms then use such demonstrations to come up with a specification.
>> import specless as sl
>> traces = [[a,b,c], [a,b,b,c], [a,a,b,b,c]]
>> dataset = sl.ArrayDataset(traces)
>> inference = sl.TPOInference()
>> specification = inference.infer(demonstrations)
"""
import os
import re
import subprocess as sp
import time
from pathlib import Path
from typing import Optional, Union

import graphviz
from IPython.display import Image, display

from specless.dataset import BaseDataset, PathToFileDataset
from specless.inference.base import InferenceAlgorithm
from specless.specification import Specification


class AutomataInferenceAlgorithm(InferenceAlgorithm):
    """The inference algorithm for inferring an automaton from a list of Traces,
    where trace is defined as a sequence of symbols, i.e. a set of strings.
    For example, ${a, b, c}$

    Args:
        InferenceAlgorithm (_type_): _description_
    """

    def __init__(
        self, binary_location: str = "dfasat/flexfringe", output_directory: str = "./"
    ) -> None:
        """FlexFringe Interface. Directly access the binary via bash commands

        Args:
            binary_location (str, optional): (absolute / relative) filepath to the
                                       flexfringe binary. Defaults to "dfasat/flexfringe".
            output_directory (str, optional): The flexfringe output directory. Defaults to "./".
        """
        super().__init__()
        self.binary_location = binary_location
        self.num_training_examples: int
        self.num_symbols: int
        self.total_symbols_in_examples: int
        self.output_filepath: str
        self.learned_model_filepath: str
        self.initial_model_filepath: str
        self.average_elapsed_time: float = 0

        self._output_filename: str = "dfa"
        self._final_output_addon_name: str = "final"
        self._learned_model_filepath: str
        self._initial_output_addon_name = "init_dfa"
        self._initial_model_filepath: str
        self._output_base_filepath: Optional[str] = None
        self._output_directory: str = output_directory
        self._flexfringe_output_dir_popt_str: str = "o"
        Path(output_directory).mkdir(parents=True, exist_ok=True)

    def infer(
        self,
        dataset: BaseDataset,
        get_help: bool = False,
        record_time: bool = True,
        go_fast: bool = False,
        **kwargs,
    ) -> Union[Specification, Exception]:
        """calls the flexfringe binary given the data in the training file

        Args:
            dataset (Dataset): dataset that contains the path to the training data
            get_help (bool, optional): Whether or not to print the flexfringe
                                    usage help memu. Defaults to False.
            record_time (bool, optional): _description_. Defaults to True.
            go_fast (bool, optional): optimizes this call to make it as fast
                                    as possible, at the expensive of usability.
                                    use for benchmarking / Hyperparam
                                    optimization. Defaults to False.
            kwargs (dict, optional) keyword arguments to pass to flexfringe
                                    controlling the learning process

        Raises:
            Exception: _description_

        Returns:
            Union[Specification, Exception]:
        """
        error_msg = "must be an instance of the PathToFileDataset class"
        assert isinstance(dataset, PathToFileDataset), error_msg
        training_file: str = dataset.filepath

        cmd: list = self._get_command(kwargs)
        output_file: str = self.learned_model_filepath

        if get_help:
            flexfringe_call = [self.binary_location] + cmd + [""]
        else:
            flexfringe_call = [self.binary_location] + cmd + [training_file]

            if not go_fast:
                # get summary statistics of learning data and save them for
                # later use of the inference interface
                with open(training_file) as fh:
                    content = fh.readlines()
                    first_line = content[0]
                    N, num_symbols_str = re.match(r"(\d*)\s(\d*)", first_line).groups()
                    self.num_training_examples = int(N)
                    self.num_symbols = int(num_symbols_str)

                    self.total_symbols_in_examples = 0
                    if self.num_training_examples > 0:
                        for line in content[1:]:
                            _, line_len, _ = re.match(
                                r"(\d)\s(\d*)\s(.*)", line
                            ).groups()
                            self.total_symbols_in_examples += int(line_len)

        if output_file is not None:
            try:
                os.remove(output_file)
            except OSError:
                pass

        if go_fast:
            stdout = sp.DEVNULL
        else:
            stdout = sp.PIPE

        if record_time:
            start_time = time.time()

        completed_process = sp.run(flexfringe_call, stdout=stdout, stderr=sp.PIPE)

        if not go_fast:
            call_string = completed_process.stdout.decode()
            print("%s" % call_string)

        if record_time:
            elapsed_time = time.time() - start_time
            n_run = int(kwargs["n"]) if "n" in kwargs else 1
            self.average_elapsed_time = elapsed_time / n_run

        if not go_fast:
            model_data = self._read_model_data(output_file)
            if model_data is not None:
                # return model_data
                return Specification()
        raise Exception("No model output generated")

    def draw_IPython(self, filename: str) -> None:
        """Draws the dot file data in a way compatible with a jupyter / IPython
        notebook

        Args:
            filename (str): The learned model dot file data
        """
        dot_file_data = self._read_model_data(filename)

        if dot_file_data == "":
            pass
        else:
            filename = Path(filename).stem
            output_file = os.path.join(self._output_directory, filename)
            g = graphviz.Source(dot_file_data, filename=output_file, format="png")
            g.render()
            display(Image(g.render()))

    def draw_initial_model(self) -> None:
        """
        Draws the initial (prefix-tree) model
        """

        dot_file = self.initial_model_filepath
        self.draw_IPython(dot_file)

    def draw_learned_model(self) -> None:
        """
        Draws the final, learned model
        """

        dot_file = self.learned_model_filepath
        self.draw_IPython(dot_file)

    @property
    def output_filepath(self) -> str:
        """The output filepath for the results of learning the model"""
        self._output_base_filepath = os.path.join(
            self._output_directory, self._output_filename
        )

        return self._output_base_filepath

    @output_filepath.setter
    def output_filepath(self, filepath: str) -> None:
        """sets output_filepath and output_directory based on the given filepath

        Args:
            filepath (str): The new filepath
        """
        (self._output_directory, self._output_base_filepath) = os.path.split(filepath)

    @property
    def learned_model_filepath(self) -> str:
        """the output filename for the fully learned model, as this is a
        different from the inputted "output-dir"

        Returns:
            str: The learned model filepath.
        """
        addon_name = self._final_output_addon_name
        self._learned_model_filepath = self._get_model_file(addon_name)

        return self._learned_model_filepath

    @learned_model_filepath.setter
    def learned_model_filepath(self, filepath: str) -> None:
        """sets the learned_model_filepath and the base model's filepath

        Args:
            filepath (str): The new learned model filepath.
        """
        addon_name = self._final_output_addon_name
        base_model_filepath = self._strip_model_file(filepath, addon_name)

        self._learned_model_filepath = filepath
        self.output_filepath = base_model_filepath

    @property
    def initial_model_filepath(self) -> str:
        """the output filename for the unlearned, initial model, as this is a
        different from the inputted "output-dir".
        In this case, it will be a prefix tree from the given learning data.

        Returns:
            str: The initial model filepath.
        """
        addon_name = self._initial_output_addon_name
        self._initial_model_filepath = self._get_model_file(addon_name)

        return self._initial_model_filepath

    @initial_model_filepath.setter
    def initial_model_filepath(self, filepath: str) -> None:
        """sets the initial_model_filepath and the base model's filepath

        Args:
            filepath (str): The new initial model filepath.
        """
        addon_name = self._initial_model_filepath
        base_model_filepath = self._strip_model_file(filepath, addon_name)

        self._learned_model_filepath = filepath
        self.output_filepath = base_model_filepath

    def _get_model_file(self, addon_name: str) -> str:
        """Gets the full model filepath, with the model type given by addon_name.

        Args:
            addon_name (str): The name to append to the base model name
                                 to access the certain model file

        Returns:
            str: The full model filepath string.
        """
        filepath = self.output_filepath
        f_dir, _ = os.path.split(filepath)

        full_model_filename = self._get_model_filename(addon_name)
        full_model_filepath = os.path.join(f_dir, full_model_filename)

        return full_model_filepath

    def _strip_model_file(self, model_filepath: str, addon_name: str) -> str:
        """Strips the full model filepath of its addon_name to get the base model
        filepath

        Args:
            model_filepath (str): The full model filepath
            addon_name (str): :      The name to strip from the full model file

        Returns:
            str: The base model filepath string.
        """
        f_dir, full_fname = os.path.split(model_filepath)
        fname, ext = os.path.splitext(full_fname)

        # base filepath is just the basename, before the "addon" model type
        # is added to the base model name
        if fname.endswith(addon_name):
            fname = fname[: -len(addon_name)]

        base_model_filepath = os.path.join(f_dir, fname)

        return base_model_filepath

    def _get_model_filename(self, addon_name: str) -> str:
        """Gets the model filename, with the model type given by addon_name.

        Args:
            addon_name (str): The name to append to the base model name
                                 to access the certain model file

        Returns:
            str: The model filename string.
        """
        filepath = self.output_filepath
        f_dir, full_fname = os.path.split(filepath)
        fname, ext = os.path.splitext(full_fname)

        full_model_filename = fname + addon_name + ".dot"

        return full_model_filename

    def _read_model_data(self, model_file: str) -> Union[str, Exception]:
        """Reads in the model data as a string.

        Args:
            model_file (str): The model filepath

        Raises:
            Exception: _description_

        Returns:
            Union[str, Exception]: The model data as a string
        """

        try:
            with open(model_file) as fh:
                return fh.read()

        except FileNotFoundError:
            raise Exception("No model file was found.")

    def _get_command(self, kwargs: dict) -> list:
        """Gets a list of popt commands to send the binary

        Args:
            kwargs (dict): The flexfringe tool keyword arguments

        Returns:
            list: The list of commands.
        """

        # default argument is to print the program's man page
        if len(kwargs) > 1:
            cmd = ["-" + key + "=" + kwargs[key] for key in kwargs]

            # need to give the output directory only if the user hasn't already
            # put that in kwargs.
            if self._flexfringe_output_dir_popt_str not in kwargs:
                cmd += ["--output-dir={}".format(self.output_filepath)]
            else:
                key = self._flexfringe_output_dir_popt_str
                self.output_filepath = kwargs[key]
        else:
            cmd = ["--help"]
            print("no learning options specified, printing tool help:")

        return cmd
