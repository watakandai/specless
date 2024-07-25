import specless as sl  # or load from specless.inference import TPOInference


def main():

    ### Partial Order Inference

    # Manually prepare a list of demonstrations
    demonstrations = [
        ["e1", "e2", "e3", "e4", "e5"],             # trace 1
        ["e1", "e4", "e2", "e3", "e5"],             # trace 2
        ["e1", "e2", "e4", "e3", "e5"],             # trace 3
    ]

    # Run the inference
    inference = sl.POInferenceAlgorithm()
    specification = inference.infer(demonstrations) # returns a Specification

    # prints the specification
    print(specification) # doctest: +ELLIPSIS

    # exports the specification to a file

    # drawws the specification to a file
    sl.draw_graph(specification, filepath='spec')

    ### Timed Partial Order Inference

    # Manually prepare a list of demonstrations
    demonstrations: list = [
        [[1, "a"], [2, "b"], [3, "c"]],
        [[4, "d"], [5, "e"], [6, "f"]],
    ]
    columns: list = ["timestamp", "symbol"]

    timedtrace_dataset = sl.ArrayDataset(demonstrations, columns)

    # Timed Partial Order Inference
    inference = sl.TPOInferenceAlgorithm()
    specification: sl.Specification = inference.infer(timedtrace_dataset)


if __name__ == "__main__":
    main()

