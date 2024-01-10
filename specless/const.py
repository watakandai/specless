from bidict import bidict

# Map of object type to integers
OBJECT_TO_IDX = {
    "unseen": 0,
    "empty": 1,
    "wall": 2,
    "floor": 3,
    "door": 4,
    "key": 5,
    "ball": 6,
    "box": 7,
    "goal": 8,
    "lava": 9,
    "agent": 10,
}

IDX_TO_OBJECT = dict(zip(OBJECT_TO_IDX.values(), OBJECT_TO_IDX.keys()))

COLOR_TO_IDX = {
    "red": 0,
    "green": 1,
    "blue": 2,
    "purple": 3,
    "yellow": 4,
    "grey": 5,
}
IDX_TO_COLOR = dict(zip(COLOR_TO_IDX.values(), COLOR_TO_IDX.keys()))
DIR_TO_STRING = bidict({0: "right", 1: "down", 2: "left", 3: "up"})
MINIGRID_TO_GRAPHVIZ_COLOR = {
    "red": "firebrick",
    "green": "darkseagreen1",
    "blue": "steelblue1",
    "purple": "mediumpurple1",
    "yellow": "yellow",
    "grey": "gray60",
}
