from abc import ABCMeta


class Strategy(metaclass=ABCMeta):
    def __init__(self) -> None:
        pass

    def action(self, state) -> None:
        pass
