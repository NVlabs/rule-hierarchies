from typing import Generic, Tuple, TypeVar, Union

import stlcg
import torch

T = TypeVar("T")


class Rule(Generic[T]):
    """
    A Rule[T] is a rule which operates on objects of type T
    Internally, rules are implemented using STL.
    The function as_stl_formula defines the rule as an stlcg.Expression
    and prepare_signals(input) transforms objects of type T into
    signals to evaluate using the stl formula
    """

    @staticmethod
    def shape_stl_signal(x: torch.Tensor) -> torch.Tensor:
        return x.flip(0).view([1, x.shape[0], 1])

    @staticmethod
    def shape_stl_signal_batch(x: torch.Tensor) -> torch.Tensor:
        return x.flip(1).view([x.shape[0], x.shape[1], 1])

    def as_stl_formula(self) -> stlcg.Expression:
        raise NotImplementedError

    def prepare_signals(self, input: T) -> Union[torch.Tensor, Tuple[torch.Tensor]]:
        """
        Given a signal, return the input to the stl formula (or a tuple of inputs to each of the subformulas)
        Note: stlcg.Expressions expect inputs to have the time dimension reversed!
        """
        raise NotImplementedError

    def set_scene_attributes(self, scene_attributes) -> None:
        self.scene_attributes = scene_attributes
