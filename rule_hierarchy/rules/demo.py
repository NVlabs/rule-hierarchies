from typing import Tuple, Union

import stlcg
import torch

from rule_hierarchy.rules.rule import Rule


class AlwaysGreater(Rule[torch.Tensor]):
    def __init__(self, rule_threshold: float) -> None:
        self.rule_threshold = rule_threshold

    def as_stl_formula(self) -> stlcg.Expression:
        stl_expression = stlcg.Expression("input", [], reversed=True)
        return stlcg.Always(stl_expression > self.rule_threshold)

    def prepare_signals(
        self, traj: torch.Tensor
    ) -> Union[torch.Tensor, Tuple[torch.Tensor]]:
        input = traj
        assert (
            input.dim() == 2
        ), "input should have shape [B,T]. B is the batch size of trajectories and T is the horizon"
        return self.shape_stl_signal_batch(input)


class AlwaysLesser(Rule[torch.Tensor]):
    def __init__(self, rule_threshold: float) -> None:
        self.rule_threshold = rule_threshold

    def as_stl_formula(self) -> stlcg.Expression:
        stl_expression = stlcg.Expression("input", [], reversed=True)
        return stlcg.Always(stl_expression < self.rule_threshold)

    def prepare_signals(
        self, traj: torch.Tensor
    ) -> Union[torch.Tensor, Tuple[torch.Tensor]]:
        input = traj
        assert (
            input.dim() == 2
        ), "input should have shape [B,T]. B is the batch size of trajectories and T is the horizon"
        return self.shape_stl_signal_batch(input)


class EventuallyAlwaysLess(Rule[torch.Tensor]):
    def __init__(self, rule_threshold: float) -> None:
        self.rule_threshold = rule_threshold

    def as_stl_formula(self) -> stlcg.Expression:
        stl_expression = stlcg.Expression("input", [], reversed=True)
        return stlcg.Eventually(
            subformula=stlcg.Always(stl_expression < self.rule_threshold),
        )

    def prepare_signals(
        self, traj: torch.Tensor
    ) -> Union[torch.Tensor, Tuple[torch.Tensor]]:
        input = traj
        assert (
            input.dim() == 2
        ), "input should have shape [B,T]. B is the batch size of trajectories and T is the horizon"
        return self.shape_stl_signal_batch(input)
