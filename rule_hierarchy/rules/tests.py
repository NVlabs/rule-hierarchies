from typing import Tuple, Union

import stlcg
import torch

from rule_hierarchy.rules.rule import Rule


class AlwaysWithin(Rule[torch.Tensor]):
    def __init__(self, rule_lower_bound: float, rule_upper_bound: float) -> None:
        self.rule_upper_bound = rule_upper_bound
        self.rule_lower_bound = rule_lower_bound

    def as_stl_formula(self) -> stlcg.Expression:
        stl_expression = stlcg.Expression("input", [], reversed=True)
        return stlcg.Always(
            (stl_expression > self.rule_lower_bound)
            & (stl_expression < self.rule_upper_bound)
        )

    def prepare_signals(
        self, traj: torch.Tensor
    ) -> Union[torch.Tensor, Tuple[torch.Tensor]]:
        assert (
            traj.dim() == 2
        ), "input should have shape [B,T]. B is the batch size of trajectories and T is the horizon"
        if traj.dim() == 2:
            signal = self.shape_stl_signal_batch(traj)
        else:
            signal = self.shape_stl_signal(traj)
        return (signal, signal)


class AvoidCircle(Rule[torch.Tensor]):
    def __init__(
        self, obs_center_x: float, obs_center_y: float, obs_radius: float
    ) -> None:
        self.obs_center_x = obs_center_x
        self.obs_center_y = obs_center_y
        self.obs_radius = obs_radius

    def as_stl_formula(self) -> stlcg.Expression:
        stl_expression = stlcg.Expression("input", [], reversed=True)
        return stlcg.Always(stl_expression > self.obs_radius)

    def prepare_signals(
        self, traj: torch.Tensor
    ) -> Union[torch.Tensor, Tuple[torch.Tensor]]:
        if traj.device.type == "cuda":
            device = traj.device.type + ":" + str(traj.device.index)
        else:
            device = traj.device.type
        distance: torch.Tensor = torch.linalg.norm(
            traj - torch.Tensor([self.obs_center_x, self.obs_center_y]).to(device),
            dim=-1,
        )
        if distance.dim() == 2:
            return self.shape_stl_signal_batch(distance)
        else:
            return self.shape_stl_signal(distance)


class StayWithinStrip(Rule[torch.Tensor]):
    def __init__(self, rule_lower_bound: float, rule_upper_bound: float) -> None:
        self.rule_upper_bound = rule_upper_bound
        self.rule_lower_bound = rule_lower_bound

    def as_stl_formula(self) -> stlcg.Expression:
        stl_expression = stlcg.Expression("input", [], reversed=True)
        return stlcg.Always(
            (stl_expression > self.rule_lower_bound)
            & (stl_expression < self.rule_upper_bound)
        )

    def prepare_signals(
        self, traj: torch.Tensor
    ) -> Union[torch.Tensor, Tuple[torch.Tensor]]:
        y = traj[..., 1]
        if y.dim() == 2:
            signal = self.shape_stl_signal_batch(y)
        else:
            signal = self.shape_stl_signal(y)
        return (signal, signal)
