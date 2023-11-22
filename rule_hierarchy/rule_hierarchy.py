from abc import ABC
from typing import Dict, Generic, List, Optional, Tuple, TypeVar

import stlcg
import torch

from rule_hierarchy.rules.rule import Rule


class AbstractRuleHierarchy(ABC):
    def __init__(
        self,
        ordered_formulae: List[stlcg.formulas.STL_Formula],
        device: Optional[str] = "cuda:0",
    ) -> None:
        """
        Evaluation of STL rule hierarchy
            Inputs:
            ord_nodes (List[STL_Formula]): list of rule classes in descending order of hierarchy
        """
        self.stl_rule_hierarchy = ordered_formulae
        self.depth = len(ordered_formulae)
        self.device = device

    def evaluate_robustness(self, signals: List[Tuple[torch.Tensor]]) -> torch.Tensor:
        """
        Evaluate the rule satisfaction for all rules in the hierarchy
        Input: signals is a list of K time-series of shape [B, T, d]  for each formula in the rulebook
        Output: Tensor of shape [B, K], where the element at index (b,k) represents the min satisfaction of the formula k by the trajectory b
        """
        rule_sat_vec = []  # Rule satisfaction vector
        for rule, inputs in zip(self.stl_rule_hierarchy, signals):
            rule_sat_vec.append(rule.robustness(inputs, scale=-1)[:, 0, 0])

        return torch.stack(rule_sat_vec, dim=1)

    # @TODO: remove reward vec from output
    def _cost(
        self, robustness: torch.Tensor, scaling: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute cost of a trajectory under the rule hierarchy
        Inputs:
            robustness (Tensor): shape [B, K], the robustness value of each trajectory in the batch under each of the K rules
            scaling (Tensor): shape [K], the scaling to apply before squashing the robustness values with tanh
        Outputs:
            Tuple containing:
            - (Tensor) cost vector (cost for each traj in the batch)
            - (Tensor) [B, K] individual squashed robustness values for each rule
        """
        if scaling is None:
            scaling = torch.ones_like(robustness[0]).to(self.device)

        rule_scales = 2.01 ** (self.depth - torch.arange(0.0, self.depth))
        rule_scales = rule_scales.to(self.device)
        reward_vec = rule_scales * (robustness > 0.0).float()

        scaled_rule_robustness = torch.tanh(robustness * scaling)
        reward = reward_vec.sum(-1) + scaled_rule_robustness.mean(-1)

        return -reward, scaled_rule_robustness

    def _diff_cost(
        self, robustness: torch.Tensor, scaling: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute differentiable cost of a trajectory under the rule hierarchy
        Inputs:
            robustness (Tensor): shape [B, K], the robustness value of each trajectory in the batch under each of the K rules
            scaling (Tensor): shape [K], the scaling to apply before squashing the robustness values with tanh
        Outputs:
            Tuple containing:
            - (Tensor) cost vector (cost for each traj in the batch)
            - (Tensor) [B, K] individual squashed robustness values for each rule
        """
        if scaling is None:
            scaling = torch.ones_like(robustness[0])

        rule_scales = 2.01 ** (self.depth - torch.arange(0.0, self.depth)).to(
            self.device
        )
        # reward_vec = (
        #     rule_scales * torch.sigmoid(30 * torch.tanh(robustness * scaling)).float()
        # )

        reward_vec = rule_scales * (torch.tanh(robustness * scaling * 30) + 1) * 0.5

        scaled_rule_robustness = torch.tanh(robustness * scaling)
        reward = reward_vec.sum(-1) + scaled_rule_robustness.mean(-1)

        return -reward, scaled_rule_robustness


T = TypeVar("T")


class RuleHierarchy(AbstractRuleHierarchy, Generic[T]):
    def __init__(
        self,
        rules: List[Rule[T]],
        scaling: List[float],
        names: Optional[List[str]] = None,
        device: Optional[str] = "cuda:0",
    ):
        """
        Takes a list of rules & list of scalings to construct a rule hierarchy
        which can evaluate trajectories.

        scalings are used to pre-multiply signals before squashing to [0,1] with sigmoid

        names are a list of strings to annotate each of the rules with names
        """
        self.names = names
        self.rules = rules
        ordered_formulae = [rule.as_stl_formula() for rule in rules]
        self.scaling = torch.Tensor(scaling).to(device)
        super().__init__(ordered_formulae, device=device)

    def _traj_robustness(self, traj: torch.Tensor):
        signals = [rule.prepare_signals(traj) for rule in self.rules]
        return self.evaluate_robustness(signals)

    def _device_check(self, traj: torch.Tensor):
        if "cuda" in self.device:
            hierarchy_device = "cuda"
        else:
            hierarchy_device = "cpu"

        if "cuda" in traj.device.type:
            traj_device = "cuda"
        else:
            traj_device = "cpu"

        assert (
            hierarchy_device == traj_device
        ), f"Rule hierarchy is on {hierarchy_device} while the trajectory is on {traj_device}.Use same device for both"

    def traj_cost(
        self, traj: torch.Tensor, get_robustness_vector: Optional[bool] = False
    ):
        self._device_check(traj)
        if get_robustness_vector:
            cost, scaled_robustness_vector = self._cost(
                self._traj_robustness(traj), self.scaling
            )
            return cost, scaled_robustness_vector

        else:
            cost, _ = self._cost(self._traj_robustness(traj), self.scaling)
            return cost

    def diff_traj_cost(
        self, traj: torch.Tensor, get_robustness_vector: Optional[bool] = False
    ):
        self._device_check(traj)
        if get_robustness_vector:
            cost, scaled_robustness_vector = self._diff_cost(
                self._traj_robustness(traj), self.scaling
            )
            return cost, scaled_robustness_vector
        else:
            cost, _ = self._diff_cost(self._traj_robustness(traj), self.scaling)
            return cost
