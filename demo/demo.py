import torch

from rule_hierarchy.rule_hierarchy import RuleHierarchy
from rule_hierarchy.rules import AlwaysGreater, AlwaysLesser

if __name__ == "__main__":
    # Define hierarchy as a list of rules, where earlier
    rules = [AlwaysGreater(1.0), AlwaysLesser(2.0)]
    scaling = [1.0, 1.0]

    rule_hierarchy = RuleHierarchy[torch.Tensor](rules, scaling)

    B, T = 2, 11  # B is the batch size and T is the horizon of the signal
    traj = torch.rand((B, T)) * 0.1 + 1.0
    loss = rule_hierarchy.traj_cost(traj.to("cuda:0"))

    print(f"Loss: {loss.cpu().numpy()}")
