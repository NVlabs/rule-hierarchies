import torch

from rule_hierarchy.rule_hierarchy import RuleHierarchy
from rule_hierarchy.rules import AlwaysGreater, AlwaysLesser

if __name__ == "__main__":
    # Define a rule hierarchy as a list of rules, where rules are
    # arranged in decreasing order of importance with the indices
    rules = [AlwaysGreater(1.0), AlwaysLesser(2.0)]
    scaling = [1.0, 1.0]
    device = "cuda:0"  # either cuda or cpu

    rule_hierarchy = RuleHierarchy[torch.Tensor](rules, scaling, device=device)

    B, T = 2, 11  # B is the batch size and T is the horizon of the signal
    traj = torch.rand((B, T)) * 0.1 + 1.0
    loss = rule_hierarchy.traj_cost(traj.to(device))

    print(f"Loss: {loss.cpu().numpy()}")
