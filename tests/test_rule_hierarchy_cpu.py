import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import unittest
from rule_hierarchy.rule_hierarchy import RuleHierarchy
from rule_hierarchy.rules import AlwaysWithin, AvoidCircle, StayWithinStrip
from test_utils import Dynamics


class RankPreservingTestCPU(unittest.TestCase):
    def setUp(self) -> None:
        rules = [AlwaysWithin(0.0, 2.0), AlwaysWithin(1.0, 3.0)]
        scaling = [1.0, 1.0]
        self.rule_hierarchy = RuleHierarchy[torch.Tensor](rules, scaling, device="cpu")

        self.B, self.T = (
            10000,
            11,
        )  # B is the batch size and T is the horizon of the signal
        self.traj_rank_1 = torch.rand((self.B, self.T)) + 1.0  # signal within [1.0,2.0]
        self.traj_rank_2 = torch.rand((self.B, self.T)) + 0.0  # signal within [0.0,1.0]
        self.traj_rank_3 = torch.rand((self.B, self.T)) + 2.0  # signal within [1.0,2.0]
        self.traj_rank_4 = torch.rand((self.B, self.T)) + 3.0  # signal within [3.0,4.0]

    def test_hierarchy_preserving(self):
        """
        Check exact rank-preserving loss
        """
        loss_rank_1: torch.Tensor = self.rule_hierarchy.traj_cost(
            self.traj_rank_1.to("cpu")
        )
        loss_rank_2: torch.Tensor = self.rule_hierarchy.traj_cost(
            self.traj_rank_2.to("cpu")
        )
        loss_rank_3: torch.Tensor = self.rule_hierarchy.traj_cost(
            self.traj_rank_3.to("cpu")
        )
        loss_rank_4: torch.Tensor = self.rule_hierarchy.traj_cost(
            self.traj_rank_4.to("cpu")
        )

        # maximum loss of higher rank traj < minimum loss of lower rank traj
        self.assertLessEqual(loss_rank_1.max().item(), loss_rank_2.min().item())
        self.assertLessEqual(loss_rank_2.max().item(), loss_rank_3.min().item())
        self.assertLessEqual(loss_rank_3.max().item(), loss_rank_4.min().item())

    def test_diff_hierarchy_preserving(self):
        """
        Check approximate differentiable rank-preserving loss
        """
        diff_loss_rank_1: torch.Tensor = self.rule_hierarchy.diff_traj_cost(
            self.traj_rank_1.to("cpu")
        )
        diff_loss_rank_2: torch.Tensor = self.rule_hierarchy.diff_traj_cost(
            self.traj_rank_2.to("cpu")
        )
        diff_loss_rank_3: torch.Tensor = self.rule_hierarchy.diff_traj_cost(
            self.traj_rank_3.to("cpu")
        )
        diff_loss_rank_4: torch.Tensor = self.rule_hierarchy.diff_traj_cost(
            self.traj_rank_4.to("cpu")
        )

        # atleast p=99% of the samples satisfy the rank-preserving property
        p = 0.99
        self.assertGreaterEqual(
            torch.sum(diff_loss_rank_1 < diff_loss_rank_2).item() / self.B, p
        )
        self.assertGreaterEqual(
            torch.sum(diff_loss_rank_2 < diff_loss_rank_3).item() / self.B, p
        )
        self.assertGreaterEqual(
            torch.sum(diff_loss_rank_3 < diff_loss_rank_4).item() / self.B, p
        )


class ContOptimizerTestGPU(unittest.TestCase):
    """
    Solve an optimization problem for avoiding a pothole (circle)
    while staying within a given lane (given by two lines). Avoiding
    pothole has higher priority than staying within the lane
    """

    def setUp(self) -> None:
        T = 20
        self.x_init = torch.zeros(2)
        self.x_init.requires_grad_(True)
        self.u_init = torch.zeros(T)
        self.dt = 0.1
        self.v = 5.0
        self.lr = 1.0
        self.grad_tol = 1e-6
        self.max_num_itr = 500
        self.vis = False

    def cont_planner(self, rule_hierarchy: RuleHierarchy) -> None:
        dyn = Dynamics(self.u_init.clone(), self.dt, self.v)
        optimizer = optim.Adam(dyn.parameters(), self.lr)
        u_grad_norm = 1e6
        itr = 0
        while u_grad_norm > self.grad_tol and itr < self.max_num_itr:
            optimizer.zero_grad()
            traj = dyn(self.x_init)
            loss = rule_hierarchy.diff_traj_cost(traj.to("cpu"))
            loss.backward()
            optimizer.step()
            u_grad_norm = torch.linalg.norm(dyn.u.grad)
            itr += 1
            if itr % 50 == 0:
                print(f"Itr {itr} | loss: {loss.item()}")
        self.u = dyn.u.data
        self.traj: torch.Tensor = dyn(self.x_init)

    def create_folder(self, folder_name: str):
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

    def visualize(
        self,
        obs_center_x: float,
        obs_center_y: float,
        obs_radius: float,
        lower_bound_y: float,
        upper_bound_y: float,
        save_file_name: str,
    ) -> None:
        fig, ax = plt.subplots()
        circle = plt.Circle(
            (obs_center_x, obs_center_y),
            obs_radius,
            edgecolor="black",
            facecolor="none",
        )
        ax.add_patch(circle)
        ax.set_xlim(0.0, 10.0)
        ax.set_ylim(-5.0, 5.0)
        ax.set_aspect("equal")

        ax.axhline(lower_bound_y, color="black")
        ax.axhline(upper_bound_y, color="black")

        traj_np = self.traj.cpu().detach().numpy()
        plt.plot(traj_np[..., 0], traj_np[..., 1], color="blue")

        self.create_folder("plots")
        plt.savefig("plots/" + save_file_name)
        plt.close()

    def test_cont_optimization_feasible(self):
        """
        Feasible to avoid the pothole and stay within the lane
        """
        obs_center_x = 5.0
        obs_center_y = 0.0
        obs_radius = 2.0
        lower_bound_y = -4.0
        upper_bound_y = 4.0
        scaling = [1.0, 1.0]
        rules = [
            AvoidCircle(obs_center_x, obs_center_y, obs_radius),
            StayWithinStrip(lower_bound_y, upper_bound_y),
        ]
        rule_hierarchy = RuleHierarchy[torch.Tensor](rules, scaling, device="cpu")
        self.cont_planner(rule_hierarchy)
        if self.vis:
            self.visualize(
                obs_center_x,
                obs_center_y,
                obs_radius,
                lower_bound_y,
                upper_bound_y,
                "test_feasible_cpu.png",
            )
        _, scaled_robustness_vector = rule_hierarchy.traj_cost(
            self.traj.to("cpu"), get_robustness_vector=True
        )
        scaled_robustness_vector.squeeze_()

        # both rules should be satisfied => both robustness should be positive
        self.assertGreaterEqual(scaled_robustness_vector[0].item(), 0.0)
        self.assertGreaterEqual(scaled_robustness_vector[1].item(), 0.0)

    def test_cont_optimization_infeasible_avoid_circle(self):
        """
        Infeasible to avoid the pothole and stay within the lane.
        Pothole avoidance has priority.
        """
        obs_center_x = 5.0
        obs_center_y = 0.0
        obs_radius = 2.0
        lower_bound_y = -2.0
        upper_bound_y = 2.0
        scaling = [1.0, 1.0]
        rules = [
            AvoidCircle(obs_center_x, obs_center_y, obs_radius),
            StayWithinStrip(lower_bound_y, upper_bound_y),
        ]
        rule_hierarchy = RuleHierarchy[torch.Tensor](rules, scaling, device="cpu")
        self.cont_planner(rule_hierarchy)
        if self.vis:
            self.visualize(
                obs_center_x,
                obs_center_y,
                obs_radius,
                lower_bound_y,
                upper_bound_y,
                "test_infeasible_avoid_pothole_cpu.png",
            )
        _, scaled_robustness_vector = rule_hierarchy.traj_cost(
            self.traj.to("cpu"), get_robustness_vector=True
        )
        scaled_robustness_vector.squeeze_()

        # both rules cannot be satisfied, only the more important one
        # for avoiding the circle should be satisfied and should have
        # positive robustness while the other rule should have negative
        self.assertGreaterEqual(scaled_robustness_vector[0].item(), 0.0)
        self.assertLessEqual(scaled_robustness_vector[1].item(), 0.0)

    def test_cont_optimization_infeasible_avoid_boundary(self):
        """
        Infeasible to avoid the pothole and stay within the lane.
        Staying within the lane has priority.
        """
        obs_center_x = 5.0
        obs_center_y = 0.0
        obs_radius = 2.0
        lower_bound_y = -2.0
        upper_bound_y = 2.0
        scaling = [1.0, 1.0]
        rules = [
            StayWithinStrip(lower_bound_y, upper_bound_y),
            AvoidCircle(obs_center_x, obs_center_y, obs_radius),
        ]
        rule_hierarchy = RuleHierarchy[torch.Tensor](rules, scaling, device="cpu")
        self.lr = 0.01
        self.cont_planner(rule_hierarchy)
        if self.vis:
            self.visualize(
                obs_center_x,
                obs_center_y,
                obs_radius,
                lower_bound_y,
                upper_bound_y,
                "test_infeasible_stay_within_lane_cpu.png",
            )
        _, scaled_robustness_vector = rule_hierarchy.traj_cost(
            self.traj.to("cpu"), get_robustness_vector=True
        )
        scaled_robustness_vector.squeeze_()

        # both rules cannot be satisfied, only the more important one
        # for staying within the lane should be satisfied and should have
        # positive robustness while the other rule should have negative
        self.assertGreaterEqual(scaled_robustness_vector[0].item(), 0.0)
        self.assertLessEqual(scaled_robustness_vector[1].item(), 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
