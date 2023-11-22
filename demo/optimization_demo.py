import os
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from rule_hierarchy.rule_hierarchy import RuleHierarchy
from rule_hierarchy.rules import AvoidCircle, StayWithinStrip
from tests.test_utils import Dynamics
from termcolor import colored
from typing import List


class StayLaneAvoidPothole:
    """
    Solve an optimization problem for avoiding a pothole (circle)
    while staying within a given lane (given by two lines).
    """

    def __init__(self, device) -> None:
        T = 20  # planning horizon
        self.x_init = torch.zeros(2)  # initial (x,y) position
        self.x_init.requires_grad_(True)
        self.u_init = torch.zeros(T)
        self.dt = 0.1  # time step
        self.v = 5.0  # constant speed
        self.lr = 1.0  # learning rate for optimization
        self.grad_tol = 1e-6  # termination grad tol
        self.max_num_itr = 500  # termination max itr
        self.device = device  # cuda or cpu

    def _cont_planner(self, rule_hierarchy: RuleHierarchy) -> None:
        """
        Continuous Planner
        """
        # vehicle is assumed to move with a constant speed self.v
        # with the only control input being the steering angle self.u
        # which is restricted between -pi/6 and pi/6
        dyn = Dynamics(self.u_init.clone(), self.dt, self.v)
        optimizer = optim.Adam(dyn.parameters(), self.lr)
        u_grad_norm = 1e6
        itr = 0
        while u_grad_norm > self.grad_tol and itr < self.max_num_itr:
            # reset optimizer
            optimizer.zero_grad()

            # forward rollout of the trajectory given the initial state
            # and the current sequence of control inputs dyn.u
            traj = dyn(self.x_init)

            # compute rule hierarchy loss
            loss = rule_hierarchy.diff_traj_cost(traj.to(self.device))

            # backpropagate
            loss.backward()

            # update the sequence of control inputs according to the gradients
            optimizer.step()

            u_grad_norm = torch.linalg.norm(dyn.u.grad)
            itr += 1
            if itr % 50 == 0:
                print(f"Itr {itr} | loss: {loss.item()}")

        # extract the sequence of control inputs
        self.u = dyn.u.data

        # extract the "optimal" trajectory
        self.traj: torch.Tensor = dyn(self.x_init)

    @staticmethod
    def _create_folder(folder_name: str) -> None:
        """
        Creates a folder if it does not exist
        """
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

    @staticmethod
    def _color_text(status: bool) -> str:
        """
        Color codes True to green and False to red
        """
        if status:
            color = "green"
        else:
            color = "red"
        return colored(text=str(status), color=color)

    @staticmethod
    def _rule_status(scaled_robustness_vector: torch.Tensor) -> List[str]:
        """
        Converts robustness vectors to color coded True / False
        """
        satisfied: List[bool] = [
            scaled_robustness_vector[i].item() > 0
            for i in range(scaled_robustness_vector.numel())
        ]
        return [
            StayLaneAvoidPothole._color_text(rule_status) for rule_status in satisfied
        ]

    def _visualize(
        self,
        obs_center_x: float,
        obs_center_y: float,
        obs_radius: float,
        lower_bound_y: float,
        upper_bound_y: float,
        save_file_name: str,
    ) -> None:
        """
        Plots the final trajectory
        """
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

        StayLaneAvoidPothole._create_folder("plots")
        plt.savefig("plots/" + save_file_name)
        plt.close()

    def feasible(self):
        """
        Feasible to avoid the pothole and stay within the lane
        Rule Hierarchy: Pothole Avoidance > Stay Within Lane
        """
        print("===========================================")
        print("Feasible: Avoid Potholes > Stay Within Lane")
        print("===========================================")

        obs_center_x = 5.0
        obs_center_y = 0.0
        obs_radius = 2.0
        lower_bound_y = -4.0
        upper_bound_y = 4.0
        scaling = [1.0, 1.0]

        # Create rule hierarchy
        rules = [
            AvoidCircle(obs_center_x, obs_center_y, obs_radius),
            StayWithinStrip(lower_bound_y, upper_bound_y),
        ]
        rule_hierarchy = RuleHierarchy[torch.Tensor](rules, scaling, device=self.device)

        # Plan with the rule hierarchy
        self._cont_planner(rule_hierarchy)

        # Visualize the optimizer's output
        self._visualize(
            obs_center_x,
            obs_center_y,
            obs_radius,
            lower_bound_y,
            upper_bound_y,
            "demo_feasible.png",
        )

        # Print status
        _, scaled_robustness_vector = rule_hierarchy.traj_cost(
            self.traj.to(self.device), get_robustness_vector=True
        )
        scaled_robustness_vector.squeeze_()
        rule_status = StayLaneAvoidPothole._rule_status(scaled_robustness_vector)
        pothole_status = rule_status[0]
        stay_within_lane_status = rule_status[1]
        print(
            f"\nPot Hole Avoidance: {pothole_status} \nStay Within Lane  : {stay_within_lane_status}\n"
        )

    def infeasible_avoid_pothole(self):
        """
        Infeasible to avoid the pothole and stay within the lane
        Rule Hierarchy: Pothole Avoidance > Stay Within Lane
        """
        print("=============================================")
        print("Infeasible: Avoid Potholes > Stay Within Lane")
        print("=============================================")

        obs_center_x = 5.0
        obs_center_y = 0.0
        obs_radius = 2.0
        lower_bound_y = -2.0
        upper_bound_y = 2.0
        scaling = [1.0, 1.0]

        # Create rule hierarchy
        rules = [
            AvoidCircle(obs_center_x, obs_center_y, obs_radius),
            StayWithinStrip(lower_bound_y, upper_bound_y),
        ]
        rule_hierarchy = RuleHierarchy[torch.Tensor](rules, scaling, device=self.device)

        # Plan with the rule hierarchy
        self._cont_planner(rule_hierarchy)

        # Visualize the optimizer's output
        self._visualize(
            obs_center_x,
            obs_center_y,
            obs_radius,
            lower_bound_y,
            upper_bound_y,
            "demo_infeasible_avoid_pothole.png",
        )

        # Print status
        _, scaled_robustness_vector = rule_hierarchy.traj_cost(
            self.traj.to(self.device), get_robustness_vector=True
        )
        scaled_robustness_vector.squeeze_()
        rule_status = StayLaneAvoidPothole._rule_status(scaled_robustness_vector)
        pothole_status = rule_status[0]
        stay_within_lane_status = rule_status[1]
        print(
            f"\nPot Hole Avoidance: {pothole_status} \nStay Within Lane  : {stay_within_lane_status}\n"
        )

    def infeasible_stay_within_lane(self):
        """
        Infeasible to avoid the pothole and stay within the lane.
        Rule Hierarchy: Stay Within Lane > Pothole Avoidance
        """
        print("=============================================")
        print("Infeasible: Stay Within Lane > Avoid Potholes")
        print("=============================================")

        obs_center_x = 5.0
        obs_center_y = 0.0
        obs_radius = 2.0
        lower_bound_y = -2.0
        upper_bound_y = 2.0
        scaling = [1.0, 1.0]

        # Create rule hierarchy
        rules = [
            StayWithinStrip(lower_bound_y, upper_bound_y),
            AvoidCircle(obs_center_x, obs_center_y, obs_radius),
        ]
        rule_hierarchy = RuleHierarchy[torch.Tensor](rules, scaling, device=self.device)
        self.lr = 0.01

        # Plan with the rule hierarchy
        self._cont_planner(rule_hierarchy)

        # Visualize the optimizer's output
        self._visualize(
            obs_center_x,
            obs_center_y,
            obs_radius,
            lower_bound_y,
            upper_bound_y,
            "demo_infeasible_stay_within_lane.png",
        )
        _, scaled_robustness_vector = rule_hierarchy.traj_cost(
            self.traj.to(self.device), get_robustness_vector=True
        )

        # Print status
        scaled_robustness_vector.squeeze_()
        rule_status = StayLaneAvoidPothole._rule_status(scaled_robustness_vector)
        pothole_status = rule_status[1]
        stay_within_lane_status = rule_status[0]
        print(
            f"\nPot Hole Avoidance: {pothole_status} \nStay Within Lane  : {stay_within_lane_status}\n"
        )


if __name__ == "__main__":
    stay_lane_avoid_pothole = StayLaneAvoidPothole("cuda:0")

    # Both pothole avoidance and staying within lane is feasible
    stay_lane_avoid_pothole.feasible()

    # Both pothole avoidance and staying within lane is infeasible
    # Pothole Avoidance > Staying Within Lane
    stay_lane_avoid_pothole.infeasible_avoid_pothole()

    # Both pothole avoidance and staying within lane is infeasible
    # Staying Within Lane > Pothole Avoidance
    stay_lane_avoid_pothole.infeasible_stay_within_lane()
