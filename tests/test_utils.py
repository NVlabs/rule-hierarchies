import torch
import torch.nn as nn


def discrete_dyn(
    x: torch.Tensor,
    u: torch.Tensor,
    v: float,
    dt: float,
):
    x_dot = torch.stack(
        [
            v * torch.cos(torch.tanh(u) * torch.pi / 6),
            v * torch.sin(torch.tanh(u) * torch.pi / 6),
        ]
    )
    return x + x_dot * dt


class Dynamics(nn.Module):
    def __init__(self, u_init: torch.Tensor, dt: float, v: float):
        super().__init__()
        self.u = nn.Parameter(u_init)
        self.T = u_init.shape[0]
        self.dt = dt
        self.v = v

    def rollout(self, x0):
        self.x = [x0]
        for t in range(self.T):
            next_x = discrete_dyn(self.x[t], self.u[t], self.v, self.dt)
            self.x.append(next_x)
        return torch.stack(self.x)

    def forward(self, x0):
        return self.rollout(x0)
