import numpy as np
import torch
import matplotlib.axes
import matplotlib.pyplot as plt
from trajdata.maps.vec_map_elements import Polyline
from typing import Optional


def project_point_on_polyline_xyz(
    xyzh: np.ndarray,
    pl: Polyline,
    num_points: Optional[int] = None,
) -> np.ndarray:
    """
    Compute the distance of points in traj_in_world from polyline pl
    """

    xyz = xyzh[:, np.newaxis, :3]

    # p0, p1 are (1, N, 3)
    p0: np.ndarray = pl.points[np.newaxis, :-1, :3]
    p1: np.ndarray = pl.points[np.newaxis, 1:, :3]

    # Compute projections of each point to each line segment in a
    # batched manner.
    line_seg_diffs: np.ndarray = p1 - p0
    point_seg_diffs: np.ndarray = xyz - p0

    dot_products: np.ndarray = (point_seg_diffs * line_seg_diffs).sum(
        axis=-1, keepdims=True
    )
    norms: np.ndarray = np.linalg.norm(line_seg_diffs, axis=-1, keepdims=True) ** 2

    # Clip ensures that the projected point stays within the line segment boundaries.
    # projs are projections of each point in xyzh on each line segment in the polyline
    projs: np.ndarray = (
        p0 + np.clip(dot_products / norms, a_min=0, a_max=1) * line_seg_diffs
    )

    return projs


def project_point_on_polyline_xyzh_pt(
    xyzh: torch.Tensor,
    pl: Polyline,
    device: Optional[str] = "cuda:0",
    has_heading: bool = True,
) -> np.ndarray:
    """Project the given points onto this Polyline.

    Args:
        xyzh (np.ndarray): Points to project, of shape (M, D)

    Returns:
        np.ndarray: The projected points, of shape (M, D)

    Note:
        D = 4 if this Polyline has headings, otherwise D = 3
    """
    # xyzh is now (M, 1, 3), we do not use heading for projection.
    xyz = xyzh[:, None, :3]
    pl_points_pt = torch.Tensor(pl.points).to(device)

    projs = project_point_on_polyline_xyz_pt(xyzh, pl, device=device)

    # 2. Find the nearest projections to the original points.
    closest_proj_idxs: int = torch.linalg.norm(xyz - projs, axis=-1).argmin(axis=-1)

    if has_heading:
        # Adding in the heading of the corresponding p0 point (which makes
        # sense as p0 to p1 is a line => same heading along it).
        return torch.concat(
            [
                projs[range(xyz.shape[0]), closest_proj_idxs],
                torch.unsqueeze(pl_points_pt[closest_proj_idxs, -1], axis=-1),
            ],
            axis=-1,
        )
    else:
        return projs[range(xyz.shape[0]), closest_proj_idxs]


def project_point_on_polyline_xyz_pt(
    xyzh: torch.Tensor,
    pl: Polyline,
    num_points: Optional[int] = None,
    device: Optional[str] = "cuda:0",
) -> torch.Tensor:
    """
    Compute the distance of points in traj_in_world from polyline pl
    """

    xyz = xyzh[:, None, :3]
    pl_points = torch.Tensor(pl.points).to(device)

    # p0, p1 are (1, N, 3)
    p0: torch.Tensor = pl_points[None, :-1, :3]
    p1: torch.Tensor = pl_points[None, 1:, :3]

    # Compute projections of each point to each line segment in a
    # batched manner.
    line_seg_diffs: torch.Tensor = p1 - p0
    point_seg_diffs: torch.Tensor = xyz - p0

    dot_products: torch.Tensor = (point_seg_diffs * line_seg_diffs).sum(
        axis=-1, keepdims=True
    )
    norms: torch.Tensor = torch.linalg.norm(line_seg_diffs, axis=-1, keepdims=True) ** 2
    if torch.min(norms).item() == 0:
        indices = torch.nonzero(norms == 0)
        for i in range(indices.shape[0]):
            norms[list(indices[i, ...])] = 1e-5

    # Clip ensures that the projected point stays within the line segment boundaries.
    # projs are projections of each point in xyzh on each line segment in the polyline
    projs: torch.Tensor = (
        p0 + torch.clip(dot_products / norms, min=0, max=1) * line_seg_diffs
    )

    return projs


def distance_from_polyline(
    xyzh: np.ndarray,
    pl: Polyline,
    num_points: Optional[int] = None,
) -> np.ndarray:
    assert (
        xyzh.ndim == 3
    ), "xyzh should have shape [B,T,4]. B is the batch size of ego trajectories, T is the horizon, and 4 is the dimension of xyzh"
    if num_points is not None:
        if pl.points.shape[0] < num_points:
            pl = pl.interpolate(num_pts=num_points)

    B = xyzh.shape[0]
    T = xyzh.shape[1]
    D = xyzh.shape[2]
    # for batch distance computation turn 3D tensor to a 2D array
    xyzh_flattened = xyzh.reshape((B * T, D))
    xyz = xyzh_flattened[:, np.newaxis, :3]
    projs = project_point_on_polyline_xyz(xyzh_flattened, pl)
    distance_from_pl: np.array = np.linalg.norm(
        xyz[..., :2] - projs[..., :2], axis=-1
    ).min(axis=-1)

    return distance_from_pl.reshape(B, T)


def distance_from_polyline_pt(
    xyzh: np.ndarray,
    pl: Polyline,
    num_points: Optional[int] = None,
    device: Optional[str] = "cuda:0",
) -> torch.Tensor:
    assert (
        xyzh.ndim == 3
    ), "xyzh should have shape [B,T,4]. B is the batch size of ego trajectories, T is the horizon, and 4 is the dimension of xyzh"
    if num_points is not None:
        if pl.points.shape[0] < num_points:
            pl = pl.interpolate(num_pts=num_points)

    B = xyzh.shape[0]
    T = xyzh.shape[1]
    D = xyzh.shape[2]
    # for batch distance computation turn 3D tensor to a 2D array
    xyzh_flattened = xyzh.reshape((B * T, D))
    xyz = torch.Tensor(xyzh_flattened[:, None, :3]).to(device)
    projs = project_point_on_polyline_xyz_pt(
        torch.Tensor(xyzh_flattened).to(device), pl
    )
    distance_from_pl: torch.Tensor = (
        torch.linalg.norm(xyz[..., :2] - projs[..., :2], axis=-1).min(axis=-1).values
    )

    return distance_from_pl.view(B, T)


def heading_distance_from_polyline(
    xyzh: np.ndarray,
    pl: Polyline,
    num_points: Optional[int] = None,
) -> np.ndarray:
    assert (
        xyzh.ndim == 3
    ), "xyzh should have shape [B,T,4]. B is the batch size of ego trajectories, T is the horizon, and 4 is the dimension of xyzh"
    if num_points is not None:
        if pl.points.shape[0] < num_points:
            pl = pl.interpolate(num_pts=num_points)

    B = xyzh.shape[0]
    T = xyzh.shape[1]
    D = xyzh.shape[2]
    # for batch distance computation turn 3D tensor to a 2D array
    xyzh_flattened = xyzh.reshape((B * T, D))
    projs = pl.project_onto(xyzh_flattened)
    projs = projs.reshape((B, T, -1))
    return np.abs(xyzh[..., -1] - projs[..., -1])


def heading_distance_from_polyline_pt(
    xyzh: np.ndarray,
    pl: Polyline,
    num_points: Optional[int] = None,
    device: Optional[str] = "cuda:0",
) -> np.ndarray:
    assert (
        xyzh.ndim == 3
    ), "xyzh should have shape [B,T,4]. B is the batch size of ego trajectories, T is the horizon, and 4 is the dimension of xyzh"
    if num_points is not None:
        if pl.points.shape[0] < num_points:
            pl = pl.interpolate(num_pts=num_points)

    B = xyzh.shape[0]
    T = xyzh.shape[1]
    D = xyzh.shape[2]

    xyzh = torch.Tensor(xyzh).to(device)
    # for batch distance computation turn 3D tensor to a 2D array
    xyzh_flattened = xyzh.view((B * T, D))
    projs = project_point_on_polyline_xyzh_pt(xyzh_flattened, pl)
    projs = projs.view((B, T, -1))
    return torch.abs(xyzh[..., -1] - projs[..., -1])


def distance_along_polyline(
    xyzh: np.ndarray,
    pl: Polyline,
    num_points: Optional[int] = None,
) -> np.ndarray:
    assert (
        xyzh.ndim == 3
    ), "xyzh should have shape [B,T,4]. B is the batch size of ego trajectories, T is the horizon, and 4 is the dimension of xyzh"
    if num_points is not None:
        if pl.points.shape[0] < num_points:
            pl = pl.interpolate(num_pts=num_points)

    B = xyzh.shape[0]
    T = xyzh.shape[1]
    D = xyzh.shape[2]
    # for batch distance computation turn 3D tensor to a 2D array
    xyzh_flattened = xyzh.reshape((B * T, D))
    xyz = xyzh_flattened[:, np.newaxis, :3]
    projs = project_point_on_polyline_xyz(xyzh_flattened, pl)

    pl_length_vec = get_polyline_length_vec(pl)

    closest_point_on_pl = np.linalg.norm(xyz - projs, axis=-1).argmin(axis=-1)
    closest_pl_points_to_proj = pl.points[closest_point_on_pl, ...]
    closest_proj_points = projs[
        np.arange(closest_pl_points_to_proj.shape[0]), closest_point_on_pl, :
    ]
    distance_between_closest_pl_and_proj_points = np.linalg.norm(
        closest_pl_points_to_proj[..., :2] - closest_proj_points[..., :2], axis=-1
    )
    distance_upto_closest_point_on_pl = pl_length_vec[closest_point_on_pl]

    distance_along = (
        distance_upto_closest_point_on_pl + distance_between_closest_pl_and_proj_points
    )
    progress_percent = distance_along / pl_length_vec[-1]

    return distance_along.reshape(B, T), progress_percent.reshape(B, T)


def distance_along_polyline_pt(
    xyzh: np.ndarray,
    pl: Polyline,
    num_points: Optional[int] = None,
    device: Optional[str] = "cuda:0",
) -> np.ndarray:
    assert (
        xyzh.ndim == 3
    ), "xyzh should have shape [B,T,4]. B is the batch size of ego trajectories, T is the horizon, and 4 is the dimension of xyzh"
    if num_points is not None:
        if pl.points.shape[0] < num_points:
            pl = pl.interpolate(num_pts=num_points)

    B = xyzh.shape[0]
    T = xyzh.shape[1]
    D = xyzh.shape[2]
    # for batch distance computation turn 3D tensor to a 2D array
    xyzh_flattened = xyzh.reshape((B * T, D))
    xyz = torch.Tensor(xyzh_flattened[:, np.newaxis, :3]).to(device)
    projs = project_point_on_polyline_xyz_pt(
        torch.Tensor(xyzh_flattened).to(device), pl
    )

    pl_length_vec = get_polyline_length_vec(pl)
    pl_length_vec = torch.Tensor(pl_length_vec).to(device)

    pl_points_pt = torch.Tensor(pl.points).to(device)
    closest_point_on_pl = torch.linalg.norm(xyz - projs, axis=-1).argmin(axis=-1)
    closest_pl_points_to_proj = pl_points_pt[closest_point_on_pl, ...]
    closest_proj_points = projs[
        torch.arange(closest_pl_points_to_proj.shape[0]), closest_point_on_pl, :
    ]
    distance_between_closest_pl_and_proj_points = torch.linalg.norm(
        closest_pl_points_to_proj[..., :2] - closest_proj_points[..., :2], axis=-1
    )
    distance_upto_closest_point_on_pl = pl_length_vec[closest_point_on_pl]

    distance_along = (
        distance_upto_closest_point_on_pl + distance_between_closest_pl_and_proj_points
    )
    progress_percent = distance_along / pl_length_vec[-1]

    return distance_along.view(B, T), progress_percent.view(B, T)


def get_polyline_length_vec(pl: Polyline):
    """
    Takes a polyline and returns a numpy array that provides polyline
    length upto each point; e.g. if pl is [(0,0),(1,0),(1,1)],
    the output is [1,1+1] = [1,2].
    """
    # p0, p1 are (1, N, 3)
    p0: np.ndarray = pl.points[np.newaxis, :-1, :3]
    p1: np.ndarray = pl.points[np.newaxis, 1:, :3]

    # Compute projections of each point to each line segment in a
    # batched manner.
    line_seg_diffs: np.ndarray = p1 - p0
    if pl.points.shape[0] > 2:
        n = line_seg_diffs.squeeze().shape[0]
        A = np.tril(np.ones(n))
        pl_length_vec = A @ np.linalg.norm(line_seg_diffs, axis=-1).squeeze()
        pl_length_vec = np.concatenate((np.zeros(1), pl_length_vec))
    else:
        pl_length_vec = np.array([np.linalg.norm(line_seg_diffs.squeeze())])
    return pl_length_vec


def get_polyline_length(pl: Polyline):
    return get_polyline_length_vec(pl)[-1]


def visualize_distance_polylines(
    pl: Polyline,
    x_lim: Optional[np.ndarray] = [-5.0, 5.0],
    y_lim: Optional[np.ndarray] = [-5.0, 5.0],
    num_grid_x=100,
    num_grid_y=100,
) -> matplotlib.axes:
    xlist = np.linspace(x_lim[0], x_lim[1], num_grid_x)
    ylist = np.linspace(y_lim[0], y_lim[1], num_grid_y)
    X, Y = np.meshgrid(xlist, ylist)
    Z = np.zeros_like(X)
    num_plot_points = num_grid_x * num_grid_y
    xyzh = np.array(
        [
            X.reshape(num_plot_points),
            Y.reshape(num_plot_points),
            np.zeros(num_plot_points),
            np.zeros(num_plot_points),
        ]
    ).transpose()
    distance_from_pl = distance_from_polyline(
        xyzh.reshape(1, num_plot_points, -1), pl, num_points=100
    )
    distance_along_pl, _ = distance_along_polyline(
        xyzh.reshape(1, num_plot_points, -1), pl, num_points=100
    )

    distance_from_pl = distance_from_pl.reshape((num_grid_x, num_grid_y))
    distance_along_pl = distance_along_pl.reshape((num_grid_x, num_grid_y))

    fig, ax = plt.subplots(dpi=150)
    cp = ax.contourf(X, Y, distance_from_pl)

    fig, ax = plt.subplots(dpi=150)
    cp = ax.contourf(X, Y, distance_along_pl)

    return ax


def test_lane_utils():
    pl_x = np.linspace(0, 10, 31)
    pl_y = np.zeros_like(pl_x)
    pl_z = np.zeros_like(pl_x)
    points = np.stack([pl_x, pl_y, pl_z]).transpose()
    pl = Polyline(points)
    xyzh = np.random.rand(729, 7, 4)
    # dist = distance_from_polyline(xyzh, pl)
    # dist_pt = distance_from_polyline_pt(xyzh, pl)
    # print(dist)
    # dist = distance_along_polyline(xyzh, pl)
    # dist_pt = distance_along_polyline_pt(xyzh, pl)
    # print(dist)
    # print(dist_pt)
    heading_error = heading_distance_from_polyline(xyzh, pl)
    heading_error_pt = heading_distance_from_polyline_pt(xyzh, pl)
    print(heading_error)
    # ax = visualize_distance_polylines(pl)
    # plt.show()


if __name__ == "__main__":
    # import pprofile
    # profiler = pprofile.Profile()
    # with profiler:
    test_lane_utils()
    # profiler.print_stats()
    # profiler.dump_stats("/tmp/profiler_stats.txt")
