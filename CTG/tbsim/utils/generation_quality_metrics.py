"""
Generation-quality metrics for CTG's SceneDiffuser (`trajdata_nuplan_scene_diff`),
computed purely from the model's own diffusion-generated output -- mirrors
ADCG-Diffusion/JARCD's `jarcd_generation_quality_metrics.py` (physical
validity / map compliance / realism), which is run alongside diversity
(APD/FPD, already computed by `SceneDiffuserTrafficModel._compute_metrics`)
inside the same PyTorch-Lightning validation pass used for training, the
same way ADCG's `forward_generation_eval` + `test_jarcd_generation_metrics.sh`
do it -- see `SceneDiffuserTrafficModel._compute_generation_quality_metrics`
in tbsim/algos/algos.py for the call site.

Every check here reuses the exact coordinate-transform and collision/
off-road primitives already used for guidance losses in guidance_loss.py
(`AgentCollisionLoss`'s disk approximation + `world_from_agent` transform,
`MapCollisionLoss`'s box-corner sampling + `raster_from_agent` transform)
and metrics.py (`batch_detect_off_road`), just reduced to a violation
*rate* instead of a differentiable penalty -- these functions take no
guidance-loss-specific state (no `init_for_batch` caching, no gradient
masking) since they only ever run at validation time, never inside the
denoising loop.

Unlike JARCD, CTG's SceneDiffuser conditions generation on each agent's
real observed history (that is the task itself, not an artifact of this
eval) -- only the *future* is diffused from noise. The real future
(`data_batch["target_positions"]`/`target_availabilities`) is used here
only as the realism-comparison reference distribution for `realism_jsd`,
exactly as JARCD's real `scene_tensor` is kept "only as the
realism-comparison reference distribution and never fed into generation"
(jarcd.py's `forward_generation_eval` docstring).
"""

from typing import Dict, Tuple

import torch

import tbsim.utils.geometry_utils as GeoUtils
import tbsim.utils.metrics as Metrics

# box sampling grid for off-road checks -- same convention/defaults as
# guidance_loss.MapCollisionLoss.
_MAP_NUM_POINTS_LW = (10, 10)


def _agent_exists_mask(data_batch: Dict) -> torch.Tensor:
    """[B, M] bool, True where an agent slot is real (not scene padding)."""
    avail = data_batch["target_availabilities"]  # [B, M, T]
    return avail.any(dim=-1)


def physical_validity_rate(
    predictions: Dict,
    data_batch: Dict,
    acc_limit: float = 4.0,
    yaw_rate_limit: float = 1.0,
    speed_limit: float = 40.0,
    collision_buffer_m: float = 0.2,
) -> Dict[str, torch.Tensor]:
    """
    Fraction of generated (sample, agent) trajectories that violate no
    kinematic limit and are involved in no agent-agent collision --
    design doc section 8.1's "물리 한계 위반 없음" validity criterion,
    ported from JARCD's `compute_j_phys`'s accel/yaw-rate/speed checks
    (jarcd_objectives.py) plus an explicit inter-agent collision check
    (JARCD's own scenes are single anchor/reaction pairs so it doesn't
    need one; CTG's SceneDiffuser jointly generates every agent in the
    scene, so agent-agent collision is part of "physical validity" here).

    Kinematic limits read directly off `predictions["trajectories"]`'s
    (x, y, vel, yaw, acc, yawvel) channels -- the same convention
    `guidance_loss.AccLimitLoss`/`SpeedLimitLoss` use -- no
    finite-differencing needed. Collision uses the same single-circle
    per-agent approximation and `world_from_agent` transform as
    `guidance_loss.AgentCollisionLoss`, evaluated per-scene (the [B, M]
    layout already keeps scenes separate, so no block-diagonal
    scene-masking is needed the way that flat-batch guidance loss does).
    """
    traj = predictions["trajectories"]  # [B, N, M, T, 6]
    B, N, M, T, _ = traj.shape
    device = traj.device
    exists = _agent_exists_mask(data_batch)  # [B, M]
    step_avail = data_batch["target_availabilities"].bool()  # [B, M, T]

    # --- kinematic limits ---
    speed = traj[..., 2].abs()
    acc = traj[..., 4].abs()
    yaw_rate = traj[..., 5].abs()
    kin_violation = (speed > speed_limit) | (acc > acc_limit) | (yaw_rate > yaw_rate_limit)  # [B, N, M, T]
    step_mask = step_avail.unsqueeze(1).expand(B, N, M, T)
    kin_violated = (kin_violation & step_mask).any(dim=-1)  # [B, N, M]

    # --- pairwise agent collision (disk approximation, common world frame) ---
    pos = predictions["positions"]  # [B, N, M, T, 2]
    yaw = predictions["yaws"]  # [B, N, M, T, 1]
    extent = data_batch["extent"][..., :2]  # [B, M, 2] (length, width)
    world_from_agent = data_batch["world_from_agent"]  # [B, M, 3, 3]

    pos_flat = pos.permute(0, 2, 1, 3, 4).reshape(B * M, N, T, 2)
    yaw_flat = yaw.permute(0, 2, 1, 3, 4).reshape(B * M, N, T, 1)
    world_pos, _ = GeoUtils.transform_agents_to_world(
        pos_flat, yaw_flat, world_from_agent.reshape(B * M, 3, 3)
    )
    world_pos = world_pos.reshape(B, M, N, T, 2).permute(0, 2, 1, 3, 4)  # [B, N, M, T, 2]

    radius = 0.5 * extent[..., 1]  # width/2, same single-disk radius convention
    pair_min_dist = radius.unsqueeze(-1) + radius.unsqueeze(-2) + collision_buffer_m  # [B, M, M]

    diff = world_pos.unsqueeze(3) - world_pos.unsqueeze(2)  # [B, N, M, M, T, 2]
    dist = diff.norm(dim=-1)  # [B, N, M, M, T]
    colliding = dist <= pair_min_dist[:, None, :, :, None]

    self_mask = (~torch.eye(M, dtype=torch.bool, device=device))[None, None, :, :, None]
    both_exist = (exists.unsqueeze(-1) & exists.unsqueeze(-2))[:, None, :, :, None]
    both_avail = (step_avail.unsqueeze(2) & step_avail.unsqueeze(1)).unsqueeze(1)  # [B, 1, M, M, T]
    colliding = colliding & self_mask & both_exist & both_avail
    collided = colliding.any(dim=-1).any(dim=-1)  # [B, N, M]

    violation = kin_violated | collided  # [B, N, M]
    valid_mask = exists[:, None, :].expand(B, N, M)
    violation = violation[valid_mask]
    if violation.numel() == 0:
        rate = torch.full((), float("nan"), device=device)
    else:
        rate = (~violation).float().mean()
    return {"generation_physical_validity_rate": rate}


def map_compliance_rate(
    predictions: Dict,
    data_batch: Dict,
    num_points_lw: Tuple[int, int] = _MAP_NUM_POINTS_LW,
) -> Dict[str, torch.Tensor]:
    """
    Fraction of generated (sample, agent, timestep) boxes that stay fully
    on the drivable region -- design doc section 8.1's map-compliance
    criterion (JARCD's `compute_j_map`, ported to CTG's actual rasterized
    `drivable_map` rather than nearest-map-point distance, since CTG has
    a real driveable-region raster available where JARCD didn't).

    Box-corner sampling + `raster_from_agent` transform is the exact
    logic `guidance_loss.MapCollisionLoss.forward` uses, just reduced to
    a violation rate over agents/timesteps instead of a per-agent loss.
    """
    traj = predictions["trajectories"]  # [B, N, M, T, 6]
    B, N, M, T, _ = traj.shape
    device = traj.device
    NA = B * M

    pos = predictions["positions"].permute(0, 2, 1, 3, 4).reshape(NA, N, T, 2)
    yaw = predictions["yaws"].permute(0, 2, 1, 3, 4).reshape(NA, N, T, 1)
    extent = data_batch["extent"][..., :2].reshape(NA, 2)
    raster_from_agent = data_batch["raster_from_agent"].reshape(NA, 3, 3)
    drivable_map = data_batch["drivable_map"].reshape(NA, *data_batch["drivable_map"].shape[-2:])
    step_avail = data_batch["target_availabilities"].bool().reshape(NA, T)
    exists = _agent_exists_mask(data_batch).reshape(NA)
    _, H, W = drivable_map.shape

    lwise = torch.linspace(-0.5, 0.5, num_points_lw[0], device=device)
    wwise = torch.linspace(-0.5, 0.5, num_points_lw[1], device=device)
    local_coords = torch.cartesian_prod(lwise, wwise)  # [P, 2]
    P = local_coords.shape[0]

    pos_flat = pos.reshape(NA * N * T, 2)
    yaw_flat = yaw.reshape(NA * N * T, 1)
    lw_flat = extent[:, None, None].expand(NA, N, T, 2).reshape(NA * N * T, 2)
    raster_from_agent_flat = raster_from_agent[:, None, None].expand(NA, N, T, 3, 3).reshape(NA * N * T, 3, 3)

    cur_loc_coords = local_coords[None].expand(NA * N * T, P, 2) * lw_flat[:, None, :]
    s = torch.sin(yaw_flat).unsqueeze(-1)
    c = torch.cos(yaw_flat).unsqueeze(-1)
    rotM = torch.cat((torch.cat((c, s), dim=-1), torch.cat((-s, c), dim=-1)), dim=-2)
    agt_coords_agent_frame = cur_loc_coords @ rotM + pos_flat.unsqueeze(-2)
    agt_coords_raster = GeoUtils.transform_points_tensor(agt_coords_agent_frame, raster_from_agent_flat)
    agt_coords_raster = agt_coords_raster.reshape(NA, N, T, P, 2).long().detach()
    agt_coords_raster[..., 0] = agt_coords_raster[..., 0].clamp(0, W - 1)
    agt_coords_raster[..., 1] = agt_coords_raster[..., 1].clamp(0, H - 1)

    # leading dim of `positions` just needs to match drivable_map's (NA);
    # all trailing (N, T, P) dims are flattened together internally by
    # batch_detect_off_road via gather -- no need to expand drivable_map.
    off_road_pts = Metrics.batch_detect_off_road(agt_coords_raster, drivable_map)  # [NA, N, T, P]
    box_off_road = off_road_pts.sum(dim=-1) > 0.5  # [NA, N, T], "off road" if any sampled corner/point is

    mask = step_avail[:, None, :].expand(NA, N, T) & exists[:, None, None].expand(NA, N, T)
    violation = box_off_road & mask
    denom = mask.float().sum()
    if denom.item() == 0:
        rate = torch.full((), float("nan"), device=device)
    else:
        rate = 1.0 - (violation.float().sum() / denom)
    return {"generation_map_compliance_rate": rate}


def _histogram(x: torch.Tensor, bins: int, max_val: float) -> torch.Tensor:
    x = x.clamp(min=-max_val, max=max_val)
    return torch.histc(x.float(), bins=bins, min=-max_val, max=max_val)


def _jsd(hist_a: torch.Tensor, hist_b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Jensen-Shannon divergence between two (unnormalized) histograms --
    direct port of JARCD's `jarcd_generation_quality_metrics._jsd`."""
    p = hist_a / hist_a.sum().clamp(min=eps)
    q = hist_b / hist_b.sum().clamp(min=eps)
    m = 0.5 * (p + q)

    def _kl(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        mask = a > 0
        return (a[mask] * (torch.log(a[mask]) - torch.log(b[mask].clamp(min=eps)))).sum()

    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def realism_jsd(
    predictions: Dict,
    data_batch: Dict,
    dt: float,
    speed_bins: int = 32,
    speed_max_mps: float = 30.0,
    accel_bins: int = 32,
    accel_max_mps2: float = 8.0,
) -> Dict[str, torch.Tensor]:
    """
    JSD between the generated batch's speed/acceleration distribution and
    this same batch's real future distribution -- direct port of JARCD's
    `_jsd`-based realism term, fed CTG's own generated vs. real
    trajectories. The real future (`data_batch["target_positions"]`) is
    used only as this reference distribution, never as conditioning for
    generation -- see this module's docstring.
    """
    traj = predictions["trajectories"]  # [B, N, M, T, 6]
    B, N, M, T, _ = traj.shape
    exists = _agent_exists_mask(data_batch)  # [B, M]
    step_avail = data_batch["target_availabilities"].bool()  # [B, M, T]
    gen_valid = (step_avail & exists.unsqueeze(-1)).unsqueeze(1).expand(B, N, M, T)

    gen_speed = traj[..., 2][gen_valid]
    gen_acc = traj[..., 4][gen_valid]

    real_pos = data_batch["target_positions"]  # [B, M, T, 2]
    real_speed = real_pos.diff(dim=2).norm(dim=-1) / dt  # [B, M, T-1]
    real_acc = real_speed.diff(dim=2) / dt  # [B, M, T-2]
    real_valid_speed = step_avail[:, :, 1:] & step_avail[:, :, :-1] & exists.unsqueeze(-1)
    real_valid_acc = real_valid_speed[:, :, 1:] & real_valid_speed[:, :, :-1]

    real_speed = real_speed[real_valid_speed]
    real_acc = real_acc[real_valid_acc]

    speed_jsd = _jsd(
        _histogram(gen_speed, speed_bins, speed_max_mps),
        _histogram(real_speed, speed_bins, speed_max_mps),
    )
    accel_jsd = _jsd(
        _histogram(gen_acc, accel_bins, accel_max_mps2),
        _histogram(real_acc, accel_bins, accel_max_mps2),
    )
    return {
        "generation_realism_speed_jsd": speed_jsd,
        "generation_realism_accel_jsd": accel_jsd,
    }
