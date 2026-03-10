"""Custom action terms for the MyoChallenge 2024 Bimanual Manipulation task.

The MPL (Modular Prosthetic Limb) uses XML-defined <position> actuators whose
joint names carry a "prosthesis/" path prefix.  mjlab's XmlPositionActuatorCfg
does exact target-name matching (without path prefix), so we bypass it entirely
and write ctrl values directly via the sim's data object.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import mujoco
import torch

from mjlab.managers.action_manager import ActionTerm, ActionTermCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

from .utils import get_bimanual_model_ids


@dataclass(kw_only=True)
class MplJointPositionActionCfg(ActionTermCfg):
    """Configuration for MPL (Modular Prosthetic Limb) position control.

    Actions are in [-1, 1] and linearly mapped to each actuator's ctrl range.
    The entity_name must be set to the bimanual entity key in the scene
    (required by the ActionTermCfg base class, though the entity is not used
    directly for control in this term).
    """

    scale: float = 1.0

    def build(self, env: ManagerBasedRlEnv) -> MplJointPositionAction:
        return MplJointPositionAction(self, env)


class MplJointPositionAction(ActionTerm):
    """Control MPL position actuators by writing ctrl directly to the sim.

    Finds all non-muscle (position) actuators in the bimanual model that
    belong to the prosthesis and writes position targets scaled to their
    ctrl ranges.
    """

    cfg: MplJointPositionActionCfg

    def __init__(self, cfg: MplJointPositionActionCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg=cfg, env=env)

        ids = get_bimanual_model_ids(env)
        self._ctrl_ids = ids["mpl_ctrl_ids"]  # global ctrl indices (tensor)

        # Store ctrl ranges for [-1, 1] → [lo, hi] mapping
        ctrl_ids_np = self._ctrl_ids.cpu().numpy()
        ctrl_range = env.sim.mj_model.actuator_ctrlrange[ctrl_ids_np]
        self._ctrl_lo = torch.tensor(ctrl_range[:, 0], device=env.device)
        self._ctrl_hi = torch.tensor(ctrl_range[:, 1], device=env.device)
        self._ctrl_mid = 0.5 * (self._ctrl_lo + self._ctrl_hi)
        self._ctrl_half = 0.5 * (self._ctrl_hi - self._ctrl_lo)

        n_mpl = len(self._ctrl_ids)
        self._action_dim = n_mpl
        self._raw_actions = torch.zeros(env.num_envs, n_mpl, device=env.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def raw_action(self) -> torch.Tensor:
        return self._raw_actions

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions[:] = actions
        clipped = torch.clamp(actions * self.cfg.scale, -1.0, 1.0)
        self._processed_actions = self._ctrl_mid + clipped * self._ctrl_half

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        self._raw_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = self._ctrl_mid

    def apply_actions(self) -> None:
        # Write directly to sim ctrl for the MPL actuator slots
        self._entity.write_ctrl_to_sim(
            self._processed_actions,
            ctrl_ids=self._ctrl_ids,
        )
