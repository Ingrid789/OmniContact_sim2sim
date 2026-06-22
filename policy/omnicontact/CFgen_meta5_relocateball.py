import numpy as np

from policy.omnicontact.CFgen_meta2_carrybox import CfGenCarryBox


class CfGenRelocateBall(CfGenCarryBox):
    """Relocate-ball generator that reuses carrybox phases except Phase14/15 grasp targets."""

    def __init__(self, pad: int = 30, step_size_linear: float = 0.016, step_size_angular: float = 0.03):
        super().__init__(pad=pad, step_size_linear=step_size_linear, step_size_angular=step_size_angular)
        
        self.cfg.update(
            phase11_pregrasp_standoff_dist=0.3,
            phase22_object_goal_standoff=0.05
        )