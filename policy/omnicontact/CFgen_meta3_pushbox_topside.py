from policy.omnicontact.CFgen_meta3_pushbox_innerside import CfGenPushBoxInnerSide
from policy.omnicontact.CFgen_meta3_pushbox_twosides import CfGenPushBoxTwoSides


class CfGenPushBoxTopSide(CfGenPushBoxTwoSides):
    """Placeholder for a top-side pushbox reference generator."""

    def generate(self, *args, **kwargs):
        raise NotImplementedError("CfGenPushBoxTopSide is not implemented yet.")
