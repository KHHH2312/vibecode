"""lineage_v2 — 0.94-class CELLECT-lite stack (Session 0+).

Public UNet pack is baseline-only. See PLAN_094.md / Desktop plan.md.
"""

__version__ = "0.1.0-session0"

from .constants import FORBIDDEN_TEST_STEMS, SPACING_ZYX_UM

__all__ = ["FORBIDDEN_TEST_STEMS", "SPACING_ZYX_UM", "__version__"]
