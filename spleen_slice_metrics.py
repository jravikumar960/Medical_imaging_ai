"""Compatibility shim: per-slice spleen metrics live in ``spleen_evaluation``."""

from spleen_evaluation import *  # noqa: F401,F403
from spleen_evaluation import (  # noqa: F401
    slice_metrics,
    per_slice_metrics,
    compute_slice_metrics,
    slice_confusion_and_metrics,
    slice_tp_fp_fn,
    per_slice_confusion,
    compute_per_slice_metrics,
    worst_slices,
    identify_worst_slices,
    rank_worst_slices,
    worst_slice,
    get_worst_slices,
    worst_slices_report,
    worst_slice_indices,
    worst_slices_for_case,
)