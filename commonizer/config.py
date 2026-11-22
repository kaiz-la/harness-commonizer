"""Tunable thresholds and weights. All editable from the GUI settings panel."""
from dataclasses import dataclass, field, asdict, fields
from typing import List


@dataclass
class Config:
    # --- wire fingerprint: attributes expected to stay stable for a carried-over wire
    fingerprint_cols: List[str] = field(default_factory=lambda: [
        "cross_section", "color", "wire_pn", "cable_pn", "cable_name",
        "print", "harness_attribute", "long_name", "remark",
    ])

    # --- connector alignment
    align_iterations: int = 3          # propagation passes
    align_min_similarity: float = 0.15  # below this a candidate pair is rejected
    align_identity_bonus: float = 0.20  # prior for a connector keeping its name
    align_knn: int = 25                 # candidates per connector in huge blocks
    align_dense_block_limit: int = 3000  # larger blocks use greedy kNN assignment instead of Hungarian

    # --- wire matching
    fuzzy_wire_min_score: float = 0.80  # same endpoints, attribute similarity needed to call it carry-over

    # --- module commonization
    w_connections: float = 1.0
    w_vercodung: float = 0.6
    w_harness: float = 0.5
    w_long_name: float = 0.3
    w_wire_profile: float = 0.3
    merge_same_vercodung_only: bool = True     # merge candidates must have the same (logically equal) Vercodung
    merge_include_empty_vercodung: bool = False  # also pair modules that both have no Vercodung
    merge_knn: int = 15
    candidate_max_df: int = 300         # blocking tokens shared by more modules than this are ignored
    merge_min_similarity: float = 0.45   # combined cosine to become a candidate
    absorb_containment: float = 0.85     # A ⊂ B share for 'Absorb'
    merge_jaccard: float = 0.50          # connection Jaccard for 'Merge'
    cluster_similarity: float = 0.55     # graph edge threshold for function clusters
    cluster_max_component: int = 60      # components bigger than this are split with HDBSCAN
    svd_components: int = 64

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})
