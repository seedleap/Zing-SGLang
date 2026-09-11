# Copyright 2026 Seedleap.ai
# SPDX-License-Identifier: Apache-2.0
"""Exact output resolutions admitted by Zing-0.5 launch profiles."""

MINWM_RESOLUTION_480P = "832x480"
MINWM_RESOLUTION_720P = "1248x704"
MINWM_RESOLUTION_720P_WIDE = "1280x704"

# Values are always (width, height).
MINWM_RESOLUTION_BUCKETS = {
    MINWM_RESOLUTION_480P: (832, 480),
    MINWM_RESOLUTION_720P: (1248, 704),
    MINWM_RESOLUTION_720P_WIDE: (1280, 704),
}

MINWM_RESOLUTION_BUCKET_ALIASES = {
    "480p": MINWM_RESOLUTION_480P,
    "720p": MINWM_RESOLUTION_720P,
    "720p-wide": MINWM_RESOLUTION_720P_WIDE,
}


def normalize_minwm_resolution_buckets(value: str) -> tuple[str, ...]:
    """Return ordered canonical bucket names from a comma-separated profile value."""

    raw_names = tuple(name.strip() for name in value.split(",") if name.strip())
    if not raw_names:
        raise ValueError("MinWM resolution buckets must not be empty")

    names = tuple(MINWM_RESOLUTION_BUCKET_ALIASES.get(name, name) for name in raw_names)
    unknown = sorted(set(names) - set(MINWM_RESOLUTION_BUCKETS))
    if unknown:
        raise ValueError(
            "Unsupported MinWM resolution buckets: "
            f"{unknown}; supported={sorted(MINWM_RESOLUTION_BUCKETS)}; "
            f"aliases={sorted(MINWM_RESOLUTION_BUCKET_ALIASES)}"
        )
    if len(set(names)) != len(names):
        raise ValueError(
            f"MinWM resolution buckets contain duplicate canonical resolutions: {value}"
        )
    return names
