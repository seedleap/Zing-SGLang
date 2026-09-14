# Copyright 2026 Seedleap.ai
# SPDX-License-Identifier: Apache-2.0
"""Exact output resolutions admitted by Zing-0.5 launch profiles."""

ZING_RESOLUTION_480P = "832x480"
ZING_RESOLUTION_720P = "1248x704"
ZING_RESOLUTION_720P_WIDE = "1280x704"

# Values are always (width, height).
ZING_RESOLUTION_BUCKETS = {
    ZING_RESOLUTION_480P: (832, 480),
    ZING_RESOLUTION_720P: (1248, 704),
    ZING_RESOLUTION_720P_WIDE: (1280, 704),
}

ZING_RESOLUTION_BUCKET_ALIASES = {
    "480p": ZING_RESOLUTION_480P,
    "720p": ZING_RESOLUTION_720P,
    "720p-wide": ZING_RESOLUTION_720P_WIDE,
}


def normalize_zing_resolution_buckets(value: str) -> tuple[str, ...]:
    """Return ordered canonical bucket names from a comma-separated profile value."""

    raw_names = tuple(name.strip() for name in value.split(",") if name.strip())
    if not raw_names:
        raise ValueError("Zing resolution buckets must not be empty")

    names = tuple(ZING_RESOLUTION_BUCKET_ALIASES.get(name, name) for name in raw_names)
    unknown = sorted(set(names) - set(ZING_RESOLUTION_BUCKETS))
    if unknown:
        raise ValueError(
            "Unsupported Zing resolution buckets: "
            f"{unknown}; supported={sorted(ZING_RESOLUTION_BUCKETS)}; "
            f"aliases={sorted(ZING_RESOLUTION_BUCKET_ALIASES)}"
        )
    if len(set(names)) != len(names):
        raise ValueError(
            f"Zing resolution buckets contain duplicate canonical resolutions: {value}"
        )
    return names
