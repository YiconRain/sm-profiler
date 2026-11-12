"""Versioned CUDA record decoding. Independent of the legacy/Triton layout."""

import struct

SCHEMA_VERSION = 2
RECORD = struct.Struct("<4Q8I")
FIELDS = (
    "start_ns",
    "end_ns",
    "payload0",
    "payload1",
    "phase",
    "site",
    "block",
    "group",
    "sm",
    "kind",
    "reserved0",
    "reserved1",
)


def decode(
    data,
    attempted,
    blocks,
    groups,
    capacity,
    expected_tracks,
    *,
    schema_version,
    sentinel_phase,
    sentinel_site,
    packed=False,
):
    """Decode one launch; coverage requires one caller-defined enclosing range.

    The caller supplies the sidecar schema and expected tracks independently of
    captured records. Site counts and semantic completeness remain application
    responsibilities. Saturated counters cannot establish an exact drop count.
    """
    if schema_version != SCHEMA_VERSION:
        raise ValueError("unsupported recorder schema")
    if any(
        type(n) is not int or not 0 <= n <= 0xFFFFFFFF
        for n in (sentinel_phase, sentinel_site)
    ):
        raise ValueError("sentinel identifiers must be uint32 values")
    if any(
        type(n) is not int or not 0 < n <= 0xFFFFFFFF
        for n in (blocks, groups, capacity)
    ):
        raise ValueError("dimensions and capacity must be positive uint32 values")
    if any(type(n) is not int or not 0 <= n <= 0xFFFFFFFF for n in attempted):
        raise ValueError("attempted counters must be uint32 values")
    if len(attempted) != blocks * groups:
        raise ValueError("v2 counter buffer size mismatch")
    count = (
        sum(min(n, capacity) for n in attempted)
        if packed
        else blocks * groups * capacity
    )
    if len(data) != count * RECORD.size:
        raise ValueError("v2 record buffer size mismatch")
    tracks = list(expected_tracks)
    if any(
        not isinstance(track, (tuple, list))
        or len(track) != 2
        or any(type(n) is not int for n in track)
        or not (0 <= track[0] < blocks and 0 <= track[1] < groups)
        for track in tracks
    ):
        raise ValueError("expected track outside recorder dimensions")
    expected = set(map(tuple, tracks))
    if not expected:
        raise ValueError("at least one independently expected track is required")
    records, counters, invalid, unclosed = [], [], 0, 0
    observed = set()
    duplicate_sentinels = 0
    outside_sentinel = 0
    recorded_tracks = set()
    cursor = 0
    for track, count in enumerate(attempted):
        block, group = divmod(track, groups)
        stored = min(count, capacity)
        track_rows = []
        sentinels = []
        if stored:
            recorded_tracks.add((block, group))
        counters.append(
            dict(
                block=block,
                group=group,
                attempted=count,
                stored=stored,
                capacity=capacity,
                dropped=max(0, count - capacity),
            )
        )
        for index in range(stored):
            offset = ((cursor if packed else track * capacity) + index) * RECORD.size
            row: dict[str, int] = dict(zip(FIELDS, RECORD.unpack_from(data, offset)))
            row["sequence"] = index
            unclosed += row["end_ns"] == 0
            bad = (
                row["start_ns"] == 0
                or row["end_ns"] < row["start_ns"]
                or row["block"] != block
                or row["group"] != group
                or row["kind"] > 1
                or row["reserved0"] != 0
                or row["reserved1"] != 0
                or (row["kind"] == 1 and row["end_ns"] != row["start_ns"])
            )
            invalid += bad
            if (
                not bad
                and row["kind"] == 0
                and row["phase"] == sentinel_phase
                and row["site"] == sentinel_site
            ):
                duplicate_sentinels += (block, group) in observed
                observed.add((block, group))
                sentinels.append(row)
            track_rows.append(row)
            records.append(row)
        if len(sentinels) == 1:
            sentinel = sentinels[0]
            outside_sentinel += sum(
                row["start_ns"] < sentinel["start_ns"]
                or row["end_ns"] > sentinel["end_ns"]
                for row in track_rows
            )
        cursor += stored
    diagnostics = dict(
        schema_version=SCHEMA_VERSION,
        counters=counters,
        dropped=sum(x["dropped"] for x in counters),
        unclosed=unclosed,
        saturated_counters=sum(n == 0xFFFFFFFF for n in attempted),
        duplicate_sentinels=duplicate_sentinels,
        outside_sentinel=outside_sentinel,
        invalid=invalid,
        missing_tracks=sorted(expected - observed),
        unexpected_tracks=sorted(recorded_tracks - expected),
    )
    diagnostics["complete"] = not any(
        diagnostics[k]
        for k in (
            "dropped",
            "unclosed",
            "invalid",
            "missing_tracks",
            "unexpected_tracks",
            "saturated_counters",
            "duplicate_sentinels",
            "outside_sentinel",
        )
    )
    return records, diagnostics


def chrome_trace(records, names, diagnostics, *, allow_partial=False):
    """Render decoded records with visible completeness and explicit partial opt-in."""
    if diagnostics.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("decoder diagnostics schema mismatch")
    if diagnostics.get("complete") is not True and not allow_partial:
        raise ValueError("incomplete capture; use allow_partial=True for diagnosis")
    valid = [row for row in records if 0 < row["start_ns"] <= row["end_ns"]]
    origin = min((row["start_ns"] for row in valid), default=0)
    events = []
    for row in valid:
        events.append(
            dict(
                name=names.get(row["phase"], str(row["phase"])),
                ph="X",
                ts=(row["start_ns"] - origin) / 1000,
                dur=(row["end_ns"] - row["start_ns"]) / 1000,
                pid=row["sm"],
                tid=f"CTA{row['block']}/group{row['group']}",
                args=row,
            )
        )
        if row["kind"] == 1:
            events[-1].update(ph="i", s="t")
            del events[-1]["dur"]
    return dict(
        schema_version=SCHEMA_VERSION,
        origin_ns=origin,
        displayTimeUnit="ns",
        diagnostics=diagnostics,
        partial=diagnostics.get("complete") is not True,
        omitted_invalid_intervals=len(records) - len(valid),
        traceEvents=events,
    )
