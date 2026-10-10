#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Estimate TactileReflex threshold candidates from controller CSV logs.

This tool reports distributions and conservative starting candidates.  It does
not certify a safe threshold and does not replace an independent hardware test.
Use separate CSVs for static holding, controlled slip/liftoff, and controlled
load-release trials whenever possible.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Iterable


def read_rows(paths: Iterable[str], holding_only: bool = True) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    numeric = {
        "Fn_max", "Fn_raw_max", "Sy_max", "Sy_raw_max", "delta_cop_y",
        "slip_on", "release_on", "protect_on", "contact_L", "contact_R",
    }
    for path in paths:
        with open(path, newline="", encoding="utf-8") as f:
            for raw in csv.DictReader(f):
                if holding_only and raw.get("state", "") != "HOLDING":
                    continue
                row: dict[str, float | str] = {"source": path}
                for key, value in raw.items():
                    if key in numeric:
                        try:
                            row[key] = float(value)
                        except (TypeError, ValueError):
                            row[key] = math.nan
                    else:
                        row[key] = value
                rows.append(row)
    return rows


def values(rows: list[dict], key: str) -> list[float]:
    out = []
    for row in rows:
        value = row.get(key, math.nan)
        if isinstance(value, (int, float)) and math.isfinite(value):
            out.append(float(value))
    return out


def percentile(xs: list[float], q: float) -> float | None:
    if not xs:
        return None
    xs = sorted(xs)
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * q / 100.0
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return xs[lo]
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def stats(xs: list[float]) -> dict:
    return {
        "n": len(xs),
        "min": min(xs) if xs else None,
        "p01": percentile(xs, 1),
        "p05": percentile(xs, 5),
        "p50": percentile(xs, 50),
        "p95": percentile(xs, 95),
        "p99": percentile(xs, 99),
        "p999": percentile(xs, 99.9),
        "max": max(xs) if xs else None,
    }


def event_values(rows: list[dict], value_key: str, event_key: str) -> list[float]:
    """Use event-edge rows when available; this avoids treating every frame as an event."""
    event_rows = [r for r in rows if float(r.get(event_key, 0) or 0) > 0]
    return values(event_rows, value_key)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--static", nargs="+", required=True,
                    help="static holding CSV(s), used for noise candidates")
    ap.add_argument("--slip", nargs="*", default=[],
                    help="controlled slip/liftoff CSV(s), optional")
    ap.add_argument("--unload", nargs="*", default=[],
                    help="controlled load-release CSV(s), optional")
    ap.add_argument("--all-frames", action="store_true",
                    help="include IDLE/CLOSING rows; default keeps HOLDING only")
    ap.add_argument("--sy_quiet_margin", type=float, default=0.02,
                    help="additive starting margin above static raw Sy P95 (default: 0.02)")
    ap.add_argument("--fn_protect_margin", type=float, default=0.20,
                    help="relative starting margin above static raw Fn P99.9 (default: 0.20)")
    ap.add_argument("--fn_stop", type=float, default=1.3,
                    help="fixed contact/closing-stop constant to include in the output (default: 1.3)")
    ap.add_argument("--out", default="threshold_candidates.json",
                    help="JSON output path")
    args = ap.parse_args()

    keep_holding = not args.all_frames
    static = read_rows(args.static, keep_holding)
    slip = read_rows(args.slip, keep_holding)
    unload = read_rows(args.unload, keep_holding)
    if not static:
        ap.error("--static contains no usable rows")

    static_sy_raw = values(static, "Sy_raw_max")
    static_sy_ema = values(static, "Sy_max")
    static_fn_raw = values(static, "Fn_raw_max")
    static_fn_ema = values(static, "Fn_max")
    static_cop = values(static, "delta_cop_y")

    slip_edge = event_values(slip, "Sy_raw_max", "slip_on")
    if not slip_edge:
        # A log made with a threshold that missed the event has no valid edge.
        # Report the observed distribution, but do not silently call it events.
        slip_observed = values(slip, "Sy_raw_max")
    else:
        slip_observed = slip_edge

    unload_edge = event_values(unload, "delta_cop_y", "release_on")
    if not unload_edge:
        unload_observed = [x for x in values(unload, "delta_cop_y") if x < 0]
    else:
        unload_observed = unload_edge

    # These are deliberately labelled candidates.  The margins only make the
    # next physical test easier; they are not an automatic safety guarantee.
    # Paper calibration uses raw Sy P95 for the quiet-shear ceiling.  The
    # controller later compares EMA Sy_max against this candidate.
    static_sy_p95 = percentile(static_sy_raw, 95)
    # The paper's slip constraint uses the high static-noise ceiling and the
    # weakest observed slip, rather than a P95/P10 heuristic.  Keep the
    # candidate null when the two distributions do not separate.
    static_sy_p999 = percentile(static_sy_raw, 99.9)
    slip_weakest = min(slip_observed) if slip_observed else None
    sy_candidate = ((static_sy_p999 + slip_weakest) / 2
                    if slip_edge and static_sy_p999 is not None
                    and slip_weakest is not None
                    and slip_weakest > static_sy_p999 else None)

    static_cop_low = percentile(static_cop, 1)
    unload_mid = percentile(unload_observed, 50)
    cop_candidate = (static_cop_low + unload_mid) / 2 if unload_edge and static_cop_low is not None and unload_mid is not None and unload_mid < static_cop_low else None

    # Paper calibration uses raw Fn P99.9 plus a margin, while runtime
    # protection compares the EMA Fn_max signal.
    fn_p999 = percentile(static_fn_raw, 99.9)
    fn_candidate = fn_p999 * (1.0 + args.fn_protect_margin) if fn_p999 is not None else None

    result = {
        "inputs": {"static": args.static, "slip": args.slip, "unload": args.unload,
                   "holding_only": keep_holding},
        "static": {
            "Sy_raw_max": stats(static_sy_raw),
            "Sy_max_ema": stats(static_sy_ema),
            "Fn_raw_max": stats(static_fn_raw),
            "Fn_max_ema": stats(static_fn_ema),
            "delta_cop_y": stats(static_cop),
        },
        "slip": {
            "event_rows_used": bool(slip_edge),
            "Sy_raw_max": stats(slip_observed),
        },
        "unload": {
            "event_rows_used": bool(unload_edge),
            "delta_cop_y": stats(unload_observed),
        },
        "candidates": {
            "sy_quiet": (static_sy_p95 + args.sy_quiet_margin) if static_sy_p95 is not None else None,
            "sy_quiet_raw_p95": static_sy_p95,
            "sy_static_raw_p999": static_sy_p999,
            "sy_slip_weakest": slip_weakest,
            "sy_th": sy_candidate,
            "cop_dec_th": cop_candidate,
            "fn_protect_limit": fn_candidate,
            "fn_protect_raw_p999": fn_p999,
            # fn_stop is a configured controller constant, not a statistic
            # estimated from the static CSV.  Keep it in the generated file
            # so the output is directly usable for a selected material/profile.
            "fn_stop": args.fn_stop,
        },
        "margins": {
            "sy_quiet_additive": args.sy_quiet_margin,
            "fn_protect_relative": args.fn_protect_margin,
        },
        "warnings": [
            "Candidates require independent physical validation and are not safety certification.",
            "fn_stop is a fixed controller constant supplied by --fn_stop; select the appropriate profile and verify both-side contact during a low-effort test.",
            "sy_quiet is static raw Sy P95 plus the requested additive starting margin; runtime release compares EMA Sy_max.",
            "fn_protect_limit is static raw Fn P99.9 plus the requested relative margin; runtime protection compares EMA Fn_max. Validate deformation and protection latency.",
            "sy_th uses static raw Sy P99.9 <= threshold <= weakest observed slip; it remains null when the distributions overlap.",
            "If event_rows_used is false, the supplied slip/unload log did not contain edge events; inspect the trial protocol before using candidates.",
        ],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(result["candidates"], indent=2, ensure_ascii=False))
    print(f"saved: {out}")
    print("These are candidates for a new physical validation trial, not final safe limits.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
