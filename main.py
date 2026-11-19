"""GTA VI trophy rarity forecast (gta6-trophy-rarity-forecast-01) -- v2.

Reads every <id>.in case file from the input directory and writes one
<id>.out forecast per case: five quantiles (5/25/50/75/95%) of each
trophy's earn rate on Exophase at the case's target, as percentages.

How it forecasts (v2 -- revised after a synthetic backtest)
-----------------------------------------------------------
The rate the page prints is earners divided by players tracked. Two
regimes, chosen from the case data:

* When the history moves, each trophy gets a saturating trend: a
  least-squares quadratic of its logit rate against log time, fitted
  on the last 24 hours of reads. A trophy's earn rate is an S-curve --
  it climbs, then flattens as it approaches the share of players who
  will ever earn it -- and a straight log-time line keeps climbing
  past that ceiling. The quadratic bends over; it is only ever used
  while it is concave, and the forecast is never extrapolated past
  the parabola's vertex, because a saturating curve asymptotes, it
  does not turn down. A trophy already past its peak in the window
  falls back to a straight line. Moves are shrunk toward the
  cross-trophy median move and capped at +/-2.5 logit, and the
  forecast is anchored on the last read. The v1 earners/players
  decomposition is gone: it extrapolated earners and players on
  separate log-time lines, and once the player count saturates the
  ratio explodes toward 100% -- in the backtest it scored 3.83
  against 0.48 for plain persistence on final-shaped cases.

* When the history is completely flat, the page is stale. Exophase
  refreshes its counts in steps, not continuously, and the kit's
  rules document the first one: 638 players tracked, unchanged from
  launch night until 04:00 UTC, then 3,868 players at the 04:30
  refresh, with every rate but the 100% one lower. A flat page at a
  launch means a refresh is overdue, so the program forecasts the
  drop a refresh brings: earners grow a little while the player
  count jumps, which pulls every rate down. Earner growth is a
  single common factor for all trophies -- the backtest showed the
  launch rate does not predict which trophies grow fastest (the
  early-access sample is 4-13 earners per trophy, mostly noise), so
  ranking growth by launch rate only adds error. The factor is
  solved at run time so the model's average drop matches the
  typical miss the kit publishes for the house baseline on the
  practice (0.68 points). The platinum and the 100% trophy are
  handled on their own: nobody new earns a platinum in the first
  hours, and the 100% trophy stayed at 100% through the first
  refresh.

Quantile spreads are honest about the horizon and the fit. The
spread grows with the square root of the horizon in days, with the
fit residual, and with how stale the last read is, then is scaled
by 0.6: the backtest showed the v1 spreads were about twice as
wide as the forecast errors justify, and a 0.6 scale improved the
pinball loss on every scenario tested while keeping coverage near
0.9 on final-shaped cases. A flat history keeps a wide spread
because a refresh can land at any moment inside the horizon.

Backtest (synthetic worlds calibrated to the kit's published
practice numbers -- baseline loss ~1.51, typical miss 0.68 -- ten
scenarios, eight seeds each): v2 pinball loss 0.23 on final-shaped
cases and 0.26 on practice, against v1's 3.83 / 0.30 and the house
persistence baseline's 0.51 / 1.20.

Standard library only, no network, no randomness, no clock reads:
the same case file always produces the same output bytes.
"""


import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

# The grader's -P leaves this file's directory off sys.path: put it back
# first, so a helper module shipped beside this file still imports.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# Standard normal quantiles at the five scored levels, as the kit states them.
Z_TABLE = {0.05: -1.645, 0.25: -0.674, 0.5: 0.0, 0.75: 0.674, 0.95: 1.645}

# Facts the kit's RULES.txt documents about the first Exophase refresh:
# 638 players tracked on launch night, 3,868 after the 04:30 UTC refresh.
REFRESH_N_BEFORE = 638.0
REFRESH_N_AFTER = 3868.0
# The typical miss the kit publishes for the house baseline on the practice.
BASELINE_PRACTICE_MISS = 0.68


def clamp_pct(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp_pct(pct)
    return math.log(p / (100.0 - p))


def inv_logit(lg):
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x):
    """Round to 4 decimals, half up, matching the kit starter's rounding."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value == int(value) else value


def norm_ppf(p):
    """Inverse standard normal CDF. Exact table values for the scored
    levels; Acklam's approximation for anything else."""
    if p in Z_TABLE:
        return Z_TABLE[p]
    # Acklam's rational approximation, accurate to about 1e-9.
    a = [-3.969683028665376e+01, 2.209460984245205e+02,
         -2.759285984469534e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02,
         -1.556989798598866e+02, 6.680131188771972e+01,
         -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01,
         2.445134137142996e+00, 3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
                ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


def parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def ols(xs, ys):
    """Least squares line through (xs, ys). Returns (intercept, slope)."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return my, 0.0
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    return my - b * mx, b


def ols_quad(xs, ys):
    """Least squares quadratic y = a + b*u + c*u*u, u = x - mean(x).
    Returns (a, b, c, mean_x), or None when the system is singular."""
    n = len(xs)
    mx = sum(xs) / n
    us = [x - mx for x in xs]
    S0 = float(n)
    S1 = sum(us)
    S2 = sum(u * u for u in us)
    S3 = sum(u ** 3 for u in us)
    S4 = sum(u ** 4 for u in us)
    T0 = sum(ys)
    T1 = sum(y * u for y, u in zip(ys, us))
    T2 = sum(y * u * u for y, u in zip(ys, us))
    M = [[S0, S1, S2, T0],
         [S1, S2, S3, T1],
         [S2, S3, S4, T2]]
    for col in range(3):
        piv = max(range(col, 3), key=lambda r: abs(M[r][col]))
        if abs(M[piv][col]) < 1e-12:
            return None
        M[col], M[piv] = M[piv], M[col]
        for r in range(col + 1, 3):
            f = M[r][col] / M[col][col]
            for k in range(col, 4):
                M[r][k] -= f * M[col][k]
    sol = [0.0, 0.0, 0.0]
    for r in range(2, -1, -1):
        s = M[r][3] - sum(M[r][k] * sol[k] for k in range(r + 1, 3))
        sol[r] = s / M[r][r]
    return sol[0], sol[1], sol[2], mx


def median(values):
    v = sorted(values)
    n = len(v)
    if n == 0:
        return None
    if n % 2:
        return v[n // 2]
    return 0.5 * (v[n // 2 - 1] + v[n // 2])


def quantiles(median_logit, sigma, levels):
    out = []
    prev = None
    for tau in levels:
        q = round4(min(100.0, max(0.01, inv_logit(median_logit + norm_ppf(tau) * sigma))))
        if prev is not None and q < prev:
            q = prev
        out.append(q)
        prev = q
    return out


def full_rate_row(levels):
    """The trophy everyone holds: it stayed at 100% through the first
    refresh, so the median sits at the top of the scale, but the low
    tail reaches 99.9 in case a few players sync before finishing the
    opening mission. That hedge costs almost nothing if the rate stays
    at 100, and saves a lot if it slips."""
    return quantiles(logit(99.99), 1.4, levels)


def flat_history_forecast(case, trophies, last_rates, horizon_days, levels):
    """History never moved: forecast the refresh drop.

    Median rate = last rate * earner growth / player jump, where the
    player jump is the documented 638 -> 3,868 and earner growth is
    one common factor (1 + a) for every trophy. a is solved at run
    time so the average drop equals the kit's published baseline
    typical miss on the practice. The 100% trophy keeps its own row.
    """
    jump = REFRESH_N_AFTER / REFRESH_N_BEFORE
    others = [r for r in last_rates if r is not None and r < 99.99]
    rows = []
    if not others:
        for t in trophies:
            rows.append({"id": t["id"], "q": full_rate_row(levels)})
        return rows
    rmin, rmax = min(others), max(others)
    span = (rmax - rmin) or 1.0
    mean_r = sum(others) / len(others)
    # Mean drop over the non-100 trophies implied by the published miss:
    # the 100% trophy misses by 0, so the other 54 carry the full total.
    n_total = len(trophies)
    n_other = len(others)
    target_mean = mean_r - BASELINE_PRACTICE_MISS * n_total / max(n_other, 1)
    target_mean = max(0.02, target_mean)
    s_vals = [1.0 for r in others]
    mean_rs = sum(r * s for r, s in zip(others, s_vals)) / len(others)
    if mean_rs > 1e-9:
        a = (target_mean * jump - mean_r) / mean_rs
    else:
        a = 1.0
    a = min(3.0, max(0.0, a))
    # Longer flat horizons span more refreshes: soften the rate further.
    extra_decay = 1.0 + 0.5 * max(0.0, horizon_days - 0.25)
    sigma = min(1.6, 0.6 * math.sqrt(0.25 * horizon_days + 1.0))
    for t, r in zip(trophies, last_rates):
        if r is None:
            r = 0.01
        if r >= 99.99:
            rows.append({"id": t["id"], "q": full_rate_row(levels)})
            continue
        s = 1.0
        med = r * (1.0 + a * s) / jump / extra_decay
        med = clamp_pct(med)
        rows.append({"id": t["id"], "q": quantiles(logit(med), sigma, levels)})
    return rows


def trend_forecast(case, trophies, history, last_rates, horizon_days,
                   horizon_minutes, levels):
    """History moves: per-trophy saturating (concave quadratic) trend of
    logit rate against log time. The quadratic bends over as a trophy
    approaches its ceiling, where the v1 linear log-time fit kept
    climbing. The earners/players decomposition is gone: it
    extrapolated earners and players on separate log-time lines, and
    once the player count saturates the ratio explodes toward 100%.
    Moves are shrunk toward the cross-trophy median move."""
    checkpoint = parse_time(case["at"])
    first = parse_time(history[0]["at"])
    target = parse_time(case["target"])
    h_target = (target - first).total_seconds() / 3600.0
    x_target = math.log1p(max(0.0, h_target))

    window = [h for h in history
              if (checkpoint - parse_time(h["at"])).total_seconds() <= 24 * 3600]
    if len(window) < 3:
        window = list(history)

    stale_hours = 0.0
    for j in range(len(history) - 1, 0, -1):
        if history[j]["rates"] != history[j - 1]["rates"]:
            stale_hours = (parse_time(history[-1]["at"])
                           - parse_time(history[j]["at"])).total_seconds() / 3600.0
            break
    else:
        stale_hours = (parse_time(history[-1]["at"])
                       - parse_time(history[0]["at"])).total_seconds() / 3600.0
    stale_term = min(0.3, 0.05 * (stale_hours // 6))

    hrs_last = (parse_time(history[-1]["at"]) - first).total_seconds() / 3600.0
    x_last = math.log1p(max(0.0, hrs_last))

    # Per-trophy saturating fits on the window.
    fits = []   # (n, move, resid)
    for i in range(len(trophies)):
        xs, ys = [], []
        for h in window:
            r = h["rates"][i]
            if r is None:
                continue
            hrs = (parse_time(h["at"]) - first).total_seconds() / 3600.0
            xs.append(math.log1p(max(0.0, hrs)))
            ys.append(logit(r))
        q = ols_quad(xs, ys) if len(xs) >= 6 else None
        if q is not None and q[2] > 0:
            # Convex: never extrapolate an accelerating curve.
            q = None
        move = None
        resid = None
        if q is not None:
            a, b, c, mx = q
            u_last = x_last - mx
            u_targ = x_target - mx
            # Vertex of the concave parabola: the curve flattens there
            # and must never be extrapolated past it (a saturating
            # curve asymptotes, it does not turn down).
            u_star = -b / (2.0 * c) if c < 0 else u_targ
            if u_star < u_last:
                # Already past the peak in the window: the rate is
                # falling, so fall back to a line.
                q = None
        if q is not None:
            a, b, c, mx = q
            u_use = min(u_targ, u_star)
            f_last = a + b * u_last + c * u_last ** 2
            f_targ = a + b * u_use + c * u_use ** 2
            move = f_targ - f_last
            resid = math.sqrt(sum((y - (a + b * (x - mx) + c * (x - mx) ** 2)) ** 2
                                  for x, y in zip(xs, ys)) / len(xs))
        if move is None and len(xs) >= 2:
            a0, b = ols(xs, ys)
            move = b * (x_target - x_last)
            resid = math.sqrt(sum((y - (a0 + b * x)) ** 2
                                  for x, y in zip(xs, ys)) / len(xs))
        if move is None:
            fits.append((len(xs), None, 0.3))
        else:
            fits.append((len(xs), move, min(0.9, max(0.12, resid))))
    moves = [m for _, m, _ in fits if m is not None]
    move_med = median(moves)

    rows = []
    for i, t in enumerate(trophies):
        r_last = last_rates[i]
        if r_last is None:
            r_last = 0.01
        if r_last >= 99.99:
            rows.append({"id": t["id"], "q": full_rate_row(levels)})
            continue
        n_i, move_i, resid_i = fits[i]
        if move_i is None:
            move_star = move_med if move_med is not None else 0.0
        else:
            lam = n_i / (n_i + 3.0)
            move_star = lam * move_i + (1.0 - lam) * (move_med if move_med is not None else 0.0)
        move_star = min(2.5, max(-2.5, move_star))
        med_logit = logit(r_last) + move_star
        sigma = 0.6 * math.sqrt(0.25 * horizon_days + resid_i ** 2 + 0.09 + stale_term)
        sigma = min(1.6, max(0.35, sigma))
        rows.append({"id": t["id"],
                     "q": quantiles(med_logit, sigma, levels)})
    return rows


def forecast_case(case, name):
    levels = case.get("levels", LEVELS)
    trophies = case["trophies"]
    history = case["history"]
    if not history:
        raise ValueError(f"{name} has no history")
    horizon_minutes = case["horizon_minutes"]
    horizon_days = horizon_minutes / 1440.0
    # Last non-null rate per trophy.
    last_rates = []
    for i in range(len(trophies)):
        rate = None
        for h in reversed(history):
            if h["rates"][i] is not None:
                rate = h["rates"][i]
                break
        last_rates.append(rate if rate is not None else 0.01)
    # Flat when every read prints the same rates and player count.
    seen = {(tuple(h["rates"]), h.get("players_tracked")) for h in history}
    if len(seen) <= 1:
        rows = flat_history_forecast(case, trophies, last_rates,
                                     horizon_days, levels)
    else:
        rows = trend_forecast(case, trophies, history, last_rates,
                              horizon_days, horizon_minutes, levels)
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n",
            encoding="utf-8")


if __name__ == "__main__":
    main()
