"""GTA VI trophy-rarity forecast (make.fun gta6-trophy-rarity-forecast-01).

Forecasts, for every GTA VI trophy on PS5, five quantiles of its Exophase
earn rate at the case target, from the case history alone. Python 3.13,
standard library only, no network, deterministic.

Model (per trophy, all in logit space, logit(p) = ln(p / (100 - p))):
  1. TREND: OLS line of logit(rate) on ln(1 + hours since release), fitted
     on the case history (last 36 h when there are enough reads). This is
     the house curve baseline's idea, but the slope is clipped and the
     extrapolation is capped so a flat launch-night history cannot produce
     a wild move.
  2. COUNTS: earners E = rate * players_tracked / 100 and tracked players
     N are each fitted as a power law in (1 + hours since release)
     (ln E and ln N on ln(1 + t)); the rate forecast is E_target / N_target.
     This separates the two forces that move a rate: players earning the
     trophy (E grows) and fresh players syncing for the first time (N grows
     and dilutes every rate, which is exactly what Exophase's first refresh
     did on launch night: 638 -> 3,868 tracked, every rate but the 100% one
     fell). Earners are cumulative, so their fitted growth is floored at 1.
  3. PRIOR: a week-one level for the trophy's kind, from its description
     and tier (see PRIOR table below). Anchors: GTA V's long-run PSN rates
     (final mission ~17% PS4 / ~29% PS5, hardest grinds ~0.2-1%) scaled
     down hard for week one, because a 200-300 h platinum chase cannot be
     finished in 7 days and most launch buyers are not trophy hunters; and
     the launch itself (Sensor Tower: ~24.6m copies day one, ~75% PS5),
     which says the Exophase tracked base keeps growing through week one,
     diluting rare trophies while early story trophies climb. The prior's
     weight grows with the horizon (0 for a 90-minute practice case, full
     for the 4.5-day final), so it never overrides what the history says
     on a short horizon.
  The three medians are blended in logit space. Story-chapter medians are
  then forced non-increasing with chapter number and the platinum median
  is forced below every other trophy (it requires all of them).
  Spread: a random-walk term 0.5 * sqrt(horizon_days) plus model
  uncertainty, the trend/prior disagreement, and a short-horizon
  refresh-risk floor (Exophase refreshes hours apart, so a 90-minute
  forecast can be jumped by a refresh the history has not seen yet).

Usage: MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
"""

import json
import math
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# Week-one (day 7 after release) prior medians, percent of tracked players.
# Keyed by trophy kind, derived from the trophy description / tier.
PRIOR_STORY_CHAPTER = {1: 38.0, 2: 27.0, 3: 18.0, 4: 11.0}
PRIOR_FINAL_MISSION = 7.0
PRIOR_PLATINUM = 0.06
PRIOR_100PCT = 0.12


def clamp(x, lo=0.01, hi=99.99):
    return min(hi, max(lo, x))


def logit(pct):
    p = clamp(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(lg):
    lg = max(-9.0, min(9.0, lg))
    return 100.0 / (1.0 + math.exp(-lg))


def round4(x):
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if float(value).is_integer() else value


def parse_time(s):
    if s is None:
        return None
    if isinstance(s, (int, float)):
        return datetime.fromtimestamp(s, tz=timezone.utc)
    txt = str(s).strip()
    if txt.endswith("Z"):
        txt = txt[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(txt)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def hours_between(a, b):
    if a is None or b is None:
        return 0.0
    return (b - a).total_seconds() / 3600.0


def ols(xs, ys):
    """Least-squares line; returns (intercept, slope) or (mean_y, 0.0)."""
    n = len(xs)
    if n == 0:
        return 0.0, 0.0
    if n < 2:
        return ys[0], 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx < 1e-12:
        return my, 0.0
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx
    return my - slope * mx, slope


def prior_for(trophy):
    """Week-one prior median (%) for a trophy, or None to trust the data."""
    name = (trophy.get("name") or "").lower()
    desc = (trophy.get("description") or "").lower()
    tier = (trophy.get("tier") or "").lower()
    text = name + " " + desc
    if tier == "platinum":
        return PRIOR_PLATINUM
    if "100% completion" in desc:
        return PRIOR_100PCT
    if "complete gone fishing" in desc:
        return None  # launch-night data has this at 100%; trust the history
    m = re.search(r"complete chapter (\d)", desc)
    if m:
        return PRIOR_STORY_CHAPTER.get(int(m.group(1)), 8.0)
    if "final mission" in desc and "complete the final mission" in desc:
        return PRIOR_FINAL_MISSION
    if "final mission for" in desc:
        return 4.5  # one character's final strand mission
    if "gold medal in all" in desc:
        return 0.5
    if "250 stunts" in desc:
        return 0.35
    if "100 locations" in desc:
        return 0.8
    if "50 challenges" in desc:
        return 1.2
    if "20 different races" in desc or "win 20" in desc:
        return 2.5
    if "8 mission strands" in desc:
        return 2.0
    if "3 collectible card sets" in desc:
        return 0.6
    if "5 bank vaults" in desc:
        return 1.0
    if "discover 100 vehicles" in desc:
        return 1.5
    if "register 10 vehicles" in desc:
        return 6.0
    if "rob 3 shops" in desc:
        return 5.0
    if "take down 5 enemies" in desc:
        return 9.0
    if "4 star wanted level" in desc:
        return 7.0
    if "rob an armored truck" in desc:
        return 6.0
    if "jewelry store without being detected" in desc:
        return 3.5
    if "illegal operation belonging to each gang" in desc:
        return 2.0
    if "explore all of leonida" in desc:
        return 3.0
    if "meditate at every spot" in desc:
        return 2.5
    if "drone race at every location" in desc:
        return 1.5
    if "try each type of activity once" in desc:
        return 12.0
    if "complete an arcade game" in desc:
        return 14.0
    if "enjoy a cocktail" in desc:
        return 10.0
    if "bowling alley" in desc:
        return 8.0
    if "spend $100,000" in desc:
        return 11.0
    if "laundering marked bills" in desc:
        return 4.0
    if "purchase all garage properties" in desc:
        return 1.5
    if "beat cal in a fishing challenge" in desc:
        return 5.0
    if "hunt a unique animal" in desc:
        return 2.5
    if "complete" in desc:
        return 10.0  # other named story / strand missions
    if tier == "gold":
        return 1.0
    if tier == "silver":
        return 2.0
    return 5.0


def forecast_case(case, name):
    if case.get("levels") != LEVELS:
        raise ValueError(f"{name} asks for levels {case.get('levels')}, not {LEVELS}")
    history = case.get("history") or []
    if not history:
        raise ValueError(f"{name} has no history")
    trophies = case["trophies"]
    horizon_days = case["horizon_minutes"] / 1440.0
    horizon_hours = case["horizon_minutes"] / 60.0

    game = case.get("game") or {}
    release = parse_time(game.get("release_date"))
    if release is None:
        release = parse_time(history[0].get("at"))
    target = parse_time(case.get("target")) or parse_time(case.get("at"))
    t_target = hours_between(release, target)

    last = history[-1]
    last_rates = last.get("rates") or []
    n_last = last.get("players_tracked") or 0

    # --- time axis for the history: x = ln(1 + hours since release) ---
    xs_all, n_all = [], []
    for read in history:
        t = hours_between(release, parse_time(read.get("at")))
        if t < 0:
            t = 0.0
        xs_all.append(math.log1p(t))
        n_all.append(read.get("players_tracked") or 0)
    x_target = math.log1p(max(0.0, t_target))
    # Prefer the last 36 h of reads for the trend fit when there are enough.
    cut = None
    for k, read in enumerate(history):
        if hours_between(parse_time(read.get("at")), parse_time(last.get("at"))) <= 36.0:
            cut = k
            break
    fit_idx = list(range(cut, len(history))) if cut is not None else list(range(len(history)))
    if len(fit_idx) < 3:
        fit_idx = list(range(len(history)))

    # Tracked-players power law: ln N on ln(1 + t), from all reads.
    ln_n_pts = [(x, math.log(n)) for x, n in zip(xs_all, n_all) if n and n > 0]
    if len(ln_n_pts) >= 2:
        a_n, b_n = ols([p[0] for p in ln_n_pts], [p[1] for p in ln_n_pts])
        b_n = max(-0.5, min(2.0, b_n))
        n_target = math.exp(a_n + b_n * x_target)
    else:
        n_target = float(n_last or 1)
    g_n = clamp(n_target / max(1.0, float(n_last or 1)), 1.0, 40.0) if n_last else 1.0

    # Prior weight: none on a 90-minute horizon, full on the 4.5-day final.
    w_prior = 0.45 * min(1.0, horizon_days / 4.5)

    medians, sigmas = [], []
    for i, trophy in enumerate(trophies):
        r_last = last_rates[i] if i < len(last_rates) else None
        if r_last is None:
            r_last = 0.0
        r_last = float(r_last)
        lg_last = logit(r_last)

        # --- trend fit on this trophy's logit rates ---
        pts = []
        for k in fit_idx:
            rates = history[k].get("rates") or []
            r = rates[i] if i < len(rates) else None
            if r is None:
                continue
            r = float(r)
            if 0.0 < r < 100.0:
                pts.append((xs_all[k], logit(r)))
        if len(pts) >= 3 and max(p[0] for p in pts) - min(p[0] for p in pts) > 0.05:
            a_t, b_t = ols([p[0] for p in pts], [p[1] for p in pts])
            b_t = max(-2.0, min(2.0, b_t))
            resid = math.sqrt(sum((y - (a_t + b_t * x)) ** 2 for x, y in pts) / len(pts))
            lg_trend = a_t + b_t * x_target
        else:
            b_t, resid = 0.0, 0.30
            lg_trend = lg_last
        lg_trend = max(lg_last - 2.2, min(lg_last + 2.2, lg_trend))

        # --- counts model: earners power law over tracked power law ---
        e_pts = []
        for k in range(len(history)):
            rates = history[k].get("rates") or []
            r = rates[i] if i < len(rates) else None
            n = n_all[k]
            if r is None or not n:
                continue
            e = float(r) / 100.0 * n
            if e >= 1.0:
                e_pts.append((xs_all[k], math.log(e)))
        if len(e_pts) >= 3 and max(p[0] for p in e_pts) - min(p[0] for p in e_pts) > 0.05:
            a_e, b_e = ols([p[0] for p in e_pts], [p[1] for p in e_pts])
            b_e = max(0.0, min(2.0, b_e))  # earners are cumulative: never fall
            e_target = math.exp(a_e + b_e * x_target)
            e_last = max(1.0, r_last / 100.0 * max(1.0, float(n_last or 1)))
            g_e = clamp(e_target / e_last, 1.0, 60.0)
            m_counts = clamp(100.0 * (e_last * g_e) / (max(1.0, float(n_last or 1)) * g_n))
        else:
            m_counts = inv_logit(lg_trend)
        lg_counts = logit(m_counts)

        # --- blend trend, counts and prior ---
        prior = prior_for(trophy)
        if prior is None or r_last >= 99.5:
            lg_med = 0.5 * (lg_trend + lg_counts)
        else:
            lg_med = (1.0 - w_prior) * 0.5 * (lg_trend + lg_counts) + w_prior * logit(prior)
        # A rate cannot credibly fall to near zero or explode in a week from
        # where 2.5 days of play put it; keep the median within reach.
        lg_med = max(lg_last - 2.5, min(lg_last + 2.5, lg_med))
        med = inv_logit(lg_med)
        if r_last >= 99.5:
            med = max(med, 99.0)
        medians.append(med)

        # --- spread ---
        disagree = abs(lg_trend - lg_counts)
        if prior is not None:
            disagree = max(disagree, abs(lg_trend - logit(prior)) * w_prior)
        sigma = math.sqrt((0.5 * math.sqrt(horizon_days)) ** 2 + 0.35 ** 2
                          + (0.25 * disagree) ** 2 + min(resid, 0.6) ** 2)
        if horizon_hours < 12.0:
            sigma = max(sigma, 0.85)  # a refresh can jump every rate at once
        sigma = max(0.35, min(1.6, sigma))
        if r_last >= 99.5:
            # Survived Exophase's first refresh at 100% (638 -> 3,868 tracked,
            # every other rate fell), so the median stays near 100, but the
            # logit scale near 100 is extreme: keep some spread in case the
            # tracked base eventually includes players who never fished.
            sigma = 0.35
        sigmas.append(sigma)

    # --- structural enforcement on medians ---
    # Story chapters non-increasing, final mission <= chapter 4, platinum
    # below every other trophy (it requires all of them).
    chapter_no = {}
    for i, trophy in enumerate(trophies):
        m = re.search(r"complete chapter (\d)", (trophy.get("description") or "").lower())
        if m:
            chapter_no[i] = int(m.group(1))
    prev = None
    for i in sorted(chapter_no, key=lambda k: chapter_no[k]):
        if prev is not None:
            medians[i] = min(medians[i], prev)
        prev = medians[i]
    final_idx = [i for i, t in enumerate(trophies)
                 if "complete the final mission" in (t.get("description") or "").lower()]
    if final_idx and chapter_no:
        last_ch = max(medians[i] for i in chapter_no)
        medians[final_idx[0]] = min(medians[final_idx[0]], last_ch)
    plat_idx = [i for i, t in enumerate(trophies) if (t.get("tier") or "") == "platinum"]
    if plat_idx:
        others = [medians[i] for i in range(len(trophies)) if i != plat_idx[0]]
        if others:
            medians[plat_idx[0]] = min(medians[plat_idx[0]], 0.9 * min(others))

    rows = []
    for i, trophy in enumerate(trophies):
        lg_med = logit(medians[i])
        qs = []
        for z in Z:
            q = inv_logit(lg_med + z * sigmas[i])
            qs.append(round4(clamp(q)))
        # keep non-decreasing after rounding/clamping
        for k in range(1, 5):
            if qs[k] < qs[k - 1]:
                qs[k] = qs[k - 1]
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
