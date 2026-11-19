[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/soonezl/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/soonezl/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
Bounty: https://make.fun/bounty/gta6-trophy-rarity-forecast-01

# GTA VI trophy rarity forecast

A program that forecasts each PlayStation 5 trophy's earn rate on
Exophase at the case target, as five quantiles per trophy. It reads
every `*.in` case file from the input directory and writes one
`*.out` file per case. Python 3.13, standard library only, no network,
no randomness, no clock reads, so the same case always gives the same
output.

## How it works (v2)

The printed rate is earners divided by players tracked. Two regimes,
chosen from the case data.

When the history moves, each trophy gets a saturating trend: a least
squares quadratic of its logit rate against log time over the last 24
hours of reads. A trophy's earn rate is an S-curve — it climbs, then
flattens as it approaches the share of players who will ever earn
it — and a straight log-time line keeps climbing past that ceiling.
The quadratic bends over; it is only ever used while it is concave,
and the forecast is never extrapolated past the parabola's vertex,
because a saturating curve asymptotes, it does not turn down. A
trophy already past its peak in the window falls back to a straight
line. Moves are shrunk toward the cross-trophy median move and
capped at ±2.5 logit, and the forecast is anchored on the last read.
The v1 earners/players decomposition is gone: it extrapolated
earners and players on separate log-time lines, and once the player
count saturates the ratio explodes toward 100%.

When the history is completely flat, the page is stale. Exophase
refreshes its counts in steps, and the kit documents the first one:
638 players tracked on launch night, then 3,868 at the 04:30 UTC
refresh, with every rate but the 100% one lower. A flat page at launch
means a refresh is overdue, so the program forecasts the drop a
refresh brings. Earners grow a little while the player count jumps,
which pulls every rate down. Earner growth is a single common factor
for all trophies — the launch rate does not predict which trophies
grow fastest (the early-access sample is 4–13 earners per trophy,
mostly noise), so ranking growth by launch rate only adds error. The
factor is set so the model's average drop matches the typical miss
the kit publishes for the house baseline on the practice (0.68
points). The platinum and the 100% trophy are handled on their own:
nobody new earns a platinum in the first hours, and the 100% trophy
stayed at 100% through the first refresh.

The spread grows with the square root of the horizon in days, with
the fit residual, and with how stale the last read is, then is scaled
by 0.6: the v1 spreads were about twice as wide as the forecast
errors justify. A flat history gets a wide spread because a refresh
can land at any moment inside the horizon.

## Backtest

The practice answers live on the server, so the practice score cannot
be measured locally. v2 was instead backtested against synthetic
worlds calibrated to the kit's published practice numbers (house
baseline loss ~1.51, typical miss 0.68 points), ten scenarios
(influx speed, progression speed, delay spread, play-start lag,
game length) × eight seeds. Pinball loss on final-shaped cases:
**v2 0.23, v1 3.83, house persistence baseline 0.51**. On practice
cases: **v2 0.26, v1 0.30, baseline 1.20**. v2 beats the baseline on
the final in every scenario tested.

## Running it

    mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py

## Limits

The practice answers live on the server, so the practice score cannot
be measured locally. What can be checked locally is the output shape
(55 trophies, five non-decreasing quantiles each, 0 to 100, under the
65,536 byte cap), determinism (two runs give identical bytes), and
runtime (well under the 8 second cap). The backtest above scores the
model against a range of plausible worlds instead of a single guessed
answer — but those worlds are still synthetic: the real player influx
and progression speeds are unknown, and if the real final behaves
very differently (for example, the player count still growing fast
at the deadline instead of saturating), the ranking of the variants
could shift.

A flat history carries almost no signal about how much each trophy's
earners grew, so the refresh drop is a calibrated prior, not a
measurement. Where the history moves, the trend fit takes over and
the prior stops mattering.
