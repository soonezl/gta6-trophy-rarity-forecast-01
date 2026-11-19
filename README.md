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

## How it works

The printed rate is earners divided by players tracked, and the
program treats those two separately.

When the history moves, each trophy gets a least squares fit of its
logit rate against log time over the last 24 hours of reads, with the
slope pulled toward the median slope across trophies. A second fit
tracks log earners and log players separately, and the two forecasts
are averaged on the logit scale. The forecast is anchored on the last
read and the extrapolated move is capped, so a 4.5 day horizon cannot
run away from two days of data.

When the history is completely flat, the page is stale. Exophase
refreshes its counts in steps, and the kit documents the first one:
638 players tracked on launch night, then 3,868 at the 04:30 UTC
refresh, with every rate but the 100% one lower. A flat page at launch
means a refresh is overdue, so the program forecasts the drop a
refresh brings. Earners grow a little while the player count jumps,
which pulls every rate down. The earner growth slope is set so the
model's average drop matches the typical miss the kit publishes for
the house baseline on the practice (0.68 points). Trophies that
already sit higher get faster earner growth than the rare ones. The
platinum and the 100% trophy are handled on their own: nobody new
earns a platinum in the first hours, and the 100% trophy stayed at
100% through the first refresh.

The spread grows with the square root of the horizon in days, with
the fit residual, and with how stale the last read is. A flat history
gets a wide spread because a refresh can land at any moment inside
the horizon.

## Running it

    mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py

## Limits

The practice answers live on the server, so the practice score cannot
be measured locally. What can be checked locally is the output shape
(55 trophies, five non-decreasing quantiles each, 0 to 100, under the
65,536 byte cap), determinism (two runs give identical bytes), and
runtime (well under the 8 second cap). The sensitivity checks in the
build notes score the model against a range of plausible refresh
outcomes instead of a single guessed answer.

A flat history carries almost no signal about how much each trophy's
earners grew, so the refresh drop is a calibrated prior, not a
measurement. Where the history moves, the trend fit takes over and
the prior stops mattering.
