# 3-Player Mahjong

## Usage

```py
import mahjax

env = mahjax.make("red_3p", observe_type="dict")
```

or load the environment class directly:

```py
from mahjax.mahjong_3p.config import RED
from mahjax.mahjong_3p.env import Mahjong3p

env = Mahjong3p(round_mode="half", observe_type="dict", game_config=RED)
```

## Rules

`red_3p` is 3-player riichi mahjong (三人麻雀) as [Tenhou](https://tenhou.net/) plays it. The rules are `red_4p`'s ([Red Mahjong](red_mahjong.md)) except for the following.

- **Tiles:** 108 tiles. 2m–8m are left out, and the red fives are one 5p and one 5s.
- **Game:** everyone starts with 35,000 points, and the game can end after South 3 once someone has 40,000. East and South have three kyoku each, and sudden death runs through West 1–3.
- **No chi.**
- **Kita (北抜き):** on their turn a player may set a north from their hand aside (`KITA`).
  - Each north set aside counts as a dora, one more if the north is a dora.
  - The player then draws a replacement tile from the dead wall, and winning on it is rinshan kaihou.
  - Before that, the others may ron the north, without chankan.
  - Once nobody rons the north, the kita counts as a call: it ends every ippatsu and interrupts the first go-around (no double riichi, nine terminals or blessings after it). A ron on the north itself keeps the winner's ippatsu.
  - Not right after a pon. In riichi only a north just drawn may be set aside, and never on the haitei draw.
- **Dora:** a 1m indicator points at 9m, and a 9m indicator at 1m.
- **Payments:** tsumo-son, so a tsumo is paid by the two others only. Honba is 100 points per payer, and the noten players pay 2,000 between them.
- **Riichi** needs at least 3 live-wall draws left.
- **Special abortive draws:** nine terminals and four kans only.

`no_red_3p` is `red_3p` with the `NO_RED` preset: no red fives, no double ron, no pao and no special abortive draws. The other `GameConfig` fields work as they do for four players.

## State, observation and actions

- The state has a row per player for three players, plus `players.kita` (the norths each player has set aside) and `round_state.kita_declared`.
- There are 88 actions: the 4-player ids plus `KITA` (87). The chi ids are never legal.
- The dict observation has the 4-player keys with three seats where those have four, plus `kita` (3,), seat-rotated like `scores`.
- The UI (`mahjax.ui`) does not support three players yet.
