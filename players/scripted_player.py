"""Scripted (hand-coded) cogame-moba player: ``python -m players.scripted_player``.

A per-hero finite-state lane-push bot, written as a legible reference
policy (Task 6.2b) — not a tuning contest entry. Structure borrows from
the coworld-ctf baseline bot: perception is parsed into an explicit
state struct, a small persistent world model sits on top of the raw
observation (static nav grid + cost fields, dead-tower memory), roles
are deterministic from the hero identity, and steering is separated
from mode selection.

Ground truth for every constant below is the vendored upstream source,
``vendor/upstream/moba.h`` (line numbers cited per table).

Observation contract (compute_observations, moba.h:460-533)
-----------------------------------------------------------
Each hero gets 510 bytes: a 484-byte "crop" region then 26 self bytes.

Self bytes (obs[484:510], moba.h:480-503): x, y, level, health*10/max,
mana*10/max, damage/50, move_speed, move_modifier (int-cast: slow 0.5
reads as 0), stun/move/q/w/e/basic-attack timers, basic cd, is_hit,
team, hero one-hot at 17+hero_type, and four reward-event flags.

Crop reliability — the stride-1 overlap quirk (moba.h:505-531): the
crop is written per cell at ``map_idx = (dy+5)*11 + (dx+5)`` — stride
1, not 4 — so indices run 0..120 while the buffer holds 484 bytes.
For each cell the tile id is written at ``map_idx`` and, if an entity
occupies the cell, health/mana/level at ``map_idx+1..3`` — which are
exactly the *tile* slots of the next three cells. Cells are visited in
increasing ``map_idx`` order, so every tile write at index i happens
AFTER any earlier cell's entity bytes landed on index i and clobbers
them. Conclusion, relied on throughout this module:

- bytes 0..120 are ALWAYS the 121 tile ids of the 11x11 crop
  (row-major, row = dy+5, col = dx+5, center = self);
- entity health/mana/level survive ONLY for the last cell (indices
  121..123, bottom-right corner) — useless, ignored;
- bytes 124..483 are never written (memset zero, moba.h:462).

So the crop yields tile-id presence only: which tile types sit where
around us. Tile ids (moba.h:28-43): EMPTY 0, WALL 1, TOWER 2 (team NOT
encoded — resolved against the static tower table via our absolute
position), RADIANT_CREEP 3, DIRE_CREEP 4, NEUTRAL 5, radiant heroes
6..10, dire heroes 11..15.

Action contract (step_players, moba.h:1503-1545)
------------------------------------------------
[vel_y, vel_x, attack_target, use_q, use_w, use_e], each 0..ACT_HIGH-1.
Velocity decodes as (a-3)/3, dest = pos + move_modifier*speed*vel with
speed 1.0 (sim/shim_common.h:17-18); move_to fails silently into
non-empty cells — players get NO pathfinding, hence the client-side
nav grid. attack_target 0/1 scan ALL hostiles (creeps, neutrals,
heroes, towers) in the vision square (radius 5); 2 scans heroes+towers
only (moba.h:1519-1525). The engine self-selects the NEAREST scanned
entity as the target for skills and basic attacks; skill flags are
tried Q, then W, then E, then basic attack (moba.h:1533-1541) and a
failed skill (cooldown/mana/target-type) falls through harmlessly.

Mode machine (per hero)
-----------------------
- PUSH: follow the hero's lane waypoints (WAYPOINTS, moba.h:71-72)
  toward the enemy ancient. Lanes mirror the engine's own hero lane
  assignment (init_moba, moba.h:1667-1744): support/carry mid-bot
  lane 2, assassin/burst lane 1, tank lane 0 (+3 for dire). Steering
  is a BFS cost field over the embedded static wall grid (derived from
  vendor/upstream/game_map.h; sha256 ec17403c...), descending one
  8-connected step per tick exactly like the engine's own bfs atn map.
- SIEGE HOLD (inside PUSH): towers scan at the same radius as heroes
  (TOWER_VISION 5, moba.h:45 — no out-ranging) but hit for 110-175 per
  shot (moba.h:65), an order of magnitude over a hero basic attack, so
  diving one without a creep wave soaking its shots is a fast death.
  When a live enemy tower is within its scan radius and no friendly
  creep wave is engaged on it, back off and wait for creeps. Ancients
  are exempt (TOWER_DAMAGE 0 for both, moba.h:65): they cannot shoot,
  so the endgame dive is free. Tower liveness is
  a small world model: a tower whose known cell is visible in the crop
  without a TOWER tile is remembered dead (kill_entity's body zeroes
  the grid cell, moba.h:632-633; called on tower death from attack,
  moba.h:735-737).
- RETREAT: health <= 3/10 sends the hero toward its own fountain
  (spawn, moba.h:1636-1643); passive +2/tick regen (moba.h:1468-1475)
  heals it; hysteresis exits at >= 8/10 so it cannot oscillate.
  Death needs no special mode: the engine respawns instantly at the
  fountain with full health (attack -> spawn_player, moba.h:723-728);
  a position teleport just resets waypoint tracking to the nearest
  waypoint. Rushers (below) use a much tighter 1/10 -> 4/10 band:
  since death is an instant full-heal teleport home, a rusher walking
  home at low health only loses race time.
- DEFEND (dire only, shared alarm): heroes report enemy-hero
  sightings near the own ancient into seat-shared state. A single
  sighting within 30 cells of the ancient, or >= 2 distinct enemy
  heroes within 45 cells in one tick (a grouped dive), arms a 90-tick
  alarm; while it is armed, dire heroes within 60 cells of the own
  ancient (and healthy enough, or already levelled past 10) rally on
  the ancient so the base towers back the fight. This exists because
  the pretrained baseline wins by 5-hero ancient dives that a pure
  lane-push FSM never answers. Radiant plays the classic v1 lane push
  bit-identically: on that side the FSM already beat the baseline
  100% of the time, and league A/B showed the extra machinery only
  hurt against rivals that keep lane pressure up.
- RUSH (dire only): the map and the pretrained baseline are radiant-
  favored — measured head-to-head, baseline-as-radiant group-dives
  the dire ancient by tick ~1300 while a symmetric lane push cracks
  the radiant base only by tick ~2100+, so dire always loses that
  race. Instead of racing on the baseline's terms, dire exploits two
  vendored facts: attacks land at L1 range <= 12 (moba.h:686-689)
  while both tower scan and player scan reach only chebyshev 5, and
  ancients deal no damage and never regenerate (moba.h:65, 1422-1425
  commented out). Cells therefore exist that poke the enemy ancient
  risk-free (POKE below), and a Dijkstra route over the wall grid
  (RUSH_ROUTE) reaches dire's poke cell while entering tower range on
  only 3 cells (~one 175-damage shot on the sprint through). The
  assassin, tank and carry take that route and grind the ancient
  down; the burst stays on its mid lane so the lanes are not entirely
  free-fed, and the support garrisons the own ancient as a sentinel
  to spot and stall dives (towers plus fountain proximity). Opponents
  cannot even observe their own ancient's health (it is not in the
  obs contract), so a backdoor race is structurally hard to answer.
  Radiant keeps the classic lane push, which already beats the
  baseline 100% of the time on that side.
- RADIANT DEFENSE (reactive, time-boxed): the co-gas rivals' standard
  dire opening is a five-hero pack down the west edge — it appears in
  every recorded win AND loss, so no route predicate can tell games
  that need a defense from games that don't. What separates them is
  the base siege itself: in 11 of 12 recorded wins no enemy hero ever
  comes within 12 cells of the radiant ancient, while in every
  recorded loss the pack camps there from tick ~330-640 and needs
  580+ more ticks to burn the ancient. An enemy hero SIGHTED within
  12 of our ancient therefore heats a 900-tick defense window
  (re-armed by every further sighting): while hot, the support
  garrisons the west staging watchpost (100,19); the other four
  heroes keep pushing their v1 lanes (recalling them was measured to
  cost more than it saves). When the window cools the support
  re-anchors to its lane. Without a sighted intrusion radiant play
  is bit-identical v1, full-episode, on 11 of the 12 recorded wins
  and every baseline game.
- PASSIVITY GATE: classic v1 lane play is dire's ground state. The
  sentinel+rush plan only arms at tick 400, and only if no enemy hero
  was ever sighted in the north-west quadrant behind dire's top lane
  (y <= 30 and x <= 60) or on our ancient. Aggressive league rivals
  race cross-map through exactly that quadrant and are sighted there
  by tick ~90 in every recorded loss (head-on into our own v1 top
  laner), while the pretrained baseline produced zero such sightings
  in 500 ticks across every battery seed — so passive-baseline games
  always arm the rush, and map-fighting opponents are answered with
  the proven v1 lane game for the whole episode. The gate is
  dire-only: a radiant seat never changes behavior.
- STUCK detour: creeps and heroes block cells; if position hasn't
  moved for a few ticks the desired step is rotated 90 degrees
  (alternating side) for a few ticks to slide around blockers.
  Rotating a blocked diagonal explores only its two perpendiculars,
  which can all be blocked (walls plus float-truncated move_to) —
  league replays showed dire heroes frozen that way at one cell for
  thousands of ticks. A dire hero still unmoved after a full detour
  cycle therefore escalates to a deterministic sweep of all 8 engine
  step directions (5 ticks each) until the position actually
  changes, and switches to the scan-everything filter so entity
  plugs (neutral camps, creep scrums) get attacked clear. Radiant
  keeps the plain v1 detour bit-identically.

Skills (moba.h:1151-1362) are fired by per-role rules on the engine's
Q->W->E priority chain; the target filter is 0 (everything) when
hostile creeps are visible — lane-clearing XP — else 2 (heroes+towers)
so sieging focuses towers and idle neutrals camps are not aggroed.

Determinism: pure function of (obs bytes, per-hero state). No RNG and
no clocks; COGAME_PLAYER_SEED is accepted for interface parity but
unused. Works for 1-hero and 5-hero seats: hero identity (team, hero
type) is read from each obs row, per-hero state is keyed by row index.
"""

from __future__ import annotations

import base64
import sys
import zlib
from collections import deque
from dataclasses import dataclass

from .client import run_policy_main, seed_from_env

# -- action space (mirrors players.random_player.ACT_HIGH) -------------------

ACT_HIGH = (7, 7, 3, 2, 2, 2)

# -- tile ids (moba.h:28-43) -------------------------------------------------

EMPTY, WALL, TOWER = 0, 1, 2
RADIANT_CREEP, DIRE_CREEP, NEUTRAL = 3, 4, 5
# heroes: 6+hero_type for radiant, 11+hero_type for dire

OBS_SIZE = 510
CROP_TILES = 121          # reliable prefix of the 484-byte crop region
CROP_W = 11
VIS = 5                   # vision_range (sim/shim_common.h:17)
SELF_OFF = 484            # 11*11*4

MAP_W = 128

# -- hero roles (init_moba, moba.h:1661-1744) --------------------------------

SUPPORT, ASSASSIN, BURST, TANK, CARRY = range(5)
# hero_type -> radiant lane (dire adds 3); moba.h:1676,1693,1710,1727,1744
LANE_FOR_HERO = (2, 1, 1, 0, 2)
# hero_type -> (base_mana, mana_gain_per_level); moba.h:1672-1743
MANA_TABLE = ((250, 50), (300, 65), (300, 90), (200, 50), (250, 50))

# team -> fountain/spawn cell (init_moba, moba.h:1636-1643)
SPAWN = ((113, 12), (15, 116))

TOWER_VISION = 5          # moba.h:45

# (y, x, team, tier) per tower index; 22 = dire ancient, 23 = radiant
# ancient (moba.h:65-70; ancient pids in c_step, moba.h:1909-1952)
TOWERS = (
    (112.015, 34.6, 0, 3),
    (96.877, 29.308, 0, 3),
    (91.215, 14.292, 0, 3),
    (113.615, 64.2, 0, 2),
    (112.138, 102.877, 0, 1),
    (85.431, 38.292, 0, 2),
    (71.708, 17.615, 0, 2),
    (51.031, 16.877, 0, 1),
    (102.415, 21.062, 0, 4),
    (28.262, 103.031, 1, 4),
    (18.723, 65.0, 1, 2),
    (18.723, 29.062, 1, 1),
    (48.754, 84.2, 1, 2),
    (59.985, 69.031, 1, 1),
    (62.046, 112.754, 1, 2),
    (41.677, 113.738, 1, 3),
    (20.569, 92.323, 1, 3),
    (36.085, 97.862, 1, 3),
    (30.908, 105.615, 1, 4),
    (104.938, 23.523, 0, 4),
    (75.831, 53.123, 0, 1),
    (78.3, 113.223, 1, 1),
    (26.538, 107.523, 1, 5),   # dire ancient
    (106.169, 19.462, 0, 5),   # radiant ancient
)
ANCIENT_IDX = (23, 22)        # per team: own ancient tower index

# (y, x) polylines, lanes 0-2 radiant->dire, 3-5 dire->radiant
# (WAYPOINTS/WAYPOINTS_N, moba.h:71-72)
WAYPOINTS = (
    (  # lane 0 (radiant top), 14 waypoints
        (96.262, 14.046), (93.615, 14.292), (58.231, 16.077),
        (43.4, 16.015), (32.108, 17.585), (24.938, 19.123),
        (22.969, 20.446), (21.062, 25.308), (20.2, 30.477),
        (19.4, 35.892), (18.477, 43.523), (18.046, 50.723),
        (20.508, 94.662), (27.677, 105.946),
    ),
    (  # lane 1 (radiant mid), 10 waypoints
        (99.523, 26.477), (98.662, 27.646), (86.046, 41.123),
        (71.338, 57.492), (66.969, 62.231), (64.077, 66.354),
        (60.2, 72.015), (51.892, 84.877), (34.662, 99.4),
        (28.169, 105.946),
    ),
    (  # lane 2 (radiant bot), 13 waypoints
        (112.015, 36.938), (112.015, 32.508), (116.2, 68.692),
        (113.738, 80.754), (113.554, 90.538), (113.215, 95.985),
        (111.492, 100.938), (109.062, 108.077), (107.185, 111.369),
        (101.738, 114.662), (90.538, 111.769), (39.4, 113.738),
        (28.169, 106.438),
    ),
    (  # lane 3 (dire top), 9 waypoints
        (20.446, 89.369), (20.631, 94.662), (18.785, 50.6),
        (22.292, 24.508), (22.815, 18.815), (27.523, 17.031),
        (35.031, 15.954), (93.615, 14.415), (103.646, 21.992),
    ),
    (  # lane 4 (dire mid), 9 waypoints
        (37.431, 96.508), (34.538, 99.277), (51.708, 84.385),
        (59.831, 71.646), (63.831, 66.108), (66.723, 61.985),
        (70.846, 57.246), (98.538, 27.523), (103.646, 22.485),
    ),
    (  # lane 5 (dire bot), 12 waypoints
        (36.938, 113.246), (39.4, 113.615), (62.292, 114.6),
        (83.646, 114.6), (90.662, 109.8), (94.877, 112.015),
        (104.262, 112.292), (109.308, 107.831), (112.231, 105.862),
        (114.446, 72.969), (111.892, 32.508), (104.138, 22.485),
    ),
)

# 128x128 wall mask (0 empty / 1 wall) from vendor/upstream/game_map.h,
# zlib-compressed + base64 (raw grid sha256
# ec17403cd2c93547f0d9da4da8d0d89ab08160153dc3a68e680e8dc9cc430478).
_WALLS_B64 = (
    "eNrtW+2SIyEIpN//pa+2NpOMCkgjmqu78cdWbda1/YCmQQM87WlPu5pYbd5D6VsH/x5UduJL"
    "TfsyfHYCUojvT032ws/xZSu8iy+/WHvgf0b//WH3ECg2InJs/aqVnsLHEXzeSb8M/3+tfy/5"
    "BJz0bzj/Inx0Tj3z/xr87j/7IWj85BHazLIJX+85DMLiM/FFH0lW8Cli1/q/REUSP3DSVyfE"
    "DofEV7tDA4waeWQCt/1H2NJ8H2gsYsv5FyiQNfsPzmMTfnx7Iz2qxSeoCVTGP3UQFOLDHxDp"
    "OcfwX0lLyC9r8fFheME1kXr8Ydz7p8CMsDMn4+KDG9E0DyTxOR50UUrwSd+O9KWLP6MKQTzO"
    "0ecPW2DdAibCtj707rE7+dIsuPEDZU+jZB/Lf9CLj0H+gPf/u8yY5V/TfKJTZbnwEbE8AwGf"
    "akvQCkn83sD7BKcxi+UE0OaX+wRm4nQth3PiXtyny/E9vl/MGcPRl4wr0RyTiP7ZwoDr14z6"
    "SIpRl1b2rn9uNXP8JqWmMgCoAoHEn2T0YGxd8SRG+aj2y+w+tD1x8TH8P5P/Lvv/IEfQqwAO"
    "n/V/eLPJ4FP86wuCSDFmzpcuNbq/1+TLNv6UEUAudQW/F8JdpSsfG3Pr7+txc/HI2z/M87d8"
    "JcILWFm/n/A0N4pSv/6mDuCRLMKrn+N3fxr6qicPUynS60e7nrEvDF5SFBzI8x+T3uvS3D0q"
    "m8xB4XN1lJDvYg0/HGxs6pKl9Wd0H47VXxFW/cFe3JQUsXJ9jBQ+f5k0zxmp9Qt3l+NuGBj8"
    "e6iJGoffz8+hFVZNWB8mDIEwvmRzPfpw6u4fYKUI7gSL4duwiajlX72xTn3ySU9CT/m0CiCx"
    "GwZxR2qC7SO0JPnBtOy3MvULGA75zHfBzwReT9x8bmAFY3yPuOqU1E9ACvAXGP8EPtbntu/t"
    "UR1+/CXFUO1CAb6pZnvN295PRSJp8P1J5G0GkCrPh84/xY+V9mdfBtlVbaoAweJr4SbFChl8"
    "rRwzPzvU4Qt/9wss8U+4+MZRGVF9TmXCZfjhZR3Dh6G55MT+3/3+XRIHf0u0tP+D/IIsvGEl"
    "vtli8lzmEVni/emeL6kcFF31+VchPr6CDpppNuELdh+67nv++rEdHxH8wmqhnX179Y3d+NNE"
    "1XkrKM0wThJtf/vNuQp+2tOe9i+2P3gjEM0="
)

# Engine step directions in engine order (ATN_MAP, moba.h:58-61, with
# move_towards mapping row 0 -> dy, row 1 -> dx, moba.h:615-617):
# S, N, E, W, SW, NW, NE, SE as (dy, dx). Reused for neighbor iteration
# so tie-breaks are deterministic and engine-like.
STEPS = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, -1), (-1, 1), (1, 1))

# -- tuning knobs ------------------------------------------------------------

RETREAT_ENTER_HP = 3      # health tenths; <= enters RETREAT
RETREAT_EXIT_HP = 8       # >= returns to PUSH (hysteresis)
RUSH_RETREAT_ENTER_HP = 1  # rushers barely retreat: death is an instant
RUSH_RETREAT_EXIT_HP = 4   # full-heal teleport home, walking back is not
WAYPOINT_REACHED = 2.5    # chebyshev cells; matches creep_ai's <2 spirit
TELEPORT_DIST = 4.0       # position jump > this = respawn/displacement
STUCK_TICKS = 4           # unmoved ticks before detouring
DETOUR_TICKS = 5          # how long one detour burst lasts
SIEGE_STANDOFF = TOWER_VISION + 1   # hold distance from a live tower
CREEP_ENGAGED_DIST = 5    # friendly creep within this of tower = engaged
ALARM_RADIUS = 45         # group-sighting distance from own ancient
ALARM_NEAR = 30           # single sighting this close to the ancient counts
ALARM_GROUP = 2           # distinct enemy heroes sighted in one tick = dive
ALARM_TICKS = 90          # alarm persistence after the last sighting
BREAKOUT_LEVEL = 10       # this well-fed, a hero pushes through anything
RECALL_RADIUS = 60        # heroes closer than this to own ancient defend
DEFEND_REJOIN_HP = 5      # retreating defenders rejoin at this health
SENTINEL_TEAM = 1         # dire only: keep a garrison hero at the ancient
SENTINEL_HERO = SUPPORT   # weakest pusher stands guard / trips the alarm
RUSHER_HEROES = (ASSASSIN, TANK, CARRY)   # dire backdoor squad; the
# burst stays on its mid lane so the lanes are not entirely free-fed
RAD_RUSHER_HEROES = (ASSASSIN, BURST, TANK, CARRY)  # radiant counter-
# backdoor squad: when the enemy pack besieges our base the mid lane
# is dead ground anyway (observed: the burst spent 500+ ticks in a
# creepless siege standoff there), so it rushes too
# Passivity gate: classic v1 lane play is dire's GROUND STATE — it is
# the proven posture against opponents that fight for the map (league
# floor 0.68-0.74 vs the co-gas rivals). Dire switches to the
# sentinel+rush plan only on positive evidence of passivity: if no
# enemy hero has been sighted sprinting the top-edge corridor (the
# cross-map attack path every observed aggressive rival uses from
# tick ~150, head-on into our own v1 top laner's lane) and none has
# been sighted on our ancient by the gate tick, the opponent is
# baseline-passive and the backdoor race is safe. Any such sighting
# before the gate keeps dire in v1 lane play for the whole episode.
RUSH_GATE_TICK = 400
NW_ZONE_Y = 30            # evidence: enemy sighted at y <= this ...
NW_ZONE_X = 60            # ... and x <= this (the north-west quadrant
# behind dire's top lane: measured over every recorded rival loss the
# cross-map conveyor is sighted there by tick ~90, head-on into our
# own v1 top laner, while the pretrained baseline produced ZERO
# sightings there in 500 ticks across all battery seeds)
INTRUSION_RADIUS = 5      # or an enemy seen this close to our ancient
# Radiant defense (v7b, re-derived from 18 recorded league games):
# the west-edge 5-hero pack is the co-gas rivals' STANDARD dire
# opening — it appears, and is sighted by tick ~120-180, in every
# recorded win AND loss, so no route/zone predicate can separate
# games that need a defense from games that don't (the v7 gate
# failed exactly this way). What does separate them is the base
# siege itself: in 11 of 12 recorded wins no enemy hero ever comes
# within 12 cells of the radiant ancient, while in the losses the
# pack camps there from tick ~330-640 and burns the ancient 580+
# ticks later. Radiant defense is therefore reactive and TIME-BOXED:
# an enemy hero sighted within RAD_INTRUSION of our ancient heats
# the defense for RAD_DEFENSE_HOT ticks (re-armed by every further
# sighting); while hot, the support garrisons the staging watchpost
# and the dive-alarm rally is enabled; when it cools, everyone
# returns to the classic v1 push. Cost is bounded and proportional:
# across the 12 recorded wins this fires exactly once (w866, tick
# ~836 — a real dive that v1 survived), and in every recorded loss
# it fires with 580+ ticks of lead time.
RAD_INTRUSION = 12        # sighted enemy this close to our ancient
RING_CAMP_TICKS = 30      # dwell ticks before a siege counts as a camp
# Aggro-mode watchman: both rivals' current versions convert their
# aggressive openings into ring sieges staged just outside the
# trigger radius (drift #6: a five-stack dwelling at (16-21,86),
# chebyshev 21 from the dire ancient, rotating into the ring only
# after our lane traffic has left the area). Detection latency, not
# radius, was the loss cause — so once aggro is sighted (tick ~90,
# proof of an aggressive racer), the dire support forgoes its lane
# and garrisons the ring-entry watchpost immediately, buying the
# siege trigger 100-300 ticks in every measured loss.
RAD_DEFENSE_HOT = 900     # defense stays hot this long per sighting
# Dire siege response: the co-gas rivals eventually adopted the safe
# poke ring themselves — as radiant they stand at (21-22, 99-106),
# outside every dire tower's reach, and grind the dire ancient while
# classic v1 lane play has literally no answer (it can never even
# engage them; five straight losses). A sighted enemy within
# RAD_INTRUSION of the DIRE ancient therefore overrides the
# passivity gate's v1 lock: the rush commits immediately (racing is
# the answer to a committed siege — the same logic as the radiant
# counter-backdoor) and the burst joins it as a fourth rusher (an
# alarm-turtled burst was observed defending for 1800 ticks while
# the undermanned rush lost the race).
# Radiant counter-backdoor: a sighted base siege proves the enemy
# pack is committed at OUR side of the map, which leaves their own
# ancient unguarded — and no opponent can even observe its health
# (not in the obs contract). The same trigger therefore also commits
# the radiant assassin/tank/carry to the mirrored backdoor rush
# (min-tower-exposure Dijkstra route to POKE[0], 2 cells inside any
# tower's range), permanently for the episode: in every recorded
# siege loss the pack needed 2200+ ticks to burn our ancient while a
# staged 3-hero poke burn needs a few hundred. Games with no sighted
# siege never rush and stay bit-identical v1.
RAD_RUSH_ROUTE = (
    (107, 15), (100, 22), (93, 29), (86, 36), (79, 43), (72, 50),
    (65, 53), (58, 58), (51, 63), (44, 69), (37, 75), (30, 76),
    (23, 80), (16, 87), (11, 94), (18, 99), (21, 102),
)
SENTINEL_POST = ((100, 19), (22, 101))  # per-team garrison spot,
# each covering the conveyor entry observed for that side ((96,15)
# staging for radiant, (20,99) entry for dire) with sightlines over
# the approach while staying adjacent to the base's guard towers
BLOCKED_ESCALATE = STUCK_TICKS + DETOUR_TICKS + 1  # sweep threshold
JAM_SKIP_TICKS = 60       # dire: jammed this long -> skip the waypoint
# Safe siege cells, per attacking team: chebyshev 5 from the enemy
# ancient (inside our attack scan, moba.h:1519-1525, and inside the
# L1<=12 attack range, moba.h:686-689) yet chebyshev >6 from every
# enemy guard tower (outside TOWER_VISION 5) — the ancient cannot
# shoot back (TOWER_DAMAGE 0), so this cell pokes it risk-free.
POKE = ((21, 102), (101, 14))
# Dire rush route to POKE[1]: min-tower-exposure Dijkstra over the
# embedded wall grid (cost 1 + 80 per covering radiant tower, towers
# as walls), dire spawn -> poke cell, sampled every 7 cells. Only 3
# cells sit inside any tower's scan radius (single tower, ~1 shot of
# 175 while sprinting through) — every other cell is out of range of
# every radiant tower.
RUSH_ROUTE = (
    (21, 110), (28, 104), (35, 97), (42, 90), (49, 84), (56, 77),
    (63, 70), (70, 66), (77, 59), (83, 55), (90, 48), (97, 41),
    (102, 34), (108, 27), (110, 20), (105, 13), (101, 14),
)
# Alternate rush corridors (same min-exposure Dijkstra, mid-map box
# penalized): a single-file squad on one 1-2 cell wide pass BODY-
# BLOCKS ITSELF (league drift #5: heroes jam-swept into each other at
# (66-72,63-70) for 250+ ticks) and hands the opponent one choke to
# picket. Splitting the squad across two disjoint corridors removes
# both failure modes. 3 exposed cells (single tower each).
RUSH_ROUTE_ALT = (
    (14, 110), (10, 103), (10, 96), (14, 89), (20, 82), (27, 75),
    (26, 68), (33, 61), (40, 54), (47, 47), (54, 40), (61, 33),
    (68, 26), (75, 23), (80, 17), (87, 10), (94, 6), (100, 13),
    (101, 14),
)
RAD_RUSH_ROUTE_ALT = (
    (107, 9), (100, 8), (93, 15), (86, 20), (79, 19), (72, 24),
    (65, 29), (58, 36), (51, 43), (44, 45), (40, 52), (33, 59),
    (26, 66), (28, 73), (22, 80), (16, 87), (11, 94), (18, 99),
    (21, 102),
)
ALT_ROUTE_HEROES = (BURST, TANK)   # squad split by hero identity


def _decode_walls() -> bytes:
    grid = zlib.decompress(base64.b64decode("".join(_WALLS_B64)))
    if len(grid) != MAP_W * MAP_W:
        raise ValueError("embedded wall grid has wrong size")
    return grid


# -- perception --------------------------------------------------------------

@dataclass
class HeroObs:
    """Decoded 510-byte observation (see module docstring for layout)."""
    x: int
    y: int
    level: int
    health10: int          # health * 10 / max_health, 0..10
    mana10: int            # mana * 10 / max_mana, 0..10
    damage50: int
    move_speed: int
    move_modifier: int     # int-cast: 0 = slowed (0.5), 1 normal, 2 hasted
    stun_timer: int
    move_timer: int
    q_timer: int
    w_timer: int
    e_timer: int
    basic_timer: int
    basic_cd: int
    is_hit: int
    team: int              # 0 radiant, 1 dire
    hero_type: int         # 0..4 support/assassin/burst/tank/carry
    crop: bytes            # the 121 reliable tile ids, row-major

    def tile(self, dy: int, dx: int) -> int:
        """Crop tile id at offset (dy, dx) from self; -1 if outside."""
        if abs(dy) > VIS or abs(dx) > VIS:
            return -1
        return self.crop[(dy + VIS) * CROP_W + (dx + VIS)]

    def max_mana(self) -> int:
        base, gain = MANA_TABLE[self.hero_type]
        return base + self.level * gain

    def mana_at_least(self, cost: int) -> bool:
        """Conservative: mana10 floor-scaled, so mana >= mana10/10*max."""
        return self.mana10 * self.max_mana() >= cost * 10


def parse_obs(obs: bytes) -> HeroObs:
    """Decode one 510-byte observation row. Robust to garbage bytes:
    every field is taken as-is; hero_type falls back to 0 (support) when
    the one-hot region is empty/ambiguous, team is clamped to 0/1."""
    if len(obs) != OBS_SIZE:
        raise ValueError(f"obs must be {OBS_SIZE} bytes, got {len(obs)}")
    e = obs[SELF_OFF:]
    onehot = e[17:22]
    hero_type = max(range(5), key=lambda i: onehot[i]) if any(onehot) else 0
    return HeroObs(
        x=e[0], y=e[1], level=e[2], health10=e[3], mana10=e[4],
        damage50=e[5], move_speed=e[6], move_modifier=e[7],
        stun_timer=e[8], move_timer=e[9], q_timer=e[10], w_timer=e[11],
        e_timer=e[12], basic_timer=e[13], basic_cd=e[14], is_hit=e[15],
        team=1 if e[16] == 1 else 0, hero_type=hero_type,
        crop=obs[:CROP_TILES])


def hostile_tiles(team: int) -> tuple[frozenset, frozenset]:
    """(enemy hero tile ids, enemy creep tile id-set) for a team."""
    if team == 0:
        return frozenset(range(11, 16)), frozenset((DIRE_CREEP,))
    return frozenset(range(6, 11)), frozenset((RADIANT_CREEP,))


def friendly_creep_tile(team: int) -> int:
    return RADIANT_CREEP if team == 0 else DIRE_CREEP


# -- navigation --------------------------------------------------------------

class NavGrid:
    """BFS cost fields over the embedded static map.

    8-connected uniform-cost BFS from each goal (the same connectivity
    as the engine's bfs, moba.h:242-320), cached per goal cell. All
    goals are static (waypoints, spawns, ancients) so the cache stays
    small. Known tower cells are treated as walls (towers block
    move_to; dead towers leave the cell empty but avoiding it is
    harmless). A wall goal cell is force-opened so fields toward the
    enemy ancient still resolve.
    """

    UNREACHED = 0xFFFF

    def __init__(self):
        walls = bytearray(_decode_walls())
        for ty, tx, _team, _tier in TOWERS:
            walls[int(ty) * MAP_W + int(tx)] = 1
        self._walls = bytes(walls)
        self._fields: dict[tuple[int, int], list[int]] = {}

    def is_wall(self, y: int, x: int) -> bool:
        if not (0 <= y < MAP_W and 0 <= x < MAP_W):
            return True
        return self._walls[y * MAP_W + x] != 0

    def _field(self, goal: tuple[int, int]) -> list[int]:
        cached = self._fields.get(goal)
        if cached is not None:
            return cached
        dist = [self.UNREACHED] * (MAP_W * MAP_W)
        gy, gx = goal
        gy = min(max(gy, 0), MAP_W - 1)
        gx = min(max(gx, 0), MAP_W - 1)
        start = gy * MAP_W + gx
        dist[start] = 0
        queue = deque((start,))
        walls = self._walls
        while queue:
            adr = queue.popleft()
            d1 = dist[adr] + 1
            y, x = adr // MAP_W, adr % MAP_W
            for dy, dx in STEPS:
                ny, nx = y + dy, x + dx
                if not (0 <= ny < MAP_W and 0 <= nx < MAP_W):
                    continue
                nadr = ny * MAP_W + nx
                if dist[nadr] != self.UNREACHED or walls[nadr]:
                    continue
                dist[nadr] = d1
                queue.append(nadr)
        self._fields[goal] = dist
        return dist

    def step_toward(self, y: int, x: int, goal: tuple[int, int],
                    exact: bool = False) -> tuple[int, int]:
        """Best (dy, dx) unit step from (y, x) toward goal.

        Descends the goal's cost field; deterministic tie-break in
        STEPS order. Falls back to the greedy sign step when the
        current cell is unreachable from the goal (e.g. standing on a
        cell our conservative wall model considers blocked).

        By default the walk stops when ADJACENT to the goal; with
        ``exact`` it stops only ON the goal cell. Poke cells need
        exact: they sit at chebyshev 5 from the enemy ancient, so an
        adjacent stop can leave the ancient at chebyshev 6 — outside
        the attack scan — and the hero pokes nothing forever
        (observed in live league losses: pokers parked at (20,103)
        for 500+ ticks with zero ancient damage)."""
        gy, gx = goal
        if (y, x) == goal if exact else (abs(y - gy) <= 1
                                         and abs(x - gx) <= 1):
            return (0, 0)
        field = self._field(goal)
        best, best_d = None, None
        if 0 <= y < MAP_W and 0 <= x < MAP_W:
            here = field[y * MAP_W + x]
        else:
            here = self.UNREACHED
        for dy, dx in STEPS:
            ny, nx = y + dy, x + dx
            if not (0 <= ny < MAP_W and 0 <= nx < MAP_W):
                continue
            d = field[ny * MAP_W + nx]
            if d == self.UNREACHED:
                continue
            if best_d is None or d < best_d:
                best, best_d = (dy, dx), d
        if best is not None and (here == self.UNREACHED or best_d < here):
            return best
        # unreachable (or already at the local minimum): greedy fallback
        sign = lambda v: (v > 0) - (v < 0)  # noqa: E731
        return (sign(gy - y), sign(gx - x))


# -- per-hero state machine --------------------------------------------------

PUSH, RETREAT = "PUSH", "RETREAT"


class HeroState:
    def __init__(self):
        self.mode = PUSH
        self.team: int | None = None        # last-seen team (from obs)
        self.wp_index: int | None = None    # next waypoint to visit
        self.last_pos: tuple[int, int] | None = None
        self.tried_move = False             # did LAST tick want to move?
        self.stuck_ticks = 0
        self.detour_left = 0
        self.detour_side = 1                # +1 / -1, alternates
        self.blocked_ticks = 0              # consecutive unmoved tries
        self.jam_pos: tuple[int, int] | None = None  # dire jam anchor
        self.jam_ticks = 0                  # monotonic sweep progress


def _rotate90(dy: int, dx: int, side: int) -> tuple[int, int]:
    """(dy, dx) rotated 90 degrees; side=+1 clockwise on screen."""
    return (side * dx, -side * dy)


class ScriptedPolicy:
    """policy(tick, obs_rows) -> [[6 ints] per hero].

    Deterministic; ``seed`` is accepted for parity with the other
    players but unused (no RNG anywhere). Per-hero state is keyed by
    row index within the seat; dead-tower memory is shared across the
    seat's heroes (their crops all feed it).
    """

    def __init__(self, seed: int | None = None):
        self.nav = NavGrid()
        self.heroes: dict[int, HeroState] = {}
        self.dead_towers: set[int] = set()
        self.alarm_ticks = 0        # shared base-threat alarm countdown
        self._tick_sightings: set[tuple[int, int]] = set()
        self.rush_on = False        # dire rush (armed by passivity gate)
        self.aggro_seen = False     # aggression evidence pre-gate
        self.dire_siege = False     # our ancient besieged: rush anyway
        self.dire_all_in = False    # besieged by ring campers: 5 race
        self._ring_flag = False     # uncovered intrusion seen this tick
        self.ring_ticks = 0         # accumulated camper-dwell ticks
        self.rad_defense_hot = 0    # radiant defense countdown (siege)
        self.rad_offense_on = False # radiant counter-backdoor (sticky)
        self._tick = 0              # current policy tick (from __call__)

    # -- world model updates ------------------------------------------------

    def _update_tower_memory(self, s: HeroObs) -> None:
        for idx, (ty, tx, _team, _tier) in enumerate(TOWERS):
            if idx in self.dead_towers:
                continue
            dy, dx = int(ty) - s.y, int(tx) - s.x
            if abs(dy) <= VIS and abs(dx) <= VIS:
                if s.tile(dy, dx) != TOWER:
                    self.dead_towers.add(idx)

    def _nearest_live_enemy_tower(self, s: HeroObs,
                                  ) -> tuple[int, int] | None:
        """(tower index, chebyshev distance) of the closest live enemy
        tower that can actually shoot, or None. Ancients (tier 5) are
        exempt: TOWER_DAMAGE is 0 for both ancients (moba.h:65), so
        there is nothing to stand off from — diving the ancient is
        free and standing off would cede endgame time."""
        best = None
        for idx, (ty, tx, team, tier) in enumerate(TOWERS):
            if team == s.team or tier == 5 or idx in self.dead_towers:
                continue
            d = max(abs(int(ty) - s.y), abs(int(tx) - s.x))
            if best is None or d < best[1]:
                best = (idx, d)
        return best

    # -- steering -----------------------------------------------------------

    def _is_rusher(self, s: HeroObs) -> bool:
        """Backdoor squad membership: dire once the passivity gate
        has committed, radiant once a base siege has been sighted
        (see module docstring). Rushers take the poke route instead
        of lane-pushing."""
        if s.team == SENTINEL_TEAM:
            if self.dire_all_in:
                return self.rush_on     # ring-poked: all five race
            if self.dire_siege:
                # dived: four race; the sentinel stays home — a dive
                # (unlike an untouchable ring-poke siege) fights
                # inside tower fire, where its body-stall matters
                return (self.rush_on
                        and s.hero_type != SENTINEL_HERO)
            return self.rush_on and s.hero_type in RUSHER_HEROES
        return self.rad_offense_on      # siege sighted: all five race

    def _lane(self, s: HeroObs) -> tuple[tuple[float, float], ...]:
        if self._is_rusher(s):
            # split across two corridors only when the squad is big
            # enough to body-block itself (siege modes, 4-5 heroes);
            # the passive-gate 3-hero rush keeps one route
            if s.team == SENTINEL_TEAM:
                alt = (self.dire_all_in
                       and s.hero_type in ALT_ROUTE_HEROES)
                return RUSH_ROUTE_ALT if alt else RUSH_ROUTE
            alt = s.hero_type in ALT_ROUTE_HEROES
            return RAD_RUSH_ROUTE_ALT if alt else RAD_RUSH_ROUTE
        return WAYPOINTS[LANE_FOR_HERO[s.hero_type] + 3 * s.team]

    def _nearest_wp(self, s: HeroObs, lane) -> int:
        return min(range(len(lane)),
                   key=lambda i: max(abs(lane[i][0] - s.y),
                                     abs(lane[i][1] - s.x)))

    def _nearest_wp_nav(self, s: HeroObs, lane) -> int:
        """Waypoint with the smallest WALKING distance (BFS field),
        not chebyshev: rushers re-anchor from arbitrary lane
        positions when the gate arms, and the straight-line-nearest
        waypoint can sit behind a jungle wall (league round 873: the
        top-lane tank anchored to a route point it could only reach
        through one neutral-camp-blocked pass and jammed for 1600
        ticks). Unreachable fields fall back to chebyshev."""
        best, best_d = 0, None
        for i, (wy, wx) in enumerate(lane):
            field = self.nav._field((int(wy), int(wx)))
            d = field[s.y * MAP_W + s.x] \
                if 0 <= s.y < MAP_W and 0 <= s.x < MAP_W \
                else NavGrid.UNREACHED
            if d == NavGrid.UNREACHED:
                d = 10000 + max(abs(wy - s.y), abs(wx - s.x))
            if best_d is None or d < best_d:
                best, best_d = i, d
        return best

    def _push_goal(self, s: HeroObs, st: HeroState) -> tuple[int, int]:
        lane = self._lane(s)
        if st.wp_index is None:
            if self._is_rusher(s):
                st.wp_index = self._nearest_wp_nav(s, lane)
            else:
                st.wp_index = self._nearest_wp(s, lane)
        while st.wp_index < len(lane):
            wy, wx = lane[st.wp_index]
            if max(abs(wy - s.y), abs(wx - s.x)) > WAYPOINT_REACHED:
                return (int(wy), int(wx))
            st.wp_index += 1
        if s.team == 1 and self.rush_on:
            return POKE[1]      # dire endgame: the safe poke cell
        if s.team == 0 and self.rad_offense_on:
            return POKE[0]      # radiant counter-backdoor poke cell
        ay, ax, _t, _tier = TOWERS[ANCIENT_IDX[1 - s.team]]
        return (int(ay), int(ax))

    # -- combat signals -----------------------------------------------------

    @staticmethod
    def _crop_scan(s: HeroObs):
        """(enemy hero dist, hostile creep dist, friendly creep cells)
        from the reliable crop tiles; dists are chebyshev or None."""
        heroes, creeps = hostile_tiles(s.team)
        friendly = friendly_creep_tile(s.team)
        hero_d = creep_d = None
        friendly_cells = []
        enemy_hero_cells = []
        for dy in range(-VIS, VIS + 1):
            yy = s.y + dy
            if not (0 <= yy < MAP_W):
                continue
            for dx in range(-VIS, VIS + 1):
                xx = s.x + dx
                if not (0 <= xx < MAP_W):
                    continue
                t = s.crop[(dy + VIS) * CROP_W + (dx + VIS)]
                if t == EMPTY or t == WALL:
                    continue
                d = max(abs(dy), abs(dx))
                if t in heroes:
                    enemy_hero_cells.append((yy, xx))
                    if hero_d is None or d < hero_d:
                        hero_d = d
                elif t in creeps:
                    if creep_d is None or d < creep_d:
                        creep_d = d
                elif t == friendly:
                    friendly_cells.append((yy, xx))
        return hero_d, creep_d, friendly_cells, enemy_hero_cells

    def _skill_flags(self, s: HeroObs, mode: str, hero_d, creep_d,
                     anc_in_scan: bool = False) -> tuple[int, int, int]:
        """(use_q, use_w, use_e) per role. The engine re-checks
        cooldown/mana/target and fails skills harmlessly, so these are
        intent flags, gated just enough to avoid wasting mana.

        ``anc_in_scan``: the ENEMY ancient is inside our attack scan
        and no hostile creeps distract the nearest-target selection.
        Damage skills all route through the generic attack path
        (moba.h:1151-1362) and hit tower-type entities, so a poking
        hero fires them at the ancient: the assassin W alone lands
        250+50/level per cast versus a ~150 basic attack — measured
        against richard's race tempo, basic-attack-only poking loses
        the burn race by ~5x."""
        enemy_hero = hero_d is not None
        hostile = enemy_hero or creep_d is not None
        h = s.hero_type
        if h == SUPPORT:      # hook / aoe heal / stun (moba.h:1151-1187)
            return (int(enemy_hero and s.mana_at_least(100)),
                    int(s.health10 <= 6 and s.mana_at_least(100)),
                    int((enemy_hero or anc_in_scan)
                        and s.mana_at_least(75)))
        if h == ASSASSIN:     # aoe minions / tp nuke / haste (1189-1233)
            return (int(creep_d is not None and s.mana_at_least(100)),
                    int(((enemy_hero and s.health10 >= 5)
                         or anc_in_scan) and s.mana_at_least(150)),
                    int(mode == RETREAT and s.mana_at_least(100)))
        if h == BURST:        # nuke / aoe / aoe stun (1235-1272)
            return (int((enemy_hero or anc_in_scan)
                        and s.mana_at_least(200)),
                    int((enemy_hero or anc_in_scan)
                        and s.mana_at_least(100)),
                    int(enemy_hero and s.mana_at_least(75)))
        if h == TANK:         # aoe dot / self heal / engage (1274-1313)
            low = s.health10 <= 5
            return (int(hostile and not low and s.mana_at_least(5)),
                    int(low and s.mana_at_least(100)),
                    int(enemy_hero and s.health10 > 6
                        and s.mana_at_least(50)))
        # CARRY: retreat slow / slow nuke / aoe (1315-1362)
        return (int(mode == RETREAT and hostile and s.mana_at_least(25)),
                int((enemy_hero or anc_in_scan) and s.mana_at_least(150)),
                int((creep_d is not None or anc_in_scan)
                    and s.mana_at_least(100)))

    # -- per-hero tick ------------------------------------------------------

    def _hero_action(self, idx: int, obs: bytes) -> list[int]:
        s = parse_obs(obs)
        st = self.heroes.setdefault(idx, HeroState())
        st.team = s.team
        self._update_tower_memory(s)

        # respawn/teleport detection resets waypoint tracking
        pos = (s.y, s.x)
        if (st.last_pos is not None
                and max(abs(pos[0] - st.last_pos[0]),
                        abs(pos[1] - st.last_pos[1])) > TELEPORT_DIST):
            st.wp_index = None
            st.stuck_ticks = 0
            st.detour_left = 0

        # mode with hysteresis. Rushers use a much tighter band:
        # death is an instant full-health teleport home (spawn_player,
        # moba.h:723-728), so a rusher walking home at low health only
        # loses race time — it fights on the march and re-anchors
        # after respawning, bailing out only at death's door.
        if self._is_rusher(s):
            enter, exit_ = RUSH_RETREAT_ENTER_HP, RUSH_RETREAT_EXIT_HP
        else:
            enter, exit_ = RETREAT_ENTER_HP, RETREAT_EXIT_HP
        if st.mode == PUSH and s.health10 <= enter:
            st.mode = RETREAT
        elif st.mode == RETREAT and s.health10 >= exit_:
            st.mode = PUSH
            st.wp_index = None      # re-anchor to the nearest waypoint

        hero_d, creep_d, friendly_creeps, enemy_heroes = self._crop_scan(s)

        # shared base-threat alarm: a GROUP of enemy heroes sighted on
        # our half re-arms the countdown. Single wanderers are ignored
        # (the base towers plus whoever is home handle them); reacting
        # to every trickler would turtle the whole team forever.
        oy, ox, _t, _tier = TOWERS[ANCIENT_IDX[s.team]]
        oy, ox = int(oy), int(ox)
        for hy, hx in enemy_heroes:
            d_anc = max(abs(hy - oy), abs(hx - ox))
            if d_anc <= ALARM_RADIUS:
                self._tick_sightings.add((hy, hx))
            if d_anc <= ALARM_NEAR:
                self.alarm_ticks = ALARM_TICKS
            # passivity-gate evidence (dire, pre-gate): an enemy
            # hero in the top conveyor corridor or on our ancient
            if (s.team == SENTINEL_TEAM and not self.rush_on
                    and self._tick <= RUSH_GATE_TICK
                    and ((hy <= NW_ZONE_Y and hx <= NW_ZONE_X)
                         or d_anc <= INTRUSION_RADIUS)):
                self.aggro_seen = True
            # radiant defense evidence: an enemy hero sighted at
            # our ancient (base siege underway) re-heats the defense
            if (s.team == SENTINEL_TEAM and d_anc <= RAD_INTRUSION
                    and not (self.dire_siege and self.dire_all_in)):
                self.dire_siege = True
                if not self.rush_on:
                    self.rush_on = True     # siege overrides the gate
                    for hst in self.heroes.values():
                        if hst.team == SENTINEL_TEAM:
                            hst.wp_index = None
                # siege flavor: a RING CAMPER pokes from outside our
                # towers' cover (nothing can ever touch it — racing
                # all-in is the only answer); a DIVER fights inside
                # tower fire, where the sentinel's body-stall still
                # earns its seat
                covered = any(
                    max(abs(int(ty) - hy), abs(int(tx) - hx))
                    <= TOWER_VISION
                    for idx, (ty, tx, tt, tier) in enumerate(TOWERS)
                    if tt == s.team and tier != 5
                    and idx not in self.dead_towers)
                if not covered:
                    # campers DWELL in uncovered ring cells; divers
                    # and wanderers merely transit them — declare
                    # all-in only on sustained presence
                    self._ring_flag = True
            if s.team == 0 and d_anc <= RAD_INTRUSION:
                self.rad_defense_hot = RAD_DEFENSE_HOT
                if not self.rad_offense_on:
                    self.rad_offense_on = True
                    for hst in self.heroes.values():
                        if hst.team == 0:
                            hst.wp_index = None  # re-anchor to route
        if len(self._tick_sightings) >= ALARM_GROUP:
            self.alarm_ticks = ALARM_TICKS

        # DEFEND overrides the push/retreat split for nearby heroes:
        # rally on the own ancient so base towers back the fight. The
        # dire sentinel garrisons the ancient permanently: the dire
        # base is the exposed one (map favors radiant dives) and a
        # scattered lane push detects a 5-hero dive only by luck.
        team_active = (self.rush_on if s.team == SENTINEL_TEAM
                       else self.rad_defense_hot > 0)
        # the sentinel garrisons only while its team is NOT all-in
        # racing: once a siege commits the race, detection has done
        # its job and a fifth poker beats a lone doomed defender
        # (one support plus towers was measured never to hold a
        # five-hero burn on either side)
        sentinel = ((team_active or (s.team == SENTINEL_TEAM
                                     and self.aggro_seen))
                    and s.hero_type == SENTINEL_HERO
                    and not (self.dire_all_in
                             if s.team == SENTINEL_TEAM
                             else self.rad_offense_on))
        defending = (sentinel and st.mode == PUSH) or (
            s.team == SENTINEL_TEAM  # radiant: sentinel-only response
            and team_active      # v1 play until the team gate trips
            and self.alarm_ticks > 0
            and not self._is_rusher(s)
            and s.level < BREAKOUT_LEVEL
            and max(abs(s.y - oy), abs(s.x - ox)) <= RECALL_RADIUS
            and (st.mode == PUSH
                 or s.health10 >= DEFEND_REJOIN_HP))

        # goal selection
        hold = False
        if defending:
            if st.mode == RETREAT and s.health10 >= DEFEND_REJOIN_HP:
                st.mode = PUSH
            st.wp_index = None      # re-anchor to the lane when alarm ends
            # the sentinel holds the north-entrance watchpost until a
            # dive is actually underway; defenders rally the ancient
            if sentinel and self.alarm_ticks == 0:
                goal = SENTINEL_POST[s.team]
            else:
                goal = (oy, ox)
        elif st.mode == RETREAT:
            goal = SPAWN[s.team]
        else:
            goal = self._push_goal(s, st)
            tower = self._nearest_live_enemy_tower(s)
            if (tower is not None and tower[1] <= SIEGE_STANDOFF
                    and (s.team == 0 or not self.rush_on
                         or s.level < BREAKOUT_LEVEL)
                    and not self._is_rusher(s)):
                tidx, tdist = tower
                ty, tx = int(TOWERS[tidx][0]), int(TOWERS[tidx][1])
                engaged = any(
                    max(abs(cy - ty), abs(cx - tx)) <= CREEP_ENGAGED_DIST
                    for cy, cx in friendly_creeps)
                if engaged:
                    goal = (ty, tx)          # push onto the tower
                elif tdist <= TOWER_VISION:
                    goal = SPAWN[s.team]     # inside range, no wave: out
                else:
                    hold = True              # at standoff: wait for creeps

        # steering
        if hold:
            dy = dx = 0
        else:
            dy, dx = self.nav.step_toward(
                s.y, s.x, goal, exact=goal in POKE)

        # stuck detection -> deterministic 90-degree detour. A tick
        # only counts as stuck if the PREVIOUS tick actually tried to
        # move and the position stayed put; coming out of an
        # intentional hold must not count as a phantom stuck tick.
        if st.tried_move and st.last_pos == pos and (dy or dx):
            st.stuck_ticks += 1
            st.blocked_ticks += 1
        else:
            st.stuck_ticks = 0
            if st.last_pos != pos:
                st.blocked_ticks = 0
        if st.detour_left > 0:
            st.detour_left -= 1
            dy, dx = _rotate90(dy, dx, st.detour_side)
        elif st.stuck_ticks >= STUCK_TICKS:
            st.detour_left = DETOUR_TICKS
            st.detour_side = -st.detour_side
            st.stuck_ticks = 0
            dy, dx = _rotate90(dy, dx, st.detour_side)
        if s.team == SENTINEL_TEAM or self.rad_offense_on:
            # Jam escape (dire, and radiant once the counter-backdoor
            # is committed). The engine's move_to fails when
            # the DESTINATION CELL after float truncation is a wall
            # (moba.h:561-566); a diagonal step next to a wall can
            # therefore fail forever for a hero whose sub-cell float
            # offset is unlucky — and only successful DIAGONAL moves
            # change that offset, so wiggling back and forth on the
            # orthogonals never unsticks it (league replays showed
            # heroes pinned that way for 2000+ ticks). The escape must
            # be monotonic: anchor the jam cell and sweep all 8 engine
            # step directions in fixed order, holding each for a few
            # ticks, ignoring incidental one-cell progress until the
            # hero gets genuinely clear of the anchor. Successful
            # off-axis diagonals reshuffle the float offset, after
            # which the normal descent works again.
            if st.jam_pos is None:
                if st.blocked_ticks >= BLOCKED_ESCALATE and (dy or dx):
                    st.jam_pos = pos
                    st.jam_ticks = 0
            elif max(abs(pos[0] - st.jam_pos[0]),
                     abs(pos[1] - st.jam_pos[1])) > 3:
                st.jam_pos = None       # genuinely clear: nav resumes
                st.blocked_ticks = 0
            if st.jam_pos is not None:
                if st.jam_ticks == JAM_SKIP_TICKS \
                        and st.wp_index is not None:
                    # a full sweep failed to clear the anchor: assume
                    # an entity-plugged choke and reroute to the next
                    # waypoint via a different descent field
                    st.wp_index += 1
                dy, dx = STEPS[(st.jam_ticks // DETOUR_TICKS) % 8]
                st.jam_ticks += 1
        st.tried_move = bool(dy or dx)
        st.last_pos = pos

        # target filter: everything while hostile creeps are around
        # (lane XP), heroes+towers otherwise (focus towers, don't
        # aggro neutral camps)
        target_filter = 0 if creep_d is not None else 2
        if st.jam_pos is not None:
            target_filter = 0   # jammed: attack entity plugs clear
        ey, ex, _et, _etr = TOWERS[ANCIENT_IDX[1 - s.team]]
        anc_in_scan = (creep_d is None
                       and max(abs(s.y - int(ey)),
                               abs(s.x - int(ex))) <= VIS)
        use_q, use_w, use_e = self._skill_flags(
            s, st.mode, hero_d, creep_d, anc_in_scan)

        return [3 + 3 * dy, 3 + 3 * dx, target_filter, use_q, use_w, use_e]

    def __call__(self, tick: int, obs_rows: list) -> list:
        if self.alarm_ticks > 0:
            self.alarm_ticks -= 1
        if self.rad_defense_hot > 0:
            self.rad_defense_hot -= 1
        if self._ring_flag:
            self._ring_flag = False
            self.ring_ticks += 1
            if self.ring_ticks >= RING_CAMP_TICKS:
                self.dire_all_in = True
        self._tick_sightings.clear()
        self._tick = tick
        if (not self.rush_on and not self.aggro_seen
                and tick >= RUSH_GATE_TICK):
            # passivity proven: arm the backdoor rush; dire waypoint
            # tracking re-anchors because lane assignments change
            # (radiant heroes are untouched: the gate is dire-only)
            self.rush_on = True
            for hst in self.heroes.values():
                if hst.team == SENTINEL_TEAM:
                    hst.wp_index = None
        return [self._hero_action(i, bytes(row))
                for i, row in enumerate(obs_rows)]


def policy_from_env() -> ScriptedPolicy:
    return ScriptedPolicy(seed_from_env(default=None))


def main() -> int:
    return run_policy_main(policy_from_env)


if __name__ == "__main__":
    sys.exit(main())
