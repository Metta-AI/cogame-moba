// Shared MOBA env configuration used by every host shim (sim/shim.c and
// sim/viewer_main.c). One definition so the viewer's re-simulation can
// never drift from the server sim's init path.
//
// Values are the trained-on env defaults, copied verbatim from upstream
// config/moba.ini [env] (fed through ocean/moba/binding.c my_vec_init).
// Do not change: policies were trained on these.
#ifndef COGAME_SHIM_COMMON_H
#define COGAME_SHIM_COMMON_H

#include "moba.h"  // vendored sim; includes game_map.h (game_map_npy)

static inline void moba_configure(MOBA* env, unsigned int seed,
                                  int num_agents) {
    env->num_agents = num_agents;             // 10 (all heroes seat-controlled)
    env->script_opponents = (num_agents == 5); // binding.c: agents_per_env = script_opponents ? 5 : 10
    env->vision_range = 5;                    // moba.ini [env] vision_range
    env->agent_speed = 1.0f;                  // moba.ini [env] agent_speed
    env->reward_death = -0.163764f;           // moba.ini [env] reward_death
    env->reward_xp = 0.00665677f;             // moba.ini [env] reward_xp
    env->reward_distance = 0.0f;              // moba.ini [env] reward_distance
    env->reward_tower = 0.642119f;            // moba.ini [env] reward_tower
#ifndef PRISTINE
    env->seed = seed;                         // patch 0002: init_moba() -> srand(seed)
#else
    (void)seed;                               // pristine build: libc default stream (== seed 1)
#endif
}

#endif  // COGAME_SHIM_COMMON_H
