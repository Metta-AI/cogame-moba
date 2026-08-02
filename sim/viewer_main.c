// cogame-moba replay viewer: re-simulates a recorded episode in the
// browser and renders it with the upstream raylib client.
//
// Built twice by sim/build_viewer.sh, always against build/src-patched:
//   - with -DMOBA_RENDER (raylib + emscripten main loop)
//       -> viewer/dist/moba_viewer.{js,wasm,data}      (browser bundle)
//   - without MOBA_RENDER (headless core, ENVIRONMENT=node, --no-entry)
//       -> build/viewer_core.{js,wasm}                 (node verification)
//
// The core API (viewer_load / viewer_seek / viewer_advance_frame / ...)
// is identical in both builds; only the raylib main loop is render-only.
// This is what lets tests/test_viewer.py prove the re-sim logic under
// node without pixels.
//
// Replay format v1 (server/cogame_moba/replay.py is the authority):
//   bytes 0-3  magic "MOBA"; byte 4 version u8 == 1;
//   bytes 5-8  header_len u32le; then header JSON (parsed JS-side);
//   then tick_count * 60 bytes (10 heroes x 6 uint8 actions, post-clamp).
// C never parses the JSON: tick_count == body_len / 60 by construction.
//
// Determinism: viewer_seek() rebuilds the sim through the exact fresh-
// instance path the server host uses (zeroed struct -> moba_configure ->
// allocate_moba -> c_reset, patch 0002 srand(seed) inside init_moba) and
// replays the recorded actions from tick 0. Replay values are stored
// post-clamp, so a direct uint8 -> float cast matches the server's
// set_actions byte-for-byte.

#include <stdlib.h>
#include <string.h>

#include "shim_common.h"  // moba_configure(): shared env defaults (+ moba.h)

#ifdef MOBA_RENDER
#include <emscripten.h>
#endif

#define VIEWER_MAGIC_LEN 9        // magic(4) + version u8 + header_len u32le
#define VIEWER_BYTES_PER_TICK 60  // 10 heroes x 6 uint8
#define VIEWER_NUM_AGENTS 10      // replays always drive all 10 heroes
#define VIEWER_FRAMES_PER_TICK 12 // upstream demo cadence: 1 sim tick per
                                  // 12 frames at 60 fps. Kept as the
                                  // interpolation-phase granularity and
                                  // the meaning of one "60Hz-equivalent
                                  // frame" (viewer_advance_frame).

// Nominal playback: 5 sim ticks/s at 1x (the upstream demo cadence),
// advanced by WALL TIME, not render-callback count — rAF callbacks fire
// per display refresh, so counting them plays 2x too fast on a 120 Hz
// display and hitches on every dropped frame.
#define VIEWER_TICK_MS (1000.0 * VIEWER_FRAMES_PER_TICK / 60.0)  // 200 ms
// Per-callback dt clamp: a backgrounded tab can sit for seconds between
// callbacks; do not burst-run that gap on return.
#define VIEWER_MAX_DT_MS 100.0

static MOBA env;
static int g_allocated = 0;

static const unsigned char* g_body = NULL;  // action log, inside JS-owned buf
static int g_total_ticks = 0;
static int g_tick = 0;         // ticks fed so far == current sim tick
static int g_playing = 0;
static int g_speed = 1;         // playback multiplier (5*speed ticks/s)
static double g_time_acc = 0.0; // speed-scaled ms toward the next tick
                                // (invariant: < VIEWER_TICK_MS between
                                // viewer_advance calls)
static unsigned int g_seed = 0;
static int g_loaded = 0;

// Interpolation phase for the render half, in units of
// VIEWER_FRAMES_PER_TICK: how far the display sits through the current
// [last, cur] entity-position interpolation window. 0..N-1 mid-sweep;
// == N means "render exactly at-tick" (tick_frac 1.0). Upstream's
// c_render interpolates from its OWN free-running renderer->frame
// counter, which assumes exactly one sim tick per 12 phase-locked
// render calls; seeks and speed changes break that assumption and
// cause per-tick lurching, so frame() overwrites the renderer counter
// from this value before every c_render call.
static int g_phase = 0;

// Rebuild the sim exactly like a fresh wasm instance's moba_init(): the
// static struct starts zeroed, moba_configure sets the trained-on
// defaults + seed, allocate_moba re-allocates everything (cold ai_paths
// cache included) and runs init_moba (srand(seed), CachedRNG refill),
// then c_reset spawns. The renderer client survives across rebuilds.
static void sim_fresh(void) {
    void* client = env.client;  // GameRenderer*, owned by the render loop
    if (g_allocated)
        free_allocated_moba(&env);
    memset(&env, 0, sizeof(env));
    env.client = client;
    moba_configure(&env, g_seed, VIEWER_NUM_AGENTS);
    allocate_moba(&env);
    c_reset(&env);
    g_allocated = 1;
    g_tick = 0;
}

static void feed_and_step(void) {
    // MultiDiscrete highs (exclusive) per action column; replays are
    // written post-clamp, but a hand-crafted body byte >= high would
    // index out of bounds inside the sim — clamp defensively.
    static const unsigned char act_max[6] = {6, 6, 2, 1, 1, 1};
    const unsigned char* a = g_body + (size_t)g_tick * VIEWER_BYTES_PER_TICK;
    for (int i = 0; i < VIEWER_BYTES_PER_TICK; i++) {
        unsigned char hi = act_max[i % 6];
        env.actions[i] = (float)(a[i] > hi ? hi : a[i]);
    }
    c_step(&env);
    g_tick++;
}

// Parse replay bytes at ptr/len (JS-owned wasm heap memory that must stay
// alive while loaded) and start a fresh sim at tick 0, paused. The header
// JSON is parsed JS-side; JS passes the seed from header.config.seed.
// Returns total tick count, or -1 on malformed bytes.
int viewer_load(const unsigned char* data, int len, unsigned int seed) {
    if (data == NULL || len < VIEWER_MAGIC_LEN)
        return -1;
    if (memcmp(data, "MOBA", 4) != 0 || data[4] != 1)
        return -1;
    unsigned int header_len = (unsigned int)data[5]
        | ((unsigned int)data[6] << 8)
        | ((unsigned int)data[7] << 16)
        | ((unsigned int)data[8] << 24);
    // Non-wrappable on wasm32: len >= VIEWER_MAGIC_LEN is established
    // above, so the subtraction is safe; adding to header_len is not.
    if (header_len > (unsigned int)(len - VIEWER_MAGIC_LEN))
        return -1;
    size_t body_len = (size_t)len - VIEWER_MAGIC_LEN - header_len;
    if (body_len % VIEWER_BYTES_PER_TICK != 0)
        return -1;

    g_body = data + VIEWER_MAGIC_LEN + header_len;
    g_total_ticks = (int)(body_len / VIEWER_BYTES_PER_TICK);
    g_seed = seed;
    g_playing = 0;
    g_time_acc = 0.0;
    g_phase = VIEWER_FRAMES_PER_TICK;  // display tick 0 exactly
    sim_fresh();
    g_loaded = 1;
    return g_total_ticks;
}

// Re-sim from tick 0 to `tick` (clamped to 0..total), no rendering.
void viewer_seek(int tick) {
    if (!g_loaded)
        return;
    if (tick < 0) tick = 0;
    if (tick > g_total_ticks) tick = g_total_ticks;
    sim_fresh();
    while (g_tick < tick)
        feed_and_step();
    g_time_acc = 0.0;
    g_phase = VIEWER_FRAMES_PER_TICK;  // display the seek target exactly
    if (g_tick >= g_total_ticks)
        g_playing = 0;  // seek-to-end lands in the "ended" state
}

// Advance the simulation by dt_ms of wall time (clamped to
// VIEWER_MAX_DT_MS). At speed s, one sim tick per VIEWER_TICK_MS/s of
// wall time — 5*s ticks/s regardless of display refresh rate. Pauses at
// end of replay instead of looping. Returns sim ticks stepped.
int viewer_advance(double dt_ms) {
    if (!g_loaded || !g_playing)
        return 0;
    if (dt_ms < 0.0)
        dt_ms = 0.0;
    if (dt_ms > VIEWER_MAX_DT_MS)
        dt_ms = VIEWER_MAX_DT_MS;  // tab-switch gap: no burst on return
    int stepped = 0;
    g_time_acc += dt_ms * (double)g_speed;
    // Epsilon: accumulated 60Hz dts round to 199.99999999999997 over 12
    // frames; a sub-nanosecond shortfall must not defer the tick a
    // whole callback.
    while (g_time_acc >= VIEWER_TICK_MS - 1e-6) {
        g_time_acc -= VIEWER_TICK_MS;
        if (g_time_acc < 0.0)
            g_time_acc = 0.0;
        if (g_tick >= g_total_ticks) {
            g_playing = 0;   // ended: JS sees playing==0 && tick==total
            g_time_acc = 0.0;
            break;
        }
        feed_and_step();
        stepped++;
        if (g_tick >= g_total_ticks) {
            g_playing = 0;
            g_time_acc = 0.0;
            break;
        }
    }
    if (g_tick >= g_total_ticks || stepped > 1) {
        // Ended (show the final state exactly), or several ticks in one
        // callback: interpolating the last tick interval is
        // meaningless — render at-tick.
        g_phase = VIEWER_FRAMES_PER_TICK;
    } else if (stepped == 1 || g_phase != VIEWER_FRAMES_PER_TICK) {
        // Fresh interpolation window (a tick just stepped) or mid-sweep:
        // g_time_acc is wall-time progress toward the next tick, which
        // is exactly the progress through the current [last, cur]
        // window; quantize it to the renderer's 12ths. From an at-tick
        // display with no new tick (else-branch not taken) the phase
        // holds at-tick — sweeping backwards would lurch.
        g_phase = (int)(g_time_acc * VIEWER_FRAMES_PER_TICK / VIEWER_TICK_MS);
        if (g_phase < 0)
            g_phase = 0;
        if (g_phase >= VIEWER_FRAMES_PER_TICK)
            g_phase = VIEWER_FRAMES_PER_TICK - 1;
    }
    return stepped;
}

// One 60Hz-equivalent frame (the pre-time-based API, kept stable for
// the node harness): 12 calls == one tick at 1x.
int viewer_advance_frame(void) {
    return viewer_advance(1000.0 / 60.0);
}

// Current interpolation phase (see g_phase). Exported for the node
// harness so the phase-lock behavior is testable without pixels.
int viewer_render_phase(void) { return g_phase; }

int viewer_tick(void) { return g_tick; }

int viewer_total_ticks(void) { return g_total_ticks; }

void viewer_set_speed(int speed) {
    if (speed >= 1 && speed <= 1024)
        g_speed = speed;
}

int viewer_get_speed(void) { return g_speed; }

void viewer_set_playing(int playing) {
    if (!g_loaded)
        return;
    if (playing && g_tick >= g_total_ticks)
        return;  // ended: JS must seek first (no silent loop)
    g_playing = playing ? 1 : 0;
    // g_time_acc is deliberately kept: it is always < VIEWER_TICK_MS
    // here (the advance loop reduces it), so no tick burst is possible,
    // and preserving it resumes the interpolation sweep exactly where
    // the pause froze it (a reset would lurch the display backwards).
}

int viewer_playing(void) { return g_playing; }

// Patch-0003 episode state, for end-of-replay display and the headless
// verification (winner must match the replay header's result).
int viewer_done(void) { return g_allocated ? env.done : 0; }

int viewer_winner(void) { return g_allocated ? env.winner : 0; }

// Final-state digest at the current tick (see sim/shim_common.h). Must
// equal the recording host's state_digest() at the same tick — the
// headless verification (tests/test_viewer.py) and replay certification
// rely on this.
unsigned int viewer_state_digest(void) {
    return g_allocated ? moba_state_digest(&env) : 0;
}

#ifdef MOBA_RENDER
// Render loop: one callback per browser animation frame. Sim cadence is
// handled by viewer_advance_frame; c_render (upstream, unchanged) lazily
// creates its GameRenderer/window on first call and interpolates entity
// positions between sim ticks.
static void frame(void) {
    if (!g_loaded)
        return;  // window/canvas appear on first frame after load
    // Advance by measured wall time (emscripten_get_now, ms): rAF fires
    // per display refresh, so callback-counting would play 2x too fast
    // on 120 Hz displays and hitch on dropped frames. The dt clamp in
    // viewer_advance absorbs tab-switch gaps; last_now keeps updating
    // even while paused so resume sees a normal dt.
    static double last_now = -1.0;
    double now = emscripten_get_now();
    double dt_ms = (last_now < 0.0) ? 0.0 : now - last_now;
    last_now = now;
    viewer_advance(dt_ms);
    if (env.client != NULL) {
        // Phase-lock upstream's interpolation counter to the true
        // inter-tick progress (see g_phase): c_render reads
        // renderer->frame for tick_frac and assumes 1 sim tick per 12
        // phase-aligned render calls, which seeks/speeds/120Hz displays
        // all violate. renderer->frame's other uses are harmless here:
        // the HUMAN_CONTROL action-clear is replay-irrelevant (every
        // tick's actions come from the log), and the end-of-render
        // increment/wrap is superseded by this per-frame overwrite.
        // VIEWER_FRAMES_PER_TICK (== upstream FRAMES) yields
        // tick_frac 1.0: render exactly at-tick.
        GameRenderer* renderer = env.client;
        renderer->frame = viewer_render_phase();
    }
    c_render(&env);
}

int main(void) {
    // 0 fps == requestAnimationFrame; don't simulate an infinite loop —
    // main returns and the runtime stays alive (EXIT_RUNTIME=0) for the
    // viewer_* exports.
    emscripten_set_main_loop(frame, 0, 0);
    return 0;
}
#endif  // MOBA_RENDER
