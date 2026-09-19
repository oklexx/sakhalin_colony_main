// Standalone harness for the unified curriculum contract (PR 1; no Python, no raylib).
//
// Contract: Python computes the allowed set, C++ stores and applies it.
// "No restriction" is an explicit all_builds=true — there is no way to express
// "restricted" that silently degrades to "everything allowed" (the old
// set_curriculum_stage() used to wipe the manual set; the old empty manual
// list at stage 0 meant "all 32" by accident).
//
// Build (from the repo root, Linux/macOS):
//   g++ -std=c++17 -O1 -Iinclude -Iinclude/third_party -o /tmp/check \
//       tests/cpp/curriculum_check.cpp \
//       src/env.cpp src/data.cpp src/resources.cpp src/game.cpp \
//       src/rewards.cpp src/rng.cpp src/earth.cpp
//   /tmp/check          # run from the repo root: configs/*.json are loaded
// Build (Windows, MSVC):
//   cl /std:c++17 /EHsc /Iinclude /Iinclude\third_party tests\cpp\curriculum_check.cpp ^
//      src\env.cpp src\data.cpp src\resources.cpp src\game.cpp src\rewards.cpp ^
//      src\rng.cpp src\earth.cpp /Fe:check.exe
#include "colony/env.h"
#include "colony/data.h"
#include <algorithm>
#include <cstdio>
#include <string>
#include <unordered_set>
#include <vector>

using namespace colony;

static int failures = 0;
static void check(bool ok, const std::string& what) {
    printf("%s  %s\n", ok ? "[ok]  " : "[FAIL]", what.c_str());
    if (!ok) failures++;
}

static std::vector<int> allowed_build_actions(ColonyEnvCpp& env) {
    std::vector<int> out;
    auto mask = env.action_mask();
    for (int i = 0; i < env.n_build(); i++)
        if (mask[A_BUILD0 + i] != 0.0f) out.push_back(i);
    return out;
}

static std::vector<std::string> allowed_build_ids(ColonyEnvCpp& env) {
    std::vector<std::string> out;
    for (int i : allowed_build_actions(env)) out.push_back(env.build_ids()[i]);
    return out;
}

static bool allowed(ColonyEnvCpp& env, int idx) {
    return idx >= 0 && env.action_mask()[A_BUILD0 + idx] != 0.0f;
}

static int find_id(ColonyEnvCpp& env, const std::string& id) {
    for (int i = 0; i < env.n_build(); i++)
        if (env.build_ids()[i] == id) return i;
    return -1;
}

static Curriculum res_weights(std::vector<double> w) {
    Curriculum c;  // all_builds=true default: buildings open, resources weighted
    c.all_resources = false;
    for (size_t j = 0; j < w.size() && j < 9; j++) c.resource_weights[j] = w[j];
    return c;
}

static std::vector<double> only(int idx) {
    std::vector<double> w(9, 0.0);
    w[idx] = 1.0;
    return w;
}

// Build `build1` on day 1 (or nothing), then 250 DAYs into autumn; returns
// (total reward, last step). Twins run the identical script — rewards differ
// only through the curriculum weights.
static std::pair<double, ColonyEnvCpp::StepOut> run_autumn(
    const std::vector<BaseData>& bd, const std::vector<BaseEvent>& ed,
    const Curriculum& cur, const RewardConfig& rc, const char* build1) {
    ColonyEnvCpp e(bd, ed, 21, 280);
    e.reset(21);
    e.game().money = 5'000'000;
    e.set_rewards(rc);
    e.set_curriculum(cur);
    if (build1) e.step(A_BUILD0 + find_id(e, build1));
    double ep = 0.0;
    ColonyEnvCpp::StepOut last;
    for (int d = 0; d < 250; d++) { last = e.step(0); ep += last.rew; }
    return {ep, last};
}

static Curriculum restricted(std::vector<std::string> ids, int stage_report = 0) {
    Curriculum c;
    c.all_builds = false;
    c.allowed_builds = std::unordered_set<std::string>(ids.begin(), ids.end());
    c.stage_report = stage_report;
    return c;
}

int main() {
    auto bd = load_base_data("configs/bases.json");
    auto ed = load_events("configs/events.json");

    // ── baseline: what this map allows without any curriculum restriction ──
    // 5M cash removes affordability from the equation, so the mask only reflects
    // curriculum + terrain/lot rules.
    ColonyEnvCpp a(bd, ed, 42, 280);  // default Curriculum(): all_builds=true
    a.reset(42);
    a.game().money = 5'000'000;
    std::vector<int> base_allowed = allowed_build_actions(a);
    printf("unrestricted env (5M cash): %d/%d build actions available\n",
           (int)base_allowed.size(), a.n_build());
    check(base_allowed.size() >= 2, "map offers >=2 buildable types");
    check(a.curriculum().all_builds, "default-constructed env is unrestricted");

    int first = base_allowed.front();
    const std::string first_id = a.build_ids()[first];
    printf("demonstrating with a placeable id: '%s'\n", first_id.c_str());

    // ── scenario from the report: one id selected ──
    ColonyEnvCpp b(bd, ed, 42, 280);
    b.reset(42);
    b.game().money = 5'000'000;                 // same rich colony as the baseline
    b.set_curriculum(restricted({first_id}));
    auto ids_b = allowed_build_ids(b);
    check(ids_b.size() == 1 && ids_b[0] == first_id,
          "restricted env allows exactly the selected id (no other building)");
    check(!allowed(b, find_id(b, "Goldmine")), "прииск (Goldmine) is masked");
    check(b.build_allowed(first_id) && !b.build_allowed("Goldmine"),
          "build_allowed() agrees with the mask");

    // ── curriculum() round-trips exactly what was set ──
    Curriculum got = b.curriculum();
    check(!got.all_builds && got.allowed_builds.size() == 1 &&
              got.allowed_builds.count(first_id) == 1,
          "curriculum() returns exactly the applied state");
    check(got.stage_report == 0, "stage_report round-trips (0)");

    // ── set_curriculum is idempotent ──
    b.set_curriculum(restricted({first_id}));
    check(allowed_build_ids(b) == ids_b, "re-applying the same state changes nothing");

    // ── the old 9d1073e regression shape: applying a state that carries a
    // stage report must not lose the allowed set ──
    b.set_curriculum(restricted({first_id}, /*stage_report=*/2));
    auto ids_after = allowed_build_ids(b);
    check(ids_after == ids_b, "state with stage_report keeps the allowed set");
    check(b.curriculum().stage_report == 2, "stage_report stored");
    check(b.dump_obs().find("stage=2") != std::string::npos,
          "dump_obs reports the stage");

    // ── superset, then shrink back ──
    if (base_allowed.size() >= 2) {
        const std::string second_id = a.build_ids()[base_allowed[1]];
        b.set_curriculum(restricted({first_id, second_id}));
        check(allowed(b, find_id(b, first_id)) && allowed(b, find_id(b, second_id)),
              "two-id set allows both");
        b.set_curriculum(restricted({first_id}));
        auto ids_back = allowed_build_ids(b);
        check(ids_back.size() == 1 && ids_back[0] == first_id,
              "shrinking back restores the exact one-id set");
    }

    // ── JSON transport (watch_champion -> GUI exe path) ──
    Curriculum j = Curriculum::from_json(
        "{\"all_builds\": false, \"allowed_builds\": [\"WaterChannel\"], \"stage\": 0}");
    check(!j.all_builds && j.allowed_builds.size() == 1 &&
              j.allowed_builds.count("WaterChannel") == 1,
          "from_json parses the transport payload");
    ColonyEnvCpp d(bd, ed, 7, 280, j);
    d.reset(7);
    d.game().money = 5'000'000;
    int wc = find_id(d, "WaterChannel");
    int gold_d = find_id(d, "Goldmine");
    if (wc >= 0 && gold_d >= 0) {
        // WaterChannel may itself be unplaceable on this map (terrain) — the
        // curriculum part is that Goldmine is masked either way.
        check(!allowed(d, gold_d), "JSON-restricted env masks Goldmine");
        (void)wc;
    }
    bool threw = false;
    try {
        Curriculum::from_json("{oops");
    } catch (const std::exception&) {
        threw = true;
    }
    check(threw, "from_json rejects malformed JSON");

    // ── a masked build action never actually builds anything ──
    ColonyEnvCpp c(bd, ed, 7, 280);
    c.reset(7);
    c.set_curriculum(restricted({first_id}));
    int gold = find_id(c, "Goldmine");
    if (gold >= 0) {
        int before = (int)c.game().bases.size();
        for (int i = 0; i < 60; i++) c.step(A_BUILD0 + gold);
        check((int)c.game().bases.size() == before,
              "locked Goldmine action is refused by the env (nothing built)");
    }

    // ── PR 2: hard gate at the Game level ──
    // Direct g.build() on a gated env refuses a locked id with the gate message.
    ColonyEnvCpp e(bd, ed, 7, 280);
    e.reset(7);
    // Разрешены WaterChannel и Road: избирательность гейта проверяем на Road
    // (проходит и гейт, и валидатор размещения на сухопутной клетке), а
    // WaterChannel на суше после централизации валидации размещения
    // законно отвергается уже НЕ гейтом (нужна вода, need_earth=2).
    e.set_curriculum(restricted({"WaterChannel", "Road"}));
    {
        Game& g = e.game();
        int n0 = (int)g.bases.size();  // 1: the Depot
        int bx = g.bases[0].x, by = g.bases[0].y;
        const int dx[4] = {1, -1, 0, 0}, dy[4] = {0, 0, 1, -1};
        int fx = -1, fy = -1, gx = -1, gy = -1;
        for (int k = 0; k < 4; k++) {
            int nx = bx + dx[k], ny = by + dy[k];
            if (!g.base_in_box(nx, ny)) {
                if (fx < 0) { fx = nx; fy = ny; } else { gx = nx; gy = ny; break; }
            }
        }
        check(fx >= 0, "free neighbour cell next to the Depot");
        g.money = 0;  // broke AND locked: the gate message must win (gate is first)
        auto r = g.build("House", fx, fy);
        check(!r.first && r.second == "Постройка закрыта курикулумом.",
              "direct g.build(House) refused by the gate, before the money check");
        check((int)g.bases.size() == n0, "refused direct build adds no base");
        // …while an allowed id goes through (selective, not a blanket refuse)
        g.money = 5'000'000;
        auto r2 = g.build("Road", fx, fy);
        check(r2.first, "direct g.build(Road) passes the gate and placement");
        check((int)g.bases.size() == n0 + 1, "allowed direct build adds a base");
        // Разрешённому id по-прежнему действует единый валидатор размещения:
        // WaterChannel на суше отклоняет размещение, а не курикулум.
        if (gx >= 0) {
            auto r3 = g.build("WaterChannel", gx, gy);
            check(!r3.first && r3.second.find("курикулум") == std::string::npos,
                  "allowed WaterChannel on land is refused by placement, not by the gate (msg: " + r3.second + ")");
        }
    }

    // reset() recreates the Game — the gate must survive it (re-seated)
    e.reset(99);
    {
        Game& g = e.game();
        int bx = g.bases[0].x, by = g.bases[0].y;
        int fx = bx + 1, fy = by;
        if (g.base_in_box(fx, fy)) { fx = bx - 1; }
        auto r = g.build("House", fx, fy);
        check(!r.first && r.second == "Постройка закрыта курикулумом.",
              "gate still enforced after reset()");
    }

    // opening the curriculum re-opens direct builds too (one gate, one state)
    {
        Curriculum all;
        all.all_builds = true;
        e.set_curriculum(all);
        Game& g = e.game();
        g.money = 5'000'000;
        int bx = g.bases[0].x, by = g.bases[0].y;
        int fx = bx + 1, fy = by;
        if (g.base_in_box(fx, fy)) { fx = bx - 1; }
        int n0 = (int)g.bases.size();
        auto r = g.build("House", fx, fy);
        check(r.first && (int)g.bases.size() == n0 + 1,
              "unrestricted state lets direct g.build(House) through");
    }

    // a raw Game without a gate builds anything (sandbox untouched)
    {
        Game sandbox(bd, ed, 7, 280);
        int bx = sandbox.bases[0].x, by = sandbox.bases[0].y;
        int fx = bx + 1, fy = by;
        if (sandbox.base_in_box(fx, fy)) { fx = bx - 1; }
        sandbox.money = 5'000'000;
        int n0 = (int)sandbox.bases.size();
        auto r = sandbox.build("House", fx, fy);
        check(r.first && (int)sandbox.bases.size() == n0 + 1,
              "sandbox Game (no gate) builds House: bases grow");
    }

    // ── PR 6 §1: a gated step is a pure refusal — no day passes ──
    // Twin envs, identical states: the gated one refuses with exactly
    // error_penalty; the open one fails on money but the day still passes
    // (daily bonuses accrue). The PR 2 reward equality is intentionally gone.
    {
        ColonyEnvCpp t1(bd, ed, 11, 280), t2(bd, ed, 11, 280);
        t1.reset(11);
        t2.reset(11);
        t1.set_curriculum(restricted({"WaterChannel"}));
        t1.game().money = 0;
        t2.game().money = 0;
        int64_t d0 = t1.game().days_alive;
        double ep = t1.reward_config().error_penalty;
        int house = find_id(t1, "House");
        auto pre = t1.obs();
        auto o1 = t1.step(A_BUILD0 + house);  // gated: pure refusal
        auto o2 = t2.step(A_BUILD0 + house);  // money failure: day passes
        check(o1.rew == ep, "gated step reward is exactly error_penalty");
        check(o1.days == d0, "gated step advances no days");
        check(o1.obs == pre, "gated step returns the current obs unchanged");
        check(o2.days == d0 + 1, "ordinary failed build still passes the day");
        check(o1.rew != o2.rew, "gated and money-fail rewards differ by design");
        check(o1.n_bases == 1 && o2.n_bases == 1,
              "neither twin builds anything");
        check(o1.ep_return == ep, "gated ep_return tracks the pure penalty");
    }

    // ── PR 6 §3: degenerate-scenario report ──
    {
        // broke + WaterChannel-only: degenerate by construction (money reason)
        ColonyEnvCpp w(bd, ed, 7, 280, restricted({"WaterChannel"}));
        w.reset(7);
        w.game().money = 0;
        std::string rep = w.degenerate_report();
        check(!rep.empty() && rep.find("WaterChannel") != std::string::npos &&
                  rep.find("price") != std::string::npos,
              "degenerate report names the building and the money reason");
        // healthy control: rich and open
        ColonyEnvCpp h(bd, ed, 42, 280);
        h.reset(42);
        h.game().money = 5'000'000;
        check(h.degenerate_report().empty(), "healthy scenario reports nothing");
        // empty allowed set with all_builds=false
        ColonyEnvCpp z(bd, ed, 42, 280, restricted({}));
        z.reset(42);
        check(z.degenerate_report().find("набор пуст") != std::string::npos,
              "empty allowed set reported");
        // the measured case (seed 7, fresh money, WaterChannel-only): degenerate
        // with the lot reason (deterministic: the map is generated from the seed)
        ColonyEnvCpp w7(bd, ed, 7, 280, restricted({"WaterChannel"}));
        w7.reset(7);
        std::string rep7 = w7.degenerate_report();
        check(!rep7.empty(), "seed-7 WaterChannel-only is degenerate (0/32)");
        check(rep7.find("WaterChannel") != std::string::npos &&
                  rep7.find("лота") != std::string::npos,
              "seed-7 WaterChannel-only: lot reason reported");
    }

    // ── PR 4 §5.6: soft resource weights ──
    // NOTE (plan-vs-reality): the plan's water/gold twins are economy-blocked,
    // not weight-blocked — WaterChannel/WaterMill need a connected water lot
    // (unplaceable on fresh maps; seeds 1..30 probed) and Goldmine needs water
    // 200 + 30 workers. The mechanism is resource-agnostic, so the twins use
    // AirStation/energy (placeable everywhere, producing from day 184).
    // Food would NOT work either: zero building-consumers ⇒ w_food = 0 always.
    {
        RewardConfig iso;  // isolate first_extraction from the other terms
        iso.need_fill_bonus = 0.0;
        iso.extraction_daily = 0.0;
        double feb = iso.first_extraction_bonus;
        // manual w_energy from bases.json (documented formula, no C++ reuse):
        int ncons = 0;
        for (const auto& d : bd) if (d.consume[ENERGY] > 0) ncons++;
        double w_energy = std::min(1.0, 0.25 * (double)ncons);
        printf("manual w_energy: n_consumers=%d w=%.2f feb=%.1f\n", ncons, w_energy, feb);

        Curriculum c_all;  // all_builds + all_resources (legacy behaviour)
        c_all.all_builds = true;
        auto [ep_all, last_all] = run_autumn(bd, ed, c_all, iso, "AirStation");
        auto [ep_nrg, last_nrg] = run_autumn(bd, ed, res_weights(only(ENERGY)), iso, "AirStation");
        auto [ep_wo, last_wo] = run_autumn(bd, ed, res_weights(only(WATER)), iso, "AirStation");
        check(ep_nrg == ep_all,
              "energy-only twin keeps every bonus (priority intact, bit-identical)");
        double delta = ep_all - ep_wo;
        check(delta > 0.0, "water-only twin loses the energy bonus (strictly less)");
        check(std::abs(delta - feb * w_energy) < 1e-9,
              "lost bonus equals first_extraction*w_energy (manual calc)");
        check(last_all.metrics.reached_resources > 0,
              "sanity: energy was actually produced");
        check(last_wo.metrics.reached_resources > 0,
              "tracking: energy reached even at weight 0");
        check(last_wo.metrics.priority_reached == 0,
              "priority: nothing reached at weight 0");
        check(last_all.metrics.priority_reached == last_all.metrics.reached_resources,
              "all-1: every reached resource is priority");
        (void)last_nrg;

        // depot-only control: with no production, weights change nothing
        auto [ep_d_all, last_d_all] = run_autumn(bd, ed, c_all, iso, nullptr);
        auto [ep_d_wo, last_d_wo] = run_autumn(bd, ed, res_weights(only(WATER)), iso, nullptr);
        check(ep_d_all == ep_d_wo, "depot-only twins identical (no production)");
        (void)last_d_all; (void)last_d_wo;

        // flag on/off with nothing locked: identical (default changes nothing)
        {
            ColonyEnvCpp f1(bd, ed, 21, 280), f2(bd, ed, 21, 280);
            f1.reset(21); f2.reset(21);
            f1.game().money = 5'000'000; f2.game().money = 5'000'000;
            RewardConfig rcf = f1.reward_config();
            rcf.priority_count_over_allowed = true;
            f2.set_rewards(rcf);
            int g = find_id(f1, "Goldmine");
            auto o1 = f1.step(A_BUILD0 + g);
            auto o2 = f2.step(A_BUILD0 + g);
            check(o1.rew == o2.rew, "flag on/off identical when nothing is locked");
        }
        // flag on + allowed={AirStation}: locked consumers stop inflating
        // weights — the station consumes nothing, so energy weight drops to 0
        {
            Curriculum cf;
            cf.all_builds = false;
            cf.allowed_builds = {"AirStation"};
            auto runf = [&](bool flag) {
                ColonyEnvCpp e(bd, ed, 21, 280);
                e.reset(21);
                e.game().money = 5'000'000;
                RewardConfig rc = iso;
                rc.priority_count_over_allowed = flag;
                e.set_rewards(rc);
                e.set_curriculum(cf);
                e.step(A_BUILD0 + find_id(e, "AirStation"));
                double ep = 0.0;
                for (int d = 0; d < 250; d++) ep += e.step(0).rew;
                return ep;
            };
            double ep_off = runf(false), ep_on = runf(true);
            check(ep_off > ep_on, "flag on: energy bonus gone (no allowed consumers)");
            check(std::abs((ep_off - ep_on) - feb * w_energy) < 1e-9,
                  "flag delta matches the manual calc");
        }
    }

    // ── PR 4 transport + observability ──
    {
        Curriculum fj = Curriculum::from_json(
            "{\"all_builds\": true, \"allowed_builds\": [], \"stage\": 0,"
            " \"all_resources\": false, \"resource_weights\": [0,0,0,0,0,0,1,0,0]}");
        check(!fj.all_resources && fj.resource_weights[6] == 1.0 &&
                  fj.resource_weights[0] == 0.0,
              "from_json parses resource weights");
        bool threw_w = false;
        try {
            Curriculum::from_json("{\"resource_weights\": [1,1,1]}");
        } catch (const std::exception&) {
            threw_w = true;
        }
        check(threw_w, "from_json rejects short resource_weights");
        check(std::string(Sunduk::resource_name(0)) == "gold" &&
                  std::string(Sunduk::resource_name(6)) == "water" &&
                  std::string(Sunduk::resource_name(99)) == "?",
              "resource_name canonical (gold/water/?)");

        ColonyEnvCpp p1(bd, ed, 21, 280);
        p1.reset(21);
        check(p1.dump_obs().find("pr=all") != std::string::npos,
              "dump_obs shows pr=all by default");
        p1.set_curriculum(res_weights(only(6)));
        check(p1.dump_obs().find("pr=water") != std::string::npos,
              "dump_obs shows pr=water");
    }

    // ── PR 5: obs v1 — the «frame» (resource weights + build_allowed bits) ──
    {
        // Секция проверяет РАСКЛАДКИ v0/v1 — версии заданы явно. Дефолт
        // Curriculum — obs v2 (канонический, 299), см. include/colony/env.h.
        Curriculum c0;
        c0.obs_version = 0;
        ColonyEnvCpp e0(bd, ed, 7, 280, c0);
        e0.reset(7);
        check(e0.obs_size() == 248, "v0 obs_size is 248");
        check((int)e0.obs().size() == 248, "v0 obs() has 248 floats");

        Curriculum c1;
        c1.obs_version = 1;
        ColonyEnvCpp e1(bd, ed, 7, 280, c1);
        e1.reset(7);
        check(e1.obs_size() == 289, "v1 obs_size is 289");
        auto o1 = e1.obs();
        check((int)o1.size() == 289, "v1 obs() has 289 floats");

        // v0 is a strict prefix of v1 (same seed, no steps yet).
        auto o0 = e0.obs();
        bool prefix = true;
        for (int i = 0; i < 248; i++)
            if (o0[i] != o1[i]) { prefix = false; break; }
        check(prefix, "v0 obs is a strict prefix of v1 obs");

        // Unrestricted tail: nine 1.0 weights + 32 1.0 bits.
        bool tail = true;
        for (int i = 248; i < 289; i++)
            if (o1[i] != 1.0f) { tail = false; break; }
        check(tail, "v1 unrestricted tail is all ones (9 weights + 32 bits)");

        // ── P0 (2026-09-17): obs v2 — dx/dy к ближайшим ресурсам (не вода) ──
        Curriculum c2;
        c2.obs_version = 2;
        ColonyEnvCpp e2(bd, ed, 7, 280, c2);
        e2.reset(7);
        check(e2.obs_size() == 299, "v2 obs_size is 299");
        auto o2 = e2.obs();
        check((int)o2.size() == 299, "v2 obs() has 299 floats");
        bool pre2 = true;
        for (int i = 0; i < 289; i++)
            if (o1[i] != o2[i]) { pre2 = false; break; }
        check(pre2, "v1 obs is a strict prefix of v2 obs");
        {
            const Game& gg = e2.game();
            const int ms = gg.map_size();
            const int bx = gg.earth.init_sel_x, by = gg.earth.init_sel_y;
            bool dirs2 = true;
            int found_slots = 0;
            for (int k = 0; k < N_NEAREST_LOTS; k++) {
                bool found = false;
                int bestx = -1, besty = -1;
                double best = 0.0;
                for (int y = 0; y < ms; y++)
                    for (int x = 0; x < ms; x++) {
                        if (gg.earth.lots()[(size_t)y * ms + x] != NEAREST_LOT_TYPES[k]) continue;
                        double dx = (double)(x - bx), dy = (double)(y - by);
                        double d2 = dx * dx + dy * dy;
                        if (!found || d2 < best) { found = true; bestx = x; besty = y; best = d2; }
                    }
                float wx = found ? (float)((double)(bestx - bx) / (double)ms) : 0.0f;
                float wy = found ? (float)((double)(besty - by) / (double)ms) : 0.0f;
                if (found) found_slots++;
                if (o2[(size_t)(289 + 2 * k)] != wx ||
                    o2[(size_t)(289 + 2 * k + 1)] != wy) dirs2 = false;
            }
            check(dirs2, "v2 dx/dy match the real nearest lots");
            check(found_slots >= 3, "v2 test map has several resource types");
        }
        bool threw_v2 = false;
        try { e1.set_curriculum(c2); } catch (const std::runtime_error&) { threw_v2 = true; }
        check(threw_v2, "set_curriculum refuses a v1->v2 change mid-run");

        // Restricted frame: weights + bits reflect the curriculum, not terrain.
        Curriculum cr;
        cr.all_builds = false;
        cr.allowed_builds.insert(first_id);
        cr.all_resources = false;
        auto w = only(6);  // water only
        for (int j = 0; j < 9; j++) cr.resource_weights[j] = w[j];
        cr.obs_version = 1;
        ColonyEnvCpp er(bd, ed, 7, 280, cr);
        er.reset(7);
        auto obv = er.obs();
        bool weights_ok = obv[248 + 6] == 1.0f;
        for (int j = 0; j < 9; j++)
            if (j != 6 && obv[248 + j] != 0.0f) weights_ok = false;
        check(weights_ok, "v1 tail shows water-only weights");
        int fi = find_id(er, first_id);
        bool bits_ok = (fi >= 0 && obv[257 + fi] == 1.0f);
        for (int i = 0; i < er.n_build(); i++)
            if (i != fi && obv[257 + i] != 0.0f) bits_ok = false;
        check(bits_ok, "v1 tail shows build_allowed bits (only the allowed id)");

        // all_resources=true renders nine 1.0s no matter what the array holds.
        Curriculum ca;
        ca.obs_version = 1;
        for (int j = 0; j < 9; j++) ca.resource_weights[j] = 0.0;  // ignored
        ColonyEnvCpp ea(bd, ed, 7, 280, ca);
        ea.reset(7);
        auto oa = ea.obs();
        bool eff = true;
        for (int j = 0; j < 9; j++)
            if (oa[248 + j] != 1.0f) eff = false;
        check(eff, "all_resources renders effective weights (nine 1.0s)");

        // Version switch mid-run is refused; same-version set works.
        bool threw = false;
        try {
            e0.set_curriculum(c1);
        } catch (const std::runtime_error&) { threw = true; }
        check(threw, "set_curriculum refuses an obs_version change");
        bool same_ok = true;
        try {
            Curriculum same = cr;
            same.allowed_builds.insert(er.build_ids()[0]);
            e1.set_curriculum(same);  // schedule-style update under v1
        } catch (const std::exception&) { same_ok = false; }
        check(same_ok && e1.curriculum().obs_version == 1,
              "set_curriculum accepts same-version updates (schedules work)");

        // obs_mask_locked_catalog: default off (catalog ignores gating).
        Curriculum cz;  // v0 default
        cz.all_builds = false;
        cz.allowed_builds.insert(first_id);
        ColonyEnvCpp eu(bd, ed, 7, 280);      // unrestricted twin
        eu.reset(7);
        ColonyEnvCpp ez(bd, ed, 7, 280, cz);  // restricted, flag off
        ez.reset(7);
        auto ou = eu.obs();
        auto oz = ez.obs();
        bool cat_same = true;
        for (int i = 75; i < 75 + 4 * eu.n_build(); i++)
            if (ou[i] != oz[i]) { cat_same = false; break; }
        check(cat_same, "flag off: restricted catalog block == unrestricted");
        bool locked_nonzero = false;
        for (int i = 0; i < eu.n_build(); i++)
            if (i != find_id(eu, first_id) && oz[75 + i * 4 + 1] != 0.0f) {
                locked_nonzero = true;
                break;
            }
        check(locked_nonzero, "flag off: locked catalog rows carry real prices");

        RewardConfig rc_mask;
        rc_mask.obs_mask_locked_catalog = true;
        ColonyEnvCpp em(bd, ed, 7, 280, cz, rc_mask);  // restricted, flag on
        em.reset(7);
        auto om = em.obs();
        int fui = find_id(em, first_id);
        bool masked = true, kept = true;
        for (int i = 0; i < em.n_build(); i++) {
            for (int k = 0; k < 4; k++) {
                float v = om[75 + i * 4 + k];
                if (i == fui) {
                    if (v != ou[75 + i * 4 + k]) kept = false;
                } else if (v != 0.0f) {
                    masked = false;
                }
            }
        }
        check(masked, "flag on: all 31 locked catalog rows are zero");
        check(kept, "flag on: the allowed row is preserved");

        // ── mechanic curriculum: fixed manager slots, hard rejection ──
        Curriculum early;
        early.enabled_mechanics = {false, false, false};
        ColonyEnvCpp gated(bd, ed, 31, 280, early);
        gated.reset(31);
        const int manager_base = A_BUILD0 + gated.n_build();
        const int64_t day0 = gated.game().days_alive;
        const int64_t money0 = gated.game().money;
        const int64_t credit0 = gated.game().credit;
        auto early_mask = gated.action_mask();
        for (int i : {0, 4, 5, 8, 9}) {
            check(early_mask[(size_t)(manager_base + i)] == 0.0f,
                  "early curriculum masks improve/preserve/credit manager action");
            auto refused = gated.step(manager_base + i);
            check(refused.days == day0 && gated.game().days_alive == day0,
                  "disabled manager action does not advance calendar");
            check(gated.game().money == money0 && gated.game().credit == credit0,
                  "disabled manager action does not mutate money/credit");
        }
        Curriculum open;
        gated.set_curriculum(open);
        ColonyEnvCpp twin(bd, ed, 31, 280, open);
        twin.reset(31);
        auto open_mask = gated.action_mask();
        auto twin_mask = twin.action_mask();
        for (int i : {0, 4, 5, 8, 9})
            check(open_mask[(size_t)(manager_base + i)] == twin_mask[(size_t)(manager_base + i)],
                  "enabled mechanic restores exact applicability mask");

        // from_json transport.
        Curriculum fj = Curriculum::from_json("{\"obs_version\": 1}");
        check(fj.obs_version == 1, "from_json reads obs_version");
        Curriculum fj0 = Curriculum::from_json("{}");
        check(fj0.obs_version == 2, "from_json defaults obs_version to 2 (canonical v4/obs-v2 default)");
        Curriculum fm = Curriculum::from_json(
            "{\"enabled_mechanics\":[\"improve_land\",\"preservation\"]}");
        check(fm.enabled_mechanics[0] && fm.enabled_mechanics[1] && !fm.enabled_mechanics[2],
              "from_json transports mechanic allow-list");
    }

    printf("\n%s (%d failure(s))\n",
           failures == 0 ? "ALL CHECKS PASSED" : "CHECKS FAILED", failures);
    return failures == 0 ? 0 : 1;
}
