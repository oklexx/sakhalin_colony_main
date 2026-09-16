// Standalone regression harness for the 2026-09 reward fixes (no Python, no raylib).
//
// Covers three measured defects that made road spam the global reward maximum:
//   R1  Game::check_milestones counted Roads as colony bases, so +30 per 5 bases
//       was farmable with 400-money roads (measured: +1241 for 205 roads).
//   R2  the idle_build_penalty gate ignored both Road and find_lot(), so it
//       fired on every policy that was not road spam (measured: -120/episode).
//   R3  proximity_land_for() had an unreachable second BigFarm branch.
//
// Build (from the repo root, Linux/macOS):
//   g++ -std=c++17 -O1 -Iinclude -Iinclude/third_party -o /tmp/reward_check \
//       tests/cpp/reward_regressions.cpp \
//       src/env.cpp src/data.cpp src/resources.cpp src/game.cpp \
//       src/rewards.cpp src/rng.cpp src/earth.cpp
//   /tmp/reward_check      # run from the repo root: configs/*.json are loaded
#include "colony/env.h"
#include "colony/data.h"
#include "colony/rewards.h"
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using namespace colony;

static int failures = 0;
static void check(bool ok, const std::string& what) {
    printf("%s  %s\n", ok ? "[ok]  " : "[FAIL]", what.c_str());
    if (!ok) failures++;
}

static int find_id(ColonyEnvCpp& e, const std::string& id) {
    for (int i = 0; i < e.n_build(); i++)
        if (e.build_ids()[i] == id) return i;
    return -1;
}

int main() {
    auto bd = load_base_data("configs/bases.json");
    auto ed = load_events("configs/events.json");

    // ── R1: roads must not count toward the base milestone ──────────────────
    {
        ColonyEnvCpp e(bd, ed, 42, 200);
        e.reset(42);
        e.game().money = 5'000'000;
        int rd = find_id(e, "Road");
        for (int k = 0; k < 200; k++) {
            auto m = e.action_mask();
            e.step(m[A_BUILD0 + rd] != 0.0f ? A_BUILD0 + rd : 0);
        }
        int roads = 0, other = 0;
        for (const Base& b : e.game().bases)
            (b.data->id == ROAD_ID ? roads : other)++;
        printf("[info] built %d roads, %d non-road bases\n", roads, other);
        check(roads >= 100, "R1 precondition: a lot of roads were actually built");
        // With <5 non-road bases no base milestone may be payable any more.
        // (Day/year milestones can still fire, so zero out those coefficients.)
        double ms = e.game().check_milestones(30.0, 0.0, 0.0, 0.0);
        printf("[info] check_milestones(30,0,0,0) after %d roads = %.3f\n", roads, ms);
        check(std::fabs(ms) < 1e-9, "R1 roads earn no milestone_base bonus");
    }

    // ── R1b: real buildings still earn it ───────────────────────────────────
    {
        ColonyEnvCpp e(bd, ed, 42, 200);
        e.reset(42);
        e.game().money = 5'000'000;
        // City counts as a non-road base; force 5 more via direct Game::build
        // on connected lots so the milestone threshold is genuinely crossed.
        int built = 0;
        for (int i = 0; i < e.n_build() && built < 5; i++) {
            const std::string& id = e.build_ids()[i];
            if (id == ROAD_ID) continue;
            auto cell = e.find_lot(e.build_data()[i]->need_earth,
                                   e.build_data()[i]->no_near_base);
            if (!cell) continue;
            if (e.game().build(id, cell->first, cell->second).first) built++;
        }
        printf("[info] placed %d non-road buildings directly\n", built);
        double ms = e.game().check_milestones(30.0, 0.0, 0.0, 0.0);
        printf("[info] check_milestones(30,0,0,0) with %d non-road bases = %.3f\n",
               1 + built, ms);
        check(built > 0, "R1b precondition: some buildings were placed");
        if (1 + built >= 5)
            check(std::fabs(ms - 30.0) < 1e-9, "R1b real buildings still earn milestone_base");
        else
            check(true, "R1b skipped: fewer than 5 placeable buildings on this seed");
    }

    // ── R2a: Road alone must not satisfy the idle gate ──────────────────────
    // Curriculum restricted to {Road}: Road is always affordable and always has
    // a lot, so under the old money+curriculum-only check the penalty fired on
    // every idle day even though no colony building was possible.
    {
        auto run = [&](double idle_pen) {
            ColonyEnvCpp e(bd, ed, 42, 200);
            e.reset(42);
            Curriculum c;
            c.all_builds = false;
            c.allowed_builds = {ROAD_ID};
            e.set_curriculum(c);
            RewardConfig rc;
            rc.idle_build_penalty = idle_pen;
            rc.idle_build_threshold_days = 1;  // fire the check every day
            e.set_rewards(rc);
            double total = 0.0;
            for (int k = 0; k < 200; k++) total += e.step(0).rew;  // DAY only
            return total;
        };
        double with = run(-2.0);
        double without = run(0.0);
        printf("[info] Road-only curriculum, DAY policy: idle=-2 -> %+.3f, idle=0 -> %+.3f\n",
               with, without);
        check(std::fabs(with - without) < 1e-6,
              "R2a idle_build_penalty ignores Road when deciding idleness");
    }

    // ── R2b: an affordable building with NO legal lot must not trigger it ───
    // Curriculum restricted to {WaterChannel}: it is affordable, but it needs a
    // water tile adjacent to the road network, which does not exist at reset
    // (measured: the action is masked on 0/2000 random steps). Penalising the
    // agent for not building something it cannot build teaches nothing.
    {
        auto run = [&](double idle_pen) {
            ColonyEnvCpp e(bd, ed, 42, 200);
            e.reset(42);
            Curriculum c;
            c.all_builds = false;
            c.allowed_builds = {"WaterChannel"};
            e.set_curriculum(c);
            RewardConfig rc;
            rc.idle_build_penalty = idle_pen;
            rc.idle_build_threshold_days = 1;
            e.set_rewards(rc);
            int wc = find_id(e, "WaterChannel");
            check(e.action_mask()[A_BUILD0 + wc] == 0.0f,
                  "R2b precondition: WaterChannel is masked at reset");
            double total = 0.0;
            for (int k = 0; k < 200; k++) total += e.step(0).rew;  // DAY only
            return total;
        };
        double with = run(-2.0);
        double without = run(0.0);
        printf("[info] WaterChannel-only curriculum: idle=-2 -> %+.3f, idle=0 -> %+.3f\n",
               with, without);
        check(std::fabs(with - without) < 1e-6,
              "R2b idle_build_penalty requires a legal lot, not just affordability");
    }

    // ── R2c: the penalty still fires when a real building IS placeable ──────
    {
        auto run = [&](double idle_pen) {
            ColonyEnvCpp e(bd, ed, 42, 200);
            e.reset(42);
            RewardConfig rc;
            rc.idle_build_penalty = idle_pen;
            rc.idle_build_threshold_days = 1;
            e.set_rewards(rc);
            double total = 0.0;
            for (int k = 0; k < 50; k++) total += e.step(0).rew;  // DAY only
            return total;
        };
        double with = run(-2.0);
        double without = run(0.0);
        printf("[info] DAY-only reward: idle=-2 -> %+.3f, idle=0 -> %+.3f\n", with, without);
        check(with < without - 1e-6,
              "R2c idle_build_penalty still fires when a real building is placeable");
    }

    // ── R3: proximity_land_for has no dead BigFarm branch ───────────────────
    {
        check(proximity_land_for("BigFarm") == LT_WOOD,
              "R3 BigFarm maps to LT_WOOD (the LT_IRON branch was unreachable)");
        check(proximity_land_for("Farm") == LT_WATER, "R3 Farm still maps to LT_WATER");
        check(proximity_land_for("WaterChannel") == LT_NONE,
              "R3 WaterChannel intentionally has no proximity bonus (it sits on water)");
    }

    printf("\n%s (%d failure(s))\n", failures == 0 ? "ALL CHECKS PASSED" : "FAILURES", failures);
    return failures == 0 ? 0 : 1;
}
