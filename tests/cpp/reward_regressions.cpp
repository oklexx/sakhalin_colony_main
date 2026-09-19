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

// Направление на ближайшую к городу клетку воды — считаем с карты (как
// ColonyEnvCpp при reset). Хвост obs v0/v1 действительно кончается парой
// (dx,dy) воды, но в obs v2 (канонический дефолт) последними идут направления
// к дереву/углю/железу/нефти/золоту — чтение хвоста там даёт ЗОЛОТО, а не воду
// (R5 при этом начинал вести дороги к золоту и «падал»).
static void water_dir_from_map(const ColonyEnvCpp& e, double& out_dx, double& out_dy) {
    const Game& g = e.game();
    const int ms = g.map_size();
    const int8_t* lots = g.earth.lots().data();
    const double bx = g.earth.init_sel_x, by = g.earth.init_sel_y;
    double best_sq = -1.0; int wx = -1, wy = -1;
    for (int y = 0; y < ms; ++y)
        for (int x = 0; x < ms; ++x)
            if (lots[(size_t)y * ms + x] == LT_WATER) {
                double d2 = (x - bx) * (x - bx) + (y - by) * (y - by);
                if (best_sq < 0 || d2 < best_sq) { best_sq = d2; wx = x; wy = y; }
            }
    if (best_sq < 0) { out_dx = out_dy = 0.0; return; }
    out_dx = (wx - bx) / (double)ms;
    out_dy = (wy - by) / (double)ms;
}
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

    // ── R4: housing bonus must stay finite ──────────────────────────────────
    // The block computes log1p((people - housing) / 10) AFTER advance_day,
    // and advance_day's overflow kills everyone beyond home capacity — so
    // (people - housing) <= 0 always. The old formula then produced:
    //   -inf at housing - people == 10  (log1p(-1)),
    //   NaN  at housing - people > 10   (log1p(x< -1)).
    // Measured (seed 42, manual_plan probe, step 1861: people=220, home=230,
    // action BUILD:House): build=-inf, and with ep_return_ += rew before the
    // isfinite guard the whole episode return stayed -inf (dashboard NaN).
    // Post-fix the bonus is 0 unless there is a real (pre-death) shortage.
    {
        auto run = [&](int64_t people, int houses) {
            ColonyEnvCpp e(bd, ed, 42, 200);
            e.reset(42);
            e.game().money = 2'000'000;
            for (int i = 0; i < houses; i++) {
                auto cell = e.find_lot(LT_EVERYWHERE, false);
                if (!cell) break;
                e.game().build("House", cell->first, cell->second);
            }
            // дома 25 дней строятся — дождаться завершения, иначе
            // now_home_places() не увидит готовую ёмкость и сценарий не сойдётся
            for (int k = 0; k < 30; k++) {
                ColonyEnvCpp::StepOut o = e.step(A_DAY);
                if (o.terminated) break;
            }
            e.game().people = people;
            int hs = find_id(e, "House");
            bool finite = true;
            double total = 0.0;
            for (int k = 0; k < 5; k++) {
                ColonyEnvCpp::StepOut out = e.step(A_BUILD0 + hs);
                if (!std::isfinite(out.rew) || !std::isfinite(out.ep_return))
                    finite = false;
                total += out.rew;
                if (out.terminated) break;
            }
            return std::make_pair(total, finite);
        };
        // (housing - people) == 10 exactly -> old code: log1p(-1) = -inf,
        // который гвард вынужденно переводил в rew=0 и СТЫРАЛ весь
        // сшаг (build+taxb+milestone). Новое поведение: бонус = 0,
        // штатная награда шага остаётся положительной.
        auto trap = run(220, 3);
        printf("[info] R4 boundary (people=220, home=230): rew5=%+.3f finite=%d\n",
               trap.first, (int)trap.second);
        check(trap.second && trap.first > 0.0,
              "R4a boundary surplus: finite, and step reward is not wiped by -inf");
        // (housing - people) > 10 -> old code: log1p(x < -1) = NaN -> rew=0
        auto over = run(200, 3);
        printf("[info] R4 surplus (people=200, home=230): rew5=%+.3f finite=%d\n",
               over.first, (int)over.second);
        check(over.second && over.first > 0.0,
              "R4b larger surplus: finite, no NaN wipe");
        // mild surplus: old code gave a small NEGATIVE 'bonus'; post-fix 0.
        auto mild = run(225, 3);
        check(mild.second && std::isfinite(mild.first),
              "R4c mild housing surplus stays finite");
        // real shortage at step start (people > home): overflow kills the
        // excess on the first advance_day, so no bonus can be positive — the
        // run must at least stay finite and not terminate instantly.
        auto shortage = run(400, 0);
        check(shortage.second, "R4d population overflow stays finite");
    }

    // ── R5: Road shaping towards water creates a positive learning gradient ─
    {
        ColonyEnvCpp e(bd, ed, 42, 200);
        e.reset(42);

        int rdb = e.road_dir_base();
        int wc_act = find_id(e, "WaterChannel");
        check(wc_act >= 0, "R5 precondition: WaterChannel exists in build catalogue");
        int wc_full_act = A_BUILD0 + wc_act;

        // WaterChannel must be masked at start
        check(e.action_mask()[wc_full_act] == 0.0f,
              "R5a WaterChannel is masked at reset (water is far)");

        // Road towards water gives strictly higher reward than DAY
        ColonyEnvCpp e_day(bd, ed, 42, 200);
        e_day.reset(42);
        auto s_day = e_day.step(A_DAY);

        double wdx = 0.0, wdy = 0.0;
        water_dir_from_map(e, wdx, wdy);
        int best_dir = -1; double best_dot = -1e9;
        int worst_dir = -1; double worst_dot = 1e9;
        for (int d = 0; d < 4; d++) {
            double dot = ROAD_DIR_DX[d] * wdx + ROAD_DIR_DY[d] * wdy;
            if (dot > best_dot) { best_dot = dot; best_dir = d; }
            if (dot < worst_dot) { worst_dot = dot; worst_dir = d; }
        }

        ColonyEnvCpp e_good(bd, ed, 42, 200);
        e_good.reset(42);
        auto s_good = e_good.step(rdb + best_dir);

        ColonyEnvCpp e_bad(bd, ed, 42, 200);
        e_bad.reset(42);
        auto s_bad = e_bad.step(rdb + worst_dir);

        printf("[info] R5 step 0: road towards water rew=%+.3f, DAY rew=%+.3f, road away rew=%+.3f\n",
               s_good.rew, s_day.rew, s_bad.rew);
        check(s_good.rew > s_day.rew,
              "R5b road towards water earns higher reward than DAY (gradient towards water)");
        check(s_good.rew > s_bad.rew,
              "R5c road towards water earns higher reward than road away");

        // Follow the road towards water until WaterChannel unmasks
        bool reached = false;
        ColonyEnvCpp e_walk(bd, ed, 42, 200);
        e_walk.reset(42);
        for (int step = 0; step < 12; step++) {
            auto m = e_walk.action_mask();
            if (m[wc_full_act] != 0.0f) {
                reached = true;
                break;
            }
            double dx = 0.0, dy = 0.0;
            water_dir_from_map(e_walk, dx, dy);
            int d_best = -1; double bs = -1e18;
            for (int d = 0; d < N_ROAD_DIRS; d++) {
                if (m[rdb + d] == 0.0f) continue;
                double sc = -(std::fabs(dx - ROAD_DIR_DX[d] * 0.05) +
                              std::fabs(dy - ROAD_DIR_DY[d] * 0.05));
                if (sc > bs) { bs = sc; d_best = d; }
            }
            if (d_best < 0) break;
            e_walk.step(rdb + d_best);
        }
        check(reached, "R5d road chain successfully unmasks WaterChannel");
    }

    printf("\n%s (%d failure(s))\n", failures == 0 ? "ALL CHECKS PASSED" : "FAILURES", failures);
    return failures == 0 ? 0 : 1;
}
