// Standalone check for the 2026-09 P0/P1 fixes (see docs/RL_DIAGNOSIS_2026_09.md).
//
// P0 — налог больше не останавливает время:
//   * unpaid annual/main tax is settled into bank debt (money first, rest → credit),
//     `check_advance()` returns ok, the calendar advances every DAY step;
//   * the episode is NOT killed by TAX_GRACE_DAYS just because a policy spent
//     its capital (the old behaviour: every spending policy died on day 365);
//   * `tax_to_debt=false` still reproduces the legacy freeze (explicit opt-in);
//   * the step log now names every hidden term (debt/born/died/lost/overflow,
//     taxdebt) and prints BOTH raw and clipped totals.
//
// P1 — наблюдаемость и маска:
//   * obs v2 (299 dims) adds dx/dy to the nearest wood/coal/iron/oil/gold lot;
//     v0/v1 remain strict prefixes (248/289) and the values match the real map;
//   * manager actions are masked by APPLICABILITY (SELL with an empty warehouse,
//     REPAY with no debt, IMPROVE without money, manual_tax …), so the mask stops
//     advertising actions that can only burn error_penalty.
//
// Build (from the repo root):
//   g++ -std=c++17 -O2 -Iinclude -Iinclude/third_party -o /tmp/p0_p1_check \
//       tests/cpp/p0_p1_check.cpp \
//       src/env.cpp src/data.cpp src/resources.cpp src/game.cpp \
//       src/rewards.cpp src/rng.cpp src/earth.cpp
// Run from the repo root:  /tmp/p0_p1_check

#include "colony/constants.h"
#include "colony/data.h"
#include "colony/env.h"
#include "colony/game.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

using namespace colony;

static int g_fail = 0;
static int g_pass = 0;

static void check(const std::string& name, bool ok, const std::string& detail = "") {
    (ok ? g_pass : g_fail)++;
    printf("[%s] %s%s\n", ok ? "PASS" : "FAIL", name.c_str(),
           detail.empty() ? "" : (" — " + detail).c_str());
}

static int find_id(const ColonyEnvCpp& e, const std::string& id) {
    for (int i = 0; i < e.n_build(); i++)
        if (e.build_ids()[(size_t)i] == id) return i;
    return -1;
}

static std::vector<float> mask_of(ColonyEnvCpp& e) { return e.action_mask(); }

namespace {

struct LotStat {
    bool found = false;
    int x = -1, y = -1;
    double d2 = 0.0;
};

// Независимый от obs() поиск ближайшего тайла данного типа (тот же порядок
// обхода и строгое <, что в obs) — эталон для проверки obs v2.
LotStat nearest_lot(const Game& g, int lot_type, int bx, int by) {
    LotStat st;
    const int8_t* lots = g.earth.lots().data();
    const int ms = g.map_size();
    for (int y = 0; y < ms; y++) {
        for (int x = 0; x < ms; x++) {
            if (lots[(size_t)y * ms + x] != lot_type) continue;
            double dx = (double)(x - bx), dy = (double)(y - by);
            double d2 = dx * dx + dy * dy;
            if (!st.found || d2 < st.d2) { st.found = true; st.x = x; st.y = y; st.d2 = d2; }
        }
    }
    return st;
}

}  // namespace

int main() {
    auto bd = load_base_data("configs/bases.json");
    auto ed = load_events("configs/events.json");

    printf("========== P0.1 unpaid tax: calendar advances, remainder -> debt ==========\n");
    {
        const int64_t seed = 42;
        ColonyEnvCpp e(bd, ed, seed, 200, Curriculum(), RewardConfig(), DIFFICULTY_NORMAL);
        e.reset(seed);
        const int farm = find_id(e, "Farm");
        for (int k = 0; k < 4; k++) e.step(A_BUILD0 + farm);  // spend the 82k capital
        Game& g = e.game();
        printf("  after 4 farms: money=%lld credit=%lld day=%lld\n",
               (long long)g.money, (long long)g.credit, (long long)g.days_alive);
        check("broke policy: check_advance() ok (time is not blocked)",
              g.check_advance().first,
              g.check_advance().second.empty() ? "" : g.check_advance().second);

        int64_t prev_day = g.days_alive;
        bool monotone = true;
        int64_t borrowed_total = 0;
        int steps_to_tax = 0;
        int64_t credit_before_tax = g.credit;
        for (int i = 0; i < 400; i++) {
            auto out = e.step(A_DAY);
            if (out.days <= prev_day) monotone = false;
            prev_day = out.days;
            if (out.tax_borrowed > 0) borrowed_total += out.tax_borrowed;
            if (borrowed_total == 0) steps_to_tax++;
            if (out.terminated) { printf("  terminated at step %d day=%lld\n", i, (long long)out.days); break; }
        }
        printf("  after 400 DAY steps: day=%lld money=%lld credit=%lld tax_borrowed=%lld\n",
               (long long)g.days_alive, (long long)g.money, (long long)g.credit,
               (long long)borrowed_total);
        check("calendar advances on EVERY step (no freeze)", monotone);
        check("episode survives the first tax date (day 365) instead of dying there",
              g.days_alive > 365, "day=" + std::to_string((long long)g.days_alive));
        check("unpaid annual tax was settled into credit (debt)",
              borrowed_total > 0 && g.credit > credit_before_tax,
              "borrowed=" + std::to_string((long long)borrowed_total));
    }

    printf("\n========== P0.2 legacy policy opt-in still freezes (control) ==========\n");
    {
        const int64_t seed = 42;
        ColonyEnvCpp e(bd, ed, seed, 200, Curriculum(), RewardConfig(), DIFFICULTY_NORMAL,
                       /*no_city_game_over=*/false, /*no_people_days=*/365,
                       /*tax_to_debt=*/false);
        e.reset(seed);
        const int farm = find_id(e, "Farm");
        for (int k = 0; k < 4; k++) e.step(A_BUILD0 + farm);
        Game& g = e.game();
        bool any_blocked = false;
        for (int i = 0; i < 400; i++) {
            auto out = e.step(A_DAY);
            (void)out;
            auto adv = g.check_advance();
            if (!adv.first && (adv.second == "annual_tax" || adv.second == "main_tax"))
                any_blocked = true;
        }
        check("tax_to_debt=false reproduces the dialogue policy (blocker visible)",
              any_blocked || g.days_alive <= 425,
              "day=" + std::to_string((long long)g.days_alive));
    }

    printf("\n========== P0.3 step log: hidden terms + raw vs clipped ==========\n");
    {
        const std::string log_path = "/tmp/p0_p1_steplog.txt";
        std::remove(log_path.c_str());
        const int64_t seed = 42;
        ColonyEnvCpp e(bd, ed, seed, 200, Curriculum(), RewardConfig(), DIFFICULTY_NORMAL);
        e.set_step_log(log_path);
        e.reset(seed);
        const int farm = find_id(e, "Farm");
        for (int k = 0; k < 4; k++) e.step(A_BUILD0 + farm);  // broke → tax goes to debt
        Game& g = e.game();
        // 1200 days of pure DAY: no spending → the annual tax snowballs into debt
        // (and the debt term / born / died / lost / overflow terms fire), which is
        // exactly the regime where the old log disagreed with the returned reward.
        double sum_returned = 0.0;
        bool saw_clamp = false;
        for (int i = 0; i < 1200; i++) {
            auto out = e.step(A_DAY);
            sum_returned += out.rew;
            if (i > 0 && std::fabs(out.rew) >= 49.999) saw_clamp = true;
            if (out.terminated) break;
        }
        FILE* f = fopen(log_path.c_str(), "rb");
        check("step log file created", f != nullptr, log_path);
        if (f) {
            std::string content;
            {
                char buf[65536];
                size_t n;
                while ((n = fread(buf, 1, sizeof(buf), f)) > 0) content.append(buf, n);
            }
            fclose(f);
            const char* first = nullptr;
            for (const char* needle : {"total_raw=", "total_clip=", "taxdebt=", "debt=",
                                       "born=", "died=", "lost=", "overflow=", "tax_borrowed="}) {
                if (content.find(needle) == std::string::npos) {
                    check(std::string("log contains ") + needle, false);
                    first = needle;
                }
            }
            if (!first) check("log names every hidden term (debt/born/died/lost/overflow/taxdebt)", true);
            // В долговой политике tax_fail_penalty не должен срабатывать:
            // компонент tax= в step-логе остаётся нулевым (цену платят
            // tax_debt_penalty и проценты).
            check("debt policy never charges tax_fail_penalty twice",
                  content.find(" tax=-5.000") == std::string::npos);
            check("log shows the tax->debt conversion line",
                  content.find("TAX->DEBT:") != std::string::npos,
                  "credit=" + std::to_string((long long)g.credit) +
                      " tax_borrowed=" + std::to_string((long long)e.last_tax_borrowed()));
            // Разбираем total_raw / total_clip из строк и сверяем клип с формулой.
            size_t pos = 0;
            int rows = 0, clamped_rows = 0;
            double max_abs_raw = 0.0;
            while ((pos = content.find("total_raw=", pos)) != std::string::npos) {
                double raw = atof(content.c_str() + pos + 10);
                size_t clip_pos = content.find("total_clip=", pos);
                size_t nl = content.find('\n', pos);
                if (clip_pos == std::string::npos || clip_pos > nl) break;
                double clip = atof(content.c_str() + clip_pos + 11);
                double want = std::max(-50.0, std::min(50.0, raw));
                if (std::fabs(want - clip) > 1e-6) {
                    check("log total_clip == clamp(total_raw)", false,
                          "raw=" + std::to_string(raw) + " clip=" + std::to_string(clip));
                    break;
                }
                if (std::fabs(raw) > 50.0 + 1e-9) clamped_rows++;
                max_abs_raw = std::max(max_abs_raw, std::fabs(raw));
                rows++;
                pos = clip_pos + 11;
            }
            check("log prints both raw and clipped totals on every step", rows >= 1000,
                  "rows=" + std::to_string(rows));
            check("clipping actually happened (why raw and clipped must both exist)",
                  clamped_rows > 0,
                  "clamped_rows=" + std::to_string(clamped_rows) +
                      " max|raw|=" + std::to_string(max_abs_raw));
            printf("  rows=%d clamped=%d  returned_sum=%.2f\n", rows, clamped_rows, sum_returned);
            check("probe saw clipped returns (|reward| at the clip bound)", saw_clamp);
        }
    }

    printf("\n========== P1.1 applicability mask ==========\n");
    {
        const int64_t seed = 42;
        ColonyEnvCpp e(bd, ed, seed, 200, Curriculum(), RewardConfig(), DIFFICULTY_NORMAL);
        e.reset(seed);
        const int MGR = A_BUILD0 + e.n_build();
        std::vector<float> m = mask_of(e);

        printf("  manager legals at reset:");
        for (int i = 0; i < N_MANAGERS; i++)
            if (m[(size_t)(MGR + i)] > 0.5f) printf(" %d", i);
        printf("\n");
        check("IMPROVE_LAND applicable at reset (82k >= BUYGOODEARTH, lot is free)",
              m[(size_t)(MGR + 0)] > 0.5f);
        check("REPAIR masked at reset (nothing damaged)", m[(size_t)(MGR + 1)] < 0.5f);
        check("REPAIR_ALL masked at reset (nothing damaged)", m[(size_t)(MGR + 2)] < 0.5f);
        check("DEMOLISH masked at reset (only the City exists)",
              m[(size_t)(MGR + 3)] < 0.5f);
        check("PRESERVE masked at reset (nothing to preserve)", m[(size_t)(MGR + 4)] < 0.5f);
        check("UNPRESERVE masked at reset (nothing preserved)", m[(size_t)(MGR + 5)] < 0.5f);
        check("SELL masked at reset (empty warehouse)", m[(size_t)(MGR + 6)] < 0.5f);
        printf("  (BUY_FOOD at reset: food=%lld -> %s)\n",
               (long long)e.game().sunduk[FOOD],
               m[(size_t)(MGR + 7)] > 0.5f ? "applicable (food < 400)" : "masked");
        check("TAKE_LOAN legal at reset (applicable: credit < cap)",
              m[(size_t)(MGR + 8)] > 0.5f);
        check("REPAY_LOAN masked at reset (no debt)", m[(size_t)(MGR + 9)] < 0.5f);
        check("manual_tax always masked (dominated by DAY)", m[(size_t)(MGR + 10)] < 0.5f);

        // Контроль: старые маски возвращаются флагом.
        RewardConfig legacy;
        legacy.mask_managers_by_applicability = false;
        ColonyEnvCpp e2(bd, ed, seed, 200, Curriculum(), legacy, DIFFICULTY_NORMAL);
        e2.reset(seed);
        std::vector<float> m2 = mask_of(e2);
        int legal_new = 0, legal_old = 0;
        for (int i = 0; i < N_MANAGERS; i++) {
            if (m[(size_t)(MGR + i)] > 0.5f) legal_new++;
            if (m2[(size_t)(MGR + i)] > 0.5f) legal_old++;
        }
        printf("  legal managers: applicability=%d, legacy=%d\n", legal_new, legal_old);
        check("applicability mask is strictly tighter than legacy", legal_new < legal_old);

        // После займа REPAY становится применимым, а TAKE_LOAN — ещё нет (лимит не выбран).
        auto loan = e.step(MGR + 8);
        (void)loan;
        std::vector<float> m3 = mask_of(e);
        check("REPAY_LOAN applicable after taking credit", m3[(size_t)(MGR + 9)] > 0.5f);

        // Продажа: кладём ресурс в сундук и проверяем, что SELL открывается.
        Game& g = e.game();
        g.sunduk[WOOD] = 500;
        std::vector<float> m4 = mask_of(e);
        check("SELL applicable once there is surplus", m4[(size_t)(MGR + 6)] > 0.5f);
        // Еды достаточно → BUY_FOOD (чистый слив денег) больше не предлагается.
        g.sunduk[FOOD] = 500;
        std::vector<float> m5 = mask_of(e);
        check("BUY_FOOD masked once food is plentiful", m5[(size_t)(MGR + 7)] < 0.5f);
        // Денег меньше цены участка → IMPROVE_LAND не предлагается.
        g.money = BUYGOODEARTH - 1;
        std::vector<float> m6 = mask_of(e);
        check("IMPROVE_LAND masked when there is no money for the lot",
              m6[(size_t)(MGR + 0)] < 0.5f);
    }

    printf("\n========== P1.2 obs v2: nearest-resource directions ==========\n");
    {
        const int64_t seed = 42;
        struct Case { int version; int expected; };
        const Case cases[3] = {{0, 248}, {1, 289}, {2, 299}};
        std::vector<std::vector<float>> frames;
        for (const Case& c : cases) {
            Curriculum curr;
            curr.obs_version = c.version;
            curr.all_builds = true;
            curr.all_resources = true;
            ColonyEnvCpp e(bd, ed, seed, 200, curr, RewardConfig(), DIFFICULTY_NORMAL);
            e.reset(seed);
            check("obs v" + std::to_string(c.version) + " size == " +
                      std::to_string(c.expected),
                  e.obs_size() == c.expected,
                  "got " + std::to_string(e.obs_size()));
            frames.push_back(e.obs());
        }
        // v0/v1 — строгие префиксы v2 (бит-в-бит).
        bool prefix_ok = true;
        for (int v = 0; v < 2; v++) {
            if (frames[(size_t)v].size() > frames[2].size()) { prefix_ok = false; break; }
            for (size_t i = 0; i < frames[(size_t)v].size(); i++)
                if (frames[(size_t)v][i] != frames[2][i]) { prefix_ok = false; break; }
        }
        check("v0/v1 are bit-exact prefixes of v2", prefix_ok);

        // Значения v2 сверяем с независимым поиском по карте.
        Curriculum curr2;
        curr2.obs_version = 2;
        curr2.all_builds = true;
        curr2.all_resources = true;
        ColonyEnvCpp e(bd, ed, seed, 200, curr2, RewardConfig(), DIFFICULTY_NORMAL);
        e.reset(seed);
        const Game& g = e.game();
        const int ms = g.map_size();
        const int bx = g.earth.init_sel_x, by = g.earth.init_sel_y;
        std::vector<float> obs = e.obs();
        bool dirs_ok = true;
        std::string dirs_detail;
        for (int k = 0; k < N_NEAREST_LOTS; k++) {
            LotStat st = nearest_lot(g, NEAREST_LOT_TYPES[k], bx, by);
            float want_dx = st.found ? (float)((double)(st.x - bx) / (double)ms) : 0.0f;
            float want_dy = st.found ? (float)((double)(st.y - by) / (double)ms) : 0.0f;
            float got_dx = obs[(size_t)(289 + 2 * k)];
            float got_dy = obs[(size_t)(289 + 2 * k + 1)];
            if (std::fabs(got_dx - want_dx) > 1e-6 || std::fabs(got_dy - want_dy) > 1e-6) {
                dirs_ok = false;
                dirs_detail += " slot" + std::to_string(k) + " want(" +
                               std::to_string(want_dx) + "," + std::to_string(want_dy) +
                               ") got(" + std::to_string(got_dx) + "," +
                               std::to_string(got_dy) + ")";
            }
        }
        check("v2 dx/dy match the real nearest lots (wood/coal/iron/oil/gold)",
              dirs_ok, dirs_detail);
        // Вода как была в v0 (индексы 246/247), так и осталась.
        LotStat water = nearest_lot(g, LT_WATER, bx, by);
        check("v0 water direction unchanged (obs[246]/[247])",
              std::fabs(obs[246] - (float)((double)(water.x - bx) / (double)ms)) < 1e-6 &&
                  std::fabs(obs[247] - (float)((double)(water.y - by) / (double)ms)) < 1e-6);
        // Ненулевые направления — иначе «наблюдаемость» фиктивна.
        int nonzero = 0;
        for (int k = 0; k < 2 * N_NEAREST_LOTS; k++)
            if (std::fabs(obs[(size_t)(289 + k)]) > 1e-9f) nonzero++;
        printf("  nonzero v2 direction components: %d/10\n", nonzero);
        check("v2 carries real signal (most slots non-zero)", nonzero >= 6);
    }

    printf("\n========== P0.4 long run: invest-everything no longer dies on day 365 ==========\n");
    {
        const int64_t seed = 42;
        auto e = std::make_unique<ColonyEnvCpp>(bd, ed, seed, 200, Curriculum(),
                                                RewardConfig(), DIFFICULTY_NORMAL);
        e->reset(seed);
        int steps = 0;
        for (int i = 0; i < 3000; i++) {
            std::vector<float> m = e->action_mask();
            int pick = A_DAY;
            for (int b = 0; b < e->n_build(); b++)
                if (m[(size_t)(A_BUILD0 + b)] > 0.5f) { pick = A_BUILD0 + b; break; }
            auto out = e->step(pick);
            steps++;
            if (out.terminated) {
                const Game& gg = e->game();
                printf("  terminated at step %d (day %lld): credit=%lld max_credit=%lld"
                       " blocker='%s'\n",
                       steps, (long long)out.days, (long long)gg.credit,
                       (long long)gg.max_credit(), gg.check_advance().second.c_str());
                check("termination is economic (debt cap), not the tax calendar",
                      gg.check_advance().second != "annual_tax" &&
                          gg.check_advance().second != "main_tax");
                break;
            }
            if (out.truncated) break;
        }
        check("spend-everything policy survives past day 365", e->game().days_alive > 365,
              "day=" + std::to_string((long long)e->game().days_alive));
    }

    printf("\n==================== %d passed, %d failed ====================\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
