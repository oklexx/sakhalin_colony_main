// Standalone check: атрибуция причины закрытого бита маски
// (docs/MONITOR_ACTIONS_2026_09.md §4 — action_mask_reason_counts()).
//
// Проверяется (R1..R5):
//   R1  паритет: reasons[i] == MR_OPEN ⇔ mask[i] == 1.0 на каждом шаге;
//       гистограмма action_mask_reason_counts() сходится с reasons (сумма == n_actions,
//       корзина open == popcount(mask));
//   R2  first-fail BUILD (W2-порядок): курикулум → деньги → нет клетки —
//       независимая перепроверка по тем же примитивам на каждом шаге;
//   R3  принудительные состояния: money=0 → все оплаченные постройки с price>0
//       закрыты ДЕНЬГАМИ (не «нет участка»); запрет постройки (all_builds=false)
//       → КУРИКУЛУМ даже при нулевом бюджете (first-fail сильнее денег);
//   R4  менеджеры: выключенная механика → curriculum; PAY_TAX всегда other;
//       DAY/WEEK всегда open;
//   R5  masked-random роллаут (2 сида × 150 шагов): R1+сумма на каждом шаге.
//
// Build (from the repo root):
//   g++ -std=c++17 -O1 -Iinclude -Iinclude/third_party -o /tmp/mask_reason_check \
//       tests/cpp/mask_reason_check.cpp \
//       src/env.cpp src/data.cpp src/resources.cpp src/game.cpp \
//       src/rewards.cpp src/rng.cpp src/earth.cpp src/lot_finder.cpp \
//       src/action_mask.cpp src/observation.cpp src/vec_env.cpp
// Run from the repo root:  /tmp/mask_reason_check
#include "colony/env.h"
#include "colony/data.h"

#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>

using namespace colony;

static int failures = 0;
static void check(bool ok, const std::string& what) {
    printf("%s  %s\n", ok ? "[ok]  " : "[FAIL] ", what.c_str());
    if (!ok) failures++;
}

static int find_idx(ColonyEnvCpp& e, const std::string& id) {
    for (int i = 0; i < e.n_build(); ++i)
        if (e.build_ids()[(size_t)i] == id) return i;
    return -1;
}

// Независимая (от action_mask) эталонная атрибуция BUILD: тот же first-fail
// W2 — курикулум → деньги → нет легальной клетки. Примитивы те же
// (build_allowed / find_lot), но порядок и запись кода проверяются здесь
// заново: регрессия в action_mask (перепутанные continue/коды) поймается.
static uint8_t expect_build_reason(ColonyEnvCpp& e, int build_i) {
    const BaseData* d = e.build_data()[(size_t)build_i];
    if (!e.build_allowed(d->id)) return (uint8_t)MR_CURRICULUM;
    if (e.game().money < d->price) return (uint8_t)MR_MONEY;
    if (!e.find_lot(*d)) return (uint8_t)MR_NO_LOT;
    return (uint8_t)MR_OPEN;
}

// R1: паритет open⇔бит + сумма гистограммы == n_actions.
// Возвращает число ошибок (0 = ок).
static int check_parity(ColonyEnvCpp& e, const std::string& where) {
    std::vector<float> mask = e.action_mask();
    std::vector<uint8_t> reasons = e.action_mask_reasons();
    int errs = 0;
    if (reasons.size() != mask.size()) {
        printf("[FAIL] %s: reasons.size=%zu != mask.size=%zu\n",
               where.c_str(), reasons.size(), mask.size());
        return 1;
    }
    for (size_t i = 0; i < mask.size(); ++i) {
        const bool bit = mask[i] != 0.0f;
        const bool open = reasons[i] == (uint8_t)MR_OPEN;
        if (bit != open) {
            if (errs < 3)
                printf("[FAIL] %s: action %zu: mask=%d reason=%u\n",
                       where.c_str(), i, (int)bit, (unsigned)reasons[i]);
            errs++;
        }
    }
    auto counts = e.action_mask_reason_counts();
    long long total = 0, open_cnt = 0;
    for (int r = 0; r < N_MASK_REASONS; ++r) total += counts[(size_t)r];
    open_cnt = counts[(size_t)MR_OPEN];
    long long popcount = std::count_if(mask.begin(), mask.end(),
                                       [](float v) { return v != 0.0f; });
    if (total != (long long)mask.size()) {
        printf("[FAIL] %s: counts sum=%lld != n_actions=%zu\n",
               where.c_str(), total, mask.size());
        errs++;
    }
    if (open_cnt != popcount) {
        printf("[FAIL] %s: counts[open]=%lld != popcount(mask)=%lld\n",
               where.c_str(), open_cnt, popcount);
        errs++;
    }
    return errs;
}

int main() {
    auto bd = load_base_data("configs/bases.json");
    auto ed = load_events("configs/events.json");

    // ── R2: first-fail на каждом шаге (2 сида × 120 masked-random шагов) ────
    printf("=== R2: first-fail BUILD на каждом шаге роллаута ===\n");
    {
        int mismatches = 0, steps = 0;
        for (int64_t seed : {42, 100}) {
            ColonyEnvCpp e(bd, ed, seed, 200);
            e.reset(seed);
            PCG64 rnd((uint64_t)(seed * 104729 + 7));
            for (int s = 0; s < 120; ++s) {
                auto mask = e.action_mask();
                auto reasons = e.action_mask_reasons();
                for (int i = 0; i < e.n_build(); ++i) {
                    const uint8_t want = expect_build_reason(e, i);
                    if (reasons[(size_t)(A_BUILD0 + i)] != want) {
                        if (mismatches < 5)
                            printf("[info] seed=%lld step=%d build=%s got=%u want=%u\n",
                                   (long long)seed, s,
                                   e.build_ids()[(size_t)i].c_str(),
                                   (unsigned)reasons[(size_t)(A_BUILD0 + i)],
                                   (unsigned)want);
                        mismatches++;
                    }
                }
                mismatches += check_parity(e, "R2");
                std::vector<int> legal;
                for (int a = 0; a < e.n_actions(); ++a)
                    if (mask[(size_t)a] != 0.0f) legal.push_back(a);
                if (legal.empty()) break;
                int act = legal[(size_t)rnd.randint(0, (int64_t)legal.size() - 1)];
                auto out = e.step(act);
                steps++;
                if (out.terminated || out.truncated) break;
            }
        }
        check(mismatches == 0,
              "R2 коды причин BUILD == независимая first-fail перепроверка (вкл. паритет)");
        printf("[info] R2: шагов=%d, расхождений=%d\n", steps, mismatches);
    }

    // ── R3: деньги=0 и запрет постройки — принудительные состояния ──────────
    printf("\n=== R3: money=0 / all_builds=false (принудительные причины) ===\n");
    {
        ColonyEnvCpp e(bd, ed, 7, 200);
        e.reset(7);
        e.game().money = 0;
        auto reasons = e.action_mask_reasons();
        int money_cnt = 0, nocell_cnt = 0, curriculum_cnt = 0;
        for (int i = 0; i < e.n_build(); ++i) {
            const BaseData* d = e.build_data()[(size_t)i];
            if (!e.build_allowed(d->id)) { curriculum_cnt++; continue; }
            if (d->price <= 0) continue;   // бесплатная постройка не про деньги
            const uint8_t r = reasons[(size_t)(A_BUILD0 + i)];
            if (r == (uint8_t)MR_MONEY) money_cnt++;
            if (r == (uint8_t)MR_NO_LOT) nocell_cnt++;
        }
        check(money_cnt > 0, "R3 при money=0 закрытые постройки получают код MR_MONEY");
        check(nocell_cnt == 0,
              "R3 при money=0 нет MR_NO_LOT у оплаченных построек (first-fail: деньги раньше участка)");
        check(curriculum_cnt == 0, "R3 на сиде 7 all_builds=true (нет MR_CURRICULUM)");

        // Теперь запрещаем всё, кроме Road, при нулевом бюджете: курикулум
        // обязан выигрывать у денег (W2-порядок).
        Curriculum c;
        c.all_builds = false;
        c.allowed_builds = {"Road"};
        e.set_curriculum(c);
        reasons = e.action_mask_reasons();
        int nonroad_curriculum = 0, nonroad_money = 0;
        for (int i = 0; i < e.n_build(); ++i) {
            if (e.build_ids()[(size_t)i] == "Road") continue;
            const uint8_t r = reasons[(size_t)(A_BUILD0 + i)];
            if (r == (uint8_t)MR_CURRICULUM) nonroad_curriculum++;
            if (r == (uint8_t)MR_MONEY) nonroad_money++;
        }
        check(nonroad_curriculum == e.n_build() - 1,
              "R3 все не-Road постройки при all_builds=false → MR_CURRICULUM (даже при money=0)");
        check(nonroad_money == 0,
              "R3 MR_CURRICULUM сильнее MR_MONEY (первый fail — курикулум)");
        // Водоканал на старте (W2): деньги есть, курикулум открыт, клетки нет
        Curriculum c2;  // снова всё открыто
        e.set_curriculum(c2);
        e.game().money = 82000;
        const int wc = find_idx(e, "WaterChannel");
        reasons = e.action_mask_reasons();
        check(wc >= 0 && reasons[(size_t)(A_BUILD0 + wc)] == (uint8_t)MR_NO_LOT,
              "R3 WaterChannel на старте (деньги есть) → MR_NO_LOT (как W2)");
    }

    // ── R4: менеджеры + DAY/WEEK ────────────────────────────────────────────
    printf("\n=== R4: DAY/WEEK open; механики; PAY_TAX ===\n");
    {
        // set_curriculum запрещает re-lock (unlock schedules additive only),
        // поэтому «выключенная механика» задаётся СРАЗУ в конструкторе:
        // включены все, кроме repair — ровно то, что должен видеть менеджер.
        Curriculum c;  // all_builds=true, механики по умолчанию все включены
        c.enabled_mechanics[(size_t)mechanic_index("repair")] = false;
        ColonyEnvCpp e(bd, ed, 42, 200, c);
        e.reset(42);
        auto reasons = e.action_mask_reasons();
        check(reasons[A_DAY] == (uint8_t)MR_OPEN && reasons[A_WEEK] == (uint8_t)MR_OPEN,
              "R4 DAY/WEEK всегда MR_OPEN");
        // PAY_TAX (слот 10) при включённой аппликабильности всегда закрыт → other
        const int pay_tax = A_BUILD0 + e.n_build() + 10;
        check(reasons[(size_t)pay_tax] == (uint8_t)MR_OTHER,
              "R4 PAY_TAX закрыт → MR_OTHER (доминируемое A_DAY, не деньги/курикулум)");
        const int mgr = A_BUILD0 + e.n_build();
        check(reasons[(size_t)(mgr + 1)] == (uint8_t)MR_CURRICULUM &&
              reasons[(size_t)(mgr + 2)] == (uint8_t)MR_CURRICULUM,
              "R4 выключенная механика repair (REPAIR/REPAIR_ALL) → MR_CURRICULUM");
        check(reasons[(size_t)(mgr + 6)] != (uint8_t)MR_CURRICULUM,
              "R4 включённая механика sell не получает MR_CURRICULUM");
    }

    // ── R5: короткий masked-random роллаут — паритет на каждом шаге ─────────
    printf("\n=== R5: masked-random 2×150 шагов — паритет каждый шаг ===\n");
    {
        int errs = 0, steps = 0;
        for (int64_t seed : {42, 2026}) {
            ColonyEnvCpp e(bd, ed, seed, 200);
            e.reset(seed);
            PCG64 rnd((uint64_t)(seed * 31 + 17));
            for (int s = 0; s < 150; ++s) {
                auto mask = e.action_mask();
                errs += check_parity(e, "R5");
                std::vector<int> legal;
                for (int a = 0; a < e.n_actions(); ++a)
                    if (mask[(size_t)a] != 0.0f) legal.push_back(a);
                if (legal.empty()) break;
                int act = legal[(size_t)rnd.randint(0, (int64_t)legal.size() - 1)];
                auto out = e.step(act);
                steps++;
                if (out.terminated || out.truncated) break;
            }
        }
        check(errs == 0, "R5 паритет open⇔бит и сумма counts == n_actions на каждом шаге");
        printf("[info] R5: шагов=%d, ошибок=%d\n", steps, errs);
    }

    printf("\n%s (%d failure(s))\n",
           failures == 0 ? "ALL CHECKS PASSED" : "FAILURES", failures);
    return failures == 0 ? 0 : 1;
}
