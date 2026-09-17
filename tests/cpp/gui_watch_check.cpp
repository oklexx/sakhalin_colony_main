// Standalone check for the 2026-09 «наблюдение за моделью через UI» path.
//
// UI-наблюдение идёт двумя путями:
//   * headless (без галочки «GUI-окно»): watch_champion.py строит CppColonyEnv —
//     это та же среда, что при обучении (obs v2, tax_to_debt=true);
//   * визуальный (галочка «GUI-окно»): watch_champion.py запускает raylib-GUI
//     (`--headless-ai`) и гоняет политику по IPC (state.json/actions.txt).
//     Вот этот env и проверяем: он обязан совпадать с тренировочным, иначе
//     watched-модель играет в ДРУГУЮ игру.
//
// Проверки:
//   A. curriculum JSON от Python (st.to_dict()) применяется в GUI-стиле:
//      obs_size == 299 (v2), маска == 49 действий.
//   B. налоговая политика GUI-наблюдения: с tax_to_debt=false (как сейчас
//      создаёт gui.cpp) «потратить всё → DAY» замирает на 365-м дне и умирает,
//      т.е. модель видит заморозку, которой при обучении уже нет.
//   C. тот же прогон с tax_to_debt=true (то, что должен передавать watch):
//      календарь идёт каждый шаг, смерть — только экономическая.
//
// Build (from the repo root):
//   g++ -std=c++17 -O2 -Iinclude -Iinclude/third_party -o /tmp/gui_watch_check \
//       tests/cpp/gui_watch_check.cpp \
//       src/env.cpp src/data.cpp src/resources.cpp src/game.cpp \
//       src/rewards.cpp src/rng.cpp src/earth.cpp
// Run from the repo root:  /tmp/gui_watch_check

#include "colony/constants.h"
#include "colony/data.h"
#include "colony/env.h"
#include "colony/game.h"

#include <cstdio>
#include <string>
#include <vector>

using namespace colony;

static int g_fail = 0, g_pass = 0;

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

// Ровно то, что Python отдаёт GUI в --curriculum: CurriculumState.to_dict().
static const char* CURRICULUM_V2 =
    R"({"all_builds":true,"allowed_builds":[],"stage":0,"all_resources":true,)"
    R"("resource_weights":[1,1,1,1,1,1,1,1,1],"obs_version":2})";

// То же самое для старой (289) модели: obs-версия приезжает из meta модели.
static const char* CURRICULUM_V1 =
    R"({"all_builds":true,"allowed_builds":[],"stage":0,"all_resources":true,)"
    R"("resource_weights":[1,1,1,1,1,1,1,1,1],"obs_version":1})";

struct RunOut {
    int steps = 0;
    int64_t day = 0, money = 0, credit = 0;
    double sum = 0.0;
    bool terminated = false;
    std::string blocker;
};

// Политика наблюдаемой модели-«транжиры»: 4 фермы, дальше DAY.
static RunOut run(bool tax_to_debt) {
    auto bd = load_base_data("configs/bases.json");
    auto ed = load_events("configs/events.json");
    Curriculum c = Curriculum::from_json(CURRICULUM_V2);
    RewardConfig rc;
    // gui.cpp: ctor(..., gui_no_city_game_over=false, gui_no_people_days=365, tax_to_debt)
    ColonyEnvCpp e(bd, ed, 42, 200, c, rc, "normal", false,
                   GAME_OVER_NO_PEOPLE_DAYS, tax_to_debt);
    e.reset(42);
    RunOut o;
    const int farm = find_id(e, "Farm");
    for (int k = 0; k < 4; k++) { auto s = e.step(A_BUILD0 + farm); o.sum += s.rew; o.steps++; }
    o.money = e.game().money;
    for (int k = 0; k < 430; k++) {
        auto s = e.step(A_DAY);
        o.steps++; o.sum += s.rew;
        o.day = s.days; o.money = s.money; o.credit = e.game().credit;
        if (s.terminated) {
            o.terminated = true;
            auto ov = e.game().game_over();
            o.blocker = ov ? ov->reason : "?";
            break;
        }
    }
    return o;
}

int main() {
    auto bd = load_base_data("configs/bases.json");
    auto ed = load_events("configs/events.json");

    printf("=========== A. curriculum JSON со стороны Python ===========\n");
    {
        Curriculum c = Curriculum::from_json(CURRICULUM_V2);
        ColonyEnvCpp e(bd, ed, 42, 200, c, RewardConfig(), "normal", false,
                       GAME_OVER_NO_PEOPLE_DAYS, /*tax_to_debt=*/true);
        e.reset(42);
        check("GUI-watch env accepts Python's curriculum JSON (obs_version=2)",
              c.obs_version == 2 && e.obs_size() == 299,
              "obs_size=" + std::to_string(e.obs_size()));
        check("GUI-watch state.json payload sizes match the policy",
              (int)e.action_mask().size() == 49 && (int)e.obs().size() == 299,
              "mask=" + std::to_string(e.action_mask().size()));
    }
    {
        // Старая модель: watch подставляет v1 → GUI-среда отдаёт 289 чисел,
        // ровно столько же ждёт нормализатор её чекпойнта.
        Curriculum c1 = Curriculum::from_json(CURRICULUM_V1);
        ColonyEnvCpp e1(bd, ed, 42, 200, c1, RewardConfig(), "normal", false,
                        GAME_OVER_NO_PEOPLE_DAYS, /*tax_to_debt=*/true);
        e1.reset(42);
        check("v1-модель наблюдается на 289 числах (state.json ↔ normalizer)",
              c1.obs_version == 1 && e1.obs_size() == 289,
              "obs_size=" + std::to_string(e1.obs_size()));
    }

    printf("\n=========== B/C. налоговая политика GUI-наблюдения ===========\n");
    RunOut dlg = run(/*tax_to_debt=*/false);
    RunOut debt = run(/*tax_to_debt=*/true);
    printf("  tax_to_debt=false (как сейчас в gui.cpp): steps=%d day=%lld money=%lld "
           "credit=%lld sum=%.1f term=%d (%s)\n",
           dlg.steps, (long long)dlg.day, (long long)dlg.money, (long long)dlg.credit,
           dlg.sum, (int)dlg.terminated, dlg.blocker.c_str());
    printf("  tax_to_debt=true  (нужно для watch):      steps=%d day=%lld money=%lld "
           "credit=%lld sum=%.1f term=%d (%s)\n",
           debt.steps, (long long)debt.day, (long long)debt.money, (long long)debt.credit,
           debt.sum, (int)debt.terminated, debt.blocker.c_str());

    check("диалоговая политика замораживает календарь (это и увидит наблюдаемая модель)",
          dlg.day == 365 && dlg.terminated && dlg.credit == 0,
          "day=" + std::to_string((long long)dlg.day));
    check("долговая политика двигает календарь весь прогон",
          debt.day == 434 && debt.credit > 0,
          "day=" + std::to_string((long long)debt.day) +
              " credit=" + std::to_string((long long)debt.credit));
    check("награды в двух режимах расходятся (разные MDP)",
          debt.sum - dlg.sum > 5.0,
          "dlg=" + std::to_string(dlg.sum) + " debt=" + std::to_string(debt.sum));

    printf("\n==================== %d passed, %d failed ====================\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
