// Standalone harness for the curriculum fix (no Python, no raylib needed).
//
// Reproduces the reported scenario «Поставил только водоканал, курикулум 0,
// чекбокс активен — а он строит прииск» and the stage-switch regression that
// caused it: ColonyEnvCpp::set_curriculum_stage() used to wipe the manual
// unlock set, so after any stage change the env allowed ALL 32 buildings.
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

int main() {
    auto bd = load_base_data("configs/bases.json");
    auto ed = load_events("configs/events.json");

    // ── baseline: what this map allows without any curriculum restriction ──
    // 5M cash removes affordability from the equation, so the mask only reflects
    // curriculum + terrain/lot rules.
    ColonyEnvCpp a(bd, ed, 42, 280, 0, {}, RewardConfig(), "normal");
    a.reset(42);
    a.game().money = 5'000'000;
    std::vector<int> base_allowed = allowed_build_actions(a);
    printf("unrestricted env (5M cash): %d/%d build actions available\n",
           (int)base_allowed.size(), a.n_build());
    check(base_allowed.size() >= 2, "map offers >=2 buildable types");

    int first = base_allowed.front();
    const std::string first_id = a.build_ids()[first];
    printf("demonstrating with a placeable id: '%s'\n", first_id.c_str());

    // ── scenario from the report: one id selected, stage 0, checkbox ON ──
    ColonyEnvCpp b(bd, ed, 42, 280, 0, {first_id}, RewardConfig(), "normal");
    b.reset(42);
    b.game().money = 5'000'000;                 // same rich colony as the baseline
    auto ids_b = allowed_build_ids(b);
    check(ids_b.size() == 1 && ids_b[0] == first_id,
          "manual-only env allows exactly the selected id (no other building)");
    check(!allowed(b, find_id(b, "Goldmine")), "прииск (Goldmine) is masked");

    // ── regression: a stage switch must not drop the manual set ──
    b.set_curriculum_stage(0);
    auto ids_after = allowed_build_ids(b);
    check(ids_after == ids_b, "set_curriculum_stage(0) keeps the manual-only set");
    check(b.unlock_ids() == std::vector<std::string>{first_id},
          "env still stores the manual unlock_ids after the stage switch");

    // stage 2 = stage presets ∪ manual set; back to 0 → manual-only again
    b.set_curriculum_stage(2);
    check(allowed(b, first), "stage 2 keeps the manual id");
    check((int)allowed_build_actions(b).size() > (int)ids_b.size(),
          "stage 2 preset adds more buildings on top of the manual set");
    b.set_curriculum_stage(0);
    auto ids_back = allowed_build_ids(b);
    check(ids_back.size() == 1 && ids_back[0] == first_id,
          "stage 2 -> 0 restores the manual-only set");

    // ── set_unlock_ids() replaces the manual set ──
    if (base_allowed.size() >= 2) {
        const std::string second_id = a.build_ids()[base_allowed[1]];
        b.set_unlock_ids({second_id});
        check(allowed(b, find_id(b, second_id)), "set_unlock_ids('" + second_id + "') unlocks it");
        check(!allowed(b, find_id(b, first_id)), "previous manual id is locked again");
        check(b.unlock_ids() == std::vector<std::string>{second_id}, "unlock_ids() reflects the change");

        b.set_unlock_ids({first_id, second_id});
        check(allowed(b, find_id(b, first_id)) && allowed(b, find_id(b, second_id)),
              "two-id manual set allows both");
    }

    // ── a masked build action never actually builds anything ──
    ColonyEnvCpp c(bd, ed, 7, 280, 0, {first_id}, RewardConfig(), "normal");
    c.reset(7);
    int gold = find_id(c, "Goldmine");
    if (gold >= 0) {
        int before = (int)c.game().bases.size();
        for (int i = 0; i < 60; i++) c.step(A_BUILD0 + gold);
        check((int)c.game().bases.size() == before,
              "locked Goldmine action is refused by the env (nothing built)");
    }

    printf("\n%s (%d failure(s))\n",
           failures == 0 ? "ALL CHECKS PASSED" : "CHECKS FAILED", failures);
    return failures == 0 ? 0 : 1;
}
