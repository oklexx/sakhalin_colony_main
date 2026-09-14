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

    printf("\n%s (%d failure(s))\n",
           failures == 0 ? "ALL CHECKS PASSED" : "CHECKS FAILED", failures);
    return failures == 0 ? 0 : 1;
}
