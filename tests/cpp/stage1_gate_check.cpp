// Гейтинг всех 11 менеджерских слотов через 8 «механик» (2026-09-21,
// docs/TWO_STAGE_TRAINING_2026_09.md). Пресет «Стадия 1: база и ресурсы»
// держит среду в минимальном режиме: DAY/WEEK/BUILD + выбранные механики.
//
// Контракт (таблица в constants.h, транспорт в rl/curriculum.py):
//   MECHANIC_NAMES  — 8 имён, порядок = индекс в enabled_mechanics;
//   MANAGER_MECHANIC — слот менеджера → механика (11 записей);
//   легаси-имена improve_land/preservation/credit остаются валидными
//   (старые --curriculum JSON и мета-файлы обязаны парситься).
//
// Сборка (из корня репозитория, как у curriculum_check.cpp):
//   g++ -std=c++17 -O1 -Iinclude -Iinclude/third_party -o /tmp/stage1
//       tests/cpp/stage1_gate_check.cpp
//       build/cpp_checks/{env,data,resources,game,rewards,rng,earth}.o
//   /tmp/stage1
#include "colony/env.h"
#include "colony/data.h"
#include <algorithm>
#include <cstdio>
#include <set>
#include <string>
#include <vector>

using namespace colony;

static int failures = 0;
static void check(bool ok, const std::string& what) {
    printf("%s  %s\n", ok ? "[ok]  " : "[FAIL]", what.c_str());
    if (!ok) failures++;
}

static int manager_action(const ColonyEnvCpp& e, int slot) {
    return A_BUILD0 + e.n_build() + slot;
}

// Маска менеджерских слотов (только сам факт гейта: применимость выключена,
// чтобы applicability-маскирование не путало картину).
static std::vector<int> manager_bits(ColonyEnvCpp& e) {
    std::vector<int> out;
    auto mask = e.action_mask();
    for (int i = 0; i < N_MANAGERS; i++)
        out.push_back(mask[manager_action(e, i)] != 0.0f ? 1 : 0);
    return out;
}

static Curriculum mechanics_off() {
    Curriculum c;  // all_builds=true по умолчанию
    c.enabled_mechanics.fill(false);
    return c;
}

static Curriculum mechanics_only(const std::string& name) {
    Curriculum c = mechanics_off();
    const int mi = mechanic_index(name);
    c.enabled_mechanics[(size_t)mi] = true;
    return c;
}

int main() {
    auto bd = load_base_data("configs/bases.json");
    auto ed = load_events("configs/events.json");
    RewardConfig rc;  // дефолты наград
    rc.mask_managers_by_applicability = false;  // изолируем гейт от применимости

    // ── 1. Таблицы консистентны ────────────────────────────────────────────
    check(N_MECHANICS == 8, "N_MECHANICS == 8");
    {
        std::set<std::string> uniq;
        bool all_unique = true;
        for (int i = 0; i < N_MECHANICS; i++)
            if (!uniq.insert(MECHANIC_NAMES[i]).second) all_unique = false;
        check(all_unique, "MECHANIC_NAMES уникальны");
        bool in_range = true;
        for (int i = 0; i < N_MANAGERS; i++)
            if (MANAGER_MECHANIC[i] < 0 || MANAGER_MECHANIC[i] >= N_MECHANICS)
                in_range = false;
        check(in_range, "MANAGER_MECHANIC покрывает все 11 слотов (индексы в диапазоне)");
        // Легаси-имена остаются валидными и указывают на свои механики.
        check(mechanic_index("improve_land") == 0, "легаси improve_land -> индекс 0");
        check(mechanic_index("preservation") == 3, "легаси preservation -> индекс 3");
        check(mechanic_index("credit") == 6, "легаси credit -> индекс 6");
        check(mechanic_index("no_such_mechanic") == -1, "неизвестное имя -> -1");
    }

    // ── 2. Пустой allow-list закрывает все 11 менеджеров, остальное живёт ──
    {
        ColonyEnvCpp base(bd, ed, 42, 280);
        base.reset(42);
        base.set_rewards(rc);
        auto mask_before = base.action_mask();

        ColonyEnvCpp e(bd, ed, 42, 280, mechanics_off());
        e.reset(42);
        e.set_rewards(rc);
        auto bits = manager_bits(e);
        check(std::all_of(bits.begin(), bits.end(), [](int b) { return b == 0; }),
              "mechanics={} маскирует все 11 менеджерских слотов");
        auto mask_now = e.action_mask();
        check(mask_now[A_DAY] == 1.0f && mask_now[A_WEEK] == 1.0f,
              "DAY/WEEK остаются доступны");
        bool builds_same = true;
        for (int i = 0; i < e.n_build(); i++)
            if ((mask_before[A_BUILD0 + i] != 0.0f) != (mask_now[A_BUILD0 + i] != 0.0f))
                builds_same = false;
        check(builds_same, "BUILD-биты не тронуты гейтом механик");
    }

    // ── 3. Выборочное включение открывает ровно свои слоты ─────────────────
    {
        ColonyEnvCpp e(bd, ed, 42, 280, mechanics_only("repair"));
        e.reset(42);
        e.set_rewards(rc);
        auto bits = manager_bits(e);
        // repair = слоты 1 (MGR:repair) и 2 (MGR:restore_all)
        check(bits[1] == 1 && bits[2] == 1, "repair открывает MGR:repair");
        check(bits[1] == 1 && bits[2] == 1 && std::count(bits.begin(), bits.end(), 1) == 2,
              "открыты ровно 2 слота repair/restore_all, остальные закрыты");
    }
    {
        ColonyEnvCpp e(bd, ed, 42, 280, mechanics_only("credit"));
        e.reset(42);
        e.set_rewards(rc);
        auto bits = manager_bits(e);
        check(bits[8] == 1 && bits[9] == 1 &&
              std::count(bits.begin(), bits.end(), 1) == 2,
              "credit открывает ровно credit_take/credit_give");
    }

    // ── 4. step() в закрытый слот: чистый error_penalty, календарь стоит ───
    {
        ColonyEnvCpp e(bd, ed, 42, 280, mechanics_off());
        e.reset(42);
        e.set_rewards(rc);
        const int64_t day0 = e.game().days_alive;
        const int sell_slot = 6;  // MGR:sell — легален по деньгам/складу, но закрыт гейтом
        auto out = e.step(manager_action(e, sell_slot));
        check(out.rew == rc.error_penalty,
              "закрытый слот стоит ровно error_penalty (без дневных бонусов)");
        check(e.game().days_alive == day0,
              "календарь не двигается на закрытом слоте");
        auto bits = manager_bits(e);
        check(bits[sell_slot] == 0, "маска закрытого слота осталась нулевой после step");
        // DAY по-прежнему проживает день.
        auto d = e.step(A_DAY);
        (void)d;
        check(e.game().days_alive == day0 + 1, "DAY проживает день после отказа");
    }

    // ── 5. Re-lock запрещён, additive-unlock разрешён ─────────────────────
    {
        ColonyEnvCpp e(bd, ed, 42, 280);  // дефолт: все механики включены
        e.reset(42);
        e.set_rewards(rc);
        bool threw = false;
        try {
            e.set_curriculum(mechanics_off());  // попытка закрыть всё
        } catch (const std::runtime_error&) {
            threw = true;
        }
        check(threw, "set_curriculum отказывается закрывать включённые механики");
    }
    {
        ColonyEnvCpp e(bd, ed, 42, 280, mechanics_off());
        e.reset(42);
        e.set_rewards(rc);
        bool ok = true;
        try {
            Curriculum c = mechanics_only("sell");  // additive: добавили sell
            e.set_curriculum(c);
        } catch (const std::runtime_error&) {
            ok = false;
        }
        check(ok, "set_curriculum разрешает additive-unlock (пусто -> sell)");
        auto bits = manager_bits(e);
        check(bits[6] == 1, "после unlock слот sell открыт");
    }

    // ── 6. from_json: легаси-имена, all, пусто, мусор ──────────────────────
    {
        auto c = Curriculum::from_json(R"({"enabled_mechanics":["all"]})");
        bool all_on = true;
        for (int i = 0; i < N_MECHANICS; i++) all_on = all_on && c.enabled_mechanics[(size_t)i];
        check(all_on, "from_json: [\"all\"] включает все механики");

        // Легаси-JSON (старые watch/GUI-прогоны): три старых имени.
        auto l = Curriculum::from_json(
            R"({"enabled_mechanics":["improve_land","preservation","credit"]})");
        check(l.enabled_mechanics[0] && l.enabled_mechanics[3] && l.enabled_mechanics[6] &&
                  std::count(l.enabled_mechanics.begin(), l.enabled_mechanics.end(), true) == 3,
              "from_json: легаси-тройка парсится в свои механики");

        auto none = Curriculum::from_json(R"({"enabled_mechanics":[]})");
        bool all_off = true;
        for (int i = 0; i < N_MECHANICS; i++) all_off = all_off && !none.enabled_mechanics[(size_t)i];
        check(all_off, "from_json: [] = все механики выключены");

        bool threw = false;
        try {
            Curriculum::from_json(R"({"enabled_mechanics":["teleport"]})");
        } catch (const std::runtime_error&) {
            threw = true;
        }
        check(threw, "from_json: неизвестное имя -> runtime_error");

        auto absent = Curriculum::from_json("{}");
        bool legacy_on = true;
        for (int i = 0; i < N_MECHANICS; i++)
            legacy_on = legacy_on && absent.enabled_mechanics[(size_t)i];
        check(legacy_on, "from_json: поле отсутствует -> легаси/все включены");
    }

    printf("\n%s: %d failure(s)\n", failures ? "FAIL" : "OK", failures);
    return failures ? 1 : 0;
}
