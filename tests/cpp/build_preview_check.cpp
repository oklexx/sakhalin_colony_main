// Проверка: подсветка «можно/нельзя строить» в игровом окне совпадает с
// вердиктом движка.
//
// gui.cpp не компилируется без raylib/Windows, поэтому логика предпросмотра
// вынесена в include/colony/build_preview.h — здесь она сравнивается с
// Game::can_build_at() по ВСЕМ клеткам карты для нескольких типов построек и
// нескольких состояний колонии (пустой старт, дорога-ветка, снесённые
// участки). Расхождение = игрок видит зелёную клетку, а постройка не встаёт.

#include <cstdio>
#include <string>
#include <vector>

#include "colony/build_preview.h"
#include "colony/data.h"
#include "colony/env.h"

using namespace colony;

static int failures = 0;

static void check_map(const Game& g, const std::vector<const BaseData*>& bds,
                      const char* phase) {
    std::vector<char> conn;
    preview::connectivity_map(g, conn);
    const int ms = g.map_size();
    for (const BaseData* bd : bds) {
        int mismatches = 0;
        int first_x = -1, first_y = -1;
        std::string eng, pre;
        for (int y = 0; y < ms; y++) {
            for (int x = 0; x < ms; x++) {
                const bool engine_ok = g.can_build_at(*bd, x, y).first;
                const bool prev_ok = preview::ok(g, *bd, x, y, conn);
                if (engine_ok != prev_ok) {
                    if (mismatches == 0) {
                        first_x = x; first_y = y;
                        eng = g.can_build_at(*bd, x, y).second;
                        pre = preview::reason(g, *bd, x, y, conn);
                    }
                    mismatches++;
                }
            }
        }
        if (mismatches) {
            failures++;
            printf("  [FAIL] %-14s %-12s расхождений: %d, первое (%d,%d)\n"
                   "         движок: '%s'\n         подсветка: '%s'\n",
                   phase, bd->id.c_str(), mismatches, first_x, first_y,
                   eng.c_str(), pre.c_str());
        } else {
            printf("  [ ok ] %-14s %-12s 1:1 с can_build_at по всем %d клеткам\n",
                   phase, bd->id.c_str(), ms * ms);
        }
    }
}

int main() {
    auto bd = load_base_data("configs/bases.json");
    auto ed = load_events("configs/events.json");
    const std::vector<std::string> want = {"Road", "House", "Farm", "Fish", "AtomStation"};

    for (int seed : {1, 7, 42}) {
        ColonyEnvCpp env(bd, ed, seed, 120);
        env.reset(seed);
        Game& g = env.game();
        std::vector<const BaseData*> bds;
        for (const std::string& id : want)
            for (const BaseData& d : bd)
                if (d.id == id) bds.push_back(&d);

        printf("seed=%d, карта %dx%d, старт в (%d,%d)\n", seed, g.map_size(),
               g.map_size(), g.earth.init_sel_x, g.earth.init_sel_y);
        check_map(g, bds, "старт");

        // Ветка дорог от города + пара зданий: появляются «дальние» клетки,
        // связанные с колонией только цепочкой дорог.
        int cx = g.earth.init_sel_x, cy = g.earth.init_sel_y;
        for (int i = 1; i <= 8; i++) g.build("Road", cx + i, cy);
        for (int i = 1; i <= 5; i++) g.build("Road", cx + 8, cy - i);
        g.build("House", cx + 8, cy - 6);
        g.build("Farm", cx + 1, cy + 1);
        check_map(g, bds, "дороги");

        // Снесённые/сгоревшие участки.
        g.destroyed_lots[(size_t)cy * g.map_size() + cx + 3] = 5;
        g.destroyed_lots[(size_t)(cy + 1) * g.map_size() + cx + 4] = 2;
        check_map(g, bds, "пожарище");
    }

    if (failures) {
        printf("ПРОВАЛ: расхождений предпросмотра — %d\n", failures);
        return 1;
    }
    printf("OK: предпросмотр застройки совпадает с движком\n");
    return 0;
}
