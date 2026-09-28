// Регресс земельного налога (правка 2026-09-28, docs/PLAN_TAX_PER_CELL.md).
//
// Правила, которые здесь закреплены:
//   T1 налог берётся с КАЖДОЙ занятой клетки — и со здания, и с дороги;
//   T2 выкупленная земля (good_earth) от налога освобождена;
//   T3 выкупить можно и под уже стоящей постройкой;
//   T4 после сноса флаг выкупа сохраняется — земля остаётся оплаченной;
//   T5 консервация (preserve) от налога НЕ освобождает;
//   T6 снос убирает клетку из налога и возвращает 10 % цены;
//   T7 цена выкупа = 10 × NALOG_EARTH (окупается за 10 лет владения).
//
// Сборка (как остальные пробы, см. scripts/cpp_checks.sh):
//   g++ -std=c++17 -O2 -Iinclude -Iinclude/third_party -o tax_cell_check
//       tests/cpp/tax_land_check.cpp src/data.cpp src/resources.cpp
//       src/game.cpp src/rng.cpp src/earth.cpp
#include <cstdio>
#include <string>
#include <utility>
#include <vector>

#include "colony/data.h"
#include "colony/game.h"

using namespace colony;

namespace {

int g_failed = 0;

void check(bool ok, const std::string& name, int64_t got, int64_t want) {
    printf("  %s %-58s got=%lld want=%lld\n", ok ? "OK  " : "FAIL", name.c_str(),
           (long long)got, (long long)want);
    if (!ok) g_failed++;
}

// Налог без оборотных частей: только клетки и фиксированные сборы.
int64_t fixed_fees() { return NALOG_ECOLOGY + NALOG_SOCIAL + NALOG_RES; }

// Первая свободная клетка вокруг центра, на которой получилось построить.
std::pair<int, int> build_near(Game& g, const char* id, int cx, int cy) {
    for (int r = 1; r <= 14; r++)
        for (int dx = -r; dx <= r; dx++)
            for (int dy = -r; dy <= r; dy++)
                if (g.build(id, cx + dx, cy + dy).first)
                    return {cx + dx, cy + dy};
    return {-1, -1};
}

}  // namespace

int main() {
    auto bd = load_base_data("configs/bases.json");
    auto ed = load_events("configs/events.json");
    Game g(bd, ed, 42, 280);
    g.set_enable_undo(false);
    g.money = 5000000;  // касса не должна мешать проверке правил
    const int cx = 140, cy = 140;

    printf("T7 цена выкупа\n");
    check(BUYGOODEARTH == 10 * NALOG_EARTH, "BUYGOODEARTH == 10 × NALOG_EARTH",
          BUYGOODEARTH, 10 * NALOG_EARTH);

    printf("T1 налог с каждой занятой клетки (здания и дороги)\n");
    const int64_t base_tax = g.annual_tax_amount();  // только Город
    check(base_tax == 1 * NALOG_EARTH + fixed_fees(), "старт: 1 клетка Города",
          base_tax, 1 * NALOG_EARTH + fixed_fees());

    auto house = build_near(g, "SmallHouse", cx, cy);
    check(g.annual_tax_amount() == base_tax + NALOG_EARTH, "+хижина → +420",
          g.annual_tax_amount(), base_tax + NALOG_EARTH);

    int64_t before_roads = g.annual_tax_amount();
    int roads = 0;
    std::vector<std::pair<int, int>> road_cells;
    for (int r = 1; r <= 14 && roads < 5; r++)
        for (int dx = -r; dx <= r && roads < 5; dx++)
            for (int dy = -r; dy <= r && roads < 5; dy++) {
                int x = cx + dx, y = cy + dy;
                if (g.build("Road", x, y).first) { road_cells.push_back({x, y}); roads++; }
            }
    check(roads == 5 && g.annual_tax_amount() == before_roads + 5 * NALOG_EARTH,
          "+5 дорог → +2100 (раньше дороги были бесплатны)",
          g.annual_tax_amount(), before_roads + 5 * NALOG_EARTH);

    printf("T2/T3 выкуп земли снимает налог, в том числе под постройкой\n");
    int64_t before_buy = g.annual_tax_amount();
    auto buy = g.good_earth(house.first, house.second);
    check(buy.first, "выкуп под стоящей хижиной разрешён", buy.first ? 1 : 0, 1);
    check(g.annual_tax_amount() == before_buy - NALOG_EARTH,
          "клетка хижины больше не облагается", g.annual_tax_amount(),
          before_buy - NALOG_EARTH);

    int64_t before_road_buy = g.annual_tax_amount();
    auto rbuy = g.good_earth(road_cells[0].first, road_cells[0].second);
    check(rbuy.first && g.annual_tax_amount() == before_road_buy - NALOG_EARTH,
          "выкуп под дорогой тоже снимает налог", g.annual_tax_amount(),
          before_road_buy - NALOG_EARTH);

    auto again = g.good_earth(house.first, house.second);
    check(!again.first, "повторный выкуп той же клетки запрещён",
          again.first ? 1 : 0, 0);

    printf("T5 консервация от налога не освобождает\n");
    auto free_house = build_near(g, "SmallHouse", cx, cy);
    int64_t before_preserve = g.annual_tax_amount();
    g.preserve(free_house.first, free_house.second);
    check(g.annual_tax_amount() == before_preserve, "налог не изменился",
          g.annual_tax_amount(), before_preserve);
    g.preserve(free_house.first, free_house.second);

    printf("T6 снос убирает клетку из налога и возвращает 10 %% цены\n");
    int64_t money_before = g.money, tax_before = g.annual_tax_amount();
    g.destroy(free_house.first, free_house.second);
    check(g.annual_tax_amount() == tax_before - NALOG_EARTH, "−420 после сноса",
          g.annual_tax_amount(), tax_before - NALOG_EARTH);
    check(g.money - money_before == 5900 / 10, "возврат 10 % цены хижины",
          g.money - money_before, 5900 / 10);

    printf("T4 после сноса выкупленная земля остаётся оплаченной\n");
    int64_t tax_with_bought = g.annual_tax_amount();
    g.destroy(house.first, house.second);  // сносим хижину на выкупленной земле
    check(g.annual_tax_amount() == tax_with_bought, "снос на выкупленной земле: налог тот же",
          g.annual_tax_amount(), tax_with_bought);
    check(g.is_good(house.first, house.second), "флаг выкупа сохранён",
          g.is_good(house.first, house.second) ? 1 : 0, 1);
    // Сразу на том же месте строить нельзя (delete_base метит клетку как
    // «сгоревшую» на 12 дней), поэтому проверяем на другой выкупленной клетке:
    // выкупаем свободную землю заранее и строим на ней.
    const BaseData* small = nullptr;
    for (const BaseData& d : bd)
        if (d.id == "SmallHouse") small = &d;
    int fx = -1, fy = -1;
    for (int r = 1; r <= 14 && fx < 0; r++)
        for (int dx = -r; dx <= r && fx < 0; dx++)
            for (int dy = -r; dy <= r && fx < 0; dy++) {
                int x = cx + dx, y = cy + dy;
                if (g.is_good(x, y)) continue;
                if (!g.can_build_at(*small, x, y).first) continue;  // не «сгоревшая»
                if (!g.good_earth(x, y).first) continue;
                fx = x; fy = y;  // выкупили свободную клетку рядом с колонией
            }
    int64_t tax_before_new = g.annual_tax_amount();
    auto fresh = g.build("SmallHouse", fx, fy);
    if (!fresh.first) printf("       (стройка не удалась: %s)\n", fresh.second.c_str());
    check(fresh.first && g.annual_tax_amount() == tax_before_new,
          "постройка на заранее выкупленной клетке налогом не облагается",
          g.annual_tax_amount(), tax_before_new);

    printf(g_failed ? "\nПРОВАЛЕНО проверок: %d\n" : "\nвсе проверки пройдены (%d)\n",
           g_failed);
    return g_failed ? 1 : 0;
}
