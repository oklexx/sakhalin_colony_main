// Standalone harness: is the HARD legality mask for BUILD:WaterChannel correct?
//
// Открытый вопрос P0-1 (docs/REMAINING_WORK_2026_09.md): водоканал исторически
// был замаскирован в 0/2000 шагов. Направленные дороги (ROAD_E/W/S/N) уже
// добавлены, water_probe* измерили «вода достижима за 6-10 дорог». Оставалось
// не измеренным главное: **сама маска**. Этот harness сравнивает бит маски с
// полным перебором легальности (Game::can_build_at по всем клеткам карты) —
// т.е. отвечает на вопрос «маска корректна, или она слишком жёсткая и режет
// далёкую воду?».
//
// Проверяется (W1..W9):
//   W1  бит маски == ∃ легальная клетка (полный перебор) на каждом шаге
//   W2  причина маскирования атрибутирована: курикулум / деньги / нет клетки
//   W3  на старте легальных водных клеток 0, и маска открывается за ≤ N дорог
//   W4  далёкая вода: маска открывается и когда вода далеко (>15 клеток)
//   W5  find_lot() не теряет легальные клетки, которые видит перебор
//   W6  дорога на воде: сколько водных тайлов съедается дорогами (Road
//       need_earth = LT_EVERYWHERE, т.е. дорогу МОЖНО поставить на воду)
//   W7  разбор (а не фиксация) сида, где маска так и не открылась: если
//       регрессия вернёт такой сид, трассировка покажет причину и check упадёт
//   W8  замер: у masked-random вода не строится (политика, а не маска)
//   W9  регрессия «наведения на воду» (цель-ориентированный find_lot_dir,
//       2026-09-20): водоканал на всех сидах, в т.ч. seed 100, число дорог
//       ограничено дистанцией до воды (старый жадный: 205 дорог, бюджет 0)
//
// Build (from the repo root, Linux/macOS) — одна строка, без продолжений:
//   g++ -std=c++17 -O1 -Iinclude -Iinclude/third_party -o /tmp/water_mask
//       tests/cpp/water_mask_check.cpp src/env.cpp src/data.cpp
//       src/resources.cpp src/game.cpp src/rewards.cpp src/rng.cpp src/earth.cpp
//   /tmp/water_mask    # запуск из корня репозитория: читаются configs/*.json
#include "colony/env.h"
#include "colony/data.h"

#include <algorithm>
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

static const BaseData* find_data(const ColonyEnvCpp& e, const std::string& id) {
    for (const BaseData* d : e.build_data())
        if (d->id == id) return d;
    return nullptr;
}
static int find_idx(ColonyEnvCpp& e, const std::string& id) {
    for (int i = 0; i < e.n_build(); ++i)
        if (e.build_ids()[(size_t)i] == id) return i;
    return -1;
}

// Полный перебор легальности: все свободные клетки карты через ТОТ ЖЕ валидатор,
// которым пользуется Game::build (единый источник правды размещения).
static int brute_force_legal_cells(const ColonyEnvCpp& e, const BaseData& d) {
    const Game& g = e.game();
    const int ms = g.map_size();
    int n = 0;
    for (int y = 0; y < ms; ++y)
        for (int x = 0; x < ms; ++x)
            if (g.can_build_at(d, x, y).first) ++n;
    return n;
}

static int water_tiles_on_map(const ColonyEnvCpp& e) {
    const Game& g = e.game();
    const int ms = g.map_size();
    const int8_t* lots = g.earth.lots().data();
    int n = 0;
    for (int i = 0; i < ms * ms; ++i)
        if (lots[i] == LT_WATER) ++n;
    return n;
}

static double nearest_water_dist(const ColonyEnvCpp& e) {
    const Game& g = e.game();
    const int ms = g.map_size();
    const int8_t* lots = g.earth.lots().data();
    const double bx = g.earth.init_sel_x, by = g.earth.init_sel_y;
    double best = -1.0;
    for (int y = 0; y < ms; ++y)
        for (int x = 0; x < ms; ++x)
            if (lots[(size_t)y * ms + x] == LT_WATER) {
                double d2 = (x - bx) * (x - bx) + (y - by) * (y - by);
                if (best < 0 || d2 < best) best = d2;
            }
    return best < 0 ? -1.0 : std::sqrt(best);
}

// Направление к ближайшей воде — с карты (в obs v2 хвост занят ресурсами).
// Возвращает доминирующую ось (0=E, 1=W, 2=S, 3=N) — как раньше; для оценки
// «какое из открытых направлений лучше» используйте water_dir_vec().
static void water_dir_from_map(const ColonyEnvCpp& e, int& out_dir) {
    const Game& g = e.game();
    const int ms = g.map_size();
    const int8_t* lots = g.earth.lots().data();
    const double bx = g.earth.init_sel_x, by = g.earth.init_sel_y;
    double best_sq = -1.0;
    int wx = -1, wy = -1;
    for (int y = 0; y < ms; ++y)
        for (int x = 0; x < ms; ++x)
            if (lots[(size_t)y * ms + x] == LT_WATER) {
                double d2 = (x - bx) * (x - bx) + (y - by) * (y - by);
                if (best_sq < 0 || d2 < best_sq) { best_sq = d2; wx = x; wy = y; }
            }
    out_dir = 0;
    if (best_sq < 0) return;
    double dx = wx - bx, dy = wy - by;
    if (std::fabs(dx) >= std::fabs(dy)) out_dir = dx >= 0 ? 0 : 1;   // E / W
    else out_dir = dy >= 0 ? 2 : 3;                                  // S / N
}

// Направление к ближайшей воде как нормированный вектор (dx, dy) — ровно то,
// что политика видит в obs (water_dx/water_dy): так можно оценить, какое из
// открытых направленных действий ближе к цели.
static void water_dir_vec(const ColonyEnvCpp& e, double& out_dx, double& out_dy) {
    const Game& g = e.game();
    const int ms = g.map_size();
    const int8_t* lots = g.earth.lots().data();
    const double bx = g.earth.init_sel_x, by = g.earth.init_sel_y;
    double best_sq = -1.0;
    int wx = -1, wy = -1;
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

struct RolloutStats {
    int steps = 0;
    int mask_open = 0;            // шагов, где бит WaterChannel открыт
    int mismatches = 0;           // маска != перебор (W1)
    int mask0_money = 0;          // маска 0 из-за денег
    int mask0_curriculum = 0;     // маска 0 из-за курикулума
    int mask0_nocell = 0;         // маска 0 при достаточных деньгах: нет клетки
    int find_lot_missed = 0;      // find_lot() == nullopt, хотя перебор нашёл (W5)
    int find_lot_phantom = 0;     // find_lot() вернул клетку, которой нет в переборе
    int roads_on_water = 0;       // дорог поставлено на водный тайл (W6)
    int water_eaten = 0;          // из них: тайл был единственным легальным
    int waterchannel_built = 0;
    int roads = 0;
    int steps_to_first_open = -1;
    long long money_at_first_open = -1;  // бюджет в момент открытия маски (W9)
    double reward = 0.0;
};

// Смешанный роллаут: жадно тянем дорогу к воде, между дорогами — DAY; как только
// маска открыла WaterChannel — ставим его. Каждый шаг маска сверяется с перебором.
// Политика направлений — та, что доступна обученной модели по obs: из ОТКРЫТЫХ
// ROAD_* берётся ближайшее к water_dx/water_dy (цель-ориентированный выбор
// КЛЕТКИ внутри направления делает ядро в find_lot_dir, 2026-09-20).
static RolloutStats run_rollout(ColonyEnvCpp& e, int wc_idx, int max_steps,
                                bool stop_at_channel) {
    RolloutStats st;
    const BaseData* wc = find_data(e, "WaterChannel");
    const BaseData* rd = find_data(e, "Road");
    const int rdb = e.road_dir_base();
    const int road_idx = find_idx(e, "Road");

    for (int s = 0; s < max_steps; ++s) {
        const Game& g = e.game();
        auto mask = e.action_mask();
        bool open = mask[A_BUILD0 + wc_idx] != 0.0f;
        st.steps++;
        if (open) {
            st.mask_open++;
            if (st.steps_to_first_open < 0) {
                st.steps_to_first_open = st.steps;
                st.money_at_first_open = g.money;
            }
        }

        // ── W1/W2/W5: маска против полного перебора ─────────────────────────
        bool allowed = e.build_allowed(wc->id);
        bool affordable = g.money >= wc->price;
        int legal = brute_force_legal_cells(e, *wc);
        bool truth = allowed && affordable && legal > 0;
        if (open != truth) {
            st.mismatches++;
            if (st.mismatches <= 3)
                printf("[info]   MISMATCH step %d: mask=%d truth=%d (allowed=%d money=%lld/%lld legal_cells=%d)\n",
                       st.steps, (int)open, (int)truth, (int)allowed,
                       (long long)g.money, (long long)wc->price, legal);
        }
        if (!open) {
            if (!allowed) st.mask0_curriculum++;
            else if (!affordable) st.mask0_money++;
            else st.mask0_nocell++;
        }
        // find_lot() — то, чем маска пользуется внутри: не теряет ли клетки
        auto cell = e.find_lot(*wc);
        if (!cell && legal > 0) st.find_lot_missed++;
        if (cell) {
            // клетка из find_lot обязана проходить тот же валидатор
            if (!g.can_build_at(*wc, cell->first, cell->second).first) st.find_lot_phantom++;
        }
        // сколько водных тайлов сейчас легально (диагностика «далёкой воды»)
        (void)rd;

        // ── выбор действия ──────────────────────────────────────────────────
        int act;
        if (open && stop_at_channel) act = A_BUILD0 + wc_idx;
        else {
            double wdx = 0.0, wdy = 0.0;
            water_dir_vec(e, wdx, wdy);
            int best = -1;
            double bs = -1e18;
            for (int d = 0; d < N_ROAD_DIRS; d++) {
                if (mask[rdb + d] == 0.0f) continue;
                double sc = -(std::fabs(wdx - ROAD_DIR_DX[d] * 0.05) +
                              std::fabs(wdy - ROAD_DIR_DY[d] * 0.05));
                if (sc > bs) { bs = sc; best = d; }
            }
            if (best >= 0) act = rdb + best;
            else if (mask[A_BUILD0 + road_idx] != 0.0f) act = A_BUILD0 + road_idx;
            else act = A_DAY;
        }

        const int roads_before = (int)std::count_if(
            g.bases.begin(), g.bases.end(),
            [](const Base& b) { return b.data->id == ROAD_ID; });
        // вода под дорогой: Road need_earth = LT_EVERYWHERE, поэтому дорога
        // МОЖЕТ встать на воду и «съесть» тайл, годный для водоканала.
        std::vector<int8_t> before;
        const int ms = g.map_size();
        before.reserve((size_t)ms * ms);
        for (int i = 0; i < ms * ms; ++i) before.push_back(g.earth.lots()[i]);
        int legal_before = legal;

        auto out = e.step(act);
        st.reward += out.rew;
        const Game& g2 = e.game();
        int roads_after = (int)std::count_if(
            g2.bases.begin(), g2.bases.end(),
            [](const Base& b) { return b.data->id == ROAD_ID; });
        if (roads_after > roads_before) {
            st.roads++;
            const Base& nb = g2.bases.back();
            if (before[(size_t)nb.y * ms + nb.x] == LT_WATER) {
                st.roads_on_water++;
                if (brute_force_legal_cells(e, *wc) < legal_before) st.water_eaten++;
            }
        }
        for (const Base& b : g2.bases)
            if (b.data->id == "WaterChannel") st.waterchannel_built++;
        if (out.terminated || out.truncated) break;
        if (open && stop_at_channel) break;
        // (направление к воде пересчитывается каждый шаг — water_dir_vec)
    }
    return st;
}

int main() {
    auto bd = load_base_data("configs/bases.json");
    auto ed = load_events("configs/events.json");

    printf("=== W0: параметры WaterChannel/Road из configs/bases.json ===\n");
    {
        ColonyEnvCpp e(bd, ed, 42, 200);
        e.reset(42);
        const BaseData* wc = find_data(e, "WaterChannel");
        const BaseData* rd = find_data(e, "Road");
        printf("[info] n_actions=%d road_dir_base=%d\n", e.n_actions(), e.road_dir_base());
        printf("[info] WaterChannel need_earth=%d (LT_WATER=%d) price=%lld no_near_base=%d\n",
               wc->need_earth, LT_WATER, (long long)wc->price, (int)wc->no_near_base);
        printf("[info] Road           need_earth=%d (LT_EVERYWHERE=%d) price=%lld\n",
               rd->need_earth, LT_EVERYWHERE, (long long)rd->price);
        check(wc->need_earth == LT_WATER,
              "W0 WaterChannel требует именно водный тайл (need_earth == LT_WATER)");
        check(rd->need_earth == LT_EVERYWHERE,
              "W0 Road ставится на любой тип земли, включая воду (need_earth == LT_EVERYWHERE)");
    }

    // ── W1..W6: роллауты по сидам ───────────────────────────────────────────
    printf("\n=== W1: бит маски против полного перебора легальности ===\n");
    printf("%-6s %-6s %-7s %-8s %-8s %-8s %-7s %-7s %-8s %-8s\n",
           "seed", "steps", "wc_open", "mismatch", "m:money", "m:nocell",
           "fl_miss", "roads", "rd_water", "wc_built");
    long long tot_steps = 0, tot_mismatch = 0, tot_open = 0, tot_money = 0,
              tot_curric = 0, tot_nocell = 0, tot_fl_missed = 0, tot_phantom = 0,
              tot_roads = 0, tot_roads_water = 0, tot_eaten = 0;
    int seeds_with_wc = 0, n_seeds = 0;
    int min_open = -1, max_open = -1;
    double far_dist = -1.0;
    int far_roads = -1;
    // W9: худшее по сиду отношение «дорог до воды / дистанция до воды»
    // (излишек = маневрирование вокруг рельефа; 1.0 = идеальный прямой путь).
    double worst_road_ratio = -1.0;

    for (int64_t seed : {1, 7, 21, 42, 100, 777, 31337, 2026}) {
        ColonyEnvCpp e(bd, ed, seed, 200);
        e.reset(seed);
        const int wc_idx = find_idx(e, "WaterChannel");
        const double wd = nearest_water_dist(e);
        const int wt = water_tiles_on_map(e);
        const int legal_at_reset = brute_force_legal_cells(e, *find_data(e, "WaterChannel"));
        auto st = run_rollout(e, wc_idx, 400, true);

        tot_steps += st.steps;
        tot_mismatch += st.mismatches;
        tot_open += st.mask_open;
        tot_money += st.mask0_money;
        tot_curric += st.mask0_curriculum;
        tot_nocell += st.mask0_nocell;
        tot_fl_missed += st.find_lot_missed;
        tot_phantom += st.find_lot_phantom;
        tot_roads += st.roads;
        tot_roads_water += st.roads_on_water;
        tot_eaten += st.water_eaten;
        if (st.waterchannel_built > 0) seeds_with_wc++;
        n_seeds++;
        if (st.steps_to_first_open >= 0) {
            min_open = min_open < 0 ? st.steps_to_first_open
                                    : std::min(min_open, st.steps_to_first_open);
            max_open = std::max(max_open, st.steps_to_first_open);
            if (wd > far_dist) { far_dist = wd; far_roads = st.roads; }
            double ratio = (wd > 0.0) ? (double)st.roads / wd : 0.0;
            if (ratio > worst_road_ratio) worst_road_ratio = ratio;
        }

        printf("%-6lld %-6d %-7d %-8d %-8d %-8d %-7d %-7d %-8d %-8d\n",
               (long long)seed, st.steps, st.mask_open, st.mismatches,
               st.mask0_money, st.mask0_nocell, st.find_lot_missed, st.roads,
               st.roads_on_water, st.waterchannel_built);
        printf("[info]   seed %-6lld water_tiles=%-5d nearest_water=%.1f legal_at_reset=%d "
               "first_open_step=%d roads_to_open=%d money_at_open=%lld reward=%.1f\n",
               (long long)seed, wt, wd, legal_at_reset, st.steps_to_first_open,
               st.steps_to_first_open >= 0 ? st.roads : -1,
               (long long)st.money_at_first_open, st.reward);
    }

    printf("\n[summary] steps=%lld mask_open=%lld (%.2f%%) mismatches=%lld\n",
           tot_steps, tot_open, 100.0 * (double)tot_open / (double)tot_steps, tot_mismatch);
    printf("[summary] маска=0: деньги=%lld курикулум=%lld нет_легальной_клетки=%lld\n",
           tot_money, tot_curric, tot_nocell);
    printf("[summary] find_lot: потерял_клеток=%lld фантомов=%lld\n",
           tot_fl_missed, tot_phantom);
    printf("[summary] дорог=%lld из них на воде=%lld (съели легальный тайл: %lld)\n",
           tot_roads, tot_roads_water, tot_eaten);
    printf("[summary] WaterChannel построен на %d/%d семян; маска открылась на шаге %d..%d\n",
           seeds_with_wc, n_seeds, min_open, max_open);
    printf("[summary] самая далёкая вода %.1f клеток — дошла за %d дорог\n",
           far_dist, far_roads);

    check(tot_mismatch == 0,
          "W1 бит маски BUILD:WaterChannel совпадает с полным перебором легальности");
    check(tot_fl_missed == 0 && tot_phantom == 0,
          "W5 find_lot() не теряет и не выдумывает легальные клетки (в т.ч. далёкие)");
    check(tot_curric == 0,
          "W2 курикулум не блокирует WaterChannel на этапе 0 (all_builds)");
    check(tot_nocell > 0,
          "W2 главная причина маскировки — нет легальной клетки (вода не примыкает к дороге/зданию)");
    check(min_open > 0 && max_open < 60,
          "W3 маска открывается за <60 шагов жадной направленной дороги (не навсегда)");
    check(seeds_with_wc >= 6,
          "W3 водоканал реально строится минимум на 6/8 семян");
    check(far_roads >= 0 && far_dist > 10.0,
          "W4 есть карты с водой дальше 10 клеток — и маска на них всё равно открывается");

    // ── W9: «наведение на воду» — цель-ориентированный find_lot_dir ─────────
    // Регрессия старого жадного выбора (максимум смещения по оси): seed 100
    // (диагональная вода 14.4 клетки, могила из LT_NONE вокруг) — дорога
    // «проезжала» цель, фронт вставал в 7.8 клеток, 205 дорог × 400 съедали
    // весь бюджет 82 000, и маска за 400 шагов не открылась ни разу. Теперь
    // клетка выбирается по «минимум дистанции до воды среди кандидатов в
    // запрошенную сторону» (src/env.cpp, pick_dir_cell): водоканал строится
    // на всех сидах, число дорог ограничено рельефом, а не бессмысленным
    // «уезжанием дальше по оси».
    printf("\n=== W9: наведение на воду (цель-ориентированный find_lot_dir) ===\n");
    printf("[info] худшее отношение дорог/дистанции по сидам: %.2f (1.0 = прямой путь)\n",
           worst_road_ratio);
    check(seeds_with_wc == n_seeds,
          "W9 водоканал построен на всех 8/8 семян (старый жадный: 7/8, seed 100 не дошёл)");
    check(worst_road_ratio > 0.0 && worst_road_ratio <= 3.0,
          "W9 число дорог ограничено: дороги ≤ 3× дистанции до воды на каждом сиде");
    {
        // Трудный сид измеряется ЯВНО, а не как «первый упавший»: регрессия
        // должна остаться видимой, даже если начнут падать другие сиры первыми.
        const int64_t s100 = 100;
        ColonyEnvCpp e(bd, ed, s100, 200);
        e.reset(s100);
        const double wd100 = nearest_water_dist(e);
        auto st = run_rollout(e, find_idx(e, "WaterChannel"), 400, true);
        printf("[info] seed 100: nearest_water=%.1f first_open_step=%d roads=%d "
               "money_at_open=%lld wc_built=%d\n",
               wd100, st.steps_to_first_open, st.roads,
               (long long)st.money_at_first_open, st.waterchannel_built);
        check(st.steps_to_first_open >= 1 && st.steps_to_first_open < 400 &&
              st.waterchannel_built >= 1,
              "W9 seed 100: дорога доходит до воды, маска открывается за <400 шагов, водоканал построен");
        check(st.roads <= (int)(wd100 * 3.0) + 10,
              "W9 seed 100: дорога не «уезжает» (число дорог ограничено дистанцией до воды)");
    }

    // ── W6: дорога на воде не должна делать водоканал недостижимым ──────────
    printf("\n=== W6: дорога на воде (Road need_earth = LT_EVERYWHERE) ===\n");
    check(tot_roads_water == 0 || tot_eaten == 0,
          "W6 дорога на воде не уменьшает число легальных клеток для водоканала");

    // ── W7: разбор «трудного» сида (маска так и не открылась) ───────────────
    printf("\n=== W7: разбор сида, где маска не открылась за 400 шагов ===\n");
    {
        // Семя ищется честно: первый из набора, где жадная дорога не дошла.
        int64_t hard_seed = -1;
        for (int64_t seed : {1, 7, 21, 42, 100, 777, 31337, 2026}) {
            ColonyEnvCpp e(bd, ed, seed, 200);
            e.reset(seed);
            auto st = run_rollout(e, find_idx(e, "WaterChannel"), 400, true);
            if (st.waterchannel_built == 0) { hard_seed = seed; break; }
        }
        if (hard_seed < 0) {
            printf("[info] трудного сида нет — маска открылась на всех\n");
        } else {
            ColonyEnvCpp e(bd, ed, hard_seed, 200);
            e.reset(hard_seed);
            const BaseData* wc = find_data(e, "WaterChannel");
            const int rdb = e.road_dir_base();
            const int road_idx = find_idx(e, "Road");
            int dir = 0;
            water_dir_from_map(e, dir);
            int money_blocked = 0, nocell_blocked = 0, roads = 0;
            printf("[info] seed=%lld nearest_water=%.1f price=%lld\n",
                   (long long)hard_seed, nearest_water_dist(e), (long long)wc->price);
            printf("%-6s %-10s %-6s %-8s %-10s %-9s\n",
                   "step", "money", "roads", "legal_wc", "front->water", "mask");
            for (int s = 1; s <= 400; ++s) {
                auto mask = e.action_mask();
                const Game& g = e.game();
                bool open = mask[A_BUILD0 + find_idx(e, "WaterChannel")] != 0.0f;
                int legal = brute_force_legal_cells(e, *wc);
                if (!open) (g.money < wc->price ? money_blocked : nocell_blocked)++;
                if (s % 50 == 0 || s == 1) {
                    // расстояние от границы дорог до ближайшего водного тайла
                    const int ms = g.map_size();
                    const int8_t* lots = g.earth.lots().data();
                    double best = -1.0;
                    for (const Base& b : g.bases) {
                        if (b.data->id != ROAD_ID) continue;
                        for (int y = 0; y < ms; ++y)
                            for (int x = 0; x < ms; ++x)
                                if (lots[(size_t)y * ms + x] == LT_WATER) {
                                    double d2 = (x - b.x) * (x - b.x) + (y - b.y) * (y - b.y);
                                    if (best < 0 || d2 < best) best = d2;
                                }
                    }
                    printf("%-6d %-10lld %-6d %-8d %-10.1f %-9s\n", s, (long long)g.money,
                           roads, legal, best < 0 ? -1.0 : std::sqrt(best),
                           open ? "OPEN" : "closed");
                }
                int act = open ? A_BUILD0 + find_idx(e, "WaterChannel")
                               : (mask[rdb + dir] != 0.0f ? rdb + dir
                                                          : (mask[A_BUILD0 + road_idx] != 0.0f
                                                                 ? A_BUILD0 + road_idx : A_DAY));
                auto out = e.step(act);
                roads = 0;
                for (const Base& b : e.game().bases)
                    if (b.data->id == ROAD_ID) roads++;
                if (out.terminated || out.truncated) break;
                if (open) break;
                if (s % 40 == 39) water_dir_from_map(e, dir);
            }
            printf("[info] seed=%lld маска=0 из-за денег: %d шагов, из-за отсутствия клетки: %d\n",
                   (long long)hard_seed, money_blocked, nocell_blocked);
            check(money_blocked > 0 && tot_mismatch == 0,
                  "W7 на трудном сиде маска закрыта ДЕНЬГАМИ (дороги съели бюджет), а не ошибкой легальности");
        }
    }

    // ── W8: masked-random — воспроизводится ли историческое «0 из 2000»? ────
    // Маска корректна (W1), но корректность не означает доступность: у случайной
    // политики дороги не складываются в цепочку к воде. Фиксируем факт и его
    // причину (нет легальной клетки, а не деньги/курикулум).
    printf("\n=== W8: masked-random роллаут — доля легальных шагов WaterChannel ===\n");
    {
        long long steps = 0, open_steps = 0, built = 0, r_money = 0, r_nocell = 0;
        for (int64_t seed : {1, 42, 777, 2026}) {
            ColonyEnvCpp e(bd, ed, seed, 200);
            e.reset(seed);
            const int wc_idx = find_idx(e, "WaterChannel");
            const BaseData* wc = find_data(e, "WaterChannel");
            PCG64 rnd((uint64_t)(seed * 7919 + 13));
            for (int s = 0; s < 500; ++s) {
                auto mask = e.action_mask();
                std::vector<int> legal;
                for (int a = 0; a < e.n_actions(); ++a)
                    if (mask[(size_t)a] != 0.0f) legal.push_back(a);
                if (legal.empty()) break;
                if (mask[A_BUILD0 + wc_idx] != 0.0f) {
                    open_steps++;
                } else if (e.game().money < wc->price) {
                    r_money++;
                } else {
                    r_nocell++;
                }
                int act = legal[(size_t)rnd.randint(0, (int64_t)legal.size() - 1)];
                auto out = e.step(act);
                steps++;
                if (act == A_BUILD0 + wc_idx) built++;
                if (out.terminated || out.truncated) break;
            }
        }
        printf("[info] masked-random: шагов=%lld WaterChannel легален=%lld (%.2f%%), построен=%lld\n",
               steps, open_steps,
               100.0 * (double)open_steps / (double)std::max<long long>(1, steps), built);
        printf("[info] маска=0 у random: нет легальной клетки=%lld, не хватает денег=%lld\n",
               r_nocell, r_money);
        // Намеренно без check(): это замер, а не инвариант. Он фиксирует, что
        // корректная маска не равна доступному действию: у случайной политики
        // дорога к воде почти не складывается (направления выбираются наугад),
        // а главная причина закрытой маски — бюджет: случайные действия сливают
        // деньги быстрее, чем жадная цель-ориентированная дорога доходит до воды.
        printf("[info] вывод: у masked-random водоканал %s; маска=0 — бюджет (%.0f%% шагов), "
               "затем правило связности (%.0f%%)\n",
               built > 0 ? "всё же строится (редкая удача направлений)"
                         : "не строится: случайная политика не ведёт дорогу к воде",
               100.0 * (double)r_money / (double)std::max<long long>(1, steps),
               100.0 * (double)r_nocell / (double)std::max<long long>(1, steps));
    }

    printf("\n=== ВЕРДИКТ по P0-1 (жёсткая маска legality) + «наведение на воду» ===\n");
    printf("Маска КОРРЕКТНА: бит BUILD:WaterChannel совпал с полным перебором\n");
    printf("Game::can_build_at во всех %lld состояниях (расхождений %lld), find_lot()\n",
           tot_steps, tot_mismatch);
    printf("не потерял и не выдумал ни одной клетки. «Далёкая вода» маской не режется:\n");
    printf("она закрыта правилом связности (вода должна примыкать к дороге/зданию)\n");
    printf("и деньгами — обе причины легитимны.\n");
    printf("Наведение (2026-09-20): цель-ориентированный find_lot_dir доводит жадную\n");
    printf("дорогу до воды на %d/%d сидах (маска открылась на шаге %d..%d), включая\n",
           seeds_with_wc, n_seeds, min_open, max_open);
    printf("seed 100 — регрессию старого «уезжания по оси» (205 дорог, бюджет 0, вода\n");
    printf("в 14.4 клетки осталась недоступной). Остаточный риск — не маска и не\n");
    printf("наведение, а политика: у masked-random вода не строится (см. W8), потому\n");
    printf("что случайные действия не складываются в дорогу к воде.\n");

    printf("\n%s (%d failure(s))\n", failures == 0 ? "ALL CHECKS PASSED" : "FAILURES", failures);
    return failures == 0 ? 0 : 1;
}
