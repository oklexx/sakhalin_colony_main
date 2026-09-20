// Standalone probe: аудит цены WaterChannel (18 310) против темпа накопления
// денег. Замер-компаньон docs/BALANCE_WATERCHANNEL_2026_09.md; не имеет
// кода возврата (семейство water_probe* — всегда 0), инварианты живут в
// water_mask_check (W9).
//
// Что измеряется:
//   E0  статика из configs/bases.json: цена, доля бюджета, эквивалент в
//       дорогах, годовой налог на старте, payback по чистой годовой цене
//       производства (для сравнения — вся таблица построек);
//   E1  по 8 сидам: дорога к воде (цель-ориентированный find_lot_dir) —
//       сколько дорог, сколько денег съедено ДО момента, когда вода стала
//       достижимой, и какой бюджет остался против цены водоканала;
//   E2  запас прочности: при каком удорожании водоканала «платёжеспособность
//       в момент достижения воды» перестанет входить в бюджет;
//   E3  темп накопления БЕЗ продаж (политика DAY-only): бюджет не растёт
//       и не падает до налога — цена ограничена не накоплением, а расходами;
//   E4  темп накопления ПОСЛЕ воды: водоканал → фермы/сады → продажа излишка
//       (политика «экономика»), чистый прирост денег в день.
//
// Build (from the repo root, Linux/macOS) — одна строка, без продолжений:
//   g++ -std=c++17 -O1 -Iinclude -Iinclude/third_party -o /tmp/water_econ \
//       tests/cpp/water_economy_probe.cpp src/env.cpp src/data.cpp \
//       src/resources.cpp src/game.cpp src/rewards.cpp src/rng.cpp src/earth.cpp
//   /tmp/water_econ    # запуск из корня репозитория: читаются configs/*.json
#include "colony/env.h"
#include "colony/data.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using namespace colony;

static int find_idx(ColonyEnvCpp& e, const std::string& id) {
    for (int i = 0; i < e.n_build(); ++i)
        if (e.build_ids()[(size_t)i] == id) return i;
    return -1;
}

static const BaseData* find_data(const ColonyEnvCpp& e, const std::string& id) {
    for (const BaseData* d : e.build_data())
        if (d->id == id) return d;
    return nullptr;
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

static int road_count(const Game& g) {
    int n = 0;
    for (const Base& b : g.bases)
        if (b.data->id == ROAD_ID) n++;
    return n;
}

// Направление к воде (как в obs: нормировано map_size) — для выбора
// лучшего из открытых направленных действий.
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

int main() {
    auto bd = load_base_data("configs/bases.json");
    auto ed = load_events("configs/events.json");

    // ── E0: статика ─────────────────────────────────────────────────────────
    printf("=== E0: статика цены WaterChannel (configs/bases.json) ===\n");
    {
        ColonyEnvCpp e(bd, ed, 42, 200);
        e.reset(42);
        const BaseData* wc = find_data(e, "WaterChannel");
        const BaseData* rd = find_data(e, "Road");
        const Game& g = e.game();
        printf("[info] INIT_MONEY=%d WaterChannel.price=%lld Road.price=%lld\n",
               INIT_MONEY, (long long)wc->price, (long long)rd->price);
        printf("[info] доля бюджета: %.1f%%; эквивалент: %.1f дорог\n",
               100.0 * (double)wc->price / (double)INIT_MONEY,
               (double)wc->price / (double)rd->price);
        printf("[info] водоканал: build_time=%lld дней, need_workers=%d, "
               "work_seasons без зимы (275 дней/год), profit water=%d/день\n",
               (long long)wc->build_time, (int)wc->need_workers,
               (int)wc->profit[WATER]);
        printf("[info] годовой налог на старте (1 здание, City): %lld\n",
               (long long)g.annual_tax_amount());
        printf("[info] максимум дорог при зарезервированных %lld на водоканал: %d\n",
               (long long)wc->price,
               (int)((INIT_MONEY - wc->price) / rd->price));

        // payback: цена / годовая чистая продажа (profit-consume, цена SALE)
        auto net_year = [](const BaseData* d) {
            int64_t v = 0;
            for (int i = 0; i < SUNDUK_SIZE; i++) v += (d->profit[i] - d->consume[i]) * SALE_SUNDUK[i];
            int days = 0;
            for (int s = 0; s < 4; s++)
                if (d->work_seasons[s]) days += SEASON_DAYS[s];
            return (double)(v * days);
        };
        const char* cls[] = {"WaterChannel", "Farm", "Garden", "Coalmine",
                             "Ironmine", "CowFarm", "Refinery"};
        printf("[info] payback (цена / годовая чистая продажа), годы:\n");
        for (const char* id : cls) {
            const BaseData* d = find_data(e, id);
            double ny = net_year(d);
            printf("[info]   %-14s %9lld / %9.0f = %5.2f\n", id,
                   (long long)d->price, ny, d->price / ny);
        }
    }

    // ── E1/E2: дорога к воде по сидам — сколько съедено до «платёжеспособного» ─
    printf("\n=== E1: бюджет в момент достижения воды (8 сидов) ===\n");
    printf("%-6s %-12s %-8s %-10s %-14s %-12s %-8s\n", "seed", "water_dist",
           "roads", "roads_cost", "money_at_open", "margin", "price");
    long long min_money_at_open = -1;
    double min_ratio = -1.0;
    for (int64_t seed : {1, 7, 21, 42, 100, 777, 31337, 2026}) {
        ColonyEnvCpp e(bd, ed, seed, 200);
        e.reset(seed);
        const int wc_idx = find_idx(e, "WaterChannel");
        const int road_idx = find_idx(e, "Road");
        const BaseData* wc = find_data(e, "WaterChannel");
        const double wd = nearest_water_dist(e);
        const int rdb = e.road_dir_base();
        int roads = 0;
        long long money_at_open = -1;
        int open_step = -1;
        for (int s = 0; s < 400; ++s) {
            auto m = e.action_mask();
            if (m[A_BUILD0 + wc_idx] != 0.0f) {
                money_at_open = e.game().money;
                open_step = s + 1;
                e.step(A_BUILD0 + wc_idx);  // ставим водоканал и останавливаемся
                break;
            }
            double wdx = 0.0, wdy = 0.0;
            water_dir_vec(e, wdx, wdy);
            int best = -1;
            double bs = -1e18;
            for (int d = 0; d < N_ROAD_DIRS; d++) {
                if (m[rdb + d] == 0.0f) continue;
                double sc = -(std::fabs(wdx - ROAD_DIR_DX[d] * 0.05) +
                              std::fabs(wdy - ROAD_DIR_DY[d] * 0.05));
                if (sc > bs) { bs = sc; best = d; }
            }
            if (best >= 0) e.step(rdb + best);
            else if (m[A_BUILD0 + road_idx] != 0.0f) e.step(A_BUILD0 + road_idx);
            else e.step(A_DAY);
            if (e.game().money < 0) break;
        }
        roads = road_count(e.game());
        long long roads_cost = (long long)roads * 400;
        long long margin = money_at_open - (long long)wc->price;
        double ratio = money_at_open > 0 ? (double)money_at_open / (double)wc->price : 0.0;
        if (min_money_at_open < 0 || money_at_open < min_money_at_open)
            min_money_at_open = money_at_open;
        if (min_ratio < 0 || ratio < min_ratio) min_ratio = ratio;
        printf("%-6lld %-12.1f %-8d %-10lld %-14lld %-12lld %8.2f\n",
               (long long)seed, wd, roads, roads_cost, money_at_open, margin, ratio);
        printf("[info]   seed %-6lld open_step=%d: дорога съела %.1f%% бюджета, "
               "на водоканал осталось %.1f цены\n",
               (long long)seed, open_step,
               100.0 * (double)roads_cost / (double)INIT_MONEY, ratio);
    }
    printf("\n[info] E2 запас: худший бюджет в момент достижения воды покрывает цену\n");
    printf("[info] в %.2f раза; «платёжеспособность в момент достижения воды» "
           "переживёт удорожание до ~%lld (сейчас 18310)\n", min_ratio,
           (long long)std::max<long long>(0, min_money_at_open));

    // ── E3: темп накопления без продаж ──────────────────────────────────────
    printf("\n=== E3: DAY-only — бюджет без единой продажи ===\n");
    {
        ColonyEnvCpp e(bd, ed, 42, 200);
        e.reset(42);
        int64_t money0 = e.game().money;
        for (int s = 0; s < 400; ++s) e.step(A_DAY);
        const Game& g = e.game();
        printf("[info] день 0: money=%lld; день %lld: money=%lld (налог 1 марта: %lld/год)\n",
               (long long)money0, (long long)g.days_alive, (long long)g.money,
               (long long)g.annual_tax_amount());
        printf("[info] вывод: до воды экономика НЕ копит (продажа — действие агента),\n");
        printf("[info] бюджет лишь тает налогом — цена водоканала не «ждёт накопления»,\n");
        printf("[info] она конкурирует только с расходами (дороги/постройки) на старте.\n");
    }

    // ── E4: темп накопления после воды ──────────────────────────────────────
    printf("\n=== E4: темп накопления ПОСЛЕ водоканала (политика «экономика») ===\n");
    {
        double tot_delta = 0.0, tot_days = 0.0, tot_steady = 0.0;
        int n_steady = 0;
        for (int64_t seed : {1, 42, 777}) {
            ColonyEnvCpp e(bd, ed, seed, 200, Curriculum(), RewardConfig(),
                           "normal", false, GAME_OVER_NO_PEOPLE_DAYS, true);
            e.reset(seed);
            const int wc_idx = find_idx(e, "WaterChannel");
            const int road_idx = find_idx(e, "Road");
            const int rdb = e.road_dir_base();
            const int mgr_sell = rdb - N_MANAGERS + 6;  // SELL_SURPLUS
            static const char* prio[] = {"Farm", "Garden", "House",
                                         "SmallHouse", "CowFarm", "Sawmill", "Coalmine"};
            long long money0 = -1;
            int64_t days0 = 0;
            long long money1 = 0;
            int64_t days1 = 0;
            bool started = false;
            // траектория для «установившегося» темпа (последние 100 дней)
            std::vector<std::pair<int64_t, long long>> hist;
            for (int s = 0; s < 900; ++s) {
                hist.push_back({e.game().days_alive, e.game().money});
                auto m = e.action_mask();
                bool has_channel = false;
                for (const Base& b : e.game().bases)
                    if (b.data->id == "WaterChannel") has_channel = true;
                int act = A_DAY;
                if (!has_channel) {
                    // Сначала вода: маски водоканала открыта — ставим,
                    // иначе тянем дорогу к воде. (Политика «экономика» из
                    // reward_v4_longrun воду НЕ добирает: бюджет уходит в
                    // постройки — это и есть ответ аудита на «а что если
                    // копить?»: накопление без воды невозможно, фермы
                    // пьют.)
                    if (m[A_BUILD0 + wc_idx] != 0.0f) act = A_BUILD0 + wc_idx;
                    else {
                        double wdx = 0.0, wdy = 0.0;
                        water_dir_vec(e, wdx, wdy);
                        int best = -1;
                        double bs = -1e18;
                        for (int d = 0; d < N_ROAD_DIRS; d++) {
                            if (m[rdb + d] == 0.0f) continue;
                            double sc = -(std::fabs(wdx - ROAD_DIR_DX[d] * 0.05) +
                                          std::fabs(wdy - ROAD_DIR_DY[d] * 0.05));
                            if (sc > bs) { bs = sc; best = d; }
                        }
                        if (best >= 0) act = rdb + best;
                        else if (m[A_BUILD0 + road_idx] != 0.0f) act = A_BUILD0 + road_idx;
                    }
                } else if (m[mgr_sell] != 0.0f) {
                    act = mgr_sell;  // излишек продаём: единственный источник денег
                } else {
                    for (const char* id : prio) {
                        int idx = find_idx(e, id);
                        if (idx >= 0 && m[A_BUILD0 + idx] != 0.0f) { act = A_BUILD0 + idx; break; }
                    }
                }
                auto out = e.step(act);
                bool channel_now = false;
                for (const Base& b : e.game().bases)
                    if (b.data->id == "WaterChannel") channel_now = true;
                if (!started && channel_now) {
                    started = true;
                    money0 = e.game().money;
                    days0 = e.game().days_alive;
                }
                money1 = e.game().money;
                days1 = e.game().days_alive;
                if (out.terminated || out.truncated) break;
            }
            if (!started) {
                printf("[info] seed %-4lld: водоканал не построен\n", (long long)seed);
                continue;
            }
            double dd = (double)(money1 - money0);
            double ddays = (double)(days1 - days0);
            tot_delta += dd;
            tot_days += ddays;
            // установившийся темп: последние ~100 дней (инвестиционная фаза
            // строек не должна маскировать итоговую экономику)
            long long m100 = 0;
            int64_t d100 = 0;
            for (int i = (int)hist.size() - 1; i >= 0; --i) {
                if (hist[(size_t)i].first >= days1 - 100) {
                    m100 = hist[(size_t)i].second;
                    d100 = hist[(size_t)i].first;
                    break;
                }
            }
            double steady = (double)(money1 - m100) / std::max(1.0, (double)(days1 - d100));
            tot_steady += steady;
            n_steady++;
            printf("[info] seed %-4lld: после водоканала %lld -> %lld за %lld дн. "
                   "(%+.1f ден/день всего; последние 100 дн.: %+.1f ден/день)\n",
                   (long long)seed, (long long)money0,
                   (long long)money1, (long long)ddays, dd / std::max(1.0, ddays), steady);
        }
        printf("[info] E4 вывод: среднее по всем дням после воды %+.1f ден/день "
               "(включает инвестиционную фазу строек);\n",
               tot_days > 0 ? tot_delta / tot_days : 0.0);
        printf("[info] установившийся темп (последние 100 дн.) %+.1f ден/день; "
               "цена 18310 отыгрывается установившимся накоплением "
               "за ~%.0f дн.\n",
               n_steady > 0 ? tot_steady / n_steady : 0.0,
               (n_steady > 0 && tot_steady / n_steady > 0.0)
                   ? 18310.0 / (tot_steady / n_steady) : 1e9);
    }

    printf("\nWATER ECONOMY PROBE: завершён (замер, код возврата 0)\n");
    return 0;
}
