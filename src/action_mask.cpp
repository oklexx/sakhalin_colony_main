// P2-7 (docs/REMAINING_WORK_2026_09.md): расщепление src/env.cpp по зонам.
// Этот файл — «маска действий»: ColonyEnvCpp::action_mask(). Чистое
// перемещение из src/env.cpp — поведение не меняется (регрессия:
// ./scripts/cpp_checks.sh, в т.ч. action_mask_bench и water_mask_check W1).
#include "colony/env.h"

#include <cstdint>
#include <unordered_map>
#include <vector>

namespace colony {

std::vector<float> ColonyEnvCpp::action_mask() {
    const int na = n_actions();
    std::vector<float> mask(na, 0.0f);
    const Game& g = game_;

    // DAY and WEEK are always available
    mask[A_DAY] = 1.0f;
    mask[A_WEEK] = 1.0f;

    // BUILD actions
    //
    // P2-11: find_lot() — это BFS по карте (O(bases + дороги) на вызов), а
    // результат зависит ТОЛЬКО от (need_earth, no_near_base): остальные входные
    // данные (карта, занятость, destroyed_lots) в рамках одного вызова маски
    // неизменны. Из 32 построек 20+ имеют need_earth = LT_EVERYWHERE, т.е.
    // раньше один и тот же BFS выполнялся десятки раз на шаг. Кеш живёт ровно
    // один вызов action_mask() (локальная переменная), поэтому протухнуть не
    // может. Замер: tests/cpp/action_mask_bench.cpp.
    std::unordered_map<int64_t, bool> lot_cache;
    auto has_lot = [&](const BaseData* d) -> bool {
        const int64_t key = ((int64_t)d->need_earth << 1) | (d->no_near_base ? 1 : 0);
        auto it = lot_cache.find(key);
        if (it != lot_cache.end()) return it->second;
        const bool ok = (bool)find_lot(*d);
        lot_cache.emplace(key, ok);
        return ok;
    };
    for (int i = 0; i < n_build_; ++i) {
        int act = A_BUILD0 + i;
        const BaseData* d = build_data_[i];

        // Curriculum unlock check
        if (!build_allowed(d->id))
            continue;

        // Money check
        if (g.money < d->price)
            continue;

        // find_lot check (BFS — also validates connectivity via base neighbor/road)
        if (!has_lot(d))
            continue;

        mask[act] = 1.0f;
    }

    // Manager actions. P1 (2026-09-17): маска по ПРИМЕНИМОСТИ — действие,
    // которое заведомо вернёт error_penalty (-2) или потратит деньги впустую,
    // не должно попадать в выбор политики. Раньше 9 менеджеров были легальны
    // безусловно, и masked-random тратил ~-1.6/шаг на «легальные» ошибки.
    // Все проверки — только чтение: good_earth()/restore_all() мутируют игру,
    // поэтому их предикаты продублированы здесь (см. обработчики в step()).
    // cfg_.mask_managers_by_applicability = false возвращает старые маски.
    const bool apply_mask = cfg_.mask_managers_by_applicability;
    for (int i = 0; i < N_MANAGERS; ++i) {
        int act = manager_base_ + i;
        // Mechanic curriculum is independent of applicability: a disabled
        // action stays zero even when its game predicate happens to be true.
        if (!mechanic_enabled_for_manager(i)) continue;
        if (i == 4) {
            // PRESERVE: только если есть непreserved здания
            bool has_unpreserved = false;
            for (const Base& b : g.bases) {
                if (!b.preserved && !b.data->no_preserve && b.data->id != DEPOT_ID) {
                    has_unpreserved = true;
                    break;
                }
            }
            if (has_unpreserved) mask[act] = 1.0f;
        } else if (i == 5) {
            // UNPRESERVE: только если есть preserved здания
            bool has_preserved = false;
            for (const Base& b : g.bases) {
                if (b.preserved) {
                    has_preserved = true;
                    break;
                }
            }
            if (has_preserved) mask[act] = 1.0f;
        } else if (!apply_mask) {
            mask[act] = 1.0f;
        } else if (i == 0) {
            // IMPROVE_LAND: нужны деньги и участок, который ещё не улучшен.
            if (g.money >= BUYGOODEARTH) {
                auto cell = find_lot(LT_EVERYWHERE, false);
                if (cell && !g.is_good(cell->first, cell->second)) mask[act] = 1.0f;
            }
        } else if (i == 1) {
            // REPAIR: есть изношенное здание и хватает хотя бы на день ремонта
            // (Game::restore ремонтирует столько, на сколько хватает денег).
            const Base* b = g.find_slowest_base();
            if (b != nullptr && b->data->restore_price_per_day() > 0 &&
                g.money >= b->data->restore_price_per_day())
                mask[act] = 1.0f;
        } else if (i == 2) {
            // REPAIR_ALL: есть повреждённые здания и денег хватает хотя бы на
            // самый дешёвый день ремонта (иначе restore_all вернёт «Недостаточно
            // денег» и action станет чистым error_penalty).
            bool damaged = false;
            int64_t min_per_day = -1;
            for (const Base& b : g.bases) {
                int64_t per_day = b.data->restore_price_per_day();
                if (per_day <= 0 || b.live_time >= b.data->live_time_total()) continue;
                damaged = true;
                if (min_per_day < 0 || per_day < min_per_day) min_per_day = per_day;
            }
            if (damaged && g.money >= min_per_day) mask[act] = 1.0f;
        } else if (i == 3) {
            // DEMOLISH: есть что сносить, и это не Город (шаг с Городом = ошибка).
            const Base* b = g.find_slowest_base();
            if (b != nullptr && b->data->id != DEPOT_ID) mask[act] = 1.0f;
        } else if (i == 6) {
            // SELL_SURPLUS: только если есть излишек сверх 200 единиц
            // (шаг продаёт max(0, sunduk-200)).
            int64_t total = 0;
            for (int r = 0; r < SUNDUK_SIZE; r++) {
                int64_t surplus = g.sunduk[r] - 200;
                if (surplus > 0) total += surplus * SALE_SUNDUK[r];
            }
            if (total > 0) mask[act] = 1.0f;
        } else if (i == 7) {
            // BUY_FOOD: шаг покупает до 400 единиц и без нужды = двойной штраф.
            int64_t need = 400 - g.sunduk[FOOD];
            if (need > 0 && g.money >= need * BUY_SUNDUK[FOOD]) mask[act] = 1.0f;
        } else if (i == 8) {
            // TAKE_LOAN: политика шага сама отказывает при credit >= 100000 и
            // при выходе за лимит банка (bank_take).
            if (g.credit < 100000 && g.credit + 50000 <= g.max_credit()) mask[act] = 1.0f;
        } else if (i == 9) {
            // REPAY_LOAN: есть долг и деньги на минимальный платёж.
            if (g.credit > 0 && g.money >= std::min<int64_t>(50000, g.credit))
                mask[act] = 1.0f;
        } else if (i == 10) {
            // MGR:manual_tax — исторический «ручной налог»: только штраф и один
            // день (налог платится автоматически ниже, в том же step()). Это
            // строго доминируемое A_DAY, поэтому маска его не открывает.
            mask[act] = 0.0f;
        }
    }

    // Directional road actions: same gate as BUILD:Road (curriculum, money,
    // legal cell) but the cell must exist in the requested direction.
    if (road_build_idx_ >= 0) {
        const BaseData* rd = build_data_[road_build_idx_];
        if (build_allowed(rd->id) && g.money >= rd->price) {
            for (int dir = 0; dir < N_ROAD_DIRS; ++dir) {
                if (find_lot_dir(*rd, ROAD_DIR_DX[dir], ROAD_DIR_DY[dir]))
                    mask[road_dir_base() + dir] = 1.0f;
            }
        }
    }

    return mask;
}

}  // namespace colony
