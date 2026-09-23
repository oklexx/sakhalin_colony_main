// P2-7 (docs/REMAINING_WORK_2026_09.md): расщепление src/env.cpp по зонам.
// Этот файл — «маска действий»: ColonyEnvCpp::action_mask(). Чистое
// перемещение из src/env.cpp — поведение не меняется (регрессия:
// ./scripts/cpp_checks.sh, в т.ч. action_mask_bench и water_mask_check W1).
//
// 2026-09 (docs/MONITOR_ACTIONS_2026_09.md §4): тот же проход теперь пишет
// и причину закрытия каждого бита (MaskReason, include/colony/constants.h):
// open ⇔ бит 1; иначе first-fail в порядке проверок W2 — курикулум → деньги →
// участок → прочая применимость. Атрибуция обязана жить ЗДЕСЬ (а не во втором
// BFS): action_masks_batch() и так гоняет find_lot на каждый env на каждый шаг.
// Регрессия паритета open⇔бит и независимая first-fail перепроверка:
// tests/cpp/mask_reason_check.cpp (R1..R5, в ./scripts/cpp_checks.sh).
#include "colony/env.h"

#include <cstdint>
#include <unordered_map>
#include <vector>

namespace colony {

std::vector<float> ColonyEnvCpp::action_mask(std::vector<uint8_t>* reasons) {
    const int na = n_actions();
    std::vector<float> mask(na, 0.0f);
    const Game& g = game_;
    // Закрыто «без особой причины» (ветка не писала код) = other; каждая
    // открытая и каждая конкретно-закрытая ветка ниже перезаписывает своё.
    std::vector<uint8_t>* rs = reasons;
    if (rs) rs->assign((size_t)na, (uint8_t)MR_OTHER);

    // DAY and WEEK are always available
    mask[A_DAY] = 1.0f;
    mask[A_WEEK] = 1.0f;
    if (rs) {
        (*rs)[A_DAY] = (uint8_t)MR_OPEN;
        (*rs)[A_WEEK] = (uint8_t)MR_OPEN;
    }

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

        // Curriculum unlock check (first-fail: раньше денег и участка)
        if (!build_allowed(d->id)) {
            if (rs) (*rs)[act] = (uint8_t)MR_CURRICULUM;
            continue;
        }

        // Money check
        if (g.money < d->price) {
            if (rs) (*rs)[act] = (uint8_t)MR_MONEY;
            continue;
        }

        // find_lot check (BFS — also validates connectivity via base neighbor/road)
        if (!has_lot(d)) {
            if (rs) (*rs)[act] = (uint8_t)MR_NO_LOT;
            continue;
        }

        mask[act] = 1.0f;
        if (rs) (*rs)[act] = (uint8_t)MR_OPEN;
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
        if (!mechanic_enabled_for_manager(i)) {
            if (rs) (*rs)[act] = (uint8_t)MR_CURRICULUM;
            continue;
        }
        if (i == 4) {
            // PRESERVE: только если есть непreserved здания
            bool has_unpreserved = false;
            for (const Base& b : g.bases) {
                if (!b.preserved && !b.data->no_preserve && b.data->id != DEPOT_ID) {
                    has_unpreserved = true;
                    break;
                }
            }
            if (has_unpreserved) {
                mask[act] = 1.0f;
                if (rs) (*rs)[act] = (uint8_t)MR_OPEN;
            }
            // иначе — «нет цели» (нечего консервировать) — уже other
        } else if (i == 5) {
            // UNPRESERVE: только если есть preserved здания
            bool has_preserved = false;
            for (const Base& b : g.bases) {
                if (b.preserved) {
                    has_preserved = true;
                    break;
                }
            }
            if (has_preserved) {
                mask[act] = 1.0f;
                if (rs) (*rs)[act] = (uint8_t)MR_OPEN;
            }
        } else if (!apply_mask) {
            mask[act] = 1.0f;
            if (rs) (*rs)[act] = (uint8_t)MR_OPEN;
        } else if (i == 0) {
            // IMPROVE_LAND: нужны деньги и участок, который ещё не улучшен.
            if (g.money >= BUYGOODEARTH) {
                auto cell = find_lot(LT_EVERYWHERE, false);
                if (cell && !g.is_good(cell->first, cell->second)) {
                    mask[act] = 1.0f;
                    if (rs) (*rs)[act] = (uint8_t)MR_OPEN;
                } else if (rs) {
                    (*rs)[act] = (uint8_t)MR_NO_LOT;  // всё вокруг уже улучшено
                }
            } else if (rs) {
                (*rs)[act] = (uint8_t)MR_MONEY;
            }
        } else if (i == 1) {
            // REPAIR: есть изношенное здание и хватает хотя бы на день ремонта
            // (Game::restore ремонтирует столько, на сколько хватает денег).
            const Base* b = g.find_slowest_base();
            if (b == nullptr || b->data->restore_price_per_day() <= 0) {
                if (rs) (*rs)[act] = (uint8_t)MR_OTHER;  // нечего ремонтировать
            } else if (g.money < b->data->restore_price_per_day()) {
                if (rs) (*rs)[act] = (uint8_t)MR_MONEY;
            } else {
                mask[act] = 1.0f;
                if (rs) (*rs)[act] = (uint8_t)MR_OPEN;
            }
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
            if (!damaged) {
                if (rs) (*rs)[act] = (uint8_t)MR_OTHER;
            } else if (g.money < min_per_day) {
                if (rs) (*rs)[act] = (uint8_t)MR_MONEY;
            } else {
                mask[act] = 1.0f;
                if (rs) (*rs)[act] = (uint8_t)MR_OPEN;
            }
        } else if (i == 3) {
            // DEMOLISH: есть что сносить, и это не Город (шаг с Городом = ошибка).
            const Base* b = g.find_slowest_base();
            if (b != nullptr && b->data->id != DEPOT_ID) {
                mask[act] = 1.0f;
                if (rs) (*rs)[act] = (uint8_t)MR_OPEN;
            }
            // иначе — нет цели для сноса — уже other
        } else if (i == 6) {
            // SELL_SURPLUS: только если есть излишек сверх 200 единиц
            // (шаг продаёт max(0, sunduk-200)).
            int64_t total = 0;
            for (int r = 0; r < SUNDUK_SIZE; r++) {
                int64_t surplus = g.sunduk[r] - 200;
                if (surplus > 0) total += surplus * SALE_SUNDUK[r];
            }
            if (total > 0) {
                mask[act] = 1.0f;
                if (rs) (*rs)[act] = (uint8_t)MR_OPEN;
            }
            // иначе — нет излишка — уже other
        } else if (i == 7) {
            // BUY_FOOD: шаг покупает до 400 единиц и без нужды = двойной штраф.
            int64_t need = 400 - g.sunduk[FOOD];
            if (need <= 0) {
                if (rs) (*rs)[act] = (uint8_t)MR_OTHER;  // еды достаточно
            } else if (g.money < need * BUY_SUNDUK[FOOD]) {
                if (rs) (*rs)[act] = (uint8_t)MR_MONEY;
            } else {
                mask[act] = 1.0f;
                if (rs) (*rs)[act] = (uint8_t)MR_OPEN;
            }
        } else if (i == 8) {
            // TAKE_LOAN: политика шага сама отказывает при credit >= 100000 и
            // при выходе за лимит банка (bank_take) — это лимит, не бюджет.
            if (g.credit < 100000 && g.credit + 50000 <= g.max_credit()) {
                mask[act] = 1.0f;
                if (rs) (*rs)[act] = (uint8_t)MR_OPEN;
            }
            // иначе — лимит банка — уже other
        } else if (i == 9) {
            // REPAY_LOAN: есть долг и деньги на минимальный платёж.
            if (g.credit <= 0) {
                if (rs) (*rs)[act] = (uint8_t)MR_OTHER;  // нечего гасить
            } else if (g.money < std::min<int64_t>(50000, g.credit)) {
                if (rs) (*rs)[act] = (uint8_t)MR_MONEY;
            } else {
                mask[act] = 1.0f;
                if (rs) (*rs)[act] = (uint8_t)MR_OPEN;
            }
        } else if (i == 10) {
            // MGR:manual_tax — исторический «ручной налог»: только штраф и один
            // день (налог платится автоматически ниже, в том же step()). Это
            // строго доминируемое A_DAY, поэтому маска его не открывает.
            mask[act] = 0.0f;
            if (rs) (*rs)[act] = (uint8_t)MR_OTHER;
        }
    }

    // Directional road actions: same gate as BUILD:Road (curriculum, money,
    // legal cell) but the cell must exist in the requested direction.
    if (road_build_idx_ >= 0) {
        const BaseData* rd = build_data_[road_build_idx_];
        if (!build_allowed(rd->id)) {
            if (rs) {
                for (int dir = 0; dir < N_ROAD_DIRS; ++dir)
                    (*rs)[road_dir_base() + dir] = (uint8_t)MR_CURRICULUM;
            }
        } else if (g.money < rd->price) {
            if (rs) {
                for (int dir = 0; dir < N_ROAD_DIRS; ++dir)
                    (*rs)[road_dir_base() + dir] = (uint8_t)MR_MONEY;
            }
        } else {
            for (int dir = 0; dir < N_ROAD_DIRS; ++dir) {
                if (find_lot_dir(*rd, ROAD_DIR_DX[dir], ROAD_DIR_DY[dir])) {
                    mask[road_dir_base() + dir] = 1.0f;
                    if (rs) (*rs)[road_dir_base() + dir] = (uint8_t)MR_OPEN;
                } else if (rs) {
                    (*rs)[road_dir_base() + dir] = (uint8_t)MR_NO_LOT;
                }
            }
        }
    }
    // road_build_idx_ < 0: направленных дорог нет в каталоге — 4 слота
    // остаются закрытыми с уже записанным кодом other.

    return mask;
}

std::vector<uint8_t> ColonyEnvCpp::action_mask_reasons() {
    std::vector<uint8_t> reasons;
    (void)action_mask(&reasons);
    return reasons;
}

std::array<int64_t, N_MASK_REASONS> ColonyEnvCpp::action_mask_reason_counts() {
    std::vector<uint8_t> reasons;
    (void)action_mask(&reasons);
    std::array<int64_t, N_MASK_REASONS> counts{};
    for (uint8_t r : reasons) {
        if (r < (uint8_t)N_MASK_REASONS) counts[(size_t)r]++;
    }
    return counts;
}

}  // namespace colony
