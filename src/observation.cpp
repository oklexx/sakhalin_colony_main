// P2-7 (docs/REMAINING_WORK_2026_09.md): расщепление src/env.cpp по зонам.
// Этот файл — «наблюдение»: obs() (×2), minimap(), dump_obs(),
// compute_catalog(), count_priority_reached() и маппинг сезонов
// python_season_to_cpp. Чистое перемещение из src/env.cpp — поведение не
// меняется (регрессия: ./scripts/cpp_checks.sh, в т.ч. все наградные зонды,
// которые живут на obs).
#include "colony/env.h"

#include <cstdio>
#include <sstream>
#include <string>
#include <vector>

namespace colony {

// Map Python season index (0=spring, 1=summer, 2=autumn, 3=winter) to C++ Season enum.
// (Файл-статик; единственный потребитель — compute_catalog ниже.)
    static Season python_season_to_cpp(int py_idx) {
        switch (py_idx) {
            case 0: return SEASON_SPRING;
            case 1: return SEASON_SUMMER;
            case 2: return SEASON_AUTUMN;
            case 3: return SEASON_WINTER;
            default: return SEASON_SPRING;
        }
    }


    void ColonyEnvCpp::compute_catalog() {
        sale_prices_.resize(SUNDUK_SIZE);
        for (int i = 0; i < SUNDUK_SIZE; i++)
            sale_prices_[i] = (float)((double)SALE_SUNDUK[i] / 1000.0);

        std::vector<std::vector<float>> catalog(n_build_, std::vector<float>(3, 0.0f));
        std::vector<std::vector<float>> season_mask(4, std::vector<float>(n_build_, 0.0f));
        for (int i = 0; i < n_build_; i++) {
            const BaseData& d = *build_data_[i];
            int64_t net = 0;
            for (int j = 0; j < SUNDUK_SIZE; j++)
                net += (d.profit[j] - d.consume[j]) * SALE_SUNDUK[j];
            int64_t season_days = 0;
            for (int s = 0; s < 4; s++)
                if (d.work_seasons[python_season_to_cpp(s)]) season_days += SEASON_DAYS[python_season_to_cpp(s)];
            catalog[i][0] = (float)std::min(10.0, (double)net * (double)season_days / 1e6);
            catalog[i][1] = (float)(std::log10((double)std::max<int64_t>(1, d.price)) / 10.0);
            catalog[i][2] = (float)((double)d.need_workers / 10.0);
            for (int s = 0; s < 4; s++)
                if (d.work_seasons[python_season_to_cpp(s)]) season_mask[s][i] = 1.0f;
        }
        catalog_by_season_.resize(4);
        for (int s = 0; s < 4; s++) {
            catalog_by_season_[s].resize((size_t)n_build_ * 4);
            for (int i = 0; i < n_build_; i++) {
                catalog_by_season_[s][i * 4 + 0] = catalog[i][0];
                catalog_by_season_[s][i * 4 + 1] = catalog[i][1];
                catalog_by_season_[s][i * 4 + 2] = catalog[i][2];
                catalog_by_season_[s][i * 4 + 3] = season_mask[s][i];
            }
        }

        std::vector<int> n_consumers(SUNDUK_SIZE, 0);
        for (int i = 0; i < n_build_; i++) {
            // PR 4 (флаг): «приоритет воды» не зависит от потребителей, которых
            // агенту строить запрещено. По умолчанию — как раньше, по всем 32.
            if (cfg_.priority_count_over_allowed && !build_allowed(build_data_[i]->id))
                continue;
            for (int j = 0; j < SUNDUK_SIZE; j++)
                if (build_data_[i]->consume[j] > 0)
                    n_consumers[j]++;
        }
        extract_weight_.resize(SUNDUK_SIZE);
        for (int j = 0; j < SUNDUK_SIZE; j++)
            extract_weight_[j] = std::min(1.0, 0.25 * (double)n_consumers[j]);
        // PR 4: мягкий ресурсный курикулум — один множитель. all_resources=true
        // (по умолчанию) ничего не меняет: старое поведение бит-в-бит.
        // compute_catalog() зовётся и из set_curriculum(), поэтому смена
        // приоритетов на ходу (по расписанию) сама пересчитывает награду.
        for (int j = 0; j < SUNDUK_SIZE; j++)
            extract_weight_[j] *= curriculum_.all_resources ? 1.0 : curriculum_.resource_weights[j];
    }


    int64_t ColonyEnvCpp::count_priority_reached() const {
        int64_t n = 0;
        for (int r : extracted_) {
            double w = curriculum_.all_resources ? 1.0 : curriculum_.resource_weights[(size_t)r];
            if (w > 0.0) n++;
        }
        return n;
    }

    void ColonyEnvCpp::update_episode_resource_metrics() {
        episode_metrics_.reached_resources = (int64_t)extracted_.size();
        episode_metrics_.priority_reached = count_priority_reached();
        episode_metrics_.reached_resource_ids.clear();
        for (int r = 0; r < SUNDUK_SIZE; ++r) {
            if (extracted_.count(r))
                episode_metrics_.reached_resource_ids.emplace_back(Sunduk::resource_name(r));
        }
    }


std::vector<float> ColonyEnvCpp::obs() const { return obs(game_); }


std::vector<float> ColonyEnvCpp::obs(const Game& g) const {
    if (obs_buf_.size() != (size_t)obs_size())
        obs_buf_.resize((size_t)obs_size());
    size_t idx = 0;
    auto push = [&](float v) { obs_buf_[idx++] = v; };

    std::vector<float> counts(n_build_, 0.0f);
    double sum_live = 0.0, min_live = 0.0;
    int n_building = 0, n_preserved = 0, n_worn = 0;
    int n_idle_sunduk = 0, n_idle_workers = 0;
    bool first = true;
    for (const Base& b : g.bases) {
        auto it = build_id_to_idx_.find(b.data->id);
        if (it != build_id_to_idx_.end()) counts[it->second] += 1.0f;
        if (b.data->live_years) {
            sum_live += (double)b.live_time;
            double lt = (double)b.live_time / (double)b.data->live_time_total();
            if (first || lt < min_live) min_live = lt;
            first = false;
            if (lt < 1.0) n_worn += 1;
        }
        if (b.build_days > 0) n_building += 1;
        if (b.preserved) n_preserved += 1;
        if (b.need_sunduk) n_idle_sunduk += 1;
        if (b.need_workers) n_idle_workers += 1;
    }
    const Sunduk& a = g.sunduk;
    int season_idx;
    switch (g.season) {
        case SEASON_SPRING: season_idx = 0; break;
        case SEASON_SUMMER: season_idx = 1; break;
        case SEASON_AUTUMN: season_idx = 2; break;
        case SEASON_WINTER: season_idx = 3; break;
        default: season_idx = 0;
    }

    push((float)((double)(g.year - START_YEAR) / 50.0));
    push((float)((double)g.month / 12.0));
    push((float)((double)g.day / 31.0));
    push((float)((double)season_idx / 3.0));
    push((float)((double)g.money / 2e5));
    push((float)((double)g.credit / 2e5));
    push((float)((double)g.people / 100.0));
    push((float)((double)g.busy_people / 100.0));
    for (int i = 0; i < SUNDUK_SIZE; i++)
        push((float)std::min(1.0, (double)a[i] / 100.0));
    push((float)((double)g.now_home_places() / 100.0));
    push((float)((double)g.now_need_workers() / 100.0));
    push((float)((double)g.free_people() / 100.0));
    push((float)(std::max(0.0, (double)(g.people - g.now_home_places())) / 100.0));
    push((float)(g.annual_tax_due() ? 1.0 : 0.0));
    push((float)(g.main_tax_due() ? 1.0 : 0.0));
    push((float)((double)g.annual_tax_amount() / 2e4));
    push((float)((double)g.main_tax_amount() / 5e5));
    push((float)((double)g.days_alive / 3650.0));
    push((float)((double)curriculum_.stage_report / 3.0));
    for (float c : counts) push((float)((double)c / 10.0));
    push((float)(sum_live / 5000.0));
    push((float)min_live);
    push((float)((double)n_building / 10.0));
    push((float)((double)n_preserved / 10.0));
    push((float)((double)n_worn / 10.0));
    push((float)((double)n_idle_sunduk / 10.0));
    push((float)((double)n_idle_workers / 10.0));
    for (float p : sale_prices_) push(p);
    // PR 5: catalog rows are per building (i*4+k); the flag zeroes the rows
    // of locked buildings (opt-in ablation, default off).
    const std::vector<float>& cat = catalog_by_season_[season_idx];
    for (int i = 0; i < n_build_; i++) {
        bool zero_row = cfg_.obs_mask_locked_catalog && !curriculum_.all_builds &&
                        curriculum_.allowed_builds.find(build_ids_[(size_t)i]) ==
                            curriculum_.allowed_builds.end();
        for (int k = 0; k < 4; k++) push(zero_row ? 0.0f : cat[(size_t)i * 4 + (size_t)k]);
    }

    // Resource balance: net production - consumption per resource across all active buildings
    // Positive = surplus, negative = deficit
    std::array<double, SUNDUK_SIZE> res_balance{};
    for (const Base& b : g.bases) {
        if (b.build_days > 0) continue;  // skip buildings under construction
        if (!b.data->season_works(g.season)) continue;  // skip off-season buildings
        for (int r = 0; r < SUNDUK_SIZE; r++) {
            res_balance[r] += (double)b.data->profit[r];
            res_balance[r] -= (double)b.data->consume[r];
        }
    }
    for (int r = 0; r < SUNDUK_SIZE; r++) {
        push((float)std::clamp(res_balance[r] / 50.0, -1.0, 1.0));
    }

    // Per-building-type idle status: 1 if any building of this type is idle (lacking resources), 0 otherwise
    std::vector<float> idle_by_type(n_build_, 0.0f);
    for (const Base& b : g.bases) {
        if (b.need_sunduk) {
            auto it = build_id_to_idx_.find(b.data->id);
            if (it != build_id_to_idx_.end()) idle_by_type[it->second] = 1.0f;
        }
    }
    for (float v : idle_by_type) push(v);

    // Tax deadline features: days to next annual tax + money/tax ratio
    // Annual tax is due on March 1 of each year (year > START_YEAR)
    int days_to_tax = 0;
    if (g.year > START_YEAR) {
        if (g.month < 3) {
            days_to_tax = (12 - g.month + 1) * 30;
        } else if (g.month == 3) {
            days_to_tax = 31 - g.day;
        } else {
            days_to_tax = (12 - g.month) * 30 + 31 - g.day;
        }
    } else {
        days_to_tax = (12 - g.month + 3) * 30 + 31 - g.day;
    }
    push((float)(days_to_tax / 365.0));
    int64_t tax_amount = g.annual_tax_amount();
    double ratio = (tax_amount > 0) ? (double)g.money / (double)tax_amount : 0.0;
    push((float)std::min(2.0, ratio));

    // Nearest water relative coordinates (dx, dy) normalized by map_size
    int ms = g.map_size();
    int base_x = g.earth.init_sel_x;
    int base_y = g.earth.init_sel_y;
    const int8_t* lots = g.earth.lots().data();

    // Один проход по карте: ближайший тайл воды (slot 0, v0) и ближайшие тайлы
    // каждого добываемого ресурса (слоты 1..N_NEAREST_LOTS, только obs v2).
    // Порядок обхода и сравнение (строго <) сохранены — значения воды в v0/v1
    // остаются бит-в-бит прежними.
    int best_x[1 + N_NEAREST_LOTS], best_y[1 + N_NEAREST_LOTS];
    double best_sq[1 + N_NEAREST_LOTS];
    for (int k = 0; k <= N_NEAREST_LOTS; k++) {
        best_x[k] = best_y[k] = -1;
        best_sq[k] = -1.0;
    }
    for (int wy = 0; wy < ms; wy++) {
        const int8_t* row = lots + (size_t)wy * ms;
        for (int wx = 0; wx < ms; wx++) {
            const int8_t t = row[wx];
            int slot = -1;
            if (t == LT_WATER) {
                slot = 0;
            } else if (curriculum_.obs_version >= 2) {
                for (int k = 0; k < N_NEAREST_LOTS; k++)
                    if (t == NEAREST_LOT_TYPES[k]) { slot = k + 1; break; }
            }
            if (slot < 0) continue;
            double dx_diff = (double)(wx - base_x);
            double dy_diff = (double)(wy - base_y);
            double dist_sq = dx_diff * dx_diff + dy_diff * dy_diff;
            if (best_sq[slot] < 0 || dist_sq < best_sq[slot]) {
                best_sq[slot] = dist_sq;
                best_x[slot] = wx;
                best_y[slot] = wy;
            }
        }
    }
    auto rel_dx = [&](int slot) -> float {
        return best_sq[slot] < 0 ? 0.0f
                                 : (float)((double)(best_x[slot] - base_x) / (double)ms);
    };
    auto rel_dy = [&](int slot) -> float {
        return best_sq[slot] < 0 ? 0.0f
                                 : (float)((double)(best_y[slot] - base_y) / (double)ms);
    };

    push(rel_dx(0));
    push(rel_dy(0));

    // PR 5, obs v1: the frame - effective resource weights (all_resources
    // renders as nine 1.0s, exactly what the economy applies) + build_allowed
    // bits (all_builds renders as all 1.0s). Appended AFTER the v0 tail so v0
    // stays a strict prefix of v1.
    if (curriculum_.obs_version >= 1) {
        for (int r = 0; r < SUNDUK_SIZE; r++)
            push(curriculum_.all_resources ? 1.0f : (float)curriculum_.resource_weights[(size_t)r]);
        for (int i = 0; i < n_build_; i++) {
            bool allowed = curriculum_.all_builds ||
                           curriculum_.allowed_builds.find(build_ids_[(size_t)i]) !=
                               curriculum_.allowed_builds.end();
            push(allowed ? 1.0f : 0.0f);
        }
    }
    // P0, obs v2: направления к ближайшему дереву/углю/железу/нефти/золоту.
    // Без них 12 из 32 построек (need_earth = 3..7) недостижимы: клетку выбирает
    // find_lot(), а в flat-obs карты нет вообще (см. docs/RL_DIAGNOSIS_2026_09.md).
    if (curriculum_.obs_version >= 2) {
        for (int k = 0; k < N_NEAREST_LOTS; k++) {
            push(rel_dx(k + 1));
            push(rel_dy(k + 1));
        }
    }

    return obs_buf_;
}


std::string ColonyEnvCpp::dump_obs() const {
    auto f = [](double v) { char b[24]; snprintf(b, 24, "%.4f", v); return std::string(b); };
    const Game& g = game_;
    std::ostringstream os;
    os << "year=" << (g.year - START_YEAR)
       << " month=" << g.month << " season="
       << (g.season == SEASON_SPRING ? 0 : g.season == SEASON_SUMMER ? 1 : g.season == SEASON_AUTUMN ? 2 : 3)
       << " money=" << g.money << " credit=" << g.credit
       << " people=" << g.people << " busy=" << g.busy_people
       << " free=" << g.free_people() << " home_places=" << g.now_home_places()
       << " need_workers=" << g.now_need_workers()
       << " tax_due=" << (g.annual_tax_due() ? 1 : 0)
       << " main_tax_due=" << (g.main_tax_due() ? 1 : 0)
       << " tax_amount=" << g.annual_tax_amount()
       << " main_tax_amount=" << g.main_tax_amount()
       << " days_alive=" << g.days_alive
       << " stage=" << curriculum_.stage_report;
    // PR 4: приоритеты ресурсов в step-логе (pr=all либо pr=water,wood,...)
    if (curriculum_.all_resources) {
        os << " pr=all";
    } else {
        os << " pr=";
        bool first = true;
        for (int r = 0; r < SUNDUK_SIZE; r++) {
            if (curriculum_.resource_weights[(size_t)r] <= 0.0) continue;
            if (!first) os << ",";
            first = false;
            os << Sunduk::resource_name(r);
        }
        if (first) os << "none";
    }
    for (int r = 0; r < SUNDUK_SIZE; r++)
        os << " res" << r << "=" << (int)g.sunduk[r];
    for (int i = 0; i < n_build_; i++) {
        const auto& d = *build_data_[i];
        int count = 0;
        for (const Base& b : g.bases) if (b.data->id == d.id) count++;
        if (count > 0) os << " " << d.id << "=" << count;
    }
    os << " | nw=" << f(net_worth(g));
    return os.str();
}


std::vector<float> ColonyEnvCpp::minimap() const {
    const Game& g = game_;
    const int ms = g.map_size();
    const int8_t* lots = g.earth.lots().data();
    const uint8_t* occ = g.occupied.data();
    const int G = 32;

    std::vector<float> out((size_t)8 * G * G, 0.0f);
    auto put = [&](int ch, int gx, int gy) {
        if (gx >= 0 && gx < G && gy >= 0 && gy < G) {
            out[(size_t)ch * G * G + (size_t)gy * G + gx] = 1.0f;
        }
    };

    for (int wy = 0; wy < ms; wy++) {
        int gy = (wy * G) / ms;
        if (gy >= G) gy = G - 1;
        for (int wx = 0; wx < ms; wx++) {
            int gx = (wx * G) / ms;
            if (gx >= G) gx = G - 1;

            int8_t lot = lots[(size_t)wy * ms + wx];
            int ch = -1;
            switch (lot) {
                case LT_NORMAL: ch = 0; break;
                case LT_WATER:  ch = 1; break;
                case LT_WOOD:   ch = 2; break;
                case LT_COAL:   ch = 3; break;
                case LT_IRON:   ch = 4; break;
                case LT_OIL:    ch = 5; break;
                case LT_GOLD:   ch = 6; break;
                default: break;
            }
            if (ch >= 0) put(ch, gx, gy);
            if (occ[(size_t)wy * ms + wx]) put(7, gx, gy);
        }
    }
    return out;
}


}  // namespace colony
