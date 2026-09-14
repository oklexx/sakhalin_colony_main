#include "colony/env.h"
#include "colony/rewards.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

#include <json.hpp>

namespace colony {

// (logging is done inline via step_log_ member in ColonyEnvCpp)

namespace {

const char* SEASON_NAMES_ENV[4] = {"spring", "summer", "autumn", "winter"};

}  // namespace

ColonyEnvCpp::ColonyEnvCpp(const std::vector<BaseData>& base_data,
                           const std::vector<BaseEvent>& events_data,
                           int64_t seed, int map_size, const Curriculum& curriculum,
                           const RewardConfig& cfg,
                           const std::string& difficulty,
                           bool no_city_game_over,
                           int64_t no_people_days)
    : base_data_(std::make_shared<const std::vector<BaseData>>(base_data)),
      events_data_(std::make_shared<const std::vector<BaseEvent>>(events_data)),
      cfg_(cfg),
      map_size_(map_size),
      curriculum_(curriculum),
      difficulty_(difficulty),
      no_city_game_over_(no_city_game_over),
      no_people_days_(no_people_days),
      game_(*base_data_, *events_data_, seed, map_size, difficulty, no_city_game_over, no_people_days) {
    // пул построек
    std::unordered_set<std::string> subset;
    for (const char* id : BUILD_SUBSET) subset.insert(id);
    for (const BaseData& d : *base_data_) {
        if (d.id != DEPOT_ID && subset.count(d.id)) {
            build_ids_.push_back(d.id);
            build_data_.push_back(&d);
        }
    }
    n_build_ = (int)build_ids_.size();
    for (int i = 0; i < n_build_; i++) build_id_to_idx_[build_ids_[i]] = i;
    manager_base_ = A_BUILD0 + n_build_;

    // курикулум уже лежит в curriculum_ (см. set_curriculum); каталог зависит от него
    compute_catalog();
    // RL-среда не использует undo — отключаем для производительности
    game_.set_enable_undo(false);
    seat_build_gate();  // PR 2: гейт на уровне игры
}

// Разобрать транспортный JSON курикулума (см. Curriculum в env.h).
Curriculum Curriculum::from_json(const std::string& text) {
    Curriculum c;
    nlohmann::json j;
    try {
        j = nlohmann::json::parse(text);
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("bad --curriculum JSON: ") + e.what());
    }
    if (!j.is_object())
        throw std::runtime_error("bad --curriculum JSON: expected an object");
    c.all_builds = j.value("all_builds", true);
    c.stage_report = j.value("stage", 0);
    if (j.contains("allowed_builds")) {
        if (!j["allowed_builds"].is_array())
            throw std::runtime_error("bad --curriculum JSON: allowed_builds must be an array");
        c.allowed_builds.clear();
        for (const auto& v : j["allowed_builds"]) {
            if (!v.is_string())
                throw std::runtime_error("bad --curriculum JSON: allowed_builds must be strings");
            c.allowed_builds.insert(v.get<std::string>());
        }
    }
    return c;
}

void ColonyEnvCpp::set_curriculum(const Curriculum& c) {
    curriculum_ = c;
    seat_build_gate();  // тот же this, но гейт дешёвый — пересадить явно
    compute_catalog();
    std::cout << "[C++ ColonyEnvCpp] set_curriculum: all_builds=" << curriculum_.all_builds
              << " allowed=" << curriculum_.allowed_builds.size()
              << " stage_report=" << curriculum_.stage_report;
    if (!curriculum_.all_builds) {
        std::cout << ":";
        std::vector<std::string> ids(curriculum_.allowed_builds.begin(), curriculum_.allowed_builds.end());
        std::sort(ids.begin(), ids.end());
        for (const auto& id : ids) std::cout << " " << id;
    }
    std::cout << std::endl;
}

void ColonyEnvCpp::set_step_log(const std::string& path) {
    step_log_path_ = path;
    if (step_log_.is_open()) {
        step_log_.close();
    }
    if (!path.empty()) {
        step_log_.open(path, std::ios::app);
        step_log_ << "=== Step log started ===" << std::endl;
    }
}

// Map Python season index (0=spring, 1=summer, 2=autumn, 3=winter) to C++ Season enum
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
        for (int i = 0; i < n_build_; i++)
            for (int j = 0; j < SUNDUK_SIZE; j++)
                if (build_data_[i]->consume[j] > 0)
                    n_consumers[j]++;
        extract_weight_.resize(SUNDUK_SIZE);
        for (int j = 0; j < SUNDUK_SIZE; j++)
            extract_weight_[j] = std::min(1.0, 0.25 * (double)n_consumers[j]);
    }

void ColonyEnvCpp::reset(int64_t seed) {
    game_ = Game(base_data_, events_data_, seed, map_size_, difficulty_, no_city_game_over_, no_people_days_);
    seat_build_gate();  // PR 2: свежий Game без гейта — вернуть его сразу
    game_.reset_milestones();
    net_worth_valid_ = false;
    cached_net_worth_ = 0.0;
    steps_ = 0;
    ep_return_ = 0.0;
    last_reward_ = 0.0;
    episode_metrics_ = EpisodeMetrics();
    last_daily_value_ = 0.0;
    last_chain_daily_ = 0.0;
    tax_due_days_ = 0;
    days_since_last_build_ = 0;
    has_ever_built_ = false;
    produced_.clear();
    building_before_.clear();
    chain_done_.clear();
    first_working_.clear();
    extracted_.clear();
    // PR 6: вырожденный сценарий виден сразу, а не как «курикулум не работает»
    std::string deg = degenerate_report();
    if (!deg.empty()) std::cout << deg << std::flush;
}

std::string ColonyEnvCpp::degenerate_report() {
    auto mask = action_mask();
    int buildable = 0;
    for (int i = 0; i < n_build_; ++i)
        if (mask[A_BUILD0 + i] != 0.0f) ++buildable;
    if (buildable > 0) return "";
    const Game& g = game_;
    std::ostringstream os;
    os << "[C++ ColonyEnvCpp] WARNING: вырожденный сценарий — 0 из " << n_build_
       << " BUILD-действий доступны на старте эпизода (seed=" << g.earth.seed()
       << ", money=" << g.money << ").\n";
    if (!curriculum_.all_builds && curriculum_.allowed_builds.empty()) {
        os << "  причина: разрешённый набор пуст (all_builds=false, allowed_builds=[])"
           << " — строить нельзя ничего.\n";
        return os.str();
    }
    os << "  причины по разрешённым зданиям:\n";
    for (int i = 0; i < n_build_; ++i) {
        const BaseData* d = build_data_[i];
        if (!build_allowed(d->id)) continue;
        std::string reason = "неизвестно (маска и отчёт разошлись — баг)";
        if (g.money < d->price) {
            reason = "price " + std::to_string(d->price) + " > money " + std::to_string(g.money);
        } else if (!find_lot(d->need_earth, d->no_near_base)) {
            reason = "нет подходящего лота (need_earth=" + std::to_string(d->need_earth) +
                     ", занятость/связность карты)";
        }
        os << "    " << d->id << ": " << reason << "\n";
    }
    return os.str();
}

std::vector<Season> ColonyEnvCpp::step_seasons(int y, int m, int d, int n_days) const {
    std::vector<Season> seasons;
    if (n_days == 1) {
        seasons.push_back(season_for_month(m));
        return seasons;
    }
    for (int i = 0; i < n_days; i++) {
        seasons.push_back(season_for_month(m));
        d += 1;
        if (d > days_in_month(y, m)) {
            d = 1;
            m += 1;
            if (m > 12) {
                m = 1;
                y += 1;
            }
        }
    }
    return seasons;
}

int ColonyEnvCpp::road_count() const {
    int n = 0;
    for (const Base& b : game_.bases)
        if (b.data->id == ROAD_ID) n++;
    return n;
}

bool ColonyEnvCpp::lot_ok(int x, int y, int need_earth, bool no_near_base) const {
    const Game& g = game_;
    if (!g.earth.in_bounds(x, y)) return false;
    if (g.base_in_box(x, y) != nullptr) return false;
    int8_t cur = g.earth.lot(x, y);
    if (!(cur >= LT_NORMAL && cur < LT_LAST)) return false;
    if (need_earth != LT_EVERYWHERE && cur != need_earth) return false;
    if (no_near_base) {
        const int dx4[4] = {1, -1, 0, 0};
        const int dy4[4] = {0, 0, 1, -1};
        for (int i = 0; i < 4; i++) {
            if (g.earth.in_bounds(x + dx4[i], y + dy4[i]) &&
                g.base_in_box(x + dx4[i], y + dy4[i]) != nullptr)
                return false;
        }
    }
    return g.cell_connected(x, y);
}

std::optional<std::pair<int, int>> ColonyEnvCpp::find_lot(int need_earth, bool no_near_base) {
    const Game& g = game_;
    const int ms = g.map_size();
    const int8_t* lots = g.earth.lots().data();
    const int32_t* idx_map = g.base_index_map().data();

    auto lot_matches = [&](int x, int y) -> bool {
        if (!g.earth.in_bounds(x, y)) return false;
        size_t idx = (size_t)y * ms + x;
        if (idx_map[idx] >= 0) return false;
        int8_t cur = lots[idx];
        if (!(cur >= LT_NORMAL && cur < LT_LAST)) return false;
        if (need_earth != LT_EVERYWHERE && cur != need_earth) return false;
        return true;
    };

    auto has_base_neighbor = [&](int x, int y) -> bool {
        if (x > 0 && idx_map[(size_t)y * ms + (x - 1)] >= 0) return true;
        if (x < ms - 1 && idx_map[(size_t)y * ms + (x + 1)] >= 0) return true;
        if (y > 0 && idx_map[(size_t)(y - 1) * ms + x] >= 0) return true;
        if (y < ms - 1 && idx_map[(size_t)(y + 1) * ms + x] >= 0) return true;
        return false;
    };

    auto is_road = [&](int x, int y) -> bool {
        size_t idx = (size_t)y * ms + x;
        int32_t bidx = idx_map[idx];
        return bidx >= 0 && bidx < (int32_t)g.bases.size() &&
               g.bases[bidx].data->id == ROAD_ID;
    };

    auto is_building = [&](int x, int y) -> bool {
        size_t idx = (size_t)y * ms + x;
        int32_t bidx = idx_map[idx];
        return bidx >= 0 && bidx < (int32_t)g.bases.size() &&
               g.bases[bidx].data->id != ROAD_ID;
    };

    // Первый случай: нет зданий (кроме City) → строить у init_sel
    bool has_real_base = false;
    for (const auto& b : g.bases) {
        if (b.data->id != "City") {
            has_real_base = true;
            break;
        }
    }
    if (!has_real_base) {
        int cx = g.earth.init_sel_x, cy = g.earth.init_sel_y;
        if (lot_matches(cx, cy)) return std::make_pair(cx, cy);
    }

    // BFS от колонии. Путь допустим, если проходит через дороги/здания.
    // Пустая земля — конечная точка (место строительства), если примыкает
    // к зданию или дорога на пути связывает её с колонией.
    std::vector<std::pair<int, int>> frontier;
    std::vector<char> visited((size_t)ms * ms, 0);
    const int dx4[4] = {1, -1, 0, 0};
    const int dy4[4] = {0, 0, 1, -1};

    auto check_and_push = [&](int nx, int ny, bool on_path_has_road) -> std::optional<std::pair<int, int>> {
        if (is_road(nx, ny)) {
            frontier.push_back({nx, ny});
            return std::nullopt;
        }
        if (is_building(nx, ny)) {
            frontier.push_back({nx, ny});
            return std::nullopt;
        }
        if (!lot_matches(nx, ny)) return std::nullopt;
        if (has_base_neighbor(nx, ny) || on_path_has_road) {
            if (no_near_base) {
                if (!has_base_neighbor(nx, ny)) return std::make_pair(nx, ny);
            } else {
                return std::make_pair(nx, ny);
            }
        }
        return std::nullopt;
    };

    for (const auto& b : g.bases) {
        for (int i = 0; i < 4; i++) {
            int nx = b.x + dx4[i], ny = b.y + dy4[i];
            if (!g.earth.in_bounds(nx, ny)) continue;
            size_t nidx = (size_t)ny * ms + nx;
            if (visited[nidx]) continue;
            visited[nidx] = 1;
            auto result = check_and_push(nx, ny, false);
            if (result) return result;
        }
    }

    while (!frontier.empty()) {
        auto [x, y] = std::move(frontier.front());
        frontier.erase(frontier.begin());
        for (int i = 0; i < 4; i++) {
            int nx = x + dx4[i], ny = y + dy4[i];
            if (!g.earth.in_bounds(nx, ny)) continue;
            size_t nidx = (size_t)ny * ms + nx;
            if (visited[nidx]) continue;
            visited[nidx] = 1;
            auto result = check_and_push(nx, ny, true);
            if (result) return result;
        }
    }
    
    // Debug: log why find_lot failed
    if (step_log_.is_open()) {
        step_log_ << "  find_lot FAILED: need_earth=" << need_earth << " no_near_base=" << no_near_base;
        step_log_ << " | bases=" << g.bases.size();
        // Count available cells
        int total_cells = 0, matching_cells = 0, valid_cells = 0;
        for (int y = 0; y < ms; y++) {
            for (int x = 0; x < ms; x++) {
                size_t idx = (size_t)y * ms + x;
                if (idx_map[idx] >= 0) continue; // occupied
                total_cells++;
                int8_t cur = lots[idx];
                if (cur >= LT_NORMAL && cur < LT_LAST) {
                    matching_cells++;
                    if (need_earth == LT_EVERYWHERE || cur == need_earth) {
                        // Check if valid (has neighbor or road)
                        bool has_nb = has_base_neighbor(x, y);
                        if (has_nb || is_road(x, y)) valid_cells++;
                    }
                }
            }
        }
        step_log_ << " | total_free=" << total_cells << " matching=" << matching_cells << " valid=" << valid_cells;
        step_log_ << "\n";
    }
    
    return std::nullopt;
}

double ColonyEnvCpp::year_production_value(const BaseData& d) const {
    int64_t v = 0;
    for (int i = 0; i < SUNDUK_SIZE; i++) v += d.profit[i] * SALE_SUNDUK[i];
    int64_t season_days = 0;
    for (int s = 0; s < 4; s++)
        if (d.work_seasons[s]) season_days += SEASON_DAYS[s];
    return (double)(v * season_days);
}

double ColonyEnvCpp::net_worth() const {
    if (net_worth_valid_) return cached_net_worth_;
    double v = (double)game_.money - (double)game_.credit;
    for (int i = 0; i < SUNDUK_SIZE; i++)
        v += (double)(game_.sunduk[i] * SALE_SUNDUK[i]);
    for (const Base& b : game_.bases) {
        const BaseData& d = *b.data;
        if (d.live_years) {
            double f = (double)b.live_time / (double)d.live_time_total();
            v += (double)d.price * (0.25 + 0.75 * f);
        } else {
            v += (double)d.price;
        }
    }
    cached_net_worth_ = v;
    net_worth_valid_ = true;
    return v;
}

double ColonyEnvCpp::net_worth(const Game& g) const {
    double v = (double)g.money - (double)g.credit;
    for (int i = 0; i < SUNDUK_SIZE; i++)
        v += (double)(g.sunduk[i] * SALE_SUNDUK[i]);
    for (const Base& b : g.bases) {
        const BaseData& d = *b.data;
        if (d.live_years) {
            double f = (double)b.live_time / (double)d.live_time_total();
            v += (double)d.price * (0.25 + 0.75 * f);
        } else {
            v += (double)d.price;
        }
    }
    return v;
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
    const std::vector<float>& cat = catalog_by_season_[season_idx];
    for (float c : cat) push(c);

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

    return obs_buf_;
}

std::vector<float> ColonyEnvCpp::action_mask() {
    const int na = n_actions();
    std::vector<float> mask(na, 0.0f);
    const Game& g = game_;

    // DAY and WEEK are always available
    mask[A_DAY] = 1.0f;
    mask[A_WEEK] = 1.0f;

    // BUILD actions
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
        auto cell = find_lot(d->need_earth, d->no_near_base);
        if (!cell)
            continue;

        mask[act] = 1.0f;
    }

    // Manager actions
    for (int i = 0; i < N_MANAGERS; ++i) {
        int act = manager_base_ + i;
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
        } else {
            mask[act] = 1.0f;
        }
    }

    return mask;
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
    const int R = minimap_radius_;
    const int N = 2 * R + 1;
    const Game& g = game_;
    const int ms = g.map_size();
    const int cx = g.earth.init_sel_x, cy = g.earth.init_sel_y;
    const int8_t* lots = g.earth.lots().data();
    const uint8_t* occ = g.occupied.data();

    std::vector<float> out((size_t)8 * N * N, 0.0f);
    auto put = [&](int ch, int x, int y) {
        out[(size_t)ch * N * N + (size_t)y * N + x] = 1.0f;
    };

    for (int dy = -R; dy <= R; dy++) {
        int wy = cy + dy;
        if (wy < 0 || wy >= ms) continue;
        for (int dx = -R; dx <= R; dx++) {
            int wx = cx + dx;
            if (wx < 0 || wx >= ms) continue;
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
            if (ch >= 0) put(ch, dx + R, dy + R);
            if (occ[(size_t)wy * ms + wx]) put(7, dx + R, dy + R);
        }
    }
    return out;
}

ColonyEnvCpp::StepOut ColonyEnvCpp::step(int action) {
    Game& g = game_;
    double rew = 0.0;

    // component accumulators for logging
    double c_build = 0, c_cost = 0, c_diversity = 0, c_proximity = 0, c_provider = 0, c_prereq = 0;
    double c_error = 0, c_tax = 0, c_tax_bonus = 0, c_novelty = 0, c_daily = 0, c_chain = 0;
    double c_milestone = 0, c_sale = 0, c_preserve = 0, c_manual_tax = 0, c_gameover = 0;
    double c_survival = 0, c_idle = 0;
    double c_extract = 0, c_loan = 0;  // v3
    std::string action_name = "?";

    // Update episode metrics peaks
    if (g.bases.size() > episode_metrics_.base_count_peak)
        episode_metrics_.base_count_peak = g.bases.size();
    if (g.people > episode_metrics_.population_peak)
        episode_metrics_.population_peak = g.people;

    // Capture ALL base uids before action for correct built_price calculation
    std::unordered_set<int64_t> uids_before;
    for (const Base& b : g.bases) uids_before.insert(b.uid);

    // PR 6: заблокированное курикулумом строительство — отказ, а не «день»:
    // день не проходит (ранний выход ниже), дневные счётчики не тикают.
    const bool gated_build =
        (action >= A_BUILD0 && action < A_BUILD0 + n_build_) &&
        !build_allowed(build_data_[action - A_BUILD0]->id);
    // Grace period: count days since the player postponed the tax.
    if (!gated_build) {
        if (g.tax_postponed_ && (g.annual_tax_due() || g.main_tax_due()))
            tax_grace_days_ += 1;
        else
            tax_grace_days_ = 0;
    }

    double net0 = cfg_.disable_net_worth ? 0.0 : net_worth();
    int y0 = g.year, m0 = g.month, d0 = g.day;

    // Track bases under construction before action (for first-production bonus)
    std::unordered_set<int64_t> building_before;
    for (const Base& b : g.bases)
        if (b.build_days > 0) building_before.insert(b.uid);
    building_before_ = building_before;

    // действия
    if (step_log_.is_open()) {
        step_log_ << "  ACTION: " << action;
        if (action == A_DAY) step_log_ << " (DAY)";
        else if (action == A_WEEK) step_log_ << " (WEEK)";
        else if (action >= A_BUILD0 && action < A_BUILD0 + n_build_) {
            const BaseData* d = build_data_[action - A_BUILD0];
            step_log_ << " (BUILD:" << d->id << " price=" << d->price << " need_earth=" << d->need_earth << ")";
        }
        step_log_ << " | money=" << g.money << " pop=" << g.people << " free=" << g.free_people() << "\n";
    }
    
    if (action == A_DAY || action == A_WEEK) {
        action_name = action == A_DAY ? "DAY" : "WEEK";
    } else if (action >= A_BUILD0 && action < A_BUILD0 + n_build_) {
        const BaseData* d = build_data_[action - A_BUILD0];
        action_name = "BUILD:" + d->id;
        // PR 6: заблокированное действие — чистый штраф и ранний выход: день не
        // проходит (без advance_day и дневных/каталожных бонусов), obs — текущий.
        // Попытка строго убыточна: раньше день проходил и tax_daily_bonus капал.
        if (gated_build) {
            rew += cfg_.error_penalty; c_error += cfg_.error_penalty;
            if (step_log_.is_open()) {
                step_log_ << "  BUILD FAILED: " << d->id << " | has_cell=NO"
                          << " | REASON: Постройка закрыта курикулумом.\n";
            }
            steps_ += 1;
            ep_return_ += rew;
            last_reward_ = rew;
            if (!std::isfinite(rew)) rew = 0.0;
            rew = std::clamp(rew, cfg_.clip_reward_min, cfg_.clip_reward_max);
            episode_metrics_.total_reward = ep_return_;
            episode_metrics_.reached_resources = (int64_t)extracted_.size();
            episode_metrics_.days_survived = g.days_alive;
            episode_metrics_.net_worth = net_worth();
            StepOut out;
            out.obs = obs();
            out.rew = rew;
            out.terminated = false;
            out.truncated = (steps_ >= MAX_STEPS);
            out.days = g.days_alive;
            out.people = g.people;
            out.money = g.money;
            out.n_bases = (int64_t)g.bases.size();
            out.seed = (int64_t)g.earth.seed();
            out.tax_due_days = tax_due_days_;
            out.tax_grace_expired = tax_grace_expired();
            out.metrics = episode_metrics_;
            out.ep_return = ep_return_;
            out.steps = steps_;
            return out;
        }
        auto cell = find_lot(d->need_earth, d->no_near_base);
        bool built = false;
        std::string build_error;
        if (cell) {
            auto build_result = g.build(d->id, cell->first, cell->second);
            built = build_result.first;
            build_error = build_result.second;
        }
        if (!built) {
            rew += cfg_.error_penalty; c_error += cfg_.error_penalty;
            if (step_log_.is_open()) {
                step_log_ << "  BUILD FAILED: " << d->id;
                step_log_ << " | has_cell=" << (cell ? "yes" : "NO");
                if (!cell) {
                    step_log_ << " | REASON: find_lot returned null (no suitable cell)";
                } else {
                    step_log_ << " | REASON: " << build_error;
                    step_log_ << " | cell=(" << cell->first << "," << cell->second << ")";
                    step_log_ << " | money=" << g.money << " price=" << d->price;
                    step_log_ << " | pop=" << g.people << " need_workers=" << d->need_workers;
                }
                step_log_ << "\n";
            }
        } else {
            days_since_last_build_ = 0;
            has_ever_built_ = true;
            episode_metrics_.total_builds++;
            bool is_road = (d->id == ROAD_ID);
            bool is_new_type = unique_build_ids_.insert(d->id).second;
            if (is_new_type)
                episode_metrics_.unique_build_types++;
            double ypv_b = year_production_value(*d);
            double build_r = cfg_.build_bonus + std::log2(1.0 + ypv_b / 1000.0);
            rew += build_r; c_build += build_r;
            double cost_r = -cfg_.build_cost_penalty * static_cast<double>(d->price);
            rew += cost_r; c_cost += cost_r;
            // diversity bonus: reward building new types (skip Road)
            if (is_new_type && !is_road) {
                rew += cfg_.diversity_bonus; c_diversity += cfg_.diversity_bonus;
            }
            // proximity bonus: check nearby land type (skip Road)
            if (!is_road && cfg_.proximity_bonus > 0.0 && cell) {
                int req = proximity_land_for(d->id);
                if (req != LT_NONE) {
                    int bx = cell->first, by = cell->second;
                    bool found = false;
                    for (int dx = -3; dx <= 3 && !found; ++dx) {
                        for (int dy = -3; dy <= 3 && !found; ++dy) {
                            if (dx == 0 && dy == 0) continue;
                            int nx = bx + dx, ny = by + dy;
                            if (g.earth.in_bounds(nx, ny) && g.earth.lot(nx, ny) == req)
                                found = true;
                        }
                    }
                    if (found) { rew += cfg_.proximity_bonus; c_proximity += cfg_.proximity_bonus; }
                }
            }
            // Provider bonus
            if (!is_road && !cfg_.disable_provider_bonus) {
                double pb = provider_bonus(g, *d);
                rew += pb; c_provider += pb;
            }
            // Prerequisite bonus
            if (!is_road && !cfg_.disable_provider_bonus) {
                double prb = prerequisite_bonus(g, *d, build_data_);
                rew += prb; c_prereq += prb;
            }
            if (!is_road && cfg_.need_fill_bonus > 0.0) {
                double nf = 0.0;
                // 1) Idle-driven: здание голодает прямо сейчас (нужен ресурс, а на складе меньше чем consume)
                for (int r = 0; r < SUNDUK_SIZE && nf == 0.0; r++) {
                    if (d->profit[r] <= 0 || extract_weight_[r] <= 0.0) continue;
                    for (const Base& bb : g.bases) {
                        if (!bb.need_sunduk) continue;
                        if (bb.data->consume[r] > 0 && g.sunduk[r] < bb.data->consume[r]) {
                            nf = cfg_.need_fill_bonus * extract_weight_[r];
                            break;
                        }
                    }
                }
                // 2) Deficit-driven (новая логика п.6): если в колонии есть отрицательный баланс ресурса
                // (потребление > производство), то постройка производителя этого ресурса тоже премируется,
                // даже если пока нет простаивающих зданий — превентивно. Бонус вполовину меньше.
                if (nf == 0.0) {
                    // считаем баланс по каждому ресурсу (как в obs: sum profit - consume активных зданий)
                    for (int r = 0; r < SUNDUK_SIZE && nf == 0.0; r++) {
                        if (d->profit[r] <= 0 || extract_weight_[r] <= 0.0) continue;
                        double balance = 0.0;
                        for (const Base& bb : g.bases) {
                            if (bb.build_days > 0) continue;
                            if (!bb.data->season_works(g.season)) continue;
                            balance += (double)bb.data->profit[r];
                            balance -= (double)bb.data->consume[r];
                        }
                        // если баланс сильно отрицательный — рекомендуем производителя
                        if (balance < -5.0) {
                            double severity = std::min(2.0, std::max(0.5, -balance / 20.0));
                            nf = cfg_.need_fill_bonus * extract_weight_[r] * 0.5 * severity;
                        }
                    }
                }
                if (nf > 0.0) { rew += nf; c_extract += nf; }
            }
            invalidate_net_worth();
        }
    } else if (action == manager_base_ + 0) {
        action_name = "MGR:improve";
        auto cell2 = find_lot(LT_EVERYWHERE, false);
        if (!cell2 || !g.good_earth(cell2->first, cell2->second).first) {
            rew += cfg_.error_penalty; c_error += cfg_.error_penalty;
        }
    } else if (action == manager_base_ + 1) {
        action_name = "MGR:repair";
        Base* b = g.find_slowest_base();
        if (b == nullptr || !g.restore(b->x, b->y).ok) {
            rew += cfg_.error_penalty; c_error += cfg_.error_penalty;
        }
        else invalidate_net_worth();
    } else if (action == manager_base_ + 2) {
        action_name = "MGR:restore_all";
        if (!g.restore_all().ok) { rew += cfg_.error_penalty; c_error += cfg_.error_penalty; }
        else invalidate_net_worth();
    } else if (action == manager_base_ + 3) {
        action_name = "MGR:destroy";
        Base* b = g.find_slowest_base();
        if (b == nullptr || b->data->id == DEPOT_ID ||
            !g.destroy(b->x, b->y).first) { rew += cfg_.error_penalty; c_error += cfg_.error_penalty; }
        else { rew += cfg_.demolish_penalty; c_preserve += cfg_.demolish_penalty; invalidate_net_worth(); }
    } else if (action == manager_base_ + 4) {
        action_name = "MGR:preserve";
        const Base* best = nullptr;
        for (const Base& b : g.bases) {
            if (!b.preserved && !b.data->no_preserve && b.data->id != DEPOT_ID) {
                if (best == nullptr || b.live_time < best->live_time) best = &b;
            }
        }
        if (best == nullptr || !g.preserve(best->x, best->y).first) { rew += cfg_.error_penalty; c_error += cfg_.error_penalty; }
        else {
            rew -= cfg_.preserve_penalty; c_preserve -= cfg_.preserve_penalty;
            invalidate_net_worth();
        }
    } else if (action == manager_base_ + 5) {
        action_name = "MGR:unpreserve";
        const Base* first_preserved = nullptr;
        for (const Base& b : g.bases) {
            if (b.preserved) { first_preserved = &b; break; }
        }
        if (first_preserved == nullptr ||
            !g.preserve(first_preserved->x, first_preserved->y).first) { rew += cfg_.error_penalty; c_error += cfg_.error_penalty; }
        else {
            rew -= cfg_.preserve_penalty; c_preserve -= cfg_.preserve_penalty;
            invalidate_net_worth();
        }
    } else if (action == manager_base_ + 6) {
        action_name = "MGR:sell";
        Sunduk counts;
        for (int i = 0; i < SUNDUK_SIZE; i++)
            counts[i] = std::max<int64_t>(0, g.sunduk[i] - 200);
        auto r = g.market_sell(counts);
        if (!r.ok || r.total <= 0) { rew += cfg_.error_penalty; c_error += cfg_.error_penalty; }
        else { double sb = cfg_.sale_bonus * std::log1p((double)r.total / 100.0); rew += sb; c_sale += sb; }
    } else if (action == manager_base_ + 7) {
        action_name = "MGR:buy_food";
        // Покупка еды — чистый слив денег (штраф).
        rew -= cfg_.buy_food_penalty; c_error -= cfg_.buy_food_penalty;
        int64_t need = std::max<int64_t>(0, 400 - g.sunduk[FOOD]);
        if (need > 0) {
            Sunduk counts; counts[FOOD] = need;
            if (!g.market_buy(counts).ok) { rew += cfg_.error_penalty; c_error += cfg_.error_penalty; }
        } else { rew += cfg_.error_penalty; c_error += cfg_.error_penalty; }
    } else if (action == manager_base_ + 8) {
        action_name = "MGR:credit_take";
        if (g.credit >= 100000 || !g.bank_take(50000).first) { rew += cfg_.error_penalty; c_error += cfg_.error_penalty; }
        else {
            rew -= cfg_.loan_penalty; c_loan -= cfg_.loan_penalty;
        }
    } else if (action == manager_base_ + 9) {
        action_name = "MGR:credit_give";
        int64_t give = std::min<int64_t>(50000, g.credit);
        if (g.credit <= 0 || !g.bank_give(give).first) { rew += cfg_.error_penalty; c_error += cfg_.error_penalty; }
    } else if (action == manager_base_ + 10) {
        action_name = "MGR:manual_tax";
        rew += cfg_.manual_tax_penalty;  c_manual_tax += cfg_.manual_tax_penalty;
    }

    g.refresh_occupied();

    // прожить день/неделю (PAY_TAX also advances 1 day to prevent spam)
    std::vector<DayResult> results;
    if (action == A_WEEK) {
        results = g.advance_week();
    } else if (action == manager_base_ + 10) {
        // PAY_TAX = advance 1 day + tax penalty (no no-op exploit)
        auto r = g.advance_day();
        if (r.ok) results.push_back(r.res);
    } else {
        auto r = g.advance_day();
        if (r.ok) results.push_back(r.res);
    }
    if (results.empty() && (action == A_DAY || action == A_WEEK || action == manager_base_ + 10)) { rew += cfg_.error_penalty; c_error += cfg_.error_penalty; }

    // налог платится автоматически после advance_day
    if (!g.tax_postponed_ && g.annual_tax_due()) {
        if (g.money >= g.annual_tax_amount()) {
            g.pay_annual_tax();
        } else {
            rew -= cfg_.tax_fail_penalty; c_tax -= cfg_.tax_fail_penalty;
        }
    } else if (!g.tax_postponed_ && g.main_tax_due()) {
        if (g.money >= g.main_tax_amount()) {
            g.pay_main_tax();
        } else {
            rew -= cfg_.tax_fail_penalty; c_tax -= cfg_.tax_fail_penalty;
        }
    }
    if (!g.annual_tax_due() && !g.main_tax_due() && !g.tax_postponed_) {
        rew += cfg_.tax_daily_bonus; c_tax_bonus += cfg_.tax_daily_bonus;
    }
    invalidate_net_worth();

    int64_t built_price = 0;
    for (const Base& b : g.bases) {
        if (uids_before.find(b.uid) == uids_before.end()) {
            built_price += b.data->price;
        }
    }
    if (!cfg_.disable_net_worth) {
        double nw = cfg_.survival_coeff * (net_worth() - net0);
        rew += nw; c_survival += nw;
    }
    for (const DayResult& r : results) {
        rew -= cfg_.debt_coeff * (double)g.credit / 1000.0;
        rew += (double)r.born * cfg_.born_bonus + (double)r.people_arrived * cfg_.born_bonus;
        rew -= (double)r.died * cfg_.death_penalty;
        rew -= (double)r.base_lost * cfg_.base_lost_penalty;
        if (r.home_overflow) rew -= cfg_.home_overflow_penalty;
        episode_metrics_.births += r.born;
        episode_metrics_.deaths += r.died;
    }

    // milestone bonuses
    {
        double ms = g.check_milestones(cfg_.milestone_base_bonus, cfg_.milestone_people_bonus,
                                      cfg_.milestone_day_bonus, cfg_.milestone_year_bonus);
        rew += ms; c_milestone += ms;
    }

    // ПРРИОРИТЕТНЫЙ REWARD: бонус за нужные здания при нехватке ресурсов
    // Это учит модели правильную стратегию: сначала жильё/еда/вода, потом производство
    {
        int64_t housing = g.now_home_places();
        int64_t people = g.people;
        int64_t food = g.sunduk[FOOD];
        int64_t water = g.sunduk[WATER];
        int64_t free = g.free_people();
        
        // Если нехватка жилья - бонус за строительство домов
        if (housing < people * 1.2 && free > 0) {
            if (action >= A_BUILD0 && action < A_BUILD0 + n_build_) {
                const int build_idx = action - A_BUILD0;
                const std::string& build_id = build_ids_[build_idx];
                if (build_id == "House" || build_id == "SmallHouse" || 
                    build_id == "BigHouse" || build_id == "SuperHouse") {
                    double housing_bonus = cfg_.housing_need_bonus * std::log1p((double)(people - housing) / 10.0);
                    rew += housing_bonus;
                    c_build += housing_bonus;
                }
            }
        }
        
        // Если нехватка еды - бонус за фермы
        if (food < 200) {
            if (action >= A_BUILD0 && action < A_BUILD0 + n_build_) {
                const int build_idx = action - A_BUILD0;
                const std::string& build_id = build_ids_[build_idx];
                if (build_id == "Farm" || build_id == "Garden" || build_id == "BigFarm") {
                    double food_shortage = std::max<double>(0.0, 200.0 - (double)food);
                    double food_bonus = cfg_.food_need_bonus * std::log1p(food_shortage / 10.0);
                    rew += food_bonus;
                    c_build += food_bonus;
                }
            }
        }
        
        // Если нехватка воды - бонус за каналы
        if (water < 100) {
            if (action >= A_BUILD0 && action < A_BUILD0 + n_build_) {
                const int build_idx = action - A_BUILD0;
                const std::string& build_id = build_ids_[build_idx];
                if (build_id == "WaterChannel") {
                    double water_shortage = std::max<double>(0.0, 100.0 - (double)water);
                    double water_bonus = cfg_.water_need_bonus * std::log1p(water_shortage / 10.0);
                    rew += water_bonus;
                    c_build += water_bonus;
                }
            }
        }
    }

    // цепочки и ежедневный доход
    std::vector<Season> seasons = step_seasons(y0, m0, d0, (int)results.size());
    std::vector<int64_t> worked_any;
    double daily_total = 0.0, chain_daily = 0.0;
    for (Season season : seasons) {
        std::vector<std::pair<int, int>> day_working;
        std::vector<std::vector<std::pair<int, int>>> producers(SUNDUK_SIZE);
        for (const Base& b : g.bases) {
            if (b.build_days == 0 && !b.preserved && b.state_empty() &&
                b.data->season_works(season)) {
                auto pos = std::make_pair(b.x, b.y);
                day_working.push_back(pos);
                worked_any.push_back(b.uid);
                const BaseData& d = *b.data;
                for (int j = 0; j < SUNDUK_SIZE; j++) {
                    daily_total +=
                        (double)((d.profit[j] - d.consume[j]) * SALE_SUNDUK[j]);
                    if (d.profit[j] > 0) producers[j].push_back(pos);
                }
            }
        }
        for (const auto& pos : day_working) {
            const Base* b = g.base_in_box(pos.first, pos.second);
            if (!b) continue;
            const BaseData& d = *b->data;
            for (int i = 0; i < SUNDUK_SIZE; i++) {
                if (d.consume[i] <= 0 || producers[i].empty()) continue;
                auto key = std::make_pair(d.id, i);
                if (!chain_done_.count(key)) {
                    chain_done_.insert(key);
                    episode_metrics_.chains_activated++;
                    double cb = cfg_.chain_bonus * std::log2(1.0 + year_production_value(d) / 1000.0);
                    rew += cb; c_chain += cb;
                }
                chain_daily += cfg_.chain_daily;
            }
        }
    }

    std::vector<char> ever_produced(SUNDUK_SIZE, 0);
    std::vector<int> day_producers(SUNDUK_SIZE, 0);
    // Note: re-evaluating production over seasons for first_extraction / extraction_daily
    for (Season season : seasons) {
        std::vector<std::vector<std::pair<int, int>>> producers(SUNDUK_SIZE);
        for (const Base& b : g.bases) {
            if (b.build_days == 0 && !b.preserved && b.state_empty() &&
                b.data->season_works(season)) {
                auto pos = std::make_pair(b.x, b.y);
                const BaseData& d = *b.data;
                for (int j = 0; j < SUNDUK_SIZE; j++) {
                    if (d.profit[j] > 0) producers[j].push_back(pos);
                }
            }
        }
        for (int j = 0; j < SUNDUK_SIZE; j++) {
            if (!producers[j].empty()) ever_produced[j] = 1;
            day_producers[j] = (int)producers[j].size();
        }
    }

    if (cfg_.first_extraction_bonus > 0.0 || cfg_.extraction_daily > 0.0) {
        for (int r = 0; r < SUNDUK_SIZE; r++) {
            double w = extract_weight_[r];
            if (w <= 0.0) continue;
            if (ever_produced[r] && !extracted_.count(r)) {
                extracted_.insert(r);
                double b = cfg_.first_extraction_bonus * w;
                rew += b; c_extract += b;
            }
            if (cfg_.extraction_daily > 0.0 && day_producers[r] > 0) {
                double sat = std::min(1.0, (double)day_producers[r] / 3.0);
                double b = cfg_.extraction_daily * w * sat * (double)seasons.size();
                rew += b; c_extract += b;
            }
        }
    }
    if (!cfg_.disable_daily_income) {
        double di = cfg_.daily_income * std::log1p(daily_total / 100.0);
        rew += di; c_daily += di;
    }
    rew += chain_daily; c_chain += chain_daily;
    last_daily_value_ = daily_total;
    last_chain_daily_ = chain_daily;
    for (int64_t uid : worked_any) {
        if (produced_.count(uid) || building_before.count(uid)) continue;
        const Base* b = nullptr;
        for (const Base& bb : g.bases)
            if (bb.uid == uid) { b = &bb; break; }
        if (!b) continue;
        produced_.insert(uid);
        if (b->data->id == ROAD_ID || b->data->id == DEPOT_ID) continue;  // skip Road and City for novelty
        if (!first_working_.count(b->data->id)) {
            first_working_.insert(b->data->id);
            rew += cfg_.novelty; c_novelty += cfg_.novelty;
        }
    }

    // автоматическое обслуживание
    for (Base& b : g.bases) {
        int64_t max_live = b.data->live_time_total();
        if (b.data->live_years && (double)b.live_time < (double)max_live * 0.3) {
            int64_t price = b.data->restore_price(b.live_time);
            if (price > 0 && g.money >= price) {
                g.save_undo();
                b.live_time = max_live;
                g.money -= price;
            }
        }
    }
    // Auto-maintenance mutated money/live_time -> invalidate cached net worth.
    invalidate_net_worth();

    // Count days actually passed (WEEK returns 7 DayResults, DAY returns 1)
    int days_passed = (int)results.size();
    if (days_passed < 1) days_passed = 1;

    // Increment idle days counter
    days_since_last_build_ += days_passed;

    // терминалы
    steps_ += 1;
    ep_return_ += rew;
    last_reward_ = rew;
    bool terminated = false, truncated = false;
    auto ov = g.game_over();

    if (g.annual_tax_due() || g.main_tax_due())
        tax_due_days_ += days_passed;
    else
        tax_due_days_ = 0;
    if (ov.has_value()) {
        terminated = true;
        rew -= cfg_.game_over_penalty; c_gameover -= cfg_.game_over_penalty;
    } else if (tax_due_days_ >= TAX_GRACE_DAYS) {
        terminated = true;
        rew -= cfg_.game_over_penalty; c_gameover -= cfg_.game_over_penalty;
    } else if (g.credit > g.max_credit()) {
        terminated = true;
        rew -= cfg_.game_over_penalty; c_gameover -= cfg_.game_over_penalty;
    } else if (steps_ >= MAX_STEPS) {
        truncated = true;
    }

    rew += cfg_.survival_bonus; c_survival += cfg_.survival_bonus;
    if (days_since_last_build_ >= cfg_.idle_build_threshold_days) {
        // Only penalize when there are affordable, unlocked buildings available
        bool any_build_available = false;
        for (int i = 0; i < n_build_; ++i) {
            const BaseData* d = build_data_[i];
            if (!build_allowed(d->id)) continue;
            if (g.money < d->price) continue;
            any_build_available = true;
            break;
        }
        if (any_build_available) {
            rew += cfg_.idle_build_penalty;
            c_idle += cfg_.idle_build_penalty;
        }
        days_since_last_build_ = 0;
    }

    if (step_log_.is_open()) {
        auto f2 = [](double v) { char b[32]; snprintf(b, 32, "%.3f", v); return std::string(b); };
        step_log_ << "STEP " << steps_ << " | " << action_name << " | total=" << f2(rew)
          << " err=" << f2(c_error) << " tax=" << f2(c_tax) << " taxb=" << f2(c_tax_bonus)
          << " build=" << f2(c_build) << " cost=" << f2(c_cost) << " div=" << f2(c_diversity)
          << " prox=" << f2(c_proximity) << " prov=" << f2(c_provider) << " preq=" << f2(c_prereq)
          << " pres=" << f2(c_preserve) << " sale=" << f2(c_sale) << " mtax=" << f2(c_manual_tax)
          << " chain=" << f2(c_chain) << " daily=" << f2(c_daily) << " nov=" << f2(c_novelty)
          << " mile=" << f2(c_milestone) << " surv=" << f2(c_survival) << " idle=" << f2(c_idle)
          << " gover=" << f2(c_gameover) << " extr=" << f2(c_extract) << " loan=" << f2(c_loan)
          << " | nw=" << net_worth() << " nw_delta=" << f2(net_worth() - net0)
          << " pop=" << g.people << " bases=" << (int)g.bases.size()
          << " money=" << g.money << " credit=" << g.credit << " day=" << g.days_alive
          << " | food=" << g.sunduk[FOOD] << " water=" << g.sunduk[WATER] << " coal=" << g.sunduk[COAL]
          << " home=" << g.now_home_places() << " free=" << g.free_people()
          << " roads=" << road_count() << "\n";
        
        // Логирование всех зданий с их статусом каждый шаг
        if (!g.bases.empty()) {
            step_log_ << "  BUILDINGS: ";
            for (const Base& b : g.bases) {
                step_log_ << b.data->id << "(" << b.build_days << "d," << b.live_time << "/" << b.data->live_time_total() << ") ";
            }
            step_log_ << "\n";
        }
        // Полный снимок наблюдения (то что видит модель)
        step_log_ << "  OBS: " << dump_obs() << "\n";
        step_log_.flush();
    }

    // Guard against NaN/Inf in raw reward (from log, division, etc.)
    if (!std::isfinite(rew)) rew = 0.0;
    // Clip raw reward before normalization
    rew = std::clamp(rew, cfg_.clip_reward_min, cfg_.clip_reward_max);

    // Fill episode metrics
    episode_metrics_.total_reward = ep_return_;
    episode_metrics_.reached_resources = (int64_t)extracted_.size();
    episode_metrics_.days_survived = g.days_alive;
    episode_metrics_.net_worth = net_worth();
    if (g.people > episode_metrics_.population_peak)
        episode_metrics_.population_peak = g.people;
    if (g.bases.size() > episode_metrics_.base_count_peak)
        episode_metrics_.base_count_peak = (int64_t)g.bases.size();

    StepOut out;
    out.obs = obs();
    out.rew = rew;
    out.terminated = terminated;
    out.truncated = truncated;
    out.days = g.days_alive;
    out.people = g.people;
    out.money = g.money;
    out.n_bases = (int64_t)g.bases.size();
    out.seed = (int64_t)g.earth.seed();
    out.tax_due_days = tax_due_days_;
    out.tax_grace_expired = tax_grace_expired();
    out.metrics = episode_metrics_;
    out.ep_return = ep_return_;
    out.steps = steps_;
    return out;
}

bool ColonyEnvCpp::tax_grace_expired() const {
    return tax_grace_days_ >= 7;
}

int ColonyEnvCpp::tax_grace_days() const {
    return tax_grace_days_;
}

// ===========================================================================
// ColonyVecEnvCpp — batched vectorized environment
// ===========================================================================

ColonyVecEnvCpp::ColonyVecEnvCpp(
    const std::vector<BaseData>& base_data,
    const std::vector<BaseEvent>& events_data,
    int n_envs, int64_t base_seed, int map_size,
    const Curriculum& curriculum,
    const RewardConfig& cfg,
    int n_threads,
    const std::string& difficulty)
    : base_data_(std::make_shared<std::vector<BaseData>>(base_data)),
      events_data_(std::make_shared<std::vector<BaseEvent>>(events_data)),
      cfg_(cfg),
      map_size_(map_size),
      curriculum_(curriculum),
      n_envs_(n_envs),
      base_seed_(base_seed),
      pool_([n_threads, n_envs]() -> size_t {
          size_t hw = std::thread::hardware_concurrency();
          if (hw == 0) hw = 4;
          int desired = n_threads > 0 ? n_threads : n_envs;
          return (size_t)std::min(desired, (int)hw);
      }()) {
    // Create N envs from shared (immutable) data
    envs_.reserve(n_envs);
    for (int i = 0; i < n_envs; ++i) {
        envs_.emplace_back(*base_data_, *events_data_, base_seed + i * 10000,
                           map_size, curriculum, cfg, difficulty);
    }
    obs_size_ = envs_[0].obs_size();
    n_actions_ = envs_[0].n_actions();

    // Pre-allocate buffers
    size_t obs_flat = (size_t)n_envs_ * obs_size_;
    obs_buffer_.resize(obs_flat, 0.0f);
    old_obs_buffer_.resize(obs_flat, 0.0f);
    rewards_.resize(n_envs_, 0.0);
    old_rew_buffer_.resize(n_envs_, 0.0);
    terminateds_.resize(n_envs_, false);
    trunceds_.resize(n_envs_, false);
    episode_return_.resize(n_envs_, 0.0);
    episode_length_.resize(n_envs_, 0);

    // Initialize RMS with correct sizes
    obs_rms_ = RunningMeanStd(obs_size_);
    rew_rms_ = RunningMeanStd(1);
}

void ColonyVecEnvCpp::do_reset(int i, int64_t seed) {
    envs_[i].reset(seed);
    std::vector<float> obs = envs_[i].obs();
    std::copy(obs.begin(), obs.end(), obs_buffer_.data() + (size_t)i * obs_size_);
    episode_return_[i] = 0.0;
    episode_length_[i] = 0;
}

void ColonyVecEnvCpp::reset_batch(const std::vector<int64_t>& seeds) {
    for (int i = 0; i < n_envs_; ++i) {
        do_reset(i, seeds[i]);
    }
    // Copy raw obs to old_obs_buffer for RMS update order
    std::copy(obs_buffer_.begin(), obs_buffer_.end(), old_obs_buffer_.begin());
    // Normalize obs for the agent (SB3 convention: reset returns normalized obs)
    if (norm_obs_) {
        obs_rms_.normalize(obs_buffer_.data(), n_envs_, obs_size_, clip_obs_);
    }
}

void ColonyVecEnvCpp::do_step(int i, int action) {
    auto out = envs_[i].step(action);
    float* obs_ptr = obs_buffer_.data() + (size_t)i * obs_size_;
    std::copy(out.obs.begin(), out.obs.end(), obs_ptr);
    rewards_[i] = out.rew;
    terminateds_[i] = out.terminated;
    trunceds_[i] = out.truncated;
    episode_return_[i] += out.rew;
    episode_length_[i] += 1;
}

void ColonyVecEnvCpp::step_async_batch(const std::vector<int>& actions) {
    std::vector<std::future<void>> futures;
    futures.reserve(n_envs_);
    for (int i = 0; i < n_envs_; ++i) {
        int action = actions[i];
        futures.push_back(pool_.submit([this, i, action]() {
            do_step(i, action);
        }));
    }
    // Wait for all tasks to complete
    for (auto& f : futures) {
        try {
            f.get();
        } catch (const std::exception&) {
            // Step failed — leave obs/reward as-is for this env
        }
    }
}

StepBatchResult ColonyVecEnvCpp::step_wait_batch() {
    // SB3 VecNormalize order:
    // 1. Update RMS with OLD obs/rewards (from previous step)
    // 2. Normalize current obs/rewards with old RMS
    // 3. Save current raw obs/rewards as old for next step

    // Save raw obs BEFORE normalization for terminal_observation
    // (use a separate buffer to avoid copying the entire obs_buffer_)
    if (raw_obs_buf_.size() != obs_buffer_.size())
        raw_obs_buf_.resize(obs_buffer_.size());
    std::copy(obs_buffer_.begin(), obs_buffer_.end(), raw_obs_buf_.begin());

    if (norm_obs_) {
        obs_rms_.update(old_obs_buffer_.data(), n_envs_, obs_size_);
        obs_rms_.normalize(obs_buffer_.data(), n_envs_, obs_size_, clip_obs_);
    }
    // Save raw rewards BEFORE normalization (RMS needs raw values)
    std::copy(rewards_.begin(), rewards_.end(), old_rew_buffer_.begin());

    if (norm_reward_) {
        for (int i = 0; i < n_envs_; ++i) {
            rew_rms_.update_scalar(old_rew_buffer_[i]);
        }
        for (int i = 0; i < n_envs_; ++i) {
            rewards_[i] = rew_rms_.normalize_reward(rewards_[i], clip_reward_);
        }
    }

    // Save current raw obs as old for next step
    std::copy(raw_obs_buf_.begin(), raw_obs_buf_.end(), old_obs_buffer_.begin());

    // Auto-reset done envs
    StepBatchResult result;
    result.obs = obs_buffer_;
    result.rewards = rewards_;
    result.terminateds = terminateds_;
    result.trunceds = trunceds_;
    result.infos.resize(n_envs_);

    for (int i = 0; i < n_envs_; ++i) {
        if (terminateds_[i] || trunceds_[i]) {
            // terminal_observation must be RAW (unnormalized), per SB3 convention
            std::vector<float> terminal_obs(raw_obs_buf_.begin() + (size_t)i * obs_size_,
                                            raw_obs_buf_.begin() + (size_t)(i + 1) * obs_size_);

            // Build info JSON
            nlohmann::json info;
            info["terminal_observation"] = terminal_obs;
            double ep_r = std::isfinite(episode_return_[i]) ? episode_return_[i] : 0.0;
            info["episode"] = {
                {"r", ep_r},
                {"l", episode_length_[i]}
            };
            result.infos[i] = info.dump();

            // Auto-reset with new seed
            int64_t new_seed = base_seed_ + (int64_t)i * 10000 +
                               envs_[i].steps() + 100000;
            do_reset(i, new_seed);

            // Update old_obs_buffer_ for the auto-reset env (raw obs for next RMS update)
            std::copy(obs_buffer_.begin() + (size_t)i * obs_size_,
                      obs_buffer_.begin() + (size_t)(i + 1) * obs_size_,
                      old_obs_buffer_.begin() + (size_t)i * obs_size_);

            // Re-normalize the new obs (fresh from reset, not from old RMS)
            if (norm_obs_) {
                float* new_obs = obs_buffer_.data() + (size_t)i * obs_size_;
                obs_rms_.normalize(new_obs, 1, obs_size_, clip_obs_);
            }
            std::copy(obs_buffer_.begin() + (size_t)i * obs_size_,
                      obs_buffer_.begin() + (size_t)(i + 1) * obs_size_,
                      result.obs.begin() + (size_t)i * obs_size_);
        } else {
            result.infos[i] = "{}";
        }
    }

    return result;
}

std::vector<float> ColonyVecEnvCpp::minimap_batch() const {
    if (envs_.empty()) return {};
    const int n = (int)envs_.size();
    const int R = minimap_radius_;
    const int N = 2 * R + 1;
    const size_t per = (size_t)8 * N * N;
    std::vector<float> out((size_t)n * per, 0.0f);
    for (int i = 0; i < n; i++) {
        std::vector<float> mm = envs_[(size_t)i].minimap();
        std::copy(mm.begin(), mm.end(), out.begin() + (size_t)i * per);
    }
    return out;
}

std::vector<float> ColonyVecEnvCpp::action_masks_batch() const {
    if (envs_.empty()) return {};
    const int n = (int)envs_.size();
    const int na = n_actions_;
    std::vector<float> out((size_t)n * na, 0.0f);
    for (int i = 0; i < n; i++) {
        // action_mask() is non-const (find_lot mutates cell_cache_), so cast
        std::vector<float> mask = const_cast<ColonyEnvCpp&>(envs_[(size_t)i]).action_mask();
        std::copy(mask.begin(), mask.end(), out.begin() + (size_t)i * na);
    }
    return out;
}

void ColonyVecEnvCpp::save_normalization(const std::string& path) {
    nlohmann::json j;
    j["obs_rms"] = obs_rms_.to_json();
    j["rew_rms"] = rew_rms_.to_json();
    j["norm_obs"] = norm_obs_;
    j["norm_reward"] = norm_reward_;
    j["clip_obs"] = clip_obs_;
    j["clip_reward"] = clip_reward_;
    std::ofstream f(path);
    f << j.dump(2);
}

void ColonyVecEnvCpp::load_normalization(const std::string& path) {
    std::ifstream f(path);
    nlohmann::json j;
    f >> j;
    obs_rms_.from_json(j["obs_rms"]);
    rew_rms_.from_json(j["rew_rms"]);
    norm_obs_ = j.value("norm_obs", true);
    norm_reward_ = j.value("norm_reward", true);
    clip_obs_ = j.value("clip_obs", 10.0);
    clip_reward_ = j.value("clip_reward", 10.0);
}

}  // namespace colony