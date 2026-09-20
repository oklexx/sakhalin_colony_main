// P2-7 (docs/REMAINING_WORK_2026_09.md): расщепление src/env.cpp по зонам.
// Этот файл — «нахождение участка»: lot_ok, find_lot (×2), find_lot_dir (×2)
// и их общий выбор клетки pick_dir_cell. Чистое перемещение из src/env.cpp —
// поведение не меняется (регрессия: ./scripts/cpp_checks.sh, в т.ч.
// road_direction_check D1-D4 и water_mask_check W0-W9).
#include "colony/env.h"

#include <cstddef>
#include <cmath>
#include <optional>
#include <vector>

namespace colony {

namespace {

// «Наведение на воду» (2026-09-20, docs/REMAINING_WORK_2026_09.md, п.2 P0-1).
// Выбор клетки для направленного действия ROAD_E/W/S/N из кандидатов BFS.
//
// Прежнее поведение (баг, зафиксирован зондом на seed 100): максимизировать
// смещение по оси от центроида колонии — «как можно дальше вдоль (dx, dy)»,
// БЕЗ понятия цели. Дорога «проезжала» диагональную воду: дистанция до неё
// колебалась (13.6 → 9.22 → 11.66 → 7.81 → 19.42) вместо монотонного спуска,
// и бюджет (205 дорог × 400) кончался до того, как вода становилась легальной
// для WaterChannel.
//
// Новое поведение (только для карт С водой; цели — тот же target, что у
// potential-based road shaping, target_water_x_/y_ из reset()):
//   1) фильтр направления: клетка обязана строго продвигать фронт в
//      запрошенную сторону (along > 0 от центроида колонии). ROAD_E и ROAD_W
//      никогда не выберут одну и ту же клетку, а при отсутствии клеток в
//      сторону действие маскируется (нечего там строить);
//   2) наведение: среди клеток в запрошенной сторону побеждает та, что
//      МИНИМИЗИРУЕТ дистанцию до ближайшей воды. Дистанции сравниваются
//      точным целочисленным dist_sq (sqrt-монотонность), без float-эпсилон:
//      масштаб ошибки не зависит от того, насколько вода далека;
//   3) тай-брейки: дальше вдоль запрошенного направления, затем меньшее
//      боковое смещение (прежнее «плотное плечо фронта»).
// Если воды на карте нет (target_x < 0) — прежний алгоритм без фильтра:
// максимум along, минимум lateral. Изменение локализовано на картах с водой.
std::optional<std::pair<int, int>> pick_dir_cell(
    const std::vector<std::pair<int, int>>& candidates, double cx, double cy,
    int dx, int dy, int target_x, int target_y) {
    if (candidates.empty()) return std::nullopt;

    if (target_x < 0) {
        // Без воды на карте — легаси: максимально вдоль (dx, dy), при равенстве —
        // плотнее к центроиду (фронт растёт компактным плечом, а не рассыпается).
        std::pair<int, int> best = candidates[0];
        double best_score = -1e18;
        for (const auto& c : candidates) {
            double along = (double)(c.first - cx) * dx + (double)(c.second - cy) * dy;
            double lateral = std::fabs((double)(c.first - cx) * dy - (double)(c.second - cy) * dx);
            double score = along - 0.01 * lateral;
            if (score > best_score) { best_score = score; best = c; }
        }
        return best;
    }

    std::optional<std::pair<int, int>> best;
    long long best_dist_sq = 0;
    double best_along = 0.0, best_lateral = 0.0;
    for (const auto& c : candidates) {
        double along = (double)(c.first - cx) * dx + (double)(c.second - cy) * dy;
        if (along <= 0.0) continue;  // фильтр: строго в запрошенную сторону
        long long ddx = (long long)c.first - (long long)target_x;
        long long ddy = (long long)c.second - (long long)target_y;
        long long dist_sq = ddx * ddx + ddy * ddy;
        double lateral = std::fabs((double)(c.first - cx) * dy - (double)(c.second - cy) * dx);
        if (!best.has_value()) {
            best = c; best_dist_sq = dist_sq; best_along = along; best_lateral = lateral;
            continue;
        }
        if (dist_sq > best_dist_sq) continue;
        if (dist_sq < best_dist_sq || along > best_along ||
            (along == best_along && lateral < best_lateral)) {
            best = c; best_dist_sq = dist_sq; best_along = along; best_lateral = lateral;
        }
    }
    return best;
}

}  // namespace

// P2-10: legacy-вход (используется только debug-биндингом debug_lot_ok).
// Раньше здесь была ВТОРАЯ копия правил размещения, и она расходилась с
// Game::can_build_at в двух местах: не знала про сгоревшие участки
// (destroyed_lots) и в no_near_base считала соседством ЛЮБУЮ постройку, включая
// дорогу (единый валидатор дорогу исключает). Теперь делегируем ему: то, что
// показывает отладка, совпадает с тем, что реально проверяет Game::build.
bool ColonyEnvCpp::lot_ok(int x, int y, int need_earth, bool no_near_base) const {
    BaseData probe;
    probe.id = "__lot_probe__";
    probe.need_earth = need_earth;
    probe.no_near_base = no_near_base;
    return game_.can_build_at(probe, x, y).first;
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

    for (size_t head = 0; head < frontier.size(); ++head) {
        auto [x, y] = frontier[head];
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


std::optional<std::pair<int, int>> ColonyEnvCpp::find_lot(const BaseData& d) {
    const Game& g = game_;
    const int ms = g.map_size();
    const int32_t* idx_map = g.base_index_map().data();

    auto is_traversable_base = [&](int x, int y) -> bool {
        size_t idx = (size_t)y * ms + x;
        int32_t bidx = idx_map[idx];
        return bidx >= 0 && bidx < (int32_t)g.bases.size();
    };

    std::vector<std::pair<int, int>> frontier;
    std::vector<char> visited((size_t)ms * ms, 0);
    const int dx4[4] = {1, -1, 0, 0};
    const int dy4[4] = {0, 0, 1, -1};

    auto consider = [&](int nx, int ny) -> std::optional<std::pair<int, int>> {
        if (is_traversable_base(nx, ny)) {
            frontier.push_back({nx, ny});
            return std::nullopt;
        }
        if (g.can_build_at(d, nx, ny).first)
            return std::make_pair(nx, ny);
        return std::nullopt;
    };

    for (const auto& b : g.bases) {
        for (int i = 0; i < 4; i++) {
            int nx = b.x + dx4[i], ny = b.y + dy4[i];
            if (!g.earth.in_bounds(nx, ny)) continue;
            size_t nidx = (size_t)ny * ms + nx;
            if (visited[nidx]) continue;
            visited[nidx] = 1;
            auto result = consider(nx, ny);
            if (result) return result;
        }
    }

    for (size_t head = 0; head < frontier.size(); ++head) {
        auto [x, y] = frontier[head];
        for (int i = 0; i < 4; i++) {
            int nx = x + dx4[i], ny = y + dy4[i];
            if (!g.earth.in_bounds(nx, ny)) continue;
            size_t nidx = (size_t)ny * ms + nx;
            if (visited[nidx]) continue;
            visited[nidx] = 1;
            auto result = consider(nx, ny);
            if (result) return result;
        }
    }
    return std::nullopt;
}

std::optional<std::pair<int, int>> ColonyEnvCpp::find_lot_dir(const BaseData& d,
                                                              int dx, int dy) {
    const Game& g = game_;
    const int ms = g.map_size();
    const int32_t* idx_map = g.base_index_map().data();

    auto is_traversable_base = [&](int x, int y) -> bool {
        size_t idx = (size_t)y * ms + x;
        int32_t bidx = idx_map[idx];
        return bidx >= 0 && bidx < (int32_t)g.bases.size();
    };

    double cx = 0.0, cy = 0.0;
    if (g.bases.empty()) {
        cx = g.earth.init_sel_x;
        cy = g.earth.init_sel_y;
    } else {
        for (const auto& b : g.bases) { cx += b.x; cy += b.y; }
        cx /= (double)g.bases.size();
        cy /= (double)g.bases.size();
    }

    std::vector<std::pair<int, int>> frontier, candidates;
    std::vector<char> visited((size_t)ms * ms, 0);
    const int dx4[4] = {1, -1, 0, 0};
    const int dy4[4] = {0, 0, 1, -1};

    auto consider = [&](int nx, int ny) {
        if (is_traversable_base(nx, ny)) {
            frontier.push_back({nx, ny});
            return;
        }
        if (g.can_build_at(d, nx, ny).first)
            candidates.push_back({nx, ny});
    };

    for (const auto& b : g.bases) {
        for (int i = 0; i < 4; i++) {
            int nx = b.x + dx4[i], ny = b.y + dy4[i];
            if (!g.earth.in_bounds(nx, ny)) continue;
            size_t nidx = (size_t)ny * ms + nx;
            if (visited[nidx]) continue;
            visited[nidx] = 1;
            consider(nx, ny);
        }
    }

    for (size_t head = 0; head < frontier.size(); ++head) {
        auto [x, y] = frontier[head];
        for (int i = 0; i < 4; i++) {
            int nx = x + dx4[i], ny = y + dy4[i];
            if (!g.earth.in_bounds(nx, ny)) continue;
            size_t nidx = (size_t)ny * ms + nx;
            if (visited[nidx]) continue;
            visited[nidx] = 1;
            consider(nx, ny);
        }
    }
    // Направление → клетка: см. pick_dir_cell (наведение на воду, 2026-09-20).
    return pick_dir_cell(candidates, cx, cy, dx, dy, target_water_x_, target_water_y_);
}

// Directional sibling of find_lot: identical legality rules, but instead of
// returning the BFS-first cell it collects every reachable legal cell and lets
// pick_dir_cell choose among them (goal-oriented: closest to the nearest water
// among the cells strictly in the requested direction; legacy "furthest along"
// when the map has no water). That gives the ROAD_E/W/S/N actions a meaning the
// agent can actually steer -- the plain Road action had none, which is why
// water stayed unreachable.
//
// Примечание (P2-7): флаг-оверлоад — мёртвый код: единственные вызовы
// find_lot_dir идут через BaseData-оверлоад (src/env.cpp: step(),
// action_mask()). При no_near_base=true правило «рядом нет базы» исключает
// ВСЕ клетки BFS (кандидаты — соседи существующих баз/дорог), поэтому
// функция всегда возвращает nullopt. Оставлен ради бит-совпадения сигнатур
// (include/colony/env.h) — удалить отдельным коммитом с зачисткой заголовка.
std::optional<std::pair<int, int>> ColonyEnvCpp::find_lot_dir(int need_earth,
                                                              bool no_near_base,
                                                              int dx, int dy) {
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
    auto is_base = [&](int x, int y) -> bool {
        return idx_map[(size_t)y * ms + x] >= 0;
    };

    // Colony centroid: the origin the direction is measured from.
    double cx = 0.0, cy = 0.0;
    if (g.bases.empty()) {
        cx = g.earth.init_sel_x; cy = g.earth.init_sel_y;
    } else {
        for (const auto& b : g.bases) { cx += b.x; cy += b.y; }
        cx /= (double)g.bases.size();
        cy /= (double)g.bases.size();
    }

    // Same frontier expansion as find_lot, but every accepted cell is kept.
    std::vector<std::pair<int, int>> frontier, candidates;
    std::vector<char> visited((size_t)ms * ms, 0);
    const int dx4[4] = {1, -1, 0, 0};
    const int dy4[4] = {0, 0, 1, -1};

    auto consider = [&](int nx, int ny, bool on_path_has_road) {
        if (is_base(nx, ny)) { frontier.push_back({nx, ny}); return; }
        if (!lot_matches(nx, ny)) return;
        if (!has_base_neighbor(nx, ny) && !on_path_has_road) return;
        if (no_near_base && has_base_neighbor(nx, ny)) return;
        candidates.push_back({nx, ny});
    };

    for (const auto& b : g.bases) {
        for (int i = 0; i < 4; i++) {
            int nx = b.x + dx4[i], ny = b.y + dy4[i];
            if (!g.earth.in_bounds(nx, ny)) continue;
            size_t nidx = (size_t)ny * ms + nx;
            if (visited[nidx]) continue;
            visited[nidx] = 1;
            consider(nx, ny, false);
        }
    }
    for (size_t head = 0; head < frontier.size(); ++head) {
        auto [x, y] = frontier[head];
        for (int i = 0; i < 4; i++) {
            int nx = x + dx4[i], ny = y + dy4[i];
            if (!g.earth.in_bounds(nx, ny)) continue;
            size_t nidx = (size_t)ny * ms + nx;
            if (visited[nidx]) continue;
            visited[nidx] = 1;
            consider(nx, ny, true);
        }
    }
    // Направление → клетка: см. pick_dir_cell (наведение на воду, 2026-09-20).
    return pick_dir_cell(candidates, cx, cy, dx, dy, target_water_x_, target_water_y_);
}

}  // namespace colony
