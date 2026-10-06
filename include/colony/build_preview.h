#pragma once

// ═══════════════════════════════════════════════════════════════════════════
// Предпросмотр застройки для игрового окна (src/gui.cpp).
//
// Зачем отдельный заголовок, а не код внутри gui.cpp: gui.cpp собирается
// только под Windows (raylib), а эта логика должна проверяться на любой
// машине. Проба tests/cpp/build_preview_check.cpp сравнивает вердикт
// preview_reason() с настоящим Game::can_build_at() по ВСЕЙ карте — если
// подсветка «зелёная, а не строится» разойдётся с движком, CI покраснеет.
//
// Почему нельзя звать can_build_at() на каждую клетку: внутри него
// cell_connected() делает BFS по дорогам, то есть подсветка области 50×50
// стоила бы 2500 обходов карты за кадр. Здесь связность считается один раз
// (connectivity_map), остальные правила — поклеточно и ровно те же.
// ═══════════════════════════════════════════════════════════════════════════

#include <string>
#include <utility>
#include <vector>

#include "colony/constants.h"
#include "colony/game.h"

namespace colony {
namespace preview {

// Тип земли, на котором в принципе можно строить (первая проверка
// Game::can_build_at: lot в [LT_NORMAL, LT_LAST)). LT_NONE — море вокруг
// острова: именно оно и даёт «дорогу лесенкой», когда нарисовано в цвет травы.
inline bool lot_is_land(int8_t lot) { return lot >= LT_NORMAL && lot < LT_LAST; }

// out[y * size + x] = 1, если пустая клетка примыкает (4-связность) к зданию
// или к дороге, соединённой цепочкой дорог со зданием.
inline void connectivity_map(const Game& g, std::vector<char>& out) {
    const int ms = g.map_size();
    out.assign((size_t)ms * ms, 0);
    const std::vector<int32_t>& idx_map = g.base_index_map();
    auto base_at = [&](int x, int y) -> const Base* {
        if (x < 0 || y < 0 || x >= ms || y >= ms) return nullptr;
        int32_t bi = idx_map[(size_t)y * ms + x];
        return bi >= 0 ? &g.bases[(size_t)bi] : nullptr;
    };
    const int dx4[4] = {1, -1, 0, 0}, dy4[4] = {0, 0, 1, -1};

    // 1) дороги, добивающиеся до настоящего здания
    std::vector<char> road_ok((size_t)ms * ms, 0);
    std::vector<std::pair<int, int>> q;
    for (const Base& b : g.bases) {
        if (b.data->id == ROAD_ID) continue;
        for (int i = 0; i < 4; i++) {
            int nx = b.x + dx4[i], ny = b.y + dy4[i];
            const Base* nb = base_at(nx, ny);
            if (nb && nb->data->id == ROAD_ID && !road_ok[(size_t)ny * ms + nx]) {
                road_ok[(size_t)ny * ms + nx] = 1;
                q.push_back({nx, ny});
            }
        }
    }
    for (size_t h = 0; h < q.size(); ++h) {
        const int x = q[h].first, y = q[h].second;
        for (int i = 0; i < 4; i++) {
            int nx = x + dx4[i], ny = y + dy4[i];
            const Base* nb = base_at(nx, ny);
            if (nb && nb->data->id == ROAD_ID && !road_ok[(size_t)ny * ms + nx]) {
                road_ok[(size_t)ny * ms + nx] = 1;
                q.push_back({nx, ny});
            }
        }
    }

    // 2) сама клетка
    for (int y = 0; y < ms; y++) {
        for (int x = 0; x < ms; x++) {
            char ok = 0;
            for (int i = 0; i < 4 && !ok; i++) {
                int nx = x + dx4[i], ny = y + dy4[i];
                const Base* nb = base_at(nx, ny);
                if (!nb) continue;
                if (nb->data->id != ROAD_ID) ok = 1;
                else if (road_ok[(size_t)ny * ms + nx]) ok = 1;
            }
            out[(size_t)y * ms + x] = ok;
        }
    }
}

// Причина, по которой здесь строить нельзя. Пустая строка — можно.
// conn — результат connectivity_map() для текущего состояния игры.
inline std::string reason(const Game& g, const BaseData& bd, int x, int y,
                          const std::vector<char>& conn) {
    const int ms = g.map_size();
    if (x < 0 || y < 0 || x >= ms || y >= ms) return "Вне карты.";
    if (g.base_in_box(x, y)) return "Это место занято.";
    const size_t idx = (size_t)y * ms + x;
    if (g.destroyed_lots[idx] > 0) return "Нельзя строить на сгоревшем участке.";
    const int8_t cur = g.earth.lot(x, y);
    if (!lot_is_land(cur)) return "Это море — строить нельзя.";
    if (bd.need_earth != LT_EVERYWHERE && cur != bd.need_earth)
        return "Неподходящий тип земли.";
    if (bd.no_near_base) {
        const int dx4[4] = {1, -1, 0, 0}, dy4[4] = {0, 0, 1, -1};
        for (int i = 0; i < 4; i++) {
            const Base* nb = g.base_in_box(x + dx4[i], y + dy4[i]);
            if (nb != nullptr && nb->data->id != ROAD_ID)
                return "Нельзя строить вплотную к зданию.";
        }
    }
    if (g.bases.empty()) return "Нет колонии.";
    if (conn.size() == (size_t)ms * ms && !conn[idx])
        return "Нет связи с колонией (нужна дорога).";
    return "";
}

inline bool ok(const Game& g, const BaseData& bd, int x, int y,
               const std::vector<char>& conn) {
    return reason(g, bd, x, y, conn).empty();
}

}  // namespace preview
}  // namespace colony
