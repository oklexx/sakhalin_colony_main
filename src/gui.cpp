#include "colony/bases.h"
#include "colony/constants.h"
#include "colony/data.h"
#include "colony/build_preview.h"
#include "colony/env.h"
#include "colony/game.h"
#include "colony/earth.h"
#include "colony/watch_ipc.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <fstream>

#include "raylib.h"

using namespace colony;

// ─── Статистика для экрана завершения ───
static std::unordered_map<std::string, int> stat_built;
static int64_t stat_earned = 0;
static int64_t stat_spent = 0;
static int64_t stat_prev_money = 0;
static bool stat_started = false;
static bool stat_tax_over = false;

// ═══════════════════════════════════════════════════════════════
// LAYOUT — точная копия promer/ui/main_window.py (1:1 с оригиналом)
// ═══════════════════════════════════════════════════════════════
static const int WIN_W = 1160;
static const int WIN_H = 720;
static const int CH = 22;            // высота текстовой строки
static const int CW = 11;            // ширина текстовой колонки
static const int TOP_H = 2 * CH + 2; // 46 — заголовок + меню
static const int BSTEP = 32;         // шаг кнопок построек (оригинал l += 32)
static const int PAL_COLS = 9;
static const int PAL_ROWS = 4;
static const int MAP_X = 0;
static const int MAP_Y = 178;  // palette bottom is PAL_Y0+PAL_ROWS*BSTEP=175; 3px gap
static const int MAP_W = 880;
static const int BOTTOM_H = 2 * CH + 2;            // 46
static const int MAP_H = WIN_H - MAP_Y - BOTTOM_H; // 500
static const int PANEL_X = MAP_W;                  // 880
static const int SIDE_W = WIN_W - PANEL_X;         // 280
static const int TILE = 25;                        // размер ячейки (оригинал CellW=25)
static const int PAL_Y0 = 2 * CH + 3;              // 47
static const int TOOL_X0 = 584;
static const int INFO_X = 296;
static const int INFO_Y = PAL_Y0;
static const int INFO_W = 280;
static const int BS = 25;                          // размер иконки постройки/инструмента

// ═══ ЦВЕТА (из ui/render.py) ═══
static const Color C_BG        = {0, 7, 0, 255};
static const Color C_TITLE     = {120, 200, 120, 255};
static const Color C_MENU      = {225, 225, 235, 255};
static const Color C_FG        = {215, 215, 205, 255};
static const Color C_FG_DIM    = {125, 135, 125, 255};
static const Color C_ACCENT    = {255, 220, 120, 255};
static const Color C_PANEL_BG  = {222, 228, 220, 255};
static const Color C_PANEL_HEAD= {132, 207, 228, 255};
static const Color C_PANEL_FG  = {20, 25, 20, 255};
static const Color C_LEGEND_BG = {224, 224, 144, 255};
static const Color C_TOOLBTN   = {30, 40, 34, 255};
static const Color C_PAL_SEL   = {36, 46, 40, 255};
static const Color C_PAL_NORM  = {20, 28, 23, 255};
static const Color C_WATER     = {131, 208, 227, 255};
static const Color C_LAND      = {186, 211, 178, 255};
static const Color C_WHITE     = {255, 255, 255, 255};
static const Color C_BLACK     = {0, 0, 0, 255};
// Море вокруг острова (LT_NONE). Раньше рисовалось тем же светло-зелёным, что
// и «ровная земля», хотя строить на нём нельзя (Game::can_build_at требует
// lot >= LT_NORMAL). Из-за этого протяжка дороги по «полю» молча обтекала
// невидимую воду — дорога выходила «лесенкой», и причина была не видна.
static const Color C_SEA       = {72, 112, 140, 255};
static const Color C_SEA_DEEP  = {54, 88, 114, 255};
// Подсветка законности постройки (предпросмотр).
static const Color C_OK_TINT   = {60, 220, 90, 90};
static const Color C_BAD_TINT  = {230, 60, 50, 95};

static const char* LOT_NAMES[9] = {
    "море (строить нельзя)", "ровная земля", "вода", "лес", "уголь",
    "железо", "нефть", "золото", "?"
};
static const char* RES_SHORT[9] = {
    "Золото", "Продовольствие", "Уголь", "Железо", "Нефть",
    "Камень", "Вода", "Дерево", "Энергия"
};

// ═══ LOT COLORS (TERRAIN_BG) ═══
static Color lot_color(int8_t lot) {
    switch (lot) {
        case LT_WATER: return C_WATER;
        case LT_NONE:  return C_SEA;   // не остров: строить нельзя
        default:       return C_LAND;
    }
}
// Клетка в принципе пригодна под застройку по типу земли (без учёта
// примыкания/денег). Первая проверка Game::can_build_at.
static bool lot_is_land(int8_t lot) { return colony::preview::lot_is_land(lot); }

// ═══ BUILDING ICON FALLBACK COLORS ═══
struct BIcon { const char* id; Color c; };
static BIcon build_palette[] = {
    {"Farm",{108,172,78}},{"Garden",{88,182,68}},{"WaterChannel",{58,128,208}},
    {"Sawmill",{148,98,58}},{"Coalmine",{78,78,78}},{"Ironmine",{158,148,138}},
    {"Refinery",{58,58,58}},{"Goldmine",{228,208,68}},{"PowerStation",{208,208,58}},
    {"HydroStation",{68,138,218}},{"Road",{178,168,148}},{"House",{198,158,118}},
    {"SmallHouse",{178,138,98}},{"Fish",{58,168,218}},{"CoalCut",{88,88,88}},
    {"HuntingLand",{118,138,78}},{"CowFarm",{158,118,78}},{"Mushroom",{178,138,98}},
    {"BigHouse",{208,168,128}},{"BigFarm",{108,188,88}},{"Apiary",{238,198,58}},
    {"Torchlight",{218,118,58}},{"Hothouse",{98,178,118}},{"SuperHouse",{218,178,138}},
    {"BigSawmill",{138,98,58}},{"WaterMill",{58,118,198}},{"BigRefinary",{68,68,68}},
    {"Puerperal",{208,168,198}},{"BigIronmine",{168,158,148}},{"AirStation",{208,218,68}},
    {"SmallAtomStation",{255,128,128}},{"AtomStation",{255,88,88}},
};
static Color build_color(const std::string& id) {
    for (auto& b : build_palette) if (id == b.id) return b.c;
    return {158, 158, 158, 255};
}

// ═══ ORIGINAL ICONS (PNG extracted from unData.dfm, ui/assets) ═══
static Texture2D baseTex[33];
static Texture2D earthTex[70];
static Texture2D toolTex[27];
static Texture2D iconTex[7];
static Texture2D seasonTex[4];
static Texture2D selEarthTex[12];
static Texture2D selNoneTex[12];
static Texture2D marketItemTex[9];

static Texture2D load_one(const char* name) {
    char a[512], b[512];
    snprintf(a, sizeof(a), "assets/%s", name);
    Texture2D t = LoadTexture(a);
    if (t.id != 0) return t;
    snprintf(b, sizeof(b), "ui/assets/%s", name);
    return LoadTexture(b);
}
static void draw_tex(const Texture2D& t, int x, int y, int sz) {
    if (t.id == 0) return;
    Rectangle src = {0, 0, (float)t.width, (float)t.height};
    Rectangle dst = {(float)x, (float)y, (float)sz, (float)sz};
    DrawTexturePro(t, src, dst, {0, 0}, 0.0f, WHITE);
}

// ═══ Text with Cyrillic-capable TTF font ═══
static Font gFont = {0};
static void text(const char* s, int x, int y, int size, Color c) {
    if (gFont.texture.id == 0) DrawText(s, x, y, size, c);
    else DrawTextEx(gFont, s, Vector2{(float)x, (float)y}, (float)size, 1.0f, c);
}

// build a list of info lines for a building (name, consume, produce, workers, housing, build time)
static std::vector<std::string> base_info_lines(const colony::BaseData* bd) {
    std::vector<std::string> L;
    L.push_back(bd->caption);
    L.push_back(TextFormat("Стоимость: %lld", (long long)bd->price));
    std::string cons, prod;
    for (int i = 0; i < 9; i++) {
        int64_t c = bd->consume[i];
        if (c > 0) { if (!cons.empty()) cons += ", "; cons += TextFormat("%s %lld", RES_SHORT[i], (long long)c); }
        int64_t p = bd->profit[i];
        if (p > 0) { if (!prod.empty()) prod += ", "; prod += TextFormat("%s %lld", RES_SHORT[i], (long long)p); }
    }
    L.push_back("Потребляет: " + (cons.empty() ? std::string("нет") : cons));
    L.push_back("Производит: " + (prod.empty() ? std::string("нет") : prod));
    L.push_back(TextFormat("Нужно рабочих: %lld", (long long)bd->need_workers));
    if (bd->home_places) L.push_back(TextFormat("Жильё: %lld", (long long)bd->home_places));
    if (bd->build_time)  L.push_back(TextFormat("Время стр-ва: %lld", (long long)bd->build_time));
    return L;
}

// draw an info box (semi-transparent panel) at (x,y); auto-clamps to screen
static void draw_info_box(const std::vector<std::string>& lines, int x, int y, int boxW) {
    int lineH = 18;
    int boxH = (int)lines.size() * lineH + 12;
    if (x + boxW > WIN_W) x = WIN_W - boxW - 4;
    if (y + boxH > WIN_H) y = WIN_H - boxH - 4;
    if (x < 0) x = 0;
    if (y < 0) y = 0;
    DrawRectangle(x, y, boxW, boxH, Color{20, 26, 18, 245});
    DrawRectangleLinesEx(Rectangle{(float)x, (float)y, (float)boxW, (float)boxH}, 1, C_ACCENT);
    for (size_t k = 0; k < lines.size(); k++)
        text(lines[k].c_str(), x + 8, y + 6 + (int)k * lineH, k == 0 ? 18 : 15,
             k == 0 ? C_WHITE : C_WHITE);
}

// dialog "age" guard so clicking outside doesn't instantly close a just-opened dialog
static int g_dlg_age = 0;
// dialog open at the end of the previous frame (used to avoid closing a dialog on the
// very frame it was opened, e.g. via a mouse click in the menu)
static int g_dlg_prev_frame = 0;
// selected base coords (resolved on use, never a dangling pointer)
static int sel_bx = -1, sel_by = -1;

// build a building (id = A_BUILD0 + i) into either a committed selection area
// or a single cell; called from the right-click building popup
static bool can_build_cell(const Game& g, const BaseData& bd, int x, int y) {
    return g.can_build_at(bd, x, y).first;
}
static bool game_over = false;

// ═══════════════════════════════════════════════════════════════
// UX-слой: сообщения, активная клетка, предпросмотр застройки
// ═══════════════════════════════════════════════════════════════
//
// Раньше строка `status` («Место занято», «Недостаточно денег», «Построено:
// 12», «Здание должно примыкать…») собиралась в обработчиках и НИГДЕ не
// выводилась — игрок не получал ни одного объяснения, почему постройка не
// встала. Теперь каждое сообщение попадает в журнал и показывается поверх
// карты несколько секунд.
struct LogMsg { std::string text; float age; bool error; };
static std::vector<LogMsg> g_log;        // последние сообщения (новые в конце)
static const size_t LOG_KEEP = 6;
static const float LOG_TTL = 6.0f;

// Сообщение-ошибка? Нужен только цвет, поэтому эвристика по началу строки.
static bool msg_is_error(const std::string& s) {
    static const char* bad[] = {
        "Нельзя", "Недостаточно", "Не ", "Нет ", "Место занято", "Нужно",
        "Здание должно", "Постройка закрыта", "Неподходящий", "Вне карты",
        "Там нет", "Эту постройку", "Нечего", "Ремонт не требуется",
    };
    for (const char* b : bad)
        if (s.rfind(b, 0) == 0) return true;
    return false;
}
static void push_log(const std::string& s) {
    if (s.empty()) return;
    if (!g_log.empty() && g_log.back().text == s) { g_log.back().age = 0.0f; return; }
    g_log.push_back({s, 0.0f, msg_is_error(s)});
    while (g_log.size() > LOG_KEEP) g_log.erase(g_log.begin());
}
static void tick_log(float dt) {
    for (auto& m : g_log) m.age += dt;
    while (!g_log.empty() && g_log.front().age > LOG_TTL) g_log.erase(g_log.begin());
}

// ─── Активная клетка ───────────────────────────────────────────
// Контракт: активная клетка существует ВСЕГДА (стартует на городе), её
// двигают WASD/стрелки и клик мышью, камера сама подтягивается следом.
static int act_x = -1, act_y = -1;
static void reset_active_cell(const Game& g) {
    act_x = g.earth.init_sel_x;
    act_y = g.earth.init_sel_y;
}
static void clamp_active_cell(const Game& g) {
    const int ms = g.map_size();
    if (act_x < 0 || act_y < 0) { reset_active_cell(g); return; }
    act_x = std::max(0, std::min(ms - 1, act_x));
    act_y = std::max(0, std::min(ms - 1, act_y));
}

// ─── Предпросмотр застройки ────────────────────────────────────
// Логика живёт в include/colony/build_preview.h и проверяется пробой
// tests/cpp/build_preview_check.cpp (сверка с Game::can_build_at по всей
// карте) — gui.cpp под Linux не собрать, а ошибаться подсветкой нельзя.
static std::vector<char> g_conn;        // 1 = клетка примыкает к колонии
static int g_conn_ms = -1;

static void refresh_connectivity(const Game& g) {
    // Считается заново каждый кадр и только когда включён режим стройки:
    // один обход карты (≤78 тыс. клеток) против 2500 BFS, если звать
    // can_build_at() на каждую клетку выделения. Кэшировать нельзя — после
    // постройки/сноса/undo карта меняется, а подсветка обязана быть точной.
    g_conn_ms = g.map_size();
    colony::preview::connectivity_map(g, g_conn);
}
static std::string preview_reason(const Game& g, const BaseData& bd, int x, int y) {
    return colony::preview::reason(g, bd, x, y, g_conn);
}
static bool preview_ok(const Game& g, const BaseData& bd, int x, int y) {
    return colony::preview::ok(g, bd, x, y, g_conn);
}

// Разделитель разрядов для денег: 48 400 читается, 48400 — нет.
static std::string money_str(int64_t v) {
    char raw[32];
    snprintf(raw, sizeof(raw), "%lld", (long long)(v < 0 ? -v : v));
    std::string s(raw), out;
    int c = 0;
    for (int i = (int)s.size() - 1; i >= 0; --i) {
        out += s[(size_t)i];
        if (++c % 3 == 0 && i > 0) out += ' ';
    }
    if (v < 0) out += '-';
    std::reverse(out.begin(), out.end());
    return out;
}

// ═══ Demo recording (behavioral cloning): --record-demo PATH ═══
//
// Задача: обучать модель на уже сыгранной человеком партии (imitation
// learning / behavioral cloning) — см. docs/IMITATION_LEARNING_2026_09.md.
// Ключевой факт, из-за которого просто "логировать клики мышью" не работает:
// RL-действия (env.step()) НЕ принимают координаты — строительство ищет
// клетку через find_lot() внутри среды, а "менеджерские" действия
// (REPAIR/DESTROY/PRESERVE/...) сами выбирают цель (find_slowest_base() и
// т.п.). Человеческий интерактивный режим этого gui.cpp обычно вызывает
// g.build()/gm.restore()/gm.destroy() НАПРЯМУЮ, в обход env.step() — значит
// obs/reward/action_mask, которые видел бы RL-агент, для этих кликов вообще
// не считаются, и клик нельзя один-в-один сопоставить (obs, action) паре.
//
// Поэтому запись демонстраций — ОТДЕЛЬНЫЙ opt-in режим (--record-demo), а не
// изменение поведения по умолчанию: когда он включён, клики по постройкам и
// по кнопкам менеджеров (см. do_build_area() и обработчики ниже) вместо
// прямого вызова Game::* идут через env.step(action) — ровно тот интерфейс,
// которым пользуется RL-политика. Расплата: точное место (x,y), которое
// выбрал человек, и произвольная сумма кредита/продажи ИГНОРИРУЮТСЯ — пишется
// только «какое решение принял человек» (тип действия), а конкретную клетку
// или сумму, как и для RL-агента, выбирает сама среда. Это не баг: RL-агент
// тоже не может указать точную клетку, так что демонстрация обязана быть
// на том же уровне абстракции, иначе датасет для BC не будет соответствовать
// пространству действий политики.
//
// Без --record-demo (по умолчанию) ничего не меняется: do_build_area() и
// остальные обработчики работают как раньше, байт в байт.
static bool g_record_demo = false;
static std::ofstream* g_demo_file = nullptr;
static int g_demo_steps_written = 0;

static void demo_record_open(const std::string& path) {
    g_demo_file = new std::ofstream(path, std::ios::app);
    if (!g_demo_file->is_open()) {
        fprintf(stderr, "ERROR: не удалось открыть --record-demo файл: %s\n", path.c_str());
        delete g_demo_file;
        g_demo_file = nullptr;
        g_record_demo = false;
        return;
    }
    g_record_demo = true;
}

// Одна строка JSONL на шаг: тот же набор полей, что и ai_write_state (obs/
// action_mask/action/reward/terminated), чтобы датасет для BC читался тем же
// парсером, что и headless-ai state.json (см. rl/bc_dataset.py). Демо-файл —
// append-only лог одного процесса (не читается конкурентно во время записи),
// поэтому atomic tmp+rename здесь не нужен, в отличие от ai_write_state.
static void demo_record_step(const std::vector<float>& obs, int action,
                             const std::vector<float>& mask, double reward,
                             bool terminated) {
    if (!g_record_demo || g_demo_file == nullptr) return;
    std::ofstream& f = *g_demo_file;
    f << "{\"action\":" << action << ",\"reward\":" << reward
      << ",\"terminated\":" << (terminated ? "true" : "false") << ",\"obs\":[";
    for (size_t i = 0; i < obs.size(); i++) { f << obs[i]; if (i + 1 < obs.size()) f << ","; }
    f << "],\"action_mask\":[";
    for (size_t i = 0; i < mask.size(); i++) { f << mask[i]; if (i + 1 < mask.size()) f << ","; }
    f << "]}\n";
    f.flush();
    g_demo_steps_written++;
}

static void demo_record_close() {
    if (g_demo_file != nullptr) {
        g_demo_file->flush();
        g_demo_file->close();
        delete g_demo_file;
        g_demo_file = nullptr;
    }
}

// Единая точка входа "человек принял решение action" для режима записи:
// снимает obs/mask ДО шага (это и есть состояние, на основе которого принято
// решение — то же самое, что видит RL-политика в rollout), делает env.step
// (тот же вызов, что использует и headless-ai, и обучение), и пишет тройку
// (obs, action, reward/terminated) в демо-файл. Вне режима записи это
// эквивалент обычного env.step(action) без побочных эффектов.
static ColonyEnvCpp::StepOut env_step_and_record(ColonyEnvCpp& env, int action) {
    if (g_record_demo) {
        std::vector<float> pre_obs = env.obs();
        std::vector<float> pre_mask = env.action_mask();
        auto out = env.step(action);
        demo_record_step(pre_obs, action, pre_mask, out.rew, out.terminated);
        return out;
    }
    return env.step(action);
}


static void do_build_area(ColonyEnvCpp& env, Game& g, int action, bool area,
                          int x0, int y0, int x1, int y1, int px, int py,
                          std::string& status) {
    int id = action - A_BUILD0;
    if (id < 0 || id >= (int)env.build_data().size()) return;
    const BaseData* bd = env.build_data()[id];
    const std::string bid = bd->id;
    int msz = g.map_size();
    int ax0, ay0, ax1, ay1;
    if (area) { ax0 = std::min(x0, x1); ay0 = std::min(y0, y1); ax1 = std::max(x0, x1); ay1 = std::max(y0, y1); }
    else { ax0 = ax1 = px; ay0 = ay1 = py; }
    // Gather all empty cells in the area
    struct Cell { int x, y; };
    std::vector<Cell> cells;
    for (int y = ay0; y <= ay1; y++)
        for (int x = ax0; x <= ax1; x++) {
            if (x < 0 || y < 0 || x >= msz || y >= msz) continue;
            if (g.base_in_box(x, y)) continue;
            cells.push_back({x, y});
        }
    if (cells.empty()) { status = "Место занято"; return; }

    if (g_record_demo) {
        // BUILD_<id> (см. заголовок блока demo recording выше): RL сама
        // выбирает клетку через find_lot() внутри env.step() — точная
        // клетка/область, которую выделил человек, здесь НЕ используется,
        // важно только «сколько построек этого типа» человек хотел поставить
        // (одна — одиночный клик, area.size() — протяжка). Каждая постройка —
        // отдельный env.step(action), т.е. отдельный день, как у RL-агента.
        int built = 0;
        for (size_t i = 0; i < cells.size(); ++i) {
            if (!env.build_allowed(bid)) { status = "Постройка закрыта курикулумом"; break; }
            auto out = env_step_and_record(env, action);
            built++;
            if (out.terminated) { game_over = true; break; }
        }
        status = built > 0 ? TextFormat("Построено (демо-режим): %d", built)
                            : "Не удалось построить (см. лог)";
        return;
    }
    // If buildings exist, start from cells adjacent to existing buildings
    // Otherwise start from top-left corner
    std::vector<Cell> frontier, remaining;
    if (!g.bases.empty()) {
        for (auto& c : cells)
            if (can_build_cell(g, *bd, c.x, c.y)) frontier.push_back(c);
            else remaining.push_back(c);
        if (frontier.empty()) { status = "Нужно строить рядом с существующими зданиями"; return; }
    } else {
        frontier.push_back(cells[0]);
        for (size_t i = 1; i < cells.size(); i++) remaining.push_back(cells[i]);
    }
    // Build: always add cells adjacent to already-built to the frontier
    int built = 0, skipped = 0;
    bool out_of_money = false;
    std::string fail_reason;  // PR 2: первая ошибка — гейт виден как гейт
    for (size_t frontier_head = 0; frontier_head < frontier.size(); ++frontier_head) {
        Cell c = frontier[frontier_head];
        auto r = g.build(bid, c.x, c.y);
        if (r.first) {
            built++;
            stat_built[bd->caption]++;
            // Add neighbors from remaining that are now adjacent to a building
            for (auto it = remaining.begin(); it != remaining.end();) {
                if (can_build_cell(g, *bd, it->x, it->y)) {
                    frontier.push_back(*it);
                    it = remaining.erase(it);
                } else it++;
            }
        } else {
            skipped++;
            if (fail_reason.empty()) fail_reason = r.second;
            // Деньги кончились — дальше смысла нет. Любая другая причина
            // (тип земли, сгоревший участок, гейт) касается ОДНОЙ клетки:
            // раньше цикл обрывался на ней и остальная часть выделения молча
            // не застраивалась — со стороны это и выглядело «лесенкой».
            if (r.second == "Недостаточно денег.") { out_of_money = true; break; }
        }
    }
    // Клетки области, до которых застройка вообще не дошла (море, чужой тип
    // земли, нет связи с колонией) — главная причина «дырявой» дороги.
    const int unreachable = (int)remaining.size();
    if (built > 0) {
        std::string msg = TextFormat("Построено: %d", built);
        if (out_of_money)
            msg += TextFormat(", дальше не хватило денег (нужно %lld ₽ за клетку)",
                              (long long)bd->price);
        if (unreachable > 0 || (skipped > 0 && !out_of_money)) {
            int bad = unreachable + (out_of_money ? 0 : skipped);
            msg += TextFormat("; пропущено клеток: %d", bad);
            if (!fail_reason.empty()) msg += " — " + fail_reason;
            else msg += " — море / не тот тип земли / нет связи с колонией";
        }
        status = msg;
    } else if (!fail_reason.empty()) {
        status = fail_reason;
    }
}
static void load_assets() {
    char p[256];
    for (int i = 0; i < 33; i++) { snprintf(p, sizeof(p), "imlBases_%02d.png", i); baseTex[i] = load_one(p); }
    for (int i = 0; i < 70; i++) { snprintf(p, sizeof(p), "imlEarth_%02d.png", i); earthTex[i] = load_one(p); }
    for (int i = 0; i < 27; i++) { snprintf(p, sizeof(p), "imlTools_%02d.png", i); toolTex[i] = load_one(p); }
    for (int i = 0; i < 7; i++)  { snprintf(p, sizeof(p), "imlIcons_%02d.png", i); iconTex[i] = load_one(p); }
    for (int i = 0; i < 4; i++)  { snprintf(p, sizeof(p), "imlSeasons_%02d.png", i); seasonTex[i] = load_one(p); }
    for (int i = 0; i < 12; i++) { snprintf(p, sizeof(p), "imlSelectEarth_%02d.png", i); selEarthTex[i] = load_one(p); }
    for (int i = 0; i < 12; i++) { snprintf(p, sizeof(p), "imlSelectNone_%02d.png", i); selNoneTex[i] = load_one(p); }
    for (int i = 0; i < 9; i++)  { snprintf(p, sizeof(p), "imlMarketItem_%02d.png", i); marketItemTex[i] = load_one(p); }
}

// building id -> imlBases index (ImageIndex из BASES.INI 3.47)
static int base_icon_index(const std::string& id) {
    static const std::pair<const char*, int> m[] = {
        {"City",10},{"Farm",0},{"Garden",1},{"WaterChannel",2},{"Sawmill",3},
        {"Coalmine",4},{"Ironmine",5},{"Refinery",6},{"Goldmine",7},
        {"PowerStation",8},{"HydroStation",9},{"Road",12},{"House",13},
        {"SmallHouse",14},{"Fish",15},{"CoalCut",11},{"HuntingLand",16},
        {"CowFarm",17},{"Mushroom",18},{"BigHouse",19},{"BigFarm",20},
        {"Apiary",21},{"Torchlight",22},{"Hothouse",23},{"SuperHouse",24},
        {"BigSawmill",25},{"WaterMill",26},{"BigRefinary",27},{"Puerperal",28},
        {"BigIronmine",29},{"AirStation",31},{"SmallAtomStation",32},{"AtomStation",30},
    };
    for (auto& e : m) if (id == e.first) return e.second;
    return -1;
}
// lot -> imlEarth index ((LotType-2)*10 + sub)
static int earth_icon_index(int8_t lot, int x, int y) {
    int sub = (x * 3 + y * 7) % 10;
    switch (lot) {
        case LT_WATER: return 0 + sub;
        case LT_WOOD:  return 10 + sub;
        case LT_COAL:  return 20 + sub;
        case LT_IRON:  return 30 + sub;
        case LT_OIL:   return 40 + sub;
        case LT_GOLD:  return 50 + sub;
        default:       return -1;
    }
}

static const char* month_ru(int m) {
    static const char* n[] = {"января","февраля","марта","апреля","мая","июня",
        "июля","августа","сентября","октября","ноября","декабря"};
    return (m >= 1 && m <= 12) ? n[m-1] : "?";
}
static const char* season_ru(Season s) {
    switch (s) {
        case SEASON_SPRING: return "весна";
        case SEASON_SUMMER: return "лето";
        case SEASON_AUTUMN: return "осень";
        case SEASON_WINTER: return "зима";
    }
    return "?";
}

// ═══ Draw building icon (original PNG, fallback to pixel-art) ═══
static void draw_build_pixel(int x, int y, int sz, const std::string& id, bool sel) {
    int bi = base_icon_index(id);
    if (sel) DrawRectangle(x - 2, y - 2, sz + 4, sz + 4, C_ACCENT);
    if (bi >= 0) {
        draw_tex(baseTex[bi], x, y, sz);
    } else {
        Color c = build_color(id);
        DrawRectangle(x, y, sz, sz, {20, 20, 20, 255});
        int bx = x + 4, by = y + 6, bw = sz - 8, bh = sz - 14;
        DrawRectangle(bx, by, bw, bh, c);
        Color dark = {(unsigned char)std::max(0,(int)c.r-60),
                      (unsigned char)std::max(0,(int)c.g-60),
                      (unsigned char)std::max(0,(int)c.b-60), 255};
        DrawRectangle(bx, by - 4, bw, 6, dark);
        int ww = sz / 8, wh = sz / 8;
        DrawRectangle(bx + 4, by + 4, ww, wh, {220, 220, 180, 255});
        DrawRectangle(bx + bw - ww - 4, by + 4, ww, wh, {220, 220, 180, 255});
        DrawRectangle(bx + bw/2 - 3, by + bh - 8, 6, 8, dark);
        DrawRectangle(x + 2, y + sz - 5, sz - 4, 3, {60, 100, 50, 255});
    }
    DrawRectangleLines(x, y, sz, sz, sel ? C_ACCENT : Color{100, 100, 100, 255});
}

// ═══ Time helpers (День / Неделя / Месяц / jump) ═══
static Rectangle g_time_btn[3];
static Rectangle g_speed_btn[4] = {};
static Rectangle g_date_strip = {0, 0, 0, 0};
static Rectangle g_tool_rect[12];

static int abs_day_of(const Game& g) {
    int d = g.day;
    for (int i = 0; i < g.month - 1; i++) d += DAYS_IN_MONTH[i];
    if (is_leap(g.year) && g.month > 2) d += 1;
    return d;
}
static int total_day_of(const Game& g) {
    return g.year * 364 + abs_day_of(g);
}
static void advance_month(ColonyEnvCpp& env) {
    int start = env.game().month;
    for (int k = 0; k < 40; k++) {
        auto o = env.step(A_DAY);
        if (o.terminated) { game_over = true; break; }
        if (env.game().month != start) break;
    }
}
static void jump_to_day(ColonyEnvCpp& env, int target) {
    int steps = 0;
    while (abs_day_of(env.game()) < target && steps < 1000) {
        if (env.game().month == 12 && env.game().day == 31) break;
        auto o = env.step(A_DAY);
        if (o.terminated) { game_over = true; break; }
        steps++;
    }
}

// ═══ Real terrain minimap (как _render_plan) ═══
static Texture2D g_mini_tex = {0};
static int g_mini_ms = 0;
static uint64_t g_mini_seed = 0;
static void draw_minimap(int mx, int my, int mw, int mh, const Game& g, const Camera2D& cam) {
    int ms = g.map_size();
    // БАГ: текстура кэшировалась только по размеру карты — после «Новая
    // карта» того же размера миникарта продолжала показывать СТАРЫЙ остров.
    if (g_mini_ms != ms || g_mini_seed != (uint64_t)g.earth.seed()) {
        g_mini_seed = (uint64_t)g.earth.seed();
        if (g_mini_tex.id) UnloadTexture(g_mini_tex);
        Image img = GenImageColor(ms, ms, BLANK);
        unsigned char* p = (unsigned char*)img.data;
        for (int y = 0; y < ms; y++)
            for (int x = 0; x < ms; x++) {
                Color c = lot_color(g.earth.lot(x, y));
                int idx = (y * ms + x) * 4;
                p[idx] = c.r; p[idx+1] = c.g; p[idx+2] = c.b; p[idx+3] = 255;
            }
        g_mini_tex = LoadTextureFromImage(img);
        UnloadImage(img);
        g_mini_ms = ms;
    }
    Rectangle src = {0, 0, (float)g_mini_tex.width, (float)g_mini_tex.height};
    Rectangle dst = {(float)mx, (float)my, (float)mw, (float)mh};
    DrawTexturePro(g_mini_tex, src, dst, {0, 0}, 0.0f, WHITE);

    float scale = (float)mw / (float)ms;
    for (const Base& b : g.bases) {
        int bx = mx + (int)(b.x * scale);
        int by = my + (int)(b.y * scale);
        // дороги — отдельным (тёмным) цветом: сеть видно как сеть,
        // а не как россыпь таких же красных точек, что и здания
        if (b.data->id == ROAD_ID) { DrawRectangle(bx, by, 2, 2, {90, 80, 70, 255}); continue; }
        if (b.data->plan) DrawRectangle(bx - 2, by - 2, 4, 4, {200, 40, 30, 255});
        else DrawRectangle(bx - 1, by - 1, 3, 3, {200, 40, 30, 255});
        if (b.is_alarm()) draw_tex(iconTex[0], bx - 7, by - 7, 14);
    }
    float vw = (float)MAP_W / cam.zoom;
    float vh = (float)MAP_H / cam.zoom;
    float vx0 = cam.target.x - vw / 2.0f;
    float vy0 = cam.target.y - vh / 2.0f;
    int rx = mx + (int)(vx0 * scale);
    int ry = my + (int)(vy0 * scale);
    int rw = (int)(vw * scale);
    int rh = (int)(vh * scale);
    DrawRectangleLines(rx, ry, rw, rh, {250, 250, 235, 255});
    DrawRectangleLines(rx - 1, ry - 1, rw + 2, rh + 2, {250, 250, 235, 255});
    DrawRectangleLines(mx, my, mw, mh, {80, 90, 110, 255});
}

// ═══════════════════════════════════════════════════════════════
// DIALOG / MENU STATE (immediate mode)
// ═══════════════════════════════════════════════════════════════
enum Dlg { DLG_NONE, DLG_MARKET, DLG_BANK, DLG_NALON, DLG_NALON_MAIN,
           DLG_SEASON, DLG_ABOUT, DLG_MESS, DLG_NEW, DLG_OPEN, DLG_SAVE,
           DLG_CONFIRM };
static Dlg cur_dlg = DLG_NONE;
static bool tax_from_dialog = false;  // рынок открыт из диалога налогов
static int open_menu = -1;
static int dlg_mode = 0;
static long long market_qty[9] = {0};
static char bank_buf[32] = {0};
static int dlg_season = 0;
static char mess_title[64] = "Сообщение";
// 4 КБ: текст помощи в UTF-8 (кириллица — 2 байта на символ)
static char mess_text[4096] = "";
static bool want_close = false;

// ═══ Headless AI mode (policy-driven) ═══
static bool headless_ai = false;
static std::string ai_actions_path;
static std::string ai_state_path;
static int ai_last_action = -1;   // last action written by Python
static bool ai_terminated = false;
static float ai_reset_timer = 0.0f;
static char ai_build_msg[128] = "";
static float ai_build_msg_timer = 0.0f;
static int ai_step_count = 0;
static FILE* ai_debug_log = nullptr;

// Разбор actions.txt — colony/watch_ipc.h (NONE / OK / INVALID, P3-4 ревью).
static ActionReadResult ai_read_action(int n_actions) {
    if (ai_actions_path.empty()) return {};
    return read_action_file(ai_actions_path, n_actions);
}

static void ai_write_state(const Game& g, const std::vector<float>& obs, int action, bool terminated, const std::vector<float>& mask, const std::vector<float>& minimap, double reward = 0.0, const std::string& error = std::string()) {
    // Write state JSON to file (includes obs, action_mask, and minimap for Python policy)
    if (ai_state_path.empty()) return;
    // Атомарно: сначала .tmp, затем rename. Драйвер (watch_champion.py) читает
    // state.json в цикле, а файл с миникартой 8×32×32 — это ~100 КБ текста:
    // раньше он регулярно читался наполовину пустым и шаг выпадал.
    const std::string tmp_path = ai_state_path + ".tmp";
    {
        std::ofstream f(tmp_path, std::ios::trunc);
        if (!f.is_open()) return;
        f << "{"
          << "\"day\":" << g.day << ","
          << "\"month\":" << g.month << ","
          << "\"year\":" << g.year << ","
          << "\"people\":" << g.people << ","
          << "\"bases\":" << g.bases.size() << ","
          << "\"money\":" << g.money << ","
          << "\"reward\":" << reward << ","
          << "\"action\":" << action << ","
          << "\"terminated\":" << (terminated ? "true" : "false") << ",";
        // Ответ на невалидное действие: шаг НЕ делался, obs — текущее состояние.
        // Строка формируется окном (без кавычек/обратных слешей) — экранировать нечего.
        if (!error.empty()) f << "\"error\":\"" << error << "\",";
        f << "\"obs\":[";
        for (size_t i = 0; i < obs.size(); i++) {
            f << obs[i];
            if (i + 1 < obs.size()) f << ",";
        }
        f << "],"
          << "\"action_mask\":[";
        for (size_t i = 0; i < mask.size(); i++) {
            f << mask[i];
            if (i + 1 < mask.size()) f << ",";
        }
        f << "],"
          << "\"minimap\":[";
        for (size_t i = 0; i < minimap.size(); i++) {
            f << minimap[i];
            if (i + 1 < minimap.size()) f << ",";
        }
        f << "]"
          << "}";
        f.flush();
        const bool ok = f.good();
        f.close();
        if (!ok) {
            std::error_code ec;
            std::filesystem::remove(tmp_path, ec);
            return;
        }
    }
    // На Windows rename не проходит, пока читатель держит целевой файл открытым
    // (в т.ч. антивирус/индексатор, который открывает файл на сканирование —
    // см. внешнее ревью, "File IPC Latency" / "Обработка исключений при
    // неполной записи JSON"): раньше было всего 5×2мс=10мс попыток, и любая
    // задержка чтения дольше этого окна проваливалась в fallback ниже.
    // Бэкофф увеличен (~15 попыток, суммарно до ~300мс) — это покрывает
    // типичную короткую блокировку файла антивирусом и почти никогда не
    // доходит до non-atomic fallback.
    for (int attempt = 0; attempt < 15; ++attempt) {
        std::error_code ec;
        std::filesystem::rename(tmp_path, ai_state_path, ec);
        if (!ec) return;
        std::this_thread::sleep_for(std::chrono::milliseconds(2 + attempt * 2));
    }
    // Крайний случай — файл занят читателем аномально долго. copy_file
    // поверх ai_state_path НЕ атомарен: читатель может увидеть файл наполовину
    // перезаписанным (обрезанный/битый JSON). Поэтому сначала копируем во
    // временный файл РЯДОМ (не в целевой путь), и уже его пытаемся
    // переименовать в цель ещё раз — rename atomic, а окно гонки остаётся
    // только в единственном случае, когда цель занята непрерывно все ~300мс+
    // ретраев, что многократно снижает (но не гарантированно устраняет —
    // читающая сторона всё равно ретраит при неудачном json.load) вероятность
    // отдать читателю обрезанный файл.
    const std::string tmp2_path = ai_state_path + ".tmp2";
    std::error_code ec;
    std::filesystem::copy_file(tmp_path, tmp2_path,
                               std::filesystem::copy_options::overwrite_existing, ec);
    if (!ec) {
        for (int attempt = 0; attempt < 5; ++attempt) {
            std::error_code ec2;
            std::filesystem::rename(tmp2_path, ai_state_path, ec2);
            if (!ec2) { std::filesystem::remove(tmp_path, ec); return; }
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
        // Совсем крайний случай: и второй rename не прошёл — публикуем
        // содержимое напрямую (старое поведение), иначе состояние вовсе не
        // обновится и драйвер уйдёт в таймаут.
        std::filesystem::copy_file(tmp2_path, ai_state_path,
                                   std::filesystem::copy_options::overwrite_existing, ec);
        std::filesystem::remove(tmp2_path, ec);
    }
    std::filesystem::remove(tmp_path, ec);
}

static bool sound_on = true;
static bool fullscreen = false;
static Camera2D cam = {0};
static Season prev_season = SEASON_SPRING;

static void ai_reset_env(ColonyEnvCpp& env) {
    int64_t new_seed = (int64_t)GetRandomValue(1, 999999999);
    env.reset(new_seed);
    game_over = false;
    ai_terminated = false;
    ai_reset_timer = 0.0f;
    sel_bx = sel_by = -1;
    reset_active_cell(env.game());
    cur_dlg = DLG_NONE;
    stat_built.clear(); stat_earned = 0; stat_spent = 0;
    stat_started = false; stat_tax_over = false;
    ai_build_msg[0] = '\0'; ai_build_msg_timer = 0.0f;
    cam.target = {(float)env.game().earth.init_sel_x * TILE,
                  (float)env.game().earth.init_sel_y * TILE};
    cam.zoom = 1.0f;
    // Clear action file so Python knows to send a new one
    {
        std::error_code ec;
        std::filesystem::remove(ai_actions_path, ec);
    }
    // Write fresh state so Python sends a new action
    ai_write_state(env.game(), env.obs(), -1, false, env.action_mask(), env.minimap(), 0.0);
}

static bool btn(int x, int y, int w, int h, const char* label, bool enabled = true) {
    Rectangle r = {(float)x, (float)y, (float)w, (float)h};
    Vector2 mp = GetMousePosition();
    bool hover = enabled && CheckCollisionPointRec(mp, r);
    if (enabled && hover && IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) return true;
    DrawRectangleRec(r, !enabled ? Color{90,90,90,255}
                     : (hover ? Color{60,120,200,255} : Color{40,80,150,255}));
    DrawRectangleLines(x, y, w, h, WHITE);
    int fs = 18, tw = (int)MeasureTextEx(gFont, label, (float)fs, 1.0f).x;
    text(label, x + (w - tw)/2, y + (h - fs)/2, fs, WHITE);
    return false;
}
static void panel(int x, int y, int w, int h, const char* title) {
    DrawRectangle(0, 0, WIN_W, WIN_H, {0, 0, 0, 160});
    DrawRectangle(x, y, w, h, {14, 18, 26, 255});
    DrawRectangle(x, y, w, 28, {40, 80, 150, 255});
    if (title) text(title, x + 10, y + 5, 18, WHITE);
    DrawRectangleLines(x, y, w, h, WHITE);
    // click outside the box closes the dialog (guard against the opening click;
    // never close on the frame the dialog was just opened)
    if (cur_dlg != DLG_NONE && g_dlg_age > 2 && cur_dlg == (Dlg)g_dlg_prev_frame
        && IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) {
        Vector2 mp = GetMousePosition();
        if (mp.x < x || mp.x > x + w || mp.y < y || mp.y > y + h) cur_dlg = DLG_NONE;
    }
}

// ─── MARKET ───
static void draw_market(ColonyEnvCpp& env) {
    Game& g = env.game();
    int w = 440, h = 520, x = (WIN_W - w)/2, y = (WIN_H - h)/2;
    panel(x, y, w, h, dlg_mode == 0 ? "Покупка" : "Продажа");
    for (int i = 0; i < 9; i++) {
        int ry = y + 40 + i * 46;
        draw_tex(toolTex[i], x + 14, ry, 28);
        int price = dlg_mode == 0 ? BUY_SUNDUK[i] : SALE_SUNDUK[i];
        text(RES_SHORT[i], x + 50, ry + 2, 18, WHITE);
        text(TextFormat("цена %lld", (long long)price), x + 50, ry + 22, 14, {190,200,210,255});
        if (btn(x + 300, ry, 28, 28, "+")) market_qty[i]++;
        text(TextFormat("%lld", (long long)market_qty[i]), x + 330, ry + 4, 18, WHITE);
        if (btn(x + 380, ry, 28, 28, "-") && market_qty[i] > 0) market_qty[i]--;
    }
    if (btn(x + 14, y + h - 44, 150, 34, dlg_mode == 0 ? "Режим: Купить" : "Режим: Продать"))
        dlg_mode = 1 - dlg_mode;
    if (dlg_mode == 1 && btn(x + 174, y + h - 44, 90, 34, "Все"))
        for (int i = 0; i < 9; i++) market_qty[i] = g.sunduk[i];
    if (btn(x + w - 184, y + h - 44, 86, 34, dlg_mode == 0 ? "Купить" : "Продать")) {
        Sunduk s{}; for (int i = 0; i < 9; i++) s[i] = market_qty[i];
        auto r = dlg_mode == 0 ? g.market_buy(s) : g.market_sell(s);
        snprintf(mess_title, sizeof(mess_title), "%s", r.ok ? "Сообщение" : "Ошибка");
        snprintf(mess_text, sizeof(mess_text), "%s", r.msg.c_str());
        for (int i = 0; i < 9; i++) market_qty[i] = 0;
        cur_dlg = DLG_MESS;
    }
    if (btn(x + w - 90, y + h - 44, 80, 34, "Отмена")) {
        for (int i = 0; i < 9; i++) market_qty[i] = 0;
        cur_dlg = DLG_NONE;
    }
}

// ─── BANK ───
static void draw_bank(ColonyEnvCpp& env) {
    Game& g = env.game();
    int w = 320, h = 200, x = (WIN_W - w)/2, y = (WIN_H - h)/2;
    panel(x, y, w, h, "Банк");
    text("Задолженность:", x + 14, y + 40, 16, WHITE);
    text(TextFormat("%lld", (long long)g.credit), x + 160, y + 40, 18, {255,120,120,255});
    text("+(0.2% в день)", x + 14, y + 64, 14, {190,200,210,255});
    text(bank_buf[0] ? bank_buf : "0", x + 14, y + 96, 20, {255,230,120,255});
    DrawRectangleLines(x + 14, y + 92, 180, 28, WHITE);
    if (btn(x + 210, y + 90, 90, 30, "Взять (T)")) {
        long long v = bank_buf[0] ? (long long)strtoll(bank_buf, nullptr, 10) : g.credit;
        if (v > 0) { auto r = g.bank_take(v); if (!r.first) { snprintf(mess_title,sizeof(mess_title),"%s","Ошибка"); snprintf(mess_text,sizeof(mess_text),"%s",r.second.c_str()); cur_dlg=DLG_MESS; } else cur_dlg=DLG_NONE; }
        bank_buf[0] = 0;
    }
    if (btn(x + 210, y + 126, 90, 30, "Вернуть (Q)")) {
        long long v = bank_buf[0] ? (long long)strtoll(bank_buf, nullptr, 10) : g.credit;
        if (v > 0) { auto r = g.bank_give(v); if (!r.first) { snprintf(mess_title,sizeof(mess_title),"%s","Ошибка"); snprintf(mess_text,sizeof(mess_text),"%s",r.second.c_str()); cur_dlg=DLG_MESS; } else cur_dlg=DLG_NONE; }
        bank_buf[0] = 0;
    }
    if (btn(x + 14, y + h - 40, 100, 30, "Отмена")) { cur_dlg = DLG_NONE; bank_buf[0] = 0; }
}

// ─── NALOG (annual / main) — 2 кнопки: Продать ресурсы / Нет ───
static void draw_nalog(ColonyEnvCpp& env, bool main_tax) {
    Game& g = env.game();
    int w = 460, h = main_tax ? 200 : 250, x = (WIN_W - w)/2, y = (WIN_H - h)/2;
    panel(x, y, w, h, main_tax ? "Главный налог" : "Налоги");
    long long total = main_tax ? g.main_tax_amount() : g.annual_tax_amount();
    text(main_tax ? "Каждые 10 лет выплачивается Главный налог."
                      : "Ежегодно следует уплатить налоги за прошедший год:",
             x + 14, y + 40, 14, WHITE);
    if (!main_tax) {
        // Разбор по статьям: видно, за что именно платим (земля/обороты/сборы).
        text(TextFormat("Земля: %lld клеток x %d = %lld   (выкупленные не в счёт)",
                        (long long)g.taxed_cells(), NALOG_EARTH,
                        (long long)(g.taxed_cells() * NALOG_EARTH)),
             x + 14, y + 58, 13, {200,200,200,255});
        text(TextFormat("Закупки %lld -> 1%% = %lld    Продажи %lld -> 2%% = %lld",
                        (long long)g.summ_buy,
                        (long long)(g.summ_buy * NALOG_BUYPERCENT / 100),
                        (long long)g.summ_sale,
                        (long long)(g.summ_sale * NALOG_SALEPERCENT / 100)),
             x + 14, y + 74, 13, {200,200,200,255});
        text(TextFormat("Сборы: %d", NALOG_ECOLOGY + NALOG_SOCIAL + NALOG_RES),
             x + 14, y + 90, 13, {200,200,200,255});
    }
    text(TextFormat("Итого: %lld", (long long)total), x + 14, y + (main_tax ? 70 : 110), 20, {255,220,120,255});
    text(TextFormat("Ваши деньги: %lld — не хватает.", (long long)g.money), x + 14, y + (main_tax ? 95 : 135), 14, {255,150,150,255});
    if (btn(x + 14, y + h - 44, 190, 34, "Продать ресурсы")) {
        dlg_mode = 1;
        tax_from_dialog = true;
        cur_dlg = DLG_MARKET;
    }
    if (btn(x + w - 204, y + h - 44, 190, 34, "Нет")) {
        game_over = true;
        stat_tax_over = true;
        cur_dlg = DLG_NONE;
    }
}

// ─── SEASON ───
static void draw_season() {
    int w = 220, h = 160, x = (WIN_W - w)/2, y = (WIN_H - h)/2;
    panel(x, y, w, h, season_ru((Season)dlg_season));
    draw_tex(seasonTex[dlg_season], x + (w - 110)/2, y + 36, 110);
    const char* txt = dlg_season == 0 ? "Наступило лето" : dlg_season == 1 ? "Наступила осень"
                   : dlg_season == 2 ? "Наступила зима" : "Наступила весна";
    text(txt, x + 20, y + 118, 18, WHITE);
    // Auto-close after ~2 seconds in headless AI mode (no user to click ОК)
    if (headless_ai && g_dlg_age > 120) { cur_dlg = DLG_NONE; return; }
    if (btn(x + (w - 80)/2, y + h - 36, 80, 28, "ОК")) cur_dlg = DLG_NONE;
}

// ─── ABOUT ───
static void draw_about() {
    int w = 400, h = 260, x = (WIN_W - w)/2, y = (WIN_H - h)/2;
    panel(x, y, w, h, "О программе");
    text("Экономическая пошаговая стратегия", x + 20, y + 44, 16, WHITE);
    text("Версия 3.47", x + 20, y + 70, 16, WHITE);
    text("http://zgsprojects.narod.ru", x + 20, y + 110, 14, {120,200,255,255});
    text("zgsuser@e-mail.ru", x + 20, y + 164, 14, {120,200,255,255});
    if (btn(x + w - 100, y + h - 40, 80, 30, "OK")) cur_dlg = DLG_NONE;
}

// ─── MESS ───
static void draw_mess() {
    // Окно подстраивается под текст: справка теперь длинная (раньше короткая
    // строка была зашита в 320x140, и любой многострочный текст вылезал).
    std::vector<std::string> lines;
    {
        std::string cur;
        for (const char* p = mess_text; *p; ++p) {
            if (*p == '\n') { lines.push_back(cur); cur.clear(); }
            else cur += *p;
        }
        lines.push_back(cur);
    }
    int lh = 19, maxw = 240;
    for (const std::string& l : lines)
        maxw = std::max(maxw, (int)MeasureTextEx(gFont, l.c_str(), 16.0f, 1.0f).x);
    int w = std::min(WIN_W - 40, maxw + 36);
    int h = std::min(WIN_H - 40, (int)lines.size() * lh + 96);
    int x = (WIN_W - w)/2, y = (WIN_H - h)/2;
    panel(x, y, w, h, mess_title);
    for (size_t i = 0; i < lines.size(); i++)
        text(lines[i].c_str(), x + 16, y + 40 + (int)i * lh, 16, WHITE);
    if (btn(x + (w - 80)/2, y + h - 40, 80, 30, "OK")) cur_dlg = DLG_NONE;
}

static void list_box(int x, int y, int w, int h, const std::vector<std::string>& items,
                     int sel, int& out_click) {
    DrawRectangle(x, y, w, h, {0, 0, 0, 255});
    DrawRectangleLines(x, y, w, h, WHITE);
    int ih = 22;
    for (int i = 0; i < (int)items.size(); i++) {
        int ry = y + i * ih;
        if (ry + ih > y + h) break;
        if (i == sel) DrawRectangle(x, ry, w, ih, {40, 80, 150, 255});
        text(items[i].c_str(), x + 6, ry + 3, 16, WHITE);
        Rectangle r = {(float)x, (float)ry, (float)w, (float)ih};
        if (CheckCollisionPointRec(GetMousePosition(), r) && IsMouseButtonPressed(MOUSE_BUTTON_LEFT))
            out_click = i;
    }
}
static std::vector<std::string> list_saves() {
    std::vector<std::string> v;
    std::error_code ec;
    if (std::filesystem::exists("saves", ec))
        for (auto& f : std::filesystem::directory_iterator("saves", ec))
            if (f.path().extension() == ".ss" || f.path().extension() == ".SS")
                v.push_back(f.path().filename().string());
    return v;
}
static void draw_newgame(ColonyEnvCpp& env) {
    int w = 460, h = 260, x = (WIN_W - w)/2, y = (WIN_H - h)/2;
    panel(x, y, w, h, "Новая игра");
    text("Выберите карту", x + 14, y + 38, 16, WHITE);
    std::vector<std::string> maps = {"Старый город", "Новый город 2", "Сахалин"};
    static int sel = 0;
    list_box(x + 14, y + 64, 300, 150, maps, sel, sel);
    if (btn(x + 330, y + 64, 110, 32, "ОК")) { env.reset(42); reset_active_cell(env.game()); sel_bx = sel_by = -1; cur_dlg = DLG_NONE; tax_from_dialog = false; stat_built.clear(); stat_earned = 0; stat_spent = 0; stat_started = false; stat_tax_over = false; ai_build_msg[0] = '\0'; ai_build_msg_timer = 0.0f; }
    if (btn(x + 330, y + 104, 110, 32, "Отмена")) cur_dlg = DLG_NONE;
}
static void draw_open(ColonyEnvCpp& env) {
    int w = 460, h = 260, x = (WIN_W - w)/2, y = (WIN_H - h)/2;
    panel(x, y, w, h, "Загрузка сохранённой игры");
    text("Выберите игру", x + 14, y + 38, 16, WHITE);
    auto items = list_saves();
    static int sel = -1;
    list_box(x + 14, y + 64, 300, 150, items, sel, sel);
    if (btn(x + 330, y + 64, 110, 32, "ОК")) { snprintf(mess_title, sizeof(mess_title), "%s", "Сообщение"); snprintf(mess_text, sizeof(mess_text), "%s",
        "Загрузка сохранений пока не реализована.\n"
        "Сейчас партию можно начать заново: Игра → Новая (F4),\n"
        "а на экране итогов — «Повторить карту» (тот же сид)."); cur_dlg=DLG_MESS; }
    if (btn(x + 330, y + 104, 110, 32, "Отмена")) cur_dlg = DLG_NONE;
}
static void draw_save() {
    int w = 460, h = 320, x = (WIN_W - w)/2, y = (WIN_H - h)/2;
    panel(x, y, w, h, "Сохранение игры");
    text("Всё сохранённые игры", x + 14, y + 38, 16, WHITE);
    auto items = list_saves();
    static int sel = -1;
    list_box(x + 14, y + 64, 300, 150, items, sel, sel);
    text("Введите название файла", x + 14, y + 224, 14, WHITE);
    if (btn(x + 330, y + 64, 110, 32, "ОК")) { snprintf(mess_title, sizeof(mess_title), "%s", "Сообщение"); snprintf(mess_text, sizeof(mess_text), "%s",
        "Сохранение пока не реализовано: состояние партии\n"
        "(дата, деньги, все постройки, износ, казна) ещё не сериализуется.\n"
        "Сид карты виден в заголовке — его можно переиграть заново."); cur_dlg=DLG_MESS; }
    if (btn(x + 330, y + 104, 110, 32, "Отмена")) cur_dlg = DLG_NONE;
}

// ─── TOP MENU (title + Игра/Сервис/?) ───
static void draw_top_menu(ColonyEnvCpp& env) {
    Vector2 mp = GetMousePosition();
    text("# Сахалинская колония 3.47", 14, 4, 18, C_TITLE);
    const char* tops[3] = {"Игра", "Сервис", "?"};
    int tx[3] = {22, 132, 275};
    for (int i = 0; i < 3; i++) {
        int w = (int)MeasureTextEx(gFont, tops[i], 18.0f, 1.0f).x + 24;
        bool hov = CheckCollisionPointRec(mp, {(float)tx[i], (float)(CH + 2), (float)w, (float)CH});
        if (hov && IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) open_menu = (open_menu == i ? -1 : i);
        text(tops[i], tx[i], CH + 2, 18, open_menu == i ? C_ACCENT : C_MENU);
    }
    if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT) && mp.y > CH + 2 + CH) open_menu = -1;

    if (open_menu == 0) {
        int dy = TOP_H;
        const char* its[] = {"Новая... (F4)", "Загрузить... (F5)", "Сохранить как... (F2)", "Выход"};
        for (int i = 0; i < 4; i++) {
            Rectangle r = {(float)tx[0], (float)(dy + i*CH), 240, CH};
            bool h = CheckCollisionPointRec(mp, r);
            if (h && IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) {
                if (i == 0) cur_dlg = DLG_NEW;
                else if (i == 1) cur_dlg = DLG_OPEN;
                else if (i == 2) cur_dlg = DLG_SAVE;
                else if (i == 3) want_close = true;
                open_menu = -1;
            }
            if (h) DrawRectangleRec(r, {40, 46, 44, 255});
            text(its[i], tx[0] + 10, dy + i*CH + 3, 18, C_MENU);
        }
        DrawRectangleLines(tx[0], dy, 240, 4*CH, C_ACCENT);
    } else if (open_menu == 1) {
        int dy = TOP_H;
        // «Звук» был рабочим с виду переключателем, который ничего не делал:
        // звуков в сборке нет вовсе. Честнее показать его выключенным.
        const char* its[] = {"Звук — нет в сборке", "Музыка (F8)", "Полный экран (F9)", "Параметры (F12)"};
        bool en[] = {false, false, true, false};
        for (int i = 0; i < 4; i++) {
            Rectangle r = {(float)tx[1], (float)(dy + i*CH), 240, CH};
            bool h = en[i] && CheckCollisionPointRec(mp, r);
            if (h && IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) {
                if (i == 0) sound_on = !sound_on;
                else if (i == 2) { fullscreen = !fullscreen; ToggleFullscreen(); }
                open_menu = -1;
            }
            if (h) DrawRectangleRec(r, {40, 46, 44, 255});
            text(its[i], tx[1] + 10, dy + i*CH + 3, 18, en[i] ? C_MENU : C_FG_DIM);
        }
        DrawRectangleLines(tx[1], dy, 240, 4*CH, C_ACCENT);
    } else if (open_menu == 2) {
        int dy = TOP_H;
        const char* its[] = {"Помощь (F1)", "О программе... (О)"};
        for (int i = 0; i < 2; i++) {
            Rectangle r = {(float)tx[2], (float)(dy + i*CH), 240, CH};
            bool h = CheckCollisionPointRec(mp, r);
            if (h && IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) {
                if (i == 0) { snprintf(mess_title, sizeof(mess_title), "%s", "Помощь"); snprintf(mess_text, sizeof(mess_text), "%s", "КАРТА И КЛЕТКА\nWASD или стрелки — двигать активную клетку (она активна всегда)\nЛКМ — выбрать клетку/здание, протяжка — выделить область\nCtrl+стрелки, средняя кнопка мыши, край экрана — сдвиг карты; Home — к городу\nКолесо или +/- — зум (к курсору мыши)\n\nСТРОИТЕЛЬСТВО\nИконка в палитре или ПКМ по карте — взять постройку в руку:\nзелёная подсветка = встанет, красная = нельзя (море, тип земли, нет связи)\nЛКМ — поставить, протяжка — заполнить область, Enter — в активную клетку\nEsc или ПКМ — выйти из режима строительства\n\nВРЕМЯ\nПробел — день, Shift+Пробел — неделя, Ctrl+Пробел — месяц\n0 — пауза, 1/2/3 — автоматический ход времени\n\nДЕЙСТВИЯ НАД АКТИВНОЙ КЛЕТКОЙ\nR — ремонт, Shift+R — ремонт всех, Delete — снести, P — консервация\nG — выкупить участок (снимает налог), F — найти изношенное\nB — купить, M — продать, K — банк, Ctrl+Z — отменить\nF1 помощь, F2 сохранить, F4 новая, F5 загрузить, F9 полный экран"); cur_dlg = DLG_MESS; }
                else cur_dlg = DLG_ABOUT;
                open_menu = -1;
            }
            if (h) DrawRectangleRec(r, {40, 46, 44, 255});
            text(its[i], tx[2] + 10, dy + i*CH + 3, 18, C_MENU);
        }
        DrawRectangleLines(tx[2], dy, 240, 2*CH, C_ACCENT);
    }
}

// ─── TOOL BUTTONS (top band, справа от палитры) ───
struct ToolDef { int img; int action; int row; int col; };
// Подписи кнопок: 12 безымянных иконок игрок мог опознать только методом тыка
// (а половина из них ещё и не работала — см. фикс `over_map` в обработчике).
static const char* TOOL_HINT[13] = {
    "",
    "Выкупить участок под активной клеткой (G) — снимает земельный налог",
    "Найти самую изношенную постройку (F)",
    "Ремонт постройки в активной клетке (R)",
    "Ремонт всех построек (Shift+R)",
    "Снести постройку в активной клетке (Delete)",
    "Отменить последнее действие (Ctrl+Z)",
    "Банк: кредит и погашение (K)",
    "Консервация / расконсервация (P)",
    "Купить ресурсы (B)",
    "Продать ресурсы (M)",
    "Прожить день (Пробел)",
    "Прожить неделю (Shift+Пробел)",
};
static void draw_tools(ColonyEnvCpp& env) {
    (void)env;
    static const ToolDef tools[12] = {
        {9,1,0,0},   {15,2,0,1},  {16,3,0,2},  {24,4,0,3},  {13,5,0,4},  {26,6,0,5},
        {25,7,1,0},  {23,8,1,1},
        {12,9,2,0},  {11,10,2,1}, {10,11,2,4}, {14,12,2,5},
    };
    int hint = -1, hx = 0, hy = 0;
    for (int i = 0; i < 12; i++) {
        int x = TOOL_X0 + tools[i].col * BSTEP;
        int y = PAL_Y0 + tools[i].row * BSTEP;
        g_tool_rect[i] = {(float)x, (float)y, (float)BS, (float)BS};
        Rectangle r = g_tool_rect[i];
        Vector2 mp = GetMousePosition();
        bool hov = CheckCollisionPointRec(mp, r);
        DrawRectangleRec(r, hov ? Color{60,120,200,255} : C_TOOLBTN);
        DrawRectangleLines(x, y, BS, BS, {70, 90, 80, 255});
        draw_tex(toolTex[tools[i].img], x + 3, y + 3, BS - 6);
        if (hov) { hint = tools[i].action; hx = x; hy = y + BS + 4; }
    }
    if (hint > 0 && hint < 13) {
        const char* t = TOOL_HINT[hint];
        int w = (int)MeasureTextEx(gFont, t, 15.0f, 1.0f).x + 16;
        int x = std::min(hx, WIN_W - w - 4);
        DrawRectangle(x, hy, w, 22, Color{20, 26, 18, 245});
        DrawRectangleLines(x, hy, w, 22, C_ACCENT);
        text(t, x + 8, hy + 3, 15, C_WHITE);
    }
}

int main(int argc, char* argv[]) {
    int64_t seed = 42;
    int map_size = 280;
    Curriculum curriculum;  // default: everything allowed
    int minimap_radius = -1;
    std::string reward_config_path;
    std::string record_demo_path;
    bool tax_policy_cli = false, tax_policy_set = false;
    // Bugfix (см. docs/CODE_REVIEW внешнего аудита, "Stale Binary Risk"):
    // раньше нераспознанный/обрезанный флаг молча проглатывался — если
    // watch_champion.py звал exe, собранный ДО появления флага (например
    // --minimap-radius), окно тихо играло по старым правилам протокола, и
    // расхождение всплывало только как необъяснимое поведение модели.
    // Теперь любой незнакомый флаг или флаг без обязательного значения —
    // фатальная ошибка с понятным сообщением в stderr (его видно в
    // gui_output.log и в описании ошибки watch_champion.py), а не тихий
    // no-op.
    static const char* kKnownFlags[] = {
        "--seed", "--map-size", "--curriculum", "--curriculum-all",
        "--minimap-radius", "--headless-ai", "--actions-file", "--state-file",
        "--reward-config", "--tax-to-debt", "--tax-dialog", "--record-demo",
    };
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        bool needs_value = (a == "--seed" || a == "--map-size" || a == "--curriculum" ||
                            a == "--minimap-radius" || a == "--actions-file" ||
                            a == "--state-file" || a == "--reward-config" ||
                            a == "--record-demo");
        if (needs_value && i + 1 >= argc) {
            fprintf(stderr, "ERROR: флаг %s требует значение, но оно не передано "
                            "(старый/несовместимый вызов exe?)\n", a.c_str());
            return 1;
        }
        if (a == "--seed") seed = std::stoll(argv[++i]);
        else if (a == "--map-size") map_size = std::stoi(argv[++i]);
        else if (a == "--curriculum") {
            try {
                curriculum = Curriculum::from_json(argv[++i]);
            } catch (const std::exception& e) {
                fprintf(stderr, "ERROR: %s\n", e.what());
                return 1;
            }
        }
        else if (a == "--curriculum-all") curriculum = Curriculum();
        else if (a == "--minimap-radius") minimap_radius = std::stoi(argv[++i]);
        else if (a == "--headless-ai") headless_ai = true;
        else if (a == "--actions-file") ai_actions_path = argv[++i];
        else if (a == "--state-file") ai_state_path = argv[++i];
        else if (a == "--reward-config") reward_config_path = argv[++i];
        // P0 (2026-09-17): налоговая политика. У GUI для человека — диалог
        // налогов («Нет» = конец игры), у наблюдения за моделью — как при
        // обучении (неоплаченный остаток → долг, календарь идёт всегда).
        // По умолчанию выбирается по режиму (см. ниже), флаги — явное переопределение.
        else if (a == "--tax-to-debt") { tax_policy_cli = true; tax_policy_set = true; }
        else if (a == "--tax-dialog") { tax_policy_cli = false; tax_policy_set = true; }
        // Behavioral cloning: запись демонстраций человека для обучения
        // (см. docs/IMITATION_LEARNING_2026_09.md). Только человеческий
        // (не --headless-ai) режим — headless и так управляется извне и его
        // шаги уже проходят через env.step().
        else if (a == "--record-demo") record_demo_path = argv[++i];
        else {
            fprintf(stderr, "ERROR: неизвестный флаг %s — exe пересобран под другой "
                            "протокол? Известные флаги:", a.c_str());
            for (const char* f : kKnownFlags) fprintf(stderr, " %s", f);
            fprintf(stderr, "\n");
            return 1;
        }
    }

    auto bd = load_base_data("configs/bases.json");
    auto ed = load_events("configs/events.json");

    // Load reward config from JSON file if provided
    colony::RewardConfig rc;
    if (!reward_config_path.empty()) {
        std::ifstream rf(reward_config_path);
        if (rf.is_open()) {
            nlohmann::json rj;
            rf >> rj;
            if (rj.contains("idle_build_penalty")) rc.idle_build_penalty = rj["idle_build_penalty"].get<double>();
            if (rj.contains("idle_build_threshold_days")) rc.idle_build_threshold_days = rj["idle_build_threshold_days"].get<int>();
            if (rj.contains("error_penalty")) rc.error_penalty = rj["error_penalty"].get<double>();
            if (rj.contains("proximity_bonus")) rc.proximity_bonus = rj["proximity_bonus"].get<double>();
            if (rj.contains("build_bonus")) rc.build_bonus = rj["build_bonus"].get<double>();
            if (rj.contains("novelty")) rc.novelty = rj["novelty"].get<double>();
            if (rj.contains("chain_bonus")) rc.chain_bonus = rj["chain_bonus"].get<double>();
            if (rj.contains("chain_daily")) rc.chain_daily = rj["chain_daily"].get<double>();
            if (rj.contains("first_extraction_bonus")) rc.first_extraction_bonus = rj["first_extraction_bonus"].get<double>();
            if (rj.contains("extraction_daily")) rc.extraction_daily = rj["extraction_daily"].get<double>();
            if (rj.contains("need_fill_bonus")) rc.need_fill_bonus = rj["need_fill_bonus"].get<double>();
            if (rj.contains("loan_penalty")) rc.loan_penalty = rj["loan_penalty"].get<double>();
            if (rj.contains("daily_income")) rc.daily_income = rj["daily_income"].get<double>();
            if (rj.contains("survival_bonus")) rc.survival_bonus = rj["survival_bonus"].get<double>();
            if (rj.contains("game_over_penalty")) rc.game_over_penalty = rj["game_over_penalty"].get<double>();
            if (rj.contains("build_cost_penalty")) rc.build_cost_penalty = rj["build_cost_penalty"].get<double>();
            if (rj.contains("disable_net_worth")) rc.disable_net_worth = rj["disable_net_worth"].get<bool>();
            if (rj.contains("disable_daily_income")) rc.disable_daily_income = rj["disable_daily_income"].get<bool>();
            if (rj.contains("disable_provider_bonus")) rc.disable_provider_bonus = rj["disable_provider_bonus"].get<bool>();
            // --- the rest of the profile (previously ignored, so the GUI watch
            // env silently played with different rewards than training) ---
            if (rj.contains("sale_bonus")) rc.sale_bonus = rj["sale_bonus"].get<double>();
            if (rj.contains("tax_daily_bonus")) rc.tax_daily_bonus = rj["tax_daily_bonus"].get<double>();
            if (rj.contains("diversity_bonus")) rc.diversity_bonus = rj["diversity_bonus"].get<double>();
            if (rj.contains("preserve_penalty")) rc.preserve_penalty = rj["preserve_penalty"].get<double>();
            if (rj.contains("demolish_penalty")) rc.demolish_penalty = rj["demolish_penalty"].get<double>();
            if (rj.contains("manual_tax_penalty")) rc.manual_tax_penalty = rj["manual_tax_penalty"].get<double>();
            if (rj.contains("survival_coeff")) rc.survival_coeff = rj["survival_coeff"].get<double>();
            if (rj.contains("milestone_base_bonus")) rc.milestone_base_bonus = rj["milestone_base_bonus"].get<double>();
            if (rj.contains("milestone_people_bonus")) rc.milestone_people_bonus = rj["milestone_people_bonus"].get<double>();
            if (rj.contains("milestone_day_bonus")) rc.milestone_day_bonus = rj["milestone_day_bonus"].get<double>();
            if (rj.contains("milestone_year_bonus")) rc.milestone_year_bonus = rj["milestone_year_bonus"].get<double>();
            if (rj.contains("clip_reward_min")) rc.clip_reward_min = rj["clip_reward_min"].get<double>();
            if (rj.contains("clip_reward_max")) rc.clip_reward_max = rj["clip_reward_max"].get<double>();
            if (rj.contains("tax_fail_penalty")) rc.tax_fail_penalty = rj["tax_fail_penalty"].get<double>();
            if (rj.contains("death_penalty")) rc.death_penalty = rj["death_penalty"].get<double>();
            if (rj.contains("base_lost_penalty")) rc.base_lost_penalty = rj["base_lost_penalty"].get<double>();
            if (rj.contains("born_bonus")) rc.born_bonus = rj["born_bonus"].get<double>();
            if (rj.contains("debt_coeff")) rc.debt_coeff = rj["debt_coeff"].get<double>();
            if (rj.contains("home_overflow_penalty")) rc.home_overflow_penalty = rj["home_overflow_penalty"].get<double>();
            if (rj.contains("housing_need_bonus")) rc.housing_need_bonus = rj["housing_need_bonus"].get<double>();
            if (rj.contains("food_need_bonus")) rc.food_need_bonus = rj["food_need_bonus"].get<double>();
            if (rj.contains("water_need_bonus")) rc.water_need_bonus = rj["water_need_bonus"].get<double>();
            if (rj.contains("buy_food_penalty")) rc.buy_food_penalty = rj["buy_food_penalty"].get<double>();
            // P0/P1 (2026-09-17): штраф за налоговый долг и маска по применимости
            if (rj.contains("tax_debt_penalty")) rc.tax_debt_penalty = rj["tax_debt_penalty"].get<double>();
            if (rj.contains("mask_managers_by_applicability"))
                rc.mask_managers_by_applicability = rj["mask_managers_by_applicability"].get<bool>();
        }
    }

    bool gui_no_city_game_over = false;
    int64_t gui_no_people_days = 365;
    {
        std::ifstream gf("configs/gui_config.json");
        if (gf.is_open()) {
            nlohmann::json gj;
            gf >> gj;
            if (gj.contains("no_city_game_over")) gui_no_city_game_over = gj["no_city_game_over"].get<bool>();
            if (gj.contains("no_people_days")) gui_no_people_days = gj["no_people_days"].get<int64_t>();
        }
    }

    // Налоговая политика: явный флаг → иначе по режиму. Человек играет с
    // диалогом налогов («Нет» = конец игры); headless-ai (окно наблюдения за
    // моделью) обязан повторять обучение, иначе watched-модель видит заморозку
    // календаря на 365-м дне, которой при обучении уже нет
    // (tests/cpp/gui_watch_check.cpp: day=365/terminated против day=434).
    const bool gui_tax_to_debt = tax_policy_set ? tax_policy_cli : headless_ai;
    // Курикулум приезжает готовым (--curriculum JSON): C++ его только применяет.
    ColonyEnvCpp env(bd, ed, seed, map_size, curriculum, rc, "normal", gui_no_city_game_over,
                     gui_no_people_days, gui_tax_to_debt);
    if (minimap_radius > 0) {
        env.set_minimap_radius(minimap_radius);
    }
    env.reset(seed);
    const Game& g = env.game();
    prev_season = g.season;
    // manager_base_ (A_BUILD0 + n_build_) — приватное поле ColonyEnvCpp;
    // формула продублирована из src/env.cpp намеренно (public API не меняем
    // ради одного opt-in режима записи демо). MGR_* — те самые действия
    // "менеджеров", которые в RL (env.cpp) сами выбирают цель/сумму.
    const int mgr_base = A_BUILD0 + env.n_build();
    const int MGR_IMPROVE_LAND = mgr_base + 0;
    const int MGR_REPAIR       = mgr_base + 1;
    const int MGR_REPAIR_ALL   = mgr_base + 2;
    const int MGR_DEMOLISH     = mgr_base + 3;
    const int MGR_PRESERVE     = mgr_base + 4;
    const int MGR_UNPRESERVE   = mgr_base + 5;

    // Демо-запись — только для интерактивного (человеческого) режима: у
    // --headless-ai уже есть свой протокол (actions.txt/state.json), где
    // КАЖДЫЙ шаг и так идёт через env.step() и не нуждается в этом опт-ине.
    if (!record_demo_path.empty()) {
        if (headless_ai) {
            fprintf(stderr, "WARNING: --record-demo игнорируется в --headless-ai "
                            "(там протокол watch_champion.py и так пишет каждый шаг)\n");
        } else {
            demo_record_open(record_demo_path);
            if (g_record_demo) {
                fprintf(stderr, "[Demo] Запись демонстрации в %s (n_actions=%d)\n",
                        record_demo_path.c_str(), env.n_actions());
            }
        }
    }

    cam.target = {(float)g.earth.init_sel_x * TILE, (float)g.earth.init_sel_y * TILE};
    cam.offset = {(float)MAP_X + MAP_W/2.0f, (float)MAP_Y + MAP_H/2.0f};
    cam.zoom = 1.0f;
    if (headless_ai) {
        cam.target = {(float)g.map_size() * TILE / 2.0f, (float)g.map_size() * TILE / 2.0f};
        cam.zoom = 0.8f;
        // Публикуем стартовое состояние сразу. Раньше state.json появлялся
        // только ПОСЛЕ первого прочитанного действия, поэтому драйвер обязан был
        // первым отправить действие «вслепую» (гонка при старте наблюдения).
        // Новый протокол: окно пишет состояние при старте и после авто-рестарта
        // карты; старый драйвер от этого не ломается (он всё равно шлёт 0).
        ai_write_state(env.game(), env.obs(), -1, false,
                       env.action_mask(), env.minimap(), 0.0);
    }

    static int sel_action = A_BUILD0;
    static bool selecting = false;
    static int sel_x0 = 0, sel_y0 = 0, sel_x1 = 0, sel_y1 = 0;
    static bool has_sel = false;
    static bool popup_open = false;
    static int popup_x = 0, popup_y = 0, prc_x = 0, prc_y = 0;
    static Rectangle g_mini_rect = {0, 0, 0, 0};
    static Rectangle g_popup_rect = {0, 0, 0, 0};
    std::string status = "";
    // Режим строительства: выбрал здание — курсор «держит» его, ЛКМ ставит,
    // Esc/ПКМ выходит. До этого здание можно было поставить только так:
    // «выдели область → ткни в палитру», причём выделение гасилось после
    // КАЖДОЙ постройки.
    static bool build_mode = false;
    static bool mini_drag = false;
    static bool follow_active = false;   // подтянуть камеру к активной клетке
    // Авто-ход времени: 0 — пауза, иначе дней в секунду.
    static const float SPEED_DPS[4] = {0.0f, 2.0f, 6.0f, 20.0f};
    static int speed_idx = 0;
    static float speed_acc = 0.0f;
    // Подтверждение сноса (ценное здание сносилось одним нажатием D).
    static bool ask_destroy = true;
    static int confirm_x = -1, confirm_y = -1;
    static std::string confirm_text;

    reset_active_cell(g);

    SetConfigFlags(FLAG_VSYNC_HINT);
    InitWindow(WIN_W, WIN_H, "Сахалинская колония 3.47");
    SetTargetFPS(60);
    load_assets();
    // Debug: log startup mode
    {
        FILE* f = fopen("ai_debug_gui.log", "a");
        if (f) {
            fprintf(f, "=== GUI STARTUP === headless=%d actions_path='%s' state_path='%s' seed=%lld map_size=%d curriculum_all=%d allowed=%zu stage=%d tax_to_debt=%d\n",
                    headless_ai, ai_actions_path.c_str(), ai_state_path.c_str(),
                    (long long)seed, map_size, (int)curriculum.all_builds,
                    curriculum.allowed_builds.size(), curriculum.stage_report,
                    (int)env.tax_to_debt());
            fclose(f);
        }
    }
    {
        static int cps[512];
        int n = 0;
        for (int c = 32; c < 127; c++) cps[n++] = c;          // ASCII
        for (int c = 0x0400; c <= 0x04FF; c++) cps[n++] = c;   // Cyrillic
        cps[n++] = (int)'ё'; cps[n++] = (int)'Ё';
        gFont = LoadFontEx("C:/Windows/Fonts/arial.ttf", 32, cps, n);
        if (gFont.texture.id == 0) gFont = GetFontDefault();
    }

    while (!WindowShouldClose() && !want_close) {
        // ─── Input ───
        Vector2 mpos = GetMousePosition();
        bool pressed = IsMouseButtonPressed(MOUSE_BUTTON_LEFT);
        bool over_map = (mpos.x >= MAP_X && mpos.x < MAP_X + MAP_W &&
                         mpos.y >= MAP_Y && mpos.y < MAP_Y + MAP_H);

        // ─── Headless AI mode: read action from file, step env ───
        if (headless_ai) {
            if (IsKeyPressed(KEY_ESCAPE)) { want_close = true; }
            // Update build notification timer
            if (ai_build_msg_timer > 0.0f) {
                ai_build_msg_timer -= GetFrameTime();
                if (ai_build_msg_timer <= 0.0f) ai_build_msg[0] = '\0';
            }

            if (game_over) {
                // Auto-reset after 5 seconds
                ai_reset_timer += GetFrameTime();
                if (ai_reset_timer > 5.0f) {
                    ai_reset_env(env);
                }
                // Draw game-over screen (reuse existing code below)
                // ... fall through to draw ...
            } else {
                const ActionReadResult rd = ai_read_action(env.n_actions());
                if (rd.kind == ActionRead::INVALID) {
                    // Файл уже удалён; отвечаем текущим состоянием с ошибкой,
                    // иначе драйвер ждал бы state.json до таймаута.
                    char err[96];
                    snprintf(err, sizeof(err), "invalid action %d (expected 0..%d)",
                             rd.action, env.n_actions() - 1);
                    if (!ai_debug_log) ai_debug_log = fopen("ai_debug_gui.log", "a");
                    if (ai_debug_log) {
                        fprintf(ai_debug_log, "INVALID: %s, path='%s'\n", err, ai_actions_path.c_str());
                        fflush(ai_debug_log);
                    }
                    ai_write_state(env.game(), env.obs(), -1, false, env.action_mask(),
                                   env.minimap(), 0.0, err);
                }
                const int action = rd.kind == ActionRead::OK ? rd.action : -1;
                if (action >= 0) {
                    ai_step_count++;
                    if (!ai_debug_log) {
                        ai_debug_log = fopen("ai_debug_gui.log", "a");
                    }
                    if (ai_debug_log) {
                        fprintf(ai_debug_log, "STEP %d: action=%d money=%lld bases=%d path='%s'\n",
                                ai_step_count, action, (long long)env.game().money,
                                (int)env.game().bases.size(), ai_actions_path.c_str());
                        fflush(ai_debug_log);
                    }
                    ai_last_action = action;
                    sel_action = action;  // highlight in palette
                    // Track bases before step to detect new buildings
                    Game& gstep = env.game();
                    std::unordered_set<int64_t> uids_before_ai;
                    for (const Base& b : gstep.bases) uids_before_ai.insert(b.uid);
                    auto out = env.step(action);
                    if (ai_debug_log) {
                        fprintf(ai_debug_log, "  -> after step: money=%lld bases=%d terminated=%d\n",
                                (long long)gstep.money, (int)gstep.bases.size(), out.terminated);
                        fflush(ai_debug_log);
                    }
                    // Update stat_built for any new buildings added by AI
                    for (const Base& b : gstep.bases) {
                        if (uids_before_ai.find(b.uid) == uids_before_ai.end()) {
                            stat_built[b.data->caption]++;
                            snprintf(ai_build_msg, sizeof(ai_build_msg), "AI построил: %s", b.data->caption.c_str());
                            ai_build_msg_timer = 3.0f;
                        }
                    }
                    if (out.terminated) {
                        game_over = true;
                        ai_terminated = true;
                    }
                    // Write state including obs, action_mask, and minimap for Python policy
                    ai_write_state(env.game(), out.obs, action, out.terminated, env.action_mask(), env.minimap(), out.rew);
                } else if (rd.kind == ActionRead::NONE) {
                    // No action available yet - log once
                    if (!ai_debug_log) {
                        ai_debug_log = fopen("ai_debug_gui.log", "a");
                    }
                    if (ai_debug_log && ai_step_count == 0) {
                        fprintf(ai_debug_log, "WAITING: no action yet, path='%s' headless=%d\n",
                                ai_actions_path.c_str(), headless_ai);
                        fflush(ai_debug_log);
                    }
                }
            }
        }


        // ─── Экран завершения: статистика + новая игра / выход ───
        if (game_over) {
            BeginDrawing();
            ClearBackground(C_BG);
            DrawRectangle(0, 0, WIN_W, WIN_H, Color{0, 0, 0, 215});
            int px = WIN_W/2 - 300, py = 40, pw = 600, ph = 580;
            panel(px, py, pw, ph, "Игра окончена");
            int lx = px + 24;
            int ly = py + 52;
            text(stat_tax_over ? "Не удалось уплатить налог — колония пала."
                               : "Колония пала.", lx, ly, 20, C_ACCENT);
            ly += 38;
            text("Построено зданий:", lx, ly, 18, C_PANEL_FG); ly += 26;
            int colw = (pw - 60) / 2;
            int row = 0;
            for (auto& kv : stat_built) {
                int cx2 = lx + (row / 18) * colw;
                int cy2 = ly + (row % 18) * 22;
                text(TextFormat("%s: %d", kv.first.c_str(), kv.second), cx2, cy2, 16, C_PANEL_FG);
                row++;
            }
            if (stat_built.empty()) text("нет", lx, ly, 16, C_PANEL_FG);
            int my = py + ph - 150;
            text(TextFormat("Заработано: %lld", (long long)stat_earned), lx, my, 18, C_PANEL_FG); my += 26;
            text(TextFormat("Потрачено:  %lld", (long long)stat_spent), lx, my, 18, C_PANEL_FG); my += 26;
            if (btn(px + 20, py + ph - 50, 170, 36, "Новая карта")) {
                int64_t new_seed = (int64_t)GetRandomValue(1, 999999999);
                env.reset(new_seed); reset_active_cell(env.game()); sel_bx = sel_by = -1; cur_dlg = DLG_NONE;
                tax_from_dialog = false;
                cam.target = {(float)env.game().earth.init_sel_x * TILE,
                              (float)env.game().earth.init_sel_y * TILE};
                cam.zoom = 1.0f;
                stat_built.clear(); stat_earned = 0; stat_spent = 0;
                stat_started = false; stat_tax_over = false;
                ai_build_msg[0] = '\0'; ai_build_msg_timer = 0.0f;
                game_over = false;
            }
            if (btn(px + pw/2 - 80, py + ph - 50, 160, 36, "Повторить карту")) {
                int64_t same_seed = (int64_t)env.game().earth.seed();
                env.reset(same_seed); reset_active_cell(env.game()); sel_bx = sel_by = -1; cur_dlg = DLG_NONE;
                tax_from_dialog = false;
                cam.target = {(float)env.game().earth.init_sel_x * TILE,
                              (float)env.game().earth.init_sel_y * TILE};
                cam.zoom = 1.0f;
                stat_built.clear(); stat_earned = 0; stat_spent = 0;
                stat_started = false; stat_tax_over = false;
                ai_build_msg[0] = '\0'; ai_build_msg_timer = 0.0f;
                game_over = false;
            }
            if (btn(px + pw - 190, py + ph - 50, 170, 36, "Выход"))
                want_close = true;
            EndDrawing();
            continue;
        }

        // cursor cell
        Vector2 wp = GetScreenToWorld2D(mpos, cam);
        int cx = (int)floor(wp.x / TILE), cy = (int)floor(wp.y / TILE);
        int nb = (int)env.build_data().size();
        Game& gm = env.game();

        if (!headless_ai) {
        // ── Esc: диалог → режим стройки → выделение ──
        if (IsKeyPressed(KEY_ESCAPE)) {
            if (cur_dlg != DLG_NONE)      cur_dlg = DLG_NONE;
            else if (popup_open)          popup_open = false;
            else if (build_mode)        { build_mode = false; status = "Режим строительства выключен"; }
            else if (has_sel || sel_bx >= 0) {
                has_sel = false; sel_bx = -1; sel_by = -1;
                sel_x0 = sel_x1 = act_x; sel_y0 = sel_y1 = act_y;
            }
        }

        // drag-pan (middle button only; LMB is reserved for area selection)
        static bool panning = false;
        if (IsMouseButtonPressed(MOUSE_BUTTON_MIDDLE)) panning = true;
        if (panning && IsMouseButtonDown(MOUSE_BUTTON_MIDDLE)) {
            Vector2 d = GetMouseDelta();
            cam.target.x -= d.x / cam.zoom;
            cam.target.y -= d.y / cam.zoom;
        }
        if (IsMouseButtonReleased(MOUSE_BUTTON_MIDDLE)) panning = false;

        // Зум к курсору мыши (а не к центру экрана) и только над картой:
        // колесо над панелью/палитрой больше не дёргает масштаб.
        float wh = (over_map && cur_dlg == DLG_NONE) ? GetMouseWheelMove() : 0.0f;
        if (wh != 0) {
            Vector2 before = GetScreenToWorld2D(mpos, cam);
            cam.zoom *= (wh > 0) ? 1.1f : 0.9f;
            if (cam.zoom < 0.4f) cam.zoom = 0.4f;
            if (cam.zoom > 3.0f) cam.zoom = 3.0f;
            Vector2 after = GetScreenToWorld2D(mpos, cam);
            cam.target.x += before.x - after.x;
            cam.target.y += before.y - after.y;
        }
        if (cur_dlg == DLG_NONE) {
            if (IsKeyPressed(KEY_EQUAL) || IsKeyPressed(KEY_KP_ADD))
                cam.zoom = std::min(3.0f, cam.zoom * 1.25f);
            if (IsKeyPressed(KEY_MINUS) || IsKeyPressed(KEY_KP_SUBTRACT))
                cam.zoom = std::max(0.4f, cam.zoom * 0.8f);
        }

        // ── Камера с клавиатуры: Ctrl+стрелки (сами стрелки двигают клетку) ──
        if (cur_dlg == DLG_NONE &&
            (IsKeyDown(KEY_LEFT_CONTROL) || IsKeyDown(KEY_RIGHT_CONTROL))) {
            float ps = 14.0f / cam.zoom;
            if (IsKeyDown(KEY_RIGHT)) cam.target.x += ps;
            if (IsKeyDown(KEY_LEFT))  cam.target.x -= ps;
            if (IsKeyDown(KEY_DOWN))  cam.target.y += ps;
            if (IsKeyDown(KEY_UP))    cam.target.y -= ps;
        }

        // ── Edge-scroll when mouse approaches map border (with small delay) ──
        {
            float ed = 28.0f;
            bool in_edge = over_map && !popup_open &&
                (mpos.x - MAP_X < ed || MAP_X + MAP_W - mpos.x < ed ||
                 mpos.y - MAP_Y < ed || MAP_Y + MAP_H - mpos.y < ed);
            static float edge_hold = 0.0f;
            if (in_edge) edge_hold += GetFrameTime();
            else edge_hold = 0.0f;
            if (in_edge && edge_hold > 0.3f) {
                float ps = 14.0f / cam.zoom;
                if (mpos.x - MAP_X < ed)         cam.target.x -= ps;
                if (MAP_X + MAP_W - mpos.x < ed) cam.target.x += ps;
                if (mpos.y - MAP_Y < ed)         cam.target.y -= ps;
                if (MAP_Y + MAP_H - mpos.y < ed) cam.target.y += ps;
            }
        }

        // ── Камеру нельзя увести за пределы карты (раньше можно было
        //    уехать в пустоту и потерять колонию) ──
        {
            float lim = (float)g.map_size() * TILE;
            cam.target.x = std::max(0.0f, std::min(lim, cam.target.x));
            cam.target.y = std::max(0.0f, std::min(lim, cam.target.y));
        }

        // ── Активная клетка: WASD и стрелки, камера едет следом ──────────
        // Клетка активна ВСЕГДА (стартует на городе) — раньше «курсор» жил
        // только под мышью и исчезал, стоило увести её с карты.
        clamp_active_cell(g);
        if (cur_dlg == DLG_NONE && !popup_open) {
            bool ctrl = IsKeyDown(KEY_LEFT_CONTROL) || IsKeyDown(KEY_RIGHT_CONTROL);
            auto rep = [](int k) { return IsKeyPressed(k) || IsKeyPressedRepeat(k); };
            int mvx = 0, mvy = 0;
            if (rep(KEY_D) || (!ctrl && rep(KEY_RIGHT))) mvx += 1;
            if (rep(KEY_A) || (!ctrl && rep(KEY_LEFT)))  mvx -= 1;
            if (rep(KEY_S) || (!ctrl && rep(KEY_DOWN)))  mvy += 1;
            if (rep(KEY_W) || (!ctrl && rep(KEY_UP)))    mvy -= 1;
            if (mvx != 0 || mvy != 0) {
                int stepn = (IsKeyDown(KEY_LEFT_SHIFT) || IsKeyDown(KEY_RIGHT_SHIFT)) ? 5 : 1;
                act_x += mvx * stepn;
                act_y += mvy * stepn;
                clamp_active_cell(g);
                follow_active = true;
                has_sel = false;          // клавиатура работает с одной клеткой
                sel_x0 = sel_x1 = act_x;
                sel_y0 = sel_y1 = act_y;
            }
            if (IsKeyPressed(KEY_HOME)) {
                act_x = g.earth.init_sel_x;
                act_y = g.earth.init_sel_y;
                cam.target = {(float)act_x * TILE, (float)act_y * TILE};
            }
        }
        // Камера подтягивается к активной клетке ТОЛЬКО когда её подвинули с
        // клавиатуры: иначе было бы невозможно осмотреть карту мышью —
        // вид отщёлкивало бы назад каждый кадр.
        if (follow_active) {
            follow_active = false;
            float halfW = (MAP_W / 2.0f) / cam.zoom, halfH = (MAP_H / 2.0f) / cam.zoom;
            float pxc = act_x * (float)TILE + TILE / 2.0f;
            float pyc = act_y * (float)TILE + TILE / 2.0f;
            float marg = TILE * 1.5f;
            if (halfW > marg) {
                if (pxc < cam.target.x - halfW + marg) cam.target.x = pxc + halfW - marg;
                if (pxc > cam.target.x + halfW - marg) cam.target.x = pxc - halfW + marg;
            }
            if (halfH > marg) {
                if (pyc < cam.target.y - halfH + marg) cam.target.y = pyc + halfH - marg;
                if (pyc > cam.target.y + halfH - marg) cam.target.y = pyc - halfH + marg;
            }
        }

        // ─── Bank text input ───
        if (cur_dlg == DLG_BANK) {
            int k = GetCharPressed();
            while (k > 0) {
                if (isdigit((char)k) && strlen(bank_buf) < 31) {
                    int l = (int)strlen(bank_buf); bank_buf[l] = (char)k; bank_buf[l + 1] = 0;
                }
                k = GetCharPressed();
            }
            if (IsKeyPressed(KEY_BACKSPACE) && strlen(bank_buf) > 0)
                bank_buf[strlen(bank_buf) - 1] = 0;
        }

        if (cur_dlg == DLG_NONE) {
            // build palette click
            for (int i = 0; i < nb && i < PAL_ROWS * PAL_COLS; i++) {
                int col = i % PAL_COLS, row = i / PAL_COLS;
                int ix = col * BSTEP, iy = PAL_Y0 + row * BSTEP;
                if (mpos.x >= ix && mpos.x < ix + BS && mpos.y >= iy && mpos.y < iy + BS) {
                    if (pressed) {
                        // PR 2: закрытое курикулумом здание нельзя даже выбрать —
                        // иначе клик давал пустой status и вывод «ограничение не работает».
                        if (!env.build_allowed(env.build_data()[i]->id)) {
                            status = "Постройка закрыта курикулумом.";
                        } else {
                            sel_action = A_BUILD0 + i;
                            build_mode = true;   // курсор «держит» постройку
                            if (has_sel) {
                                // есть выделенная область — ставим сразу в неё
                                do_build_area(env, gm, A_BUILD0 + i, true,
                                              sel_x0, sel_y0, sel_x1, sel_y1, prc_x, prc_y, status);
                            } else {
                                status = std::string(env.build_data()[i]->caption) +
                                         " — ЛКМ по карте ставит, Esc отменяет";
                            }
                        }
                    }
                }
            }
            // tool button click
            for (int i = 0; i < 12; i++) {
                if (pressed && CheckCollisionPointRec(mpos, g_tool_rect[i])) {
                    static const int acts[12] = {1,2,3,4,5,6,7,8,9,10,11,12};
                    int act = acts[i];
                    // BANK/MARKET: диалоги принимают произвольную сумму
                    // кредита/количество каждого ресурса, которую печатает
                    // человек — этому нет соответствия среди 49 фиксированных
                    // RL-действий (см. docs/IMITATION_LEARNING_2026_09.md,
                    // раздел "Что не пишется в демо"). g.market_*/g.bank_*
                    // также меняют состояние в обход env.step(), поэтому не
                    // отражались бы в записанной траектории вовсе — открывать
                    // эти диалоги в режиме записи запрещено, чтобы не оставлять
                    // молчаливо неучтённые скачки состояния между шагами демо.
                    if (act == 7) {
                        if (g_record_demo) status = "Банк недоступен в режиме записи демо";
                        else cur_dlg = DLG_BANK;
                    }
                    else if (act == 9) {
                        if (g_record_demo) status = "Рынок недоступен в режиме записи демо";
                        else { dlg_mode = 0; cur_dlg = DLG_MARKET; }
                    }
                    else if (act == 10) {
                        if (g_record_demo) status = "Рынок недоступен в режиме записи демо";
                        else { dlg_mode = 1; cur_dlg = DLG_MARKET; }
                    }
                    else if (act == 11) { auto o = env_step_and_record(env, A_DAY); if (o.terminated) game_over = true; }
                    else if (act == 12) { auto o = env_step_and_record(env, A_WEEK); if (o.terminated) game_over = true; }
                    // БАГ: раньше здесь стояло `else if (over_map && ...)`,
                    // а при клике по кнопке тулбара мышь по определению НЕ над
                    // картой — все восемь «клеточных» кнопок молча ничего не
                    // делали. Теперь они работают с АКТИВНОЙ клеткой.
                    else {
                        const int tx_ = act_x, ty_ = act_y;
                        if (act == 1) {
                            // IMPROVE_LAND (manager+0): RL сама решает, какую
                            // клетку улучшать (find_lot); клик (cx,cy) в
                            // режиме записи — просто жест "нужно улучшить
                            // землю", см. заголовок блока demo recording выше.
                            if (g_record_demo) {
                                auto o = env_step_and_record(env, MGR_IMPROVE_LAND);
                                if (o.terminated) game_over = true;
                            } else {
                                auto r = gm.good_earth(tx_, ty_); if (!r.first) status = r.second;
                            }
                        }
                        else if (act == 2) {
                            Base* b = gm.find_slowest_base();
                            if (b) {
                                cam.target.x = (float)b->x*TILE; cam.target.y = (float)b->y*TILE;
                                act_x = b->x; act_y = b->y; sel_bx = b->x; sel_by = b->y;
                                status = "Слабейшее: " + b->data->caption;
                            }
                            else status = "Нет изношенных построек";
                        }
                        else if (act == 3) {
                            // REPAIR (manager+1): RL чинит find_slowest_base(),
                            // не обязательно ту клетку, что кликнул человек.
                            if (g_record_demo) {
                                auto o = env_step_and_record(env, MGR_REPAIR);
                                if (o.terminated) game_over = true;
                            } else {
                                auto r = gm.restore(tx_, ty_); if (!r.ok) status = r.msg;
                            }
                        }
                        else if (act == 4) {
                            // REPAIR_ALL (manager+2): 1:1 с человеческой кнопкой.
                            if (g_record_demo) {
                                auto o = env_step_and_record(env, MGR_REPAIR_ALL);
                                if (o.terminated) game_over = true;
                            } else {
                                auto r = gm.restore_all(); if (!r.ok) status = r.msg;
                            }
                        }
                        else if (act == 5) {
                            // DEMOLISH (manager+3): RL сносит find_slowest_base().
                            if (g_record_demo) {
                                auto o = env_step_and_record(env, MGR_DEMOLISH);
                                if (o.terminated) game_over = true;
                            } else {
                                const Base* b = gm.base_in_box(tx_, ty_);
                                if (!b) status = "Там нет постройки";
                                else if (ask_destroy) {
                                    confirm_x = tx_; confirm_y = ty_;
                                    confirm_text = "Снести «" + b->data->caption + "»?";
                                    cur_dlg = DLG_CONFIRM;
                                } else {
                                    auto r = gm.destroy(tx_, ty_);
                                    status = r.first ? "Постройка снесена" : r.second;
                                }
                            }
                        }
                        else if (act == 6) {
                            // UNDO: нет соответствующего RL-действия (откат
                            // предыдущего шага сломал бы записанную траекторию).
                            if (g_record_demo) {
                                status = "Отмена недоступна в режиме записи демо";
                            } else if (!gm.undo()) {
                                status = "Нечего отменять";
                            }
                        }
                        else if (act == 8) {
                            // PRESERVE/UNPRESERVE: одна кнопка в GUI переключает
                            // консервацию туда-обратно (Game::preserve делает
                            // b->preserved = !b->preserved), а RL это два разных
                            // действия (manager+4 сама находит наименее изношенное
                            // незаконсервированное, manager+5 — снимает с первого
                            // законсервированного). Чтобы не перепутать намерение
                            // человека, смотрим состояние ДО клика: снятие
                            // консервации с уже законсервированного здания
                            // логируется как manager+5, а не manager+4.
                            if (g_record_demo) {
                                const Base* clicked = gm.base_in_box(tx_, ty_);
                                int mgr_action = (clicked != nullptr && clicked->preserved)
                                    ? MGR_UNPRESERVE : MGR_PRESERVE;
                                auto o = env_step_and_record(env, mgr_action);
                                if (o.terminated) game_over = true;
                            } else {
                                auto r = gm.preserve(tx_, ty_); if (!r.first) status = r.second;
                            }
                        }
                    }
                }
            }
            // ── ЛКМ: клик — активная клетка, протяжка — область ──
            // В режиме строительства отпускание кнопки СРАЗУ ставит здание
            // (клик — одну штуку, протяжка — всю область). Вне режима клик
            // выбирает клетку/здание и не гасит ничего лишнего.
            if (!popup_open) {
                if (pressed && over_map) {
                    selecting = true;
                    sel_x0 = sel_x1 = cx; sel_y0 = sel_y1 = cy;
                    act_x = cx; act_y = cy; clamp_active_cell(g);
                }
                if (selecting && IsMouseButtonDown(MOUSE_BUTTON_LEFT) && over_map) {
                    sel_x1 = cx; sel_y1 = cy;
                    act_x = cx; act_y = cy; clamp_active_cell(g);
                }
                if (IsMouseButtonReleased(MOUSE_BUTTON_LEFT) && selecting) {
                    selecting = false;
                    const bool single = (sel_x0 == sel_x1 && sel_y0 == sel_y1);
                    const bool in_map = sel_x1 >= 0 && sel_y1 >= 0 &&
                                        sel_x1 < g.map_size() && sel_y1 < g.map_size();
                    const Base* b = (single && in_map) ? g.base_in_box(sel_x1, sel_y1) : nullptr;
                    int bi = sel_action - A_BUILD0;
                    if (build_mode && in_map && bi >= 0 && bi < nb) {
                        has_sel = !single;
                        sel_bx = -1; sel_by = -1;
                        do_build_area(env, gm, sel_action, !single,
                                      sel_x0, sel_y0, sel_x1, sel_y1, sel_x1, sel_y1, status);
                    } else if (b) {
                        // клик по зданию — карточка здания справа внизу
                        sel_bx = sel_x1; sel_by = sel_y1; has_sel = false;
                        status = b->data->caption;
                    } else {
                        sel_bx = -1; sel_by = -1;
                        has_sel = true;   // остаётся до Esc или нового выделения
                    }
                }
            }

            // ── Миникарта: клик И перетаскивание переносят вид ──
            if (g_mini_rect.width > 0 &&
                ((pressed && CheckCollisionPointRec(mpos, g_mini_rect)) ||
                 (mini_drag && IsMouseButtonDown(MOUSE_BUTTON_LEFT)))) {
                mini_drag = true;
                float fx = (mpos.x - g_mini_rect.x) / g_mini_rect.width;
                float fy = (mpos.y - g_mini_rect.y) / g_mini_rect.height;
                fx = std::max(0.0f, std::min(1.0f, fx));
                fy = std::max(0.0f, std::min(1.0f, fy));
                int msz = gm.map_size();
                cam.target.x = fx * msz * TILE;
                cam.target.y = fy * msz * TILE;
            }
            if (IsMouseButtonReleased(MOUSE_BUTTON_LEFT)) mini_drag = false;

            // ── ПКМ = отмена режима стройки, иначе меню зданий ──
            if (IsMouseButtonPressed(MOUSE_BUTTON_RIGHT)) {
                if (popup_open) {
                    popup_open = false;
                } else if (build_mode) {
                    build_mode = false;
                    status = "Режим строительства выключен";
                } else if (over_map) {
                    int pcols = 6, pcell = 44;
                    int prows = (nb + pcols - 1) / pcols;
                    int pw = pcols * pcell, ph = prows * pcell;
                    int bx = (int)mpos.x, by = (int)mpos.y;
                    bx = std::max(4, std::min(bx, WIN_W - pw - 4));
                    by = std::max(4, std::min(by, WIN_H - ph - 4));
                    g_popup_rect = {(float)bx, (float)by, (float)pw, (float)ph};
                    popup_x = bx; popup_y = by;
                    prc_x = cx; prc_y = cy;
                    popup_open = true;
                }
            }
            if (popup_open && IsKeyPressed(KEY_ESCAPE)) popup_open = false;

            // ── Popup: LMB on an icon selects & builds (in area or at ПКМ cell) ──
            if (popup_open && pressed) {
                int lx = (int)mpos.x - (int)g_popup_rect.x;
                int ly = (int)mpos.y - (int)g_popup_rect.y;
                if (lx >= 0 && ly >= 0 && lx <= (int)g_popup_rect.width && ly <= (int)g_popup_rect.height) {
                    int pcell = 44, pcols = (int)g_popup_rect.width / pcell;
                    int cc = lx / pcell; if (cc >= pcols) cc = pcols - 1;
                    int ci = (ly / pcell) * pcols + cc;
                    if (ci >= 0 && ci < nb) {
                        // PR 2: как в палитре — закрытое нельзя выбрать.
                        if (!env.build_allowed(env.build_data()[ci]->id)) {
                            status = "Постройка закрыта курикулумом.";
                        } else {
                            do_build_area(env, gm, A_BUILD0 + ci, has_sel,
                                          sel_x0, sel_y0, sel_x1, sel_y1, prc_x, prc_y, status);
                            sel_action = A_BUILD0 + ci;
                            build_mode = true;   // можно продолжать ставить ЛКМ
                        }
                    }
                    popup_open = false;
                } else {
                    popup_open = false;
                }
            }
            // mini-map click -> recenter
            // (rect computed in draw; approximate using panel coords)
            // time buttons
            for (int i = 0; i < 3; i++) {
                if (pressed && CheckCollisionPointRec(mpos, g_time_btn[i])) {
                    if (i == 0) { auto o = env_step_and_record(env, A_DAY); if (o.terminated) game_over = true; }
                    else if (i == 1) { auto o = env_step_and_record(env, A_WEEK); if (o.terminated) game_over = true; }
                    else advance_month(env);
                }
            }
            // кнопки скорости авто-хода
            for (int i = 0; i < 4; i++) {
                if (pressed && g_speed_btn[i].width > 0 &&
                    CheckCollisionPointRec(mpos, g_speed_btn[i])) {
                    speed_idx = i; speed_acc = 0.0f;
                    status = (i == 0) ? "Пауза" : TextFormat("Авто-ход x%d", i);
                }
            }
            // date strip click -> jump
            if (pressed && g_date_strip.width > 0 && CheckCollisionPointRec(mpos, g_date_strip)) {
                int leap = is_leap(g.year);
                int days_year = leap ? 365 : 364;
                int target = (int)((mpos.x - g_date_strip.x) * days_year / g_date_strip.width);
                target = std::max(0, std::min(days_year, target));
                jump_to_day(env, target);
            }

            // ─── Keyboard ───
            // Дублирует обработчики тулбара (act==11/12/1/3/4/5/6/8/7/9/10)
            // один в один, включая ветвление по g_record_demo — иначе
            // человек, играющий горячими клавишами вместо кликов по
            // тулбару, молча выпадал бы из записи демо (см.
            // docs/IMITATION_LEARNING_2026_09.md).
            //
            // Раскладка переехала: WASD заняты движением активной клетки, а
            // действия над клеткой теперь применяются к АКТИВНОЙ клетке и не
            // требуют держать мышь над картой (раньше U/A/R молчали, если
            // курсор ушёл на панель).
            const bool shift = IsKeyDown(KEY_LEFT_SHIFT) || IsKeyDown(KEY_RIGHT_SHIFT);
            const bool ctrl  = IsKeyDown(KEY_LEFT_CONTROL) || IsKeyDown(KEY_RIGHT_CONTROL);
            const int kx = act_x, ky = act_y;

            if (IsKeyPressed(KEY_SPACE) || IsKeyPressed(KEY_N)) {
                if (ctrl) advance_month(env);
                else if (shift) { auto o = env_step_and_record(env, A_WEEK); if (o.terminated) game_over = true; }
                else { auto o = env_step_and_record(env, A_DAY); if (o.terminated) game_over = true; }
            }
            if (IsKeyPressed(KEY_ENTER) || IsKeyPressed(KEY_KP_ENTER)) {
                int bi = sel_action - A_BUILD0;
                if (build_mode && bi >= 0 && bi < nb) {
                    do_build_area(env, gm, sel_action, has_sel,
                                  sel_x0, sel_y0, sel_x1, sel_y1, kx, ky, status);
                } else {
                    auto o = env_step_and_record(env, A_DAY); if (o.terminated) game_over = true;
                }
            }
            // скорость времени: 0 — пауза, 1/2/3 — авто-ход
            for (int s = 0; s < 4; s++)
                if (IsKeyPressed(KEY_ZERO + s)) {
                    speed_idx = s; speed_acc = 0.0f;
                    status = s == 0 ? "Пауза" : TextFormat("Авто-ход ×%d", s);
                }
            {
                if (IsKeyPressed(KEY_G)) {
                    if (g_record_demo) {
                        auto o = env_step_and_record(env, MGR_IMPROVE_LAND);
                        if (o.terminated) game_over = true;
                    } else {
                        auto r = gm.good_earth(kx, ky); if (!r.first) status = r.second;
                        else status = "Участок выкуплен — налога за него больше нет";
                    }
                }
                if (IsKeyPressed(KEY_F)) {
                    Base* b = gm.find_slowest_base();
                    if (b) {
                        cam.target.x = (float)b->x*TILE; cam.target.y = (float)b->y*TILE;
                        act_x = b->x; act_y = b->y;
                        sel_bx = b->x; sel_by = b->y;
                        status = "Слабейшее: " + b->data->caption;
                    } else status = "Нет изношенных построек";
                }
                if (IsKeyPressed(KEY_R)) {
                    if (shift) {
                        if (g_record_demo) {
                            auto o = env_step_and_record(env, MGR_REPAIR_ALL);
                            if (o.terminated) game_over = true;
                        } else {
                            auto r = gm.restore_all();
                            status = r.ok ? TextFormat("Отремонтировано построек: %lld за %s ₽",
                                                       (long long)r.days, money_str(r.price).c_str())
                                          : r.msg;
                        }
                    } else if (g_record_demo) {
                        auto o = env_step_and_record(env, MGR_REPAIR);
                        if (o.terminated) game_over = true;
                    } else {
                        auto r = gm.restore(kx, ky);
                        status = r.ok ? TextFormat("Ремонт: %s ₽", money_str(r.price).c_str()) : r.msg;
                    }
                }
                if (IsKeyPressed(KEY_P)) {
                    if (g_record_demo) {
                        const Base* clicked = gm.base_in_box(kx, ky);
                        int mgr_action = (clicked != nullptr && clicked->preserved)
                            ? MGR_UNPRESERVE : MGR_PRESERVE;
                        auto o = env_step_and_record(env, mgr_action);
                        if (o.terminated) game_over = true;
                    } else {
                        auto r = gm.preserve(kx, ky); if (!r.first) status = r.second;
                        else status = "Консервация переключена";
                    }
                }
                // Снос — Delete (или X): с подтверждением, одной клавишей
                // больше не теряется здание за 380 000 ₽.
                if (IsKeyPressed(KEY_DELETE) || IsKeyPressed(KEY_X)) {
                    if (g_record_demo) {
                        auto o = env_step_and_record(env, MGR_DEMOLISH);
                        if (o.terminated) game_over = true;
                    } else {
                        const Base* b = gm.base_in_box(kx, ky);
                        if (!b) status = "Там нет постройки";
                        else if (ask_destroy) {
                            confirm_x = kx; confirm_y = ky;
                            confirm_text = "Снести «" + b->data->caption + "»?";
                            cur_dlg = DLG_CONFIRM;
                        } else {
                            auto r = gm.destroy(kx, ky);
                            status = r.first ? "Постройка снесена" : r.second;
                        }
                    }
                }
                if (IsKeyPressed(KEY_U) || (ctrl && IsKeyPressed(KEY_Z))) {
                    if (g_record_demo) status = "Отмена недоступна в режиме записи демо";
                    else status = gm.undo() ? "Последнее действие отменено" : "Нечего отменять";
                }
            }
            if (IsKeyPressed(KEY_B)) {
                if (g_record_demo) status = "Рынок недоступен в режиме записи демо";
                else { dlg_mode = 0; cur_dlg = DLG_MARKET; }
            }
            if (IsKeyPressed(KEY_M)) {
                if (g_record_demo) status = "Рынок недоступен в режиме записи демо";
                else { dlg_mode = 1; cur_dlg = DLG_MARKET; }
            }
            if (IsKeyPressed(KEY_K)) {
                if (g_record_demo) status = "Банк недоступен в режиме записи демо";
                else cur_dlg = DLG_BANK;
            }

            // ─── Menu shortcuts ───
            if (IsKeyPressed(KEY_F4)) cur_dlg = DLG_NEW;
            if (IsKeyPressed(KEY_F5)) cur_dlg = DLG_OPEN;
            if (IsKeyPressed(KEY_F2)) cur_dlg = DLG_SAVE;
            if (IsKeyPressed(KEY_F1)) {
                snprintf(mess_title, sizeof(mess_title), "%s", "Помощь");
                snprintf(mess_text, sizeof(mess_text), "%s", "КАРТА И КЛЕТКА\nWASD или стрелки — двигать активную клетку (она активна всегда)\nЛКМ — выбрать клетку/здание, протяжка — выделить область\nCtrl+стрелки, средняя кнопка мыши, край экрана — сдвиг карты; Home — к городу\nКолесо или +/- — зум (к курсору мыши)\n\nСТРОИТЕЛЬСТВО\nИконка в палитре или ПКМ по карте — взять постройку в руку:\nзелёная подсветка = встанет, красная = нельзя (море, тип земли, нет связи)\nЛКМ — поставить, протяжка — заполнить область, Enter — в активную клетку\nEsc или ПКМ — выйти из режима строительства\n\nВРЕМЯ\nПробел — день, Shift+Пробел — неделя, Ctrl+Пробел — месяц\n0 — пауза, 1/2/3 — автоматический ход времени\n\nДЕЙСТВИЯ НАД АКТИВНОЙ КЛЕТКОЙ\nR — ремонт, Shift+R — ремонт всех, Delete — снести, P — консервация\nG — выкупить участок (снимает налог), F — найти изношенное\nB — купить, M — продать, K — банк, Ctrl+Z — отменить\nF1 помощь, F2 сохранить, F4 новая, F5 загрузить, F9 полный экран");
                cur_dlg = DLG_MESS;
            }
            if (IsKeyPressed(KEY_F7)) sound_on = !sound_on;
            if (IsKeyPressed(KEY_F9)) { fullscreen = !fullscreen; ToggleFullscreen(); }
        }
        }

        // ─── Auto-pay tax if player has enough money (no dialog needed) ───
        if (g.annual_tax_due() && g.money >= g.annual_tax_amount()) {
            gm.pay_annual_tax();
        } else if (g.main_tax_due() && g.money >= g.main_tax_amount()) {
            gm.pay_main_tax();
        }

        // ─── After selling from tax dialog: check if tax can now be paid ───
        if (tax_from_dialog && cur_dlg == DLG_NONE) {
            tax_from_dialog = false;
            if (g.annual_tax_due() && g.money >= g.annual_tax_amount()) {
                gm.pay_annual_tax();
            } else if (g.main_tax_due() && g.money >= g.main_tax_amount()) {
                gm.pay_main_tax();
            }
            // if still can't pay, dialog will re-appear below
        }

        // ─── Auto dialogs (season change / tax due) ───
        if (cur_dlg == DLG_NONE) {
            if (g.season != prev_season) {
                dlg_season = (int)g.season; prev_season = g.season; cur_dlg = DLG_SEASON;
            } else if (g.annual_tax_due()) cur_dlg = DLG_NALON;
            else if (g.main_tax_due()) cur_dlg = DLG_NALON_MAIN;
        }

        // ─── Авто-ход времени (пауза / ×1 / ×2 / ×3) ───
        // Раньше время двигалось только кнопками «День/Неделя/Месяц»: на
        // 10 000 игровых дней это десятки тысяч кликов.
        if (!headless_ai && speed_idx > 0) {
            if (cur_dlg != DLG_NONE || game_over || popup_open) {
                speed_acc = 0.0f;
            } else {
                speed_acc += GetFrameTime() * SPEED_DPS[speed_idx];
                int budget = 0;
                while (speed_acc >= 1.0f && budget < 40 && !game_over && cur_dlg == DLG_NONE) {
                    speed_acc -= 1.0f;
                    budget++;
                    auto o = env_step_and_record(env, A_DAY);
                    if (o.terminated) { game_over = true; speed_idx = 0; }
                }
                if (speed_acc > 2.0f) speed_acc = 0.0f;
            }
        }

        // ─── Сообщения: показать то, что раньше молча терялось ───
        tick_log(GetFrameTime());
        if (!status.empty()) { push_log(status); status.clear(); }

        // ─── Статистика: учёт заработано / потрачено ───
        if (!stat_started) { stat_prev_money = g.money; stat_started = true; }
        if (!game_over) {
            int64_t dm = g.money - stat_prev_money;
            if (dm > 0) stat_earned += dm;
            else if (dm < 0) stat_spent += -dm;
        }
        stat_prev_money = g.money;

        // ═══ DRAW ═══
        BeginDrawing();
        ClearBackground(C_BG);

        // ─── Build palette (top band) ───
        {
            int hf = (int)(GetTime() * 12.0) % 12;
            for (int i = 0; i < nb && i < PAL_ROWS * PAL_COLS; i++) {
                int col = i % PAL_COLS, row = i / PAL_COLS;
                int ix = col * BSTEP, iy = PAL_Y0 + row * BSTEP;
                bool sel = (sel_action == A_BUILD0 + i);
                DrawRectangle(ix, iy, BS, BS, sel ? C_PAL_SEL : C_PAL_NORM);
                if (sel) DrawRectangleLines(ix, iy, BS, BS, C_ACCENT);
                draw_build_pixel(ix, iy, BS, env.build_data()[i]->id, false);
                if (mpos.x >= ix && mpos.x < ix + BS && mpos.y >= iy && mpos.y < iy + BS)
                    draw_tex(selEarthTex[hf], ix, iy, BS);
                // PR 2: закрытые курикулумом иконки затемнены (поверх ховера).
                if (!env.build_allowed(env.build_data()[i]->id))
                    DrawRectangle(ix, iy, BS, BS, {0, 0, 0, 140});
                // Не по карману — приглушаем и помечаем уголком, чтобы не
                // кликать вслепую в то, на что всё равно не хватит денег.
                else if (env.build_data()[i]->price > g.money) {
                    DrawRectangle(ix, iy, BS, BS, {0, 0, 0, 90});
                    DrawRectangle(ix + BS - 5, iy, 5, 5, {220, 70, 60, 255});
                }
                // рамка активного здания в режиме стройки — ярче
                if (sel && build_mode) DrawRectangleLinesEx({(float)ix - 1, (float)iy - 1,
                                                             (float)BS + 2, (float)BS + 2}, 2, C_ACCENT);
            }
        }
        // ─── Tool buttons ───
        draw_tools(env);

        {
            Vector2 mp = GetMousePosition();
            const colony::BaseData* info_bd = nullptr;
            for (int i = 0; i < nb && i < PAL_ROWS * PAL_COLS; i++) {
                int col = i % PAL_COLS, row = i / PAL_COLS;
                int ix = col * BSTEP, iy = PAL_Y0 + row * BSTEP;
                if (mp.x >= ix && mp.x < ix + BS && mp.y >= iy && mp.y < iy + BS) {
                    info_bd = env.build_data()[i];
                    break;
                }
            }
            if (!info_bd && popup_open) {
                int pcell = 44, pcols = (int)g_popup_rect.width / pcell;
                int lx = (int)mp.x - (int)g_popup_rect.x;
                int ly = (int)mp.y - (int)g_popup_rect.y;
                if (lx >= 0 && ly >= 0 && lx < (int)g_popup_rect.width && ly < (int)g_popup_rect.height) {
                    int ci = (ly / pcell) * pcols + (lx / pcell);
                    if (ci >= 0 && ci < nb) info_bd = env.build_data()[ci];
                }
            }
            if (info_bd) {
                auto L = base_info_lines(info_bd);
                if (!env.build_allowed(info_bd->id))
                    L.push_back("ЗАКРЫТО КУРИКУЛУМОМ");
                else if (info_bd->price > g.money)
                    L.push_back(TextFormat("НЕ ХВАТАЕТ %s ₽", money_str(info_bd->price - g.money).c_str()));
                L.push_back("ЛКМ — взять в руку, потом клик по карте");
                draw_info_box(L, INFO_X, INFO_Y, INFO_W);
            }
        }

        // separator under top band
        DrawRectangle(0, MAP_Y - 6, WIN_W, 1, {70, 90, 80, 255});

        // ─── MAP ───
        int prev_ok_cells = 0, prev_bad_cells = 0;
        bool prev_shown = false;
        {
            int vx = MAP_X, vy = MAP_Y, vw = MAP_W, vh = MAP_H;
            BeginScissorMode(vx, vy, vw, vh);
            DrawRectangle(vx, vy, vw, vh, C_SEA_DEEP);
            cam.offset = {(float)vx + vw/2.0f, (float)vy + vh/2.0f};
            int msz = g.map_size();
            float halfW = (vw/2.0f)/cam.zoom, halfH = (vh/2.0f)/cam.zoom;
            int x0 = (int)floor((cam.target.x - halfW)/TILE) - 1;
            int x1 = (int)ceil((cam.target.x + halfW)/TILE) + 1;
            int y0 = (int)floor((cam.target.y - halfH)/TILE) - 1;
            int y1 = (int)ceil((cam.target.y + halfH)/TILE) + 1;
            x0 = std::max(0, x0); y0 = std::max(0, y0);
            x1 = std::min(msz - 1, x1); y1 = std::min(msz - 1, y1);
            BeginMode2D(cam);
            for (int y = y0; y <= y1; y++) {
                for (int x = x0; x <= x1; x++) {
                    int8_t lot = g.earth.lot(x, y);
                    int ei = earth_icon_index(lot, x, y);
                    if (ei >= 0) {
                        draw_tex(earthTex[ei], x*TILE, y*TILE, TILE);
                    } else {
                        Color c = lot_color(lot);
                        unsigned int hsh = (x*7919 + y*6271) & 0xFF;
                        c.r = (unsigned char)std::max(0, std::min(255, (int)c.r + (int)(hsh%9) - 4));
                        c.g = (unsigned char)std::max(0, std::min(255, (int)c.g + (int)(hsh%9) - 4));
                        c.b = (unsigned char)std::max(0, std::min(255, (int)c.b + (int)(hsh%9) - 4));
                        DrawRectangle(x*TILE, y*TILE, TILE, TILE, c);
                        // Море (LT_NONE): рябь + береговая линия со стороны
                        // суши. Без этого остров и море были одного цвета, и
                        // дорога «необъяснимо» обтекала воду лесенкой.
                        if (lot == LT_NONE) {
                            if (((x * 5 + y * 3) % 7) == 0)
                                DrawRectangle(x*TILE + 5, y*TILE + TILE/2, TILE - 10, 2, C_SEA_DEEP);
                            const int dxs[4] = {1, -1, 0, 0}, dys[4] = {0, 0, 1, -1};
                            for (int k = 0; k < 4; k++) {
                                int nx = x + dxs[k], ny = y + dys[k];
                                if (!g.earth.in_bounds(nx, ny)) continue;
                                if (!lot_is_land(g.earth.lot(nx, ny))) continue;
                                if (dxs[k] == 1)  DrawRectangle(x*TILE + TILE - 2, y*TILE, 2, TILE, C_SEA_DEEP);
                                if (dxs[k] == -1) DrawRectangle(x*TILE, y*TILE, 2, TILE, C_SEA_DEEP);
                                if (dys[k] == 1)  DrawRectangle(x*TILE, y*TILE + TILE - 2, TILE, 2, C_SEA_DEEP);
                                if (dys[k] == -1) DrawRectangle(x*TILE, y*TILE, TILE, 2, C_SEA_DEEP);
                            }
                        }
                    }
                }
            }
            for (const Base& b : g.bases) {
                int bi = base_icon_index(b.data->id);
                int bx = b.x * TILE, by = b.y * TILE;
                if (bi >= 0) draw_tex(baseTex[bi], bx, by, TILE);
                else { Color bc = build_color(b.data->id); DrawRectangle(bx, by, TILE, TILE, bc); DrawRectangleLines(bx, by, TILE, TILE, C_WHITE); }
                if (b.build_days > 0) {
                    DrawRectangle(bx - 8, by - 8, 16, 5, Color{255, 200, 0, 128});
                    draw_tex(iconTex[1], bx + 4, by + 4, 14);
                } else {
                    if (b.preserved)    draw_tex(iconTex[3], bx + 4, by + 4, 14);
                    if (b.need_sunduk)  draw_tex(iconTex[4], bx + 14, by + 4, 14);
                    if (b.need_workers) draw_tex(iconTex[6], bx + 4, by + 14, 14);
                }
                if (g.is_good(b.x, b.y)) draw_tex(iconTex[2], bx + 14, by + 14, 14);
                if (b.build_days == 0 && !b.data->season_works(g.season)) draw_tex(iconTex[5], bx + 14, by + 4, 14);
                if (b.is_alarm())        draw_tex(iconTex[0], bx + 14, by + 14, 14);
            }
            // ── Предпросмотр застройки: зелёный = встанет, красный = нет ──
            // Главный ответ на «дорога строится лесенкой»: теперь ещё ДО
            // клика видно, какие клетки выделенной области непригодны (море,
            // чужой тип земли, нет связи с колонией) — дорога обтекает их не
            // «непонятно почему», а по видимой причине.
            if (build_mode && !popup_open) {
                int bidx = sel_action - A_BUILD0;
                if (bidx >= 0 && bidx < nb) {
                    const BaseData* pbd = env.build_data()[bidx];
                    refresh_connectivity(g);
                    int px0, py0, px1, py1;
                    if (selecting || has_sel) {
                        px0 = std::min(sel_x0, sel_x1); px1 = std::max(sel_x0, sel_x1);
                        py0 = std::min(sel_y0, sel_y1); py1 = std::max(sel_y0, sel_y1);
                    } else if (over_map) {
                        px0 = px1 = cx; py0 = py1 = cy;
                    } else {
                        px0 = px1 = act_x; py0 = py1 = act_y;
                    }
                    const int MAXP = 4000;   // защита от гигантской протяжки
                    if ((int64_t)(px1 - px0 + 1) * (py1 - py0 + 1) <= MAXP) {
                        for (int y = py0; y <= py1; y++)
                            for (int x = px0; x <= px1; x++) {
                                if (x < 0 || y < 0 || x >= msz || y >= msz) continue;
                                bool ok = preview_ok(g, *pbd, x, y);
                                DrawRectangle(x*TILE, y*TILE, TILE, TILE,
                                              ok ? C_OK_TINT : C_BAD_TINT);
                                if (ok) {
                                    prev_ok_cells++;
                                    int bi2 = base_icon_index(pbd->id);
                                    if (bi2 >= 0 && cam.zoom > 0.6f) {
                                        Rectangle src = {0, 0, (float)baseTex[bi2].width,
                                                         (float)baseTex[bi2].height};
                                        Rectangle dst = {(float)(x*TILE), (float)(y*TILE),
                                                         (float)TILE, (float)TILE};
                                        if (baseTex[bi2].id)
                                            DrawTexturePro(baseTex[bi2], src, dst, {0,0}, 0.0f,
                                                           Color{255,255,255,130});
                                    }
                                } else {
                                    prev_bad_cells++;
                                }
                            }
                        prev_shown = true;
                    }
                }
            }
            // area selection = every cell inside the rectangle gets an animated frame
            if (selecting || has_sel) {
                int x0 = std::min(sel_x0, sel_x1), x1 = std::max(sel_x0, sel_x1);
                int y0 = std::min(sel_y0, sel_y1), y1 = std::max(sel_y0, sel_y1);
                int f = (int)(GetTime() * 12.0) % 12;
                if ((int64_t)(x1 - x0 + 1) * (y1 - y0 + 1) <= 4000) {
                    for (int y = y0; y <= y1; y++)
                        for (int x = x0; x <= x1; x++)
                            draw_tex(selEarthTex[f], x*TILE, y*TILE, TILE);
                }
                DrawRectangleLinesEx({(float)(x0*TILE), (float)(y0*TILE),
                                      (float)((x1-x0+1)*TILE), (float)((y1-y0+1)*TILE)},
                                     2.0f, C_ACCENT);
            }
            // hover = animated frame under the mouse
            if (!popup_open && over_map && cx >= 0 && cy >= 0 && cx < msz && cy < msz) {
                int px = cx * TILE, py = cy * TILE;
                int f = (int)(GetTime() * 12.0) % 12;
                int8_t fl = g.earth.lot(cx, cy);
                draw_tex(fl == LT_NONE ? selNoneTex[f] : selEarthTex[f], px, py, TILE);
            }
            // ── АКТИВНАЯ КЛЕТКА: видна всегда, даже когда мышь ушла с карты ──
            {
                int px = act_x * TILE, py = act_y * TILE;
                float pulse = 0.5f + 0.5f * (float)sin(GetTime() * 4.0);
                Color ac = {255, 230, 90, (unsigned char)(140 + 100 * pulse)};
                DrawRectangleLinesEx({(float)px, (float)py, (float)TILE, (float)TILE}, 2.0f, ac);
                int corn = 7;
                DrawRectangle(px - 1, py - 1, corn, 3, ac);
                DrawRectangle(px - 1, py - 1, 3, corn, ac);
                DrawRectangle(px + TILE - corn + 1, py - 1, corn, 3, ac);
                DrawRectangle(px + TILE - 2, py - 1, 3, corn, ac);
                DrawRectangle(px - 1, py + TILE - 2, corn, 3, ac);
                DrawRectangle(px - 1, py + TILE - corn + 1, 3, corn, ac);
                DrawRectangle(px + TILE - corn + 1, py + TILE - 2, corn, 3, ac);
                DrawRectangle(px + TILE - 2, py + TILE - corn + 1, 3, corn, ac);
            }
            // выбранное здание — отдельная постоянная рамка
            if (sel_bx >= 0 && sel_by >= 0 && sel_bx < msz && sel_by < msz) {
                DrawRectangleLinesEx({(float)(sel_bx*TILE) - 2, (float)(sel_by*TILE) - 2,
                                      (float)TILE + 4, (float)TILE + 4}, 2.0f,
                                     Color{120, 220, 255, 230});
            }
            EndMode2D();
            EndScissorMode();
            DrawRectangleLines(vx, vy, vw, vh, {60, 80, 70, 255});

            // ── Подсказка режима стройки: что, сколько и почём ──
            if (prev_shown) {
                int bidx = sel_action - A_BUILD0;
                const BaseData* pbd = env.build_data()[bidx];
                int64_t cost = (int64_t)prev_ok_cells * pbd->price;
                bool poor = cost > g.money;
                std::vector<std::string> L;
                L.push_back(pbd->caption);
                if (prev_ok_cells + prev_bad_cells > 1)
                    L.push_back(TextFormat("Клеток: %d подходит, %d нельзя",
                                           prev_ok_cells, prev_bad_cells));
                L.push_back(TextFormat("Цена: %s ₽%s", money_str(cost).c_str(),
                                       poor ? "  — НЕ ХВАТАЕТ ДЕНЕГ" : ""));
                if (pbd->build_time)
                    L.push_back(TextFormat("Стройка: %lld дн.", (long long)pbd->build_time));
                if (prev_ok_cells == 0) {
                    int hx = over_map ? cx : act_x, hy = over_map ? cy : act_y;
                    std::string why = preview_reason(g, *pbd, hx, hy);
                    if (!why.empty()) L.push_back(why);
                }
                int tipx = (over_map ? (int)mpos.x + 18 : MAP_X + 12);
                int tipy = (over_map ? (int)mpos.y + 18 : MAP_Y + 12);
                draw_info_box(L, tipx, tipy, 300);
            }

            // ── Журнал сообщений (раньше всё это молча терялось) ──
            if (!g_log.empty()) {
                int lh = 19;
                int n = (int)g_log.size();
                int boxh = n * lh + 8;
                int bx = MAP_X + 8, by = MAP_Y + MAP_H - boxh - 8;
                DrawRectangle(bx, by, 430, boxh, Color{10, 16, 12, 170});
                for (int i = 0; i < n; i++) {
                    const LogMsg& m = g_log[(size_t)i];
                    float fade = m.age > LOG_TTL - 1.0f ? (LOG_TTL - m.age) : 1.0f;
                    if (fade < 0.0f) fade = 0.0f;
                    unsigned char a = (unsigned char)(255 * fade);
                    Color c = m.error ? Color{255, 140, 120, a} : Color{190, 240, 180, a};
                    text(m.text.c_str(), bx + 8, by + 4 + i * lh, 16, c);
                }
            }

            // ── Карточка выбранного здания ──
            if (sel_bx >= 0 && sel_by >= 0) {
                const Base* b = g.base_in_box(sel_bx, sel_by);
                if (b) {
                    std::vector<std::string> L;
                    L.push_back(TextFormat("%s (%d,%d)", b->data->caption.c_str(), b->x, b->y));
                    if (b->build_days > 0)
                        L.push_back(TextFormat("Строится, осталось %lld дн.", (long long)b->build_days));
                    if (b->data->live_years) {
                        int64_t full = b->data->live_time_total();
                        int pct = full > 0 ? (int)(100 * b->live_time / full) : 100;
                        L.push_back(TextFormat("Состояние: %d%%%s", pct, b->is_alarm() ? " — ИЗНОШЕНО" : ""));
                        int64_t rp = b->data->restore_price(b->live_time);
                        if (rp > 0) L.push_back(TextFormat("Ремонт (R): %s ₽", money_str(rp).c_str()));
                    }
                    if (b->preserved)     L.push_back("Законсервировано (P — снять)");
                    if (b->need_sunduk)   L.push_back("Не хватает ресурсов");
                    if (b->need_workers)  L.push_back("Не хватает рабочих");
                    if (b->build_days == 0 && !b->data->season_works(g.season))
                        L.push_back("Не работает в этот сезон");
                    L.push_back("R ремонт · P консервация · Del снести");
                    draw_info_box(L, MAP_X + MAP_W - 290, MAP_Y + 8, 280);
                } else {
                    sel_bx = sel_by = -1;
                }
            }
        }

        // ─── RIGHT PANEL ───
        DrawRectangle(PANEL_X, TOP_H, SIDE_W, WIN_H - BOTTOM_H - TOP_H, C_PANEL_BG);
        {
            int px = PANEL_X + 10;
            int pw = SIDE_W - 20;
            int sy = TOP_H + 4;

            // Mini-map
            int plan_w = pw;
            int plan_h = std::min(210, WIN_H - BOTTOM_H - TOP_H - 12*CH - 20);
            g_mini_rect = {(float)px, (float)sy, (float)plan_w, (float)plan_h};
            draw_minimap(px, sy, plan_w, plan_h, g, cam);
            sy += plan_h + 6;

            // Date strip (season colors)
            {
                static const Color SC[4] = {{186,216,148,255},{233,235,158,255},{255,177,140,255},{186,224,226,255}};
                int strip_y = sy, stripH = 6;
                bool leap = is_leap(g.year);
                int days_year = leap ? 365 : 364;
                int ad = abs_day_of(g);
                int d1 = 31 + DAYS_IN_MONTH[1] + (leap ? 1 : 0);
                int d2 = d1 + 31 + 30 + 31;
                int d3 = d2 + 30 + 31 + 31;
                int d4 = d3 + 30 + 31 + 30;
                DrawRectangle(px, strip_y, pw, stripH, {0,0,0,255});
                auto seg = [&](int a, int b, int ci){
                    int x0 = px + pw*a/days_year, x1 = px + pw*b/days_year;
                    if (x1 > x0) DrawRectangle(x0, strip_y, x1 - x0, stripH, SC[ci]);
                };
                seg(0, d1, 3); seg(d1, d2, 0); seg(d2, d3, 1); seg(d3, d4, 2); seg(d4, days_year, 3);
                int mxk = px + pw*ad/days_year;
                DrawRectangle(mxk - 2, strip_y - 2, 4, 4, {200,30,30,255});
                g_date_strip = {(float)px, (float)strip_y, (float)pw, (float)stripH};
                text(TextFormat("%d %s %d", g.day, month_ru(g.month), g.year), px, strip_y + 10, 18, C_PANEL_FG);
                sy = strip_y + 34;

                const char* labs[3] = {"День", "Неделя", "Месяц"};
                int bw = (pw - 8)/3, bh = 22, by = sy;
                for (int i = 0; i < 3; i++) {
                    int bx = px + i*(bw+4);
                    g_time_btn[i] = {(float)bx, (float)by, (float)bw, (float)bh};
                    Rectangle r = g_time_btn[i];
                    Vector2 mpp = GetMousePosition();
                    bool hov = CheckCollisionPointRec(mpp, r);
                    DrawRectangleRec(r, hov ? Color{60,120,200,255} : C_PANEL_HEAD);
                    DrawRectangleLines(bx, by, bw, bh, {255,255,255,255});
                    int tw = (int)MeasureTextEx(gFont, labs[i], 16.0f, 1.0f).x;
                    text(labs[i], bx + (bw - tw)/2, by + 3, 16, {10,15,12,255});
                }
                sy = by + bh + 4;
                // ── Скорость хода времени: пауза / ×1 / ×2 / ×3 ──
                {
                    const char* sl[4] = {"II", "x1", "x2", "x3"};
                    int sw = (pw - 12) / 4, sh = 20;
                    for (int i = 0; i < 4; i++) {
                        int bx = px + i*(sw+4);
                        Rectangle r = {(float)bx, (float)sy, (float)sw, (float)sh};
                        g_speed_btn[i] = r;
                        bool hov = CheckCollisionPointRec(GetMousePosition(), r);
                        bool on = (speed_idx == i);
                        DrawRectangleRec(r, on ? Color{40,120,60,255}
                                               : (hov ? Color{60,120,200,255} : Color{200,206,198,255}));
                        DrawRectangleLines(bx, sy, sw, sh, {255,255,255,255});
                        int tw = (int)MeasureTextEx(gFont, sl[i], 15.0f, 1.0f).x;
                        text(sl[i], bx + (sw - tw)/2, sy + 2, 15, on ? C_WHITE : Color{10,15,12,255});
                    }
                    sy += sh + 6;
                }
            }

            // Resources (imlTools 0..8)
            for (int i = 0; i < SUNDUK_SIZE; i++) {
                draw_tex(toolTex[i], px, sy, 18);
                text(TextFormat("%s: %lld", RES_SHORT[i], (long long)g.sunduk[i]), px + 24, sy + 2, 16, C_PANEL_FG);
                sy += 20;
            }
            sy += 4;
            // Money / People (imlTools 17 / 18)
            draw_tex(toolTex[17], px, sy, 18);
            text(TextFormat("Деньги: %lld", (long long)g.money), px + 24, sy + 2, 16, {120,90,0,255});
            sy += 22;
            draw_tex(toolTex[18], px, sy, 18);
            text(TextFormat("Люди: %lld", (long long)g.people), px + 24, sy + 2, 16, C_PANEL_FG);
            sy += 22;
            text(TextFormat("Жильё:         %lld", (long long)g.now_home_places()), px, sy, 16, C_PANEL_FG); sy += 20;
            text(TextFormat("Рабочие места: %lld", (long long)g.now_need_workers()), px, sy, 16, C_PANEL_FG); sy += 20;
            text(TextFormat("Не работают:   %lld", (long long)(g.people - g.busy_people)), px, sy, 16, C_PANEL_FG); sy += 20;
            text(TextFormat("Кредит:        %lld", (long long)g.credit), px, sy, 16, C_PANEL_FG); sy += 20;
        }

        // ─── BOTTOM: status + legend ───
        {
            int by = WIN_H - BOTTOM_H;
            DrawRectangle(0, by, WIN_W, BOTTOM_H, C_BG);
            // Информация о клетке: под мышью, а если мышь ушла с карты — об
            // активной клетке (раньше в этом случае показывалась просто дата).
            char st[256];
            const bool hov_map = over_map && cx >= 0 && cy >= 0 &&
                                 cx < g.map_size() && cy < g.map_size();
            {
                int ix = hov_map ? cx : act_x, iy = hov_map ? cy : act_y;
                cx = ix; cy = iy;
                int8_t lot = g.earth.lot(cx, cy);
                snprintf(st, sizeof(st), "%s (%d,%d): %s",
                         hov_map ? "Клетка" : "Активная клетка", cx, cy,
                         LOT_NAMES[lot < 0 || lot > 8 ? 8 : lot]);
                if (g.is_good(cx, cy)) { int l = (int)strlen(st); snprintf(st + l, sizeof(st) - l, ", выкуплена (без налога)"); }
                const Base* b = g.base_in_box(cx, cy);
                if (b) {
                    int l = (int)strlen(st);
                    snprintf(st + l, sizeof(st) - l, " | %s", b->data->caption.c_str());
                    if (b->build_days) { l = (int)strlen(st); snprintf(st + l, sizeof(st) - l, ", строится"); }
                    if (b->preserved)  { l = (int)strlen(st); snprintf(st + l, sizeof(st) - l, ", блок"); }
                    if (b->need_sunduk){ l = (int)strlen(st); snprintf(st + l, sizeof(st) - l, ", нужны ресурсы"); }
                    if (b->need_workers){l = (int)strlen(st); snprintf(st + l, sizeof(st) - l, ", нужны рабочие"); }
                    if (b->is_alarm()) { l = (int)strlen(st); snprintf(st + l, sizeof(st) - l, ", ИЗНОШЕНА"); }
                } else if (g.destroyed_lots[(size_t)cy * g.map_size() + cx]) {
                    int l = (int)strlen(st); snprintf(st + l, sizeof(st) - l, ", сгоревший участок");
                }
            }
            text(st, 6, by + 4, 16, C_ACCENT);
            // режим стройки / скорость — справа в той же строке
            {
                char rg[160];
                int bi = sel_action - A_BUILD0;
                const char* bn = (build_mode && bi >= 0 && bi < nb)
                                   ? env.build_data()[bi]->caption.c_str() : "";
                snprintf(rg, sizeof(rg), "%s%s%s",
                         build_mode ? "Строим: " : "", bn,
                         speed_idx > 0 ? (speed_idx == 1 ? "   ▶ x1" : (speed_idx == 2 ? "   ▶ x2" : "   ▶ x3"))
                                       : "   II пауза");
                int rw = (int)MeasureTextEx(gFont, rg, 16.0f, 1.0f).x;
                text(rg, WIN_W - rw - 10, by + 4, 16, build_mode ? C_ACCENT : C_FG_DIM);
            }

            // AI build notification
            if (headless_ai && ai_build_msg_timer > 0.0f && ai_build_msg[0]) {
                int msg_w = (int)MeasureTextEx(gFont, ai_build_msg, 16.0f, 1.0f).x;
                DrawRectangle(WIN_W/2 - msg_w/2 - 10, by, msg_w + 20, BOTTOM_H, Color{40, 80, 40, 220});
                text(ai_build_msg, WIN_W/2 - msg_w/2, by + 4, 16, {150, 255, 150, 255});
            }

            // legend strip
            int ly = by + CH + 2;
            DrawRectangle(0, ly, WIN_W, CH, C_LEGEND_BG);
            text("Карта:", 6, ly + 2, 16, {40, 50, 30, 255});
            int lx = 6 + 7 * CW;
            // В легенде появилось МОРЕ: именно его невидимость и давала
            // «дорогу лесенкой» — застройка обтекала воду, покрашенную в
            // цвет травы.
            int lott[] = {LT_NONE, LT_WATER, LT_NORMAL, LT_WOOD, LT_COAL, LT_IRON, LT_OIL, LT_GOLD};
            const char* lname[] = {"море (нельзя)","вода","земля","лес","уголь","железо","нефть","золото"};
            for (int i = 0; i < 8; i++) {
                int t = 16;
                if (lott[i] == LT_NORMAL)    DrawRectangle(lx, ly + 2, t, t, C_LAND);
                else if (lott[i] == LT_NONE) { DrawRectangle(lx, ly + 2, t, t, C_SEA);
                                               DrawRectangleLines(lx, ly + 2, t, t, C_SEA_DEEP); }
                else draw_tex(earthTex[earth_icon_index((int8_t)lott[i], 0, 0)], lx, ly + 2, t);
                text(lname[i], lx + t + 4, ly + 2, 16, {40, 50, 30, 255});
                lx += t + 6 + (int)(MeasureTextEx(gFont, lname[i], 16.0f, 1.0f).x) + 4;
            }
            text("WASD — клетка · ЛКМ — поставить · Esc — отмена · F1 — помощь",
                 lx + 6, ly + 2, 14, {70, 70, 40, 255});
        }

        // ─── MENU (dropdowns on top) ───
        draw_top_menu(env);

        // ─── BUILDING POPUP (right-click) ───
        if (popup_open) {
            Rectangle r = g_popup_rect;
            DrawRectangleRec(r, Color{20, 26, 18, 245});
            DrawRectangleLinesEx(r, 2, C_ACCENT);
            int pcell = 44, pcols = (int)r.width / pcell;
            int hf = (int)(GetTime() * 12.0) % 12;
            for (int i = 0; i < nb; i++) {
                int c = i % pcols, rw = i / pcols;
                int ix = (int)r.x + c*pcell, iy = (int)r.y + rw*pcell;
                if (sel_action == A_BUILD0 + i) DrawRectangleLines(ix, iy, pcell, pcell, C_ACCENT);
                draw_build_pixel(ix + 4, iy + 4, pcell - 8, env.build_data()[i]->id, false);
                int hx = (int)mpos.x - ix, hy = (int)mpos.y - iy;
                if (hx >= 0 && hy >= 0 && hx < pcell && hy < pcell)
                    draw_tex(selEarthTex[hf], ix, iy, pcell);
                // PR 2: закрытые курикулумом иконки затемнены (поверх ховера).
                if (!env.build_allowed(env.build_data()[i]->id))
                    DrawRectangle(ix, iy, pcell, pcell, {0, 0, 0, 140});
            }
        }

        // ─── MODAL DIALOGS ───
        {
            static int prev_dlg = DLG_NONE;
            if (cur_dlg != prev_dlg) { g_dlg_age = 0; prev_dlg = cur_dlg; }
            else if (cur_dlg != DLG_NONE) g_dlg_age++;
        }
        if (cur_dlg != DLG_NONE) {
            switch (cur_dlg) {
                case DLG_MARKET:      draw_market(env); break;
                case DLG_BANK:       draw_bank(env); break;
                case DLG_NALON:      draw_nalog(env, false); break;
                case DLG_NALON_MAIN: draw_nalog(env, true); break;
                case DLG_SEASON:     draw_season(); break;
                case DLG_ABOUT:      draw_about(); break;
                case DLG_MESS:       draw_mess(); break;
                case DLG_NEW:        draw_newgame(env); break;
                case DLG_OPEN:       draw_open(env); break;
                case DLG_SAVE:       draw_save(); break;
                case DLG_CONFIRM: {
                    // Подтверждение сноса: одно нажатие больше не уносит
                    // постройку за сотни тысяч рублей.
                    int w = 460, h = 190, x = (WIN_W - w)/2, y = (WIN_H - h)/2;
                    panel(x, y, w, h, "Подтверждение");
                    text(confirm_text.c_str(), x + 16, y + 44, 18, C_WHITE);
                    const Base* cb = (confirm_x >= 0) ? gm.base_in_box(confirm_x, confirm_y) : nullptr;
                    if (cb) text(TextFormat("Клетка (%d,%d). Вернётся половина остаточной стоимости.",
                                            confirm_x, confirm_y), x + 16, y + 70, 15, C_FG_DIM);
                    Rectangle cbx = {(float)(x + 16), (float)(y + 104), 18, 18};
                    DrawRectangleRec(cbx, ask_destroy ? Color{40,80,150,255} : Color{60,60,60,255});
                    DrawRectangleLinesEx(cbx, 1, WHITE);
                    if (!ask_destroy) text("v", x + 20, y + 104, 16, C_WHITE);
                    text("спрашивать каждый раз", x + 42, y + 105, 15, C_FG_DIM);
                    if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT) &&
                        CheckCollisionPointRec(mpos, cbx)) ask_destroy = !ask_destroy;
                    if (btn(x + w - 250, y + h - 52, 110, 34, "Снести")) {
                        auto r = gm.destroy(confirm_x, confirm_y);
                        push_log(r.first ? "Постройка снесена" : r.second);
                        confirm_x = confirm_y = -1;
                        cur_dlg = DLG_NONE;
                    }
                    if (btn(x + w - 130, y + h - 52, 110, 34, "Отмена")) {
                        confirm_x = confirm_y = -1;
                        cur_dlg = DLG_NONE;
                    }
                    break;
                }
                default: break;
            }
        }

        g_dlg_prev_frame = (int)cur_dlg;
        EndDrawing();
    }

    if (g_record_demo) {
        fprintf(stderr, "[Demo] Записано шагов: %d -> %s\n",
                g_demo_steps_written, record_demo_path.c_str());
    }
    demo_record_close();
    CloseWindow();
    return 0;
}
