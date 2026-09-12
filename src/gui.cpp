#include "colony/bases.h"
#include "colony/constants.h"
#include "colony/data.h"
#include "colony/env.h"
#include "colony/game.h"
#include "colony/earth.h"

#include <algorithm>
#include <cmath>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <sstream>
#include <string>
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

static const char* LOT_NAMES[9] = {
    "ничего", "ровная земля", "вода", "лес", "уголь",
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
        default:       return C_LAND;
    }
}

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
static bool has_neighbor_building(const Game& g, int x, int y) {
    for (int dy = -1; dy <= 1; dy++)
        for (int dx = -1; dx <= 1; dx++) {
            if (dx == 0 && dy == 0) continue;
            if (g.base_in_box(x + dx, y + dy)) return true;
        }
    return false;
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
    // If buildings exist, start from cells adjacent to existing buildings
    // Otherwise start from top-left corner
    std::vector<Cell> frontier, remaining;
    if (!g.bases.empty()) {
        for (auto& c : cells)
            if (has_neighbor_building(g, c.x, c.y)) frontier.push_back(c);
            else remaining.push_back(c);
        if (frontier.empty()) { status = "Нужно строить рядом с существующими зданиями"; return; }
    } else {
        frontier.push_back(cells[0]);
        for (size_t i = 1; i < cells.size(); i++) remaining.push_back(cells[i]);
    }
    // Build: always add cells adjacent to already-built to the frontier
    int built = 0, skipped = 0;
    while (!frontier.empty()) {
        Cell c = frontier[0];
        frontier.erase(frontier.begin());
        auto r = g.build(bid, c.x, c.y);
        if (r.first) {
            built++;
            stat_built[bd->caption]++;
            // Add neighbors from remaining that are now adjacent to a building
            for (auto it = remaining.begin(); it != remaining.end();) {
                if (has_neighbor_building(g, it->x, it->y)) {
                    frontier.push_back(*it);
                    it = remaining.erase(it);
                } else it++;
            }
        } else {
            skipped++;
            status = r.second;
            break;
        }
    }
    if (built > 0 && skipped > 0)
        status = TextFormat("Построено: %d, не хватило денег на %d", built, skipped);
    else if (built > 0)
        status = TextFormat("Построено: %d", built);
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
static bool game_over = false;
static Rectangle g_time_btn[3];
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
static void draw_minimap(int mx, int my, int mw, int mh, const Game& g, const Camera2D& cam) {
    int ms = g.map_size();
    if (g_mini_ms != ms) {
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
           DLG_SEASON, DLG_ABOUT, DLG_MESS, DLG_NEW, DLG_OPEN, DLG_SAVE };
static Dlg cur_dlg = DLG_NONE;
static bool tax_from_dialog = false;  // рынок открыт из диалога налогов
static int open_menu = -1;
static int dlg_mode = 0;
static long long market_qty[9] = {0};
static char bank_buf[32] = {0};
static int dlg_season = 0;
static char mess_title[64] = "Сообщение";
static char mess_text[256] = "";
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

static int ai_read_action() {
    // Read action from file. Returns -1 if no new action available.
    if (ai_actions_path.empty()) return -1;
    std::ifstream f(ai_actions_path);
    if (!f.is_open()) return -1;
    int action = -1;
    f >> action;
    f.close();
    // Delete the file so Python knows the action was consumed.
    // Using remove() avoids the race condition with truncation:
    // Python may write a new action between f.close() and an ofstream open.
    std::filesystem::remove(ai_actions_path);
    return action;
}

static void ai_write_state(const Game& g, const std::vector<float>& obs, int action, bool terminated, const std::vector<float>& mask, const std::vector<float>& minimap, double reward = 0.0) {
    // Write state JSON to file (includes obs, action_mask, and minimap for Python policy)
    std::ofstream f(ai_state_path);
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
      << "\"terminated\":" << (terminated ? "true" : "false") << ","
      << "\"obs\":[";
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
    f.close();
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
    cur_dlg = DLG_NONE;
    stat_built.clear(); stat_earned = 0; stat_spent = 0;
    stat_started = false; stat_tax_over = false;
    ai_build_msg[0] = '\0'; ai_build_msg_timer = 0.0f;
    cam.target = {(float)env.game().earth.init_sel_x * TILE,
                  (float)env.game().earth.init_sel_y * TILE};
    cam.zoom = 1.0f;
    // Clear action file so Python knows to send a new one
    std::filesystem::remove(ai_actions_path);
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
        strcpy(mess_title, r.ok ? "Сообщение" : "Ошибка");
        strncpy(mess_text, r.msg.c_str(), sizeof(mess_text) - 1);
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
        if (v > 0) { auto r = g.bank_take(v); if (!r.first) { strcpy(mess_title,"Ошибка"); strncpy(mess_text,r.second.c_str(),sizeof(mess_text)-1); cur_dlg=DLG_MESS; } else cur_dlg=DLG_NONE; }
        bank_buf[0] = 0;
    }
    if (btn(x + 210, y + 126, 90, 30, "Вернуть (Q)")) {
        long long v = bank_buf[0] ? (long long)strtoll(bank_buf, nullptr, 10) : g.credit;
        if (v > 0) { auto r = g.bank_give(v); if (!r.first) { strcpy(mess_title,"Ошибка"); strncpy(mess_text,r.second.c_str(),sizeof(mess_text)-1); cur_dlg=DLG_MESS; } else cur_dlg=DLG_NONE; }
        bank_buf[0] = 0;
    }
    if (btn(x + 14, y + h - 40, 100, 30, "Отмена")) { cur_dlg = DLG_NONE; bank_buf[0] = 0; }
}

// ─── NALOG (annual / main) — 2 кнопки: Продать ресурсы / Нет ───
static void draw_nalog(ColonyEnvCpp& env, bool main_tax) {
    Game& g = env.game();
    int w = 420, h = 200, x = (WIN_W - w)/2, y = (WIN_H - h)/2;
    panel(x, y, w, h, main_tax ? "Главный налог" : "Налоги");
    long long total = main_tax ? g.main_tax_amount() : g.annual_tax_amount();
    text(main_tax ? "Каждые 10 лет выплачивается Главный налог."
                      : "Ежегодно следует уплатить налоги за прошедший год:",
             x + 14, y + 40, 14, WHITE);
    text(TextFormat("Итого: %lld", (long long)total), x + 14, y + 70, 20, {255,220,120,255});
    text(TextFormat("Ваши деньги: %lld — не хватает.", (long long)g.money), x + 14, y + 95, 14, {255,150,150,255});
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
    int w = 320, h = 140, x = (WIN_W - w)/2, y = (WIN_H - h)/2;
    panel(x, y, w, h, mess_title);
    text(mess_text, x + 16, y + 44, 16, WHITE);
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
    if (btn(x + 330, y + 64, 110, 32, "ОК")) { env.reset(42); sel_bx = sel_by = -1; cur_dlg = DLG_NONE; tax_from_dialog = false; stat_built.clear(); stat_earned = 0; stat_spent = 0; stat_started = false; stat_tax_over = false; ai_build_msg[0] = '\0'; ai_build_msg_timer = 0.0f; }
    if (btn(x + 330, y + 104, 110, 32, "Отмена")) cur_dlg = DLG_NONE;
}
static void draw_open(ColonyEnvCpp& env) {
    int w = 460, h = 260, x = (WIN_W - w)/2, y = (WIN_H - h)/2;
    panel(x, y, w, h, "Загрузка сохранённой игры");
    text("Выберите игру", x + 14, y + 38, 16, WHITE);
    auto items = list_saves();
    static int sel = -1;
    list_box(x + 14, y + 64, 300, 150, items, sel, sel);
    if (btn(x + 330, y + 64, 110, 32, "ОК")) { strcpy(mess_title,"Сообщение"); strcpy(mess_text,"Загрузка в C++ не реализована"); cur_dlg=DLG_MESS; }
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
    if (btn(x + 330, y + 64, 110, 32, "ОК")) { strcpy(mess_title,"Сообщение"); strcpy(mess_text,"Сохранение в C++ не реализовано"); cur_dlg=DLG_MESS; }
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
        const char* its[] = {"Звук (F7)", "Музыка (F8)", "Полный экран (F9)", "Параметры (F12)"};
        bool en[] = {true, false, true, false};
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
                if (i == 0) { strcpy(mess_title,"Помощь"); strcpy(mess_text,"Стрелки — сдвиг карты, Пробел — день, W — неделя,\nB — купить, S — продать, K — банк, G — улучшить,\nF — поиск, R — восстановить, P — блок, D — разобрать,\nU — отмена, ЛКМ по карте — зажать и выделить область, ПКМ — меню зданий (выбор ЛКМ строит)."); cur_dlg = DLG_MESS; }
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
static void draw_tools(ColonyEnvCpp& env) {
    (void)env;
    static const ToolDef tools[12] = {
        {9,1,0,0},   {15,2,0,1},  {16,3,0,2},  {24,4,0,3},  {13,5,0,4},  {26,6,0,5},
        {25,7,1,0},  {23,8,1,1},
        {12,9,2,0},  {11,10,2,1}, {10,11,2,4}, {14,12,2,5},
    };
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
    }
}

int main(int argc, char* argv[]) {
    int64_t seed = 42;
    int map_size = 280;
    int curriculum_stage = 0;
    int minimap_radius = -1;
    std::string reward_config_path;
    std::vector<std::string> unlock_ids;  // ручной набор из вкладки «Курикулум»
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--seed" && i+1 < argc) seed = std::stoll(argv[++i]);
        else if (a == "--map-size" && i+1 < argc) map_size = std::stoi(argv[++i]);
        else if (a == "--stage" && i+1 < argc) curriculum_stage = std::stoi(argv[++i]);
        else if (a == "--unlock-id" && i+1 < argc) unlock_ids.push_back(argv[++i]);
        else if (a == "--unlock-ids" && i+1 < argc) {
            // CSV, как в UI (unlock_ids): --unlock-ids WaterChannel,Road
            std::string csv = argv[++i];
            std::stringstream ss(csv);
            std::string item;
            while (std::getline(ss, item, ',')) {
                // trim
                size_t b = item.find_first_not_of(" \t");
                size_t e = item.find_last_not_of(" \t");
                if (b == std::string::npos) continue;
                unlock_ids.push_back(item.substr(b, e - b + 1));
            }
        }
        else if (a == "--minimap-radius" && i+1 < argc) minimap_radius = std::stoi(argv[++i]);
        else if (a == "--headless-ai") headless_ai = true;
        else if (a == "--actions-file" && i+1 < argc) ai_actions_path = argv[++i];
        else if (a == "--state-file" && i+1 < argc) ai_state_path = argv[++i];
        else if (a == "--reward-config" && i+1 < argc) reward_config_path = argv[++i];
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

    // Ручной набор зданий (--unlock-ids) применяется вместе с этапом: на этапе 0
    // без него открывались все 32 здания, и наблюдение за моделью, обученной на
    // одном водоканале, показывало постройку прииска.
    ColonyEnvCpp env(bd, ed, seed, map_size, curriculum_stage, unlock_ids, rc, "normal", gui_no_city_game_over, gui_no_people_days);
    if (minimap_radius > 0) {
        env.set_minimap_radius(minimap_radius);
    }
    env.reset(seed);
    const Game& g = env.game();
    prev_season = g.season;

    cam.target = {(float)g.earth.init_sel_x * TILE, (float)g.earth.init_sel_y * TILE};
    cam.offset = {(float)MAP_X + MAP_W/2.0f, (float)MAP_Y + MAP_H/2.0f};
    cam.zoom = 1.0f;
    if (headless_ai) {
        cam.target = {(float)g.map_size() * TILE / 2.0f, (float)g.map_size() * TILE / 2.0f};
        cam.zoom = 0.8f;
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

    SetConfigFlags(FLAG_VSYNC_HINT);
    InitWindow(WIN_W, WIN_H, "Сахалинская колония 3.47");
    SetTargetFPS(60);
    load_assets();
    // Debug: log startup mode
    {
        FILE* f = fopen("ai_debug_gui.log", "a");
        if (f) {
            fprintf(f, "=== GUI STARTUP === headless=%d actions_path='%s' state_path='%s' seed=%lld map_size=%d stage=%d unlock_ids=%zu\n",
                    headless_ai, ai_actions_path.c_str(), ai_state_path.c_str(),
                    (long long)seed, map_size, curriculum_stage, unlock_ids.size());
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
                int action = ai_read_action();
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
                } else {
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
                env.reset(new_seed); sel_bx = sel_by = -1; cur_dlg = DLG_NONE;
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
                env.reset(same_seed); sel_bx = sel_by = -1; cur_dlg = DLG_NONE;
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
        // ── Esc closes any open dialog ──
        if (cur_dlg != DLG_NONE && IsKeyPressed(KEY_ESCAPE)) {
            cur_dlg = DLG_NONE;
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

        float wh = GetMouseWheelMove();
        if (wh != 0) {
            cam.zoom *= (wh > 0) ? 1.1f : 0.9f;
            if (cam.zoom < 0.4f) cam.zoom = 0.4f;
            if (cam.zoom > 3.0f) cam.zoom = 3.0f;
        }

        // ── Keyboard arrow panning ──
        {
            float ps = 14.0f / cam.zoom;
            if (IsKeyDown(KEY_RIGHT)) cam.target.x += ps;
            if (IsKeyDown(KEY_LEFT))  cam.target.x -= ps;
            if (IsKeyDown(KEY_DOWN))  cam.target.y += ps;
            if (IsKeyDown(KEY_UP))    cam.target.y -= ps;
        }

        // ── Edge-scroll when mouse approaches map border (with small delay) ──
        {
            float ed = 28.0f;
            bool in_edge = over_map &&
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
                        sel_action = A_BUILD0 + i;
                        status = env.build_data()[i]->caption;
                        if (has_sel) {
                            do_build_area(env, gm, A_BUILD0 + i, true,
                                          sel_x0, sel_y0, sel_x1, sel_y1, prc_x, prc_y, status);
                            has_sel = false;
                        }
                    }
                }
            }
            // tool button click
            for (int i = 0; i < 12; i++) {
                if (pressed && CheckCollisionPointRec(mpos, g_tool_rect[i])) {
                    static const int acts[12] = {1,2,3,4,5,6,7,8,9,10,11,12};
                    int act = acts[i];
                    if (act == 7) cur_dlg = DLG_BANK;
                    else if (act == 9) { dlg_mode = 0; cur_dlg = DLG_MARKET; }
                    else if (act == 10) { dlg_mode = 1; cur_dlg = DLG_MARKET; }
                    else if (act == 11) { auto o = env.step(A_DAY); if (o.terminated) game_over = true; }
                    else if (act == 12) { auto o = env.step(A_WEEK); if (o.terminated) game_over = true; }
                    else if (over_map && cx >= 0 && cy >= 0 && cx < g.map_size() && cy < g.map_size()) {
                        if (act == 1) { auto r = gm.good_earth(cx, cy); if (!r.first) status = r.second; }
                        else if (act == 2) {
                            Base* b = gm.find_slowest_base();
                            if (b) { cam.target.x = (float)b->x*TILE; cam.target.y = (float)b->y*TILE; status = "Слабейшее: " + b->data->caption; }
                            else status = "Нет изношенных построек";
                        }
                        else if (act == 3) { auto r = gm.restore(cx, cy); if (!r.ok) status = r.msg; }
                        else if (act == 4) { auto r = gm.restore_all(); if (!r.ok) status = r.msg; }
                        else if (act == 5) { auto r = gm.destroy(cx, cy); if (!r.first) status = r.second; }
                        else if (act == 6) { if (!gm.undo()) status = "Нечего отменять"; }
                        else if (act == 8) { auto r = gm.preserve(cx, cy); if (!r.first) status = r.second; }
                    }
                }
            }
            // ── LMB drag = select area ONLY (no build yet) ──
            if (!popup_open) {
                if (pressed && over_map) {
                    selecting = true; has_sel = false; sel_bx = -1; sel_by = -1;
                    sel_x0 = sel_x1 = cx; sel_y0 = sel_y1 = cy;
                }
                if (selecting && IsMouseButtonDown(MOUSE_BUTTON_LEFT) && over_map) {
                    sel_x1 = cx; sel_y1 = cy;
                }
                if (IsMouseButtonReleased(MOUSE_BUTTON_LEFT) && selecting) {
                    selecting = false;
                    if (sel_x0 == sel_x1 && sel_y0 == sel_y1) {
                        // single-cell click: if a building is there, select it for info
                        const Base* b = (cx >= 0 && cy >= 0 && cx < g.map_size() && cy < g.map_size())
                                           ? g.base_in_box(cx, cy) : nullptr;
                        if (b) { sel_bx = cx; sel_by = cy; has_sel = false; status = b->data->caption; }
                        else   { sel_bx = -1; sel_by = -1; has_sel = true; sel_x0 = sel_x1 = cx; sel_y0 = sel_y1 = cy; }
                    } else {
                        has_sel = true; sel_bx = -1; sel_by = -1;
                    }
                }
            }

            // ── Minimap LMB click = recenter big map to that point ──
            if (pressed && g_mini_rect.width > 0 && CheckCollisionPointRec(mpos, g_mini_rect)) {
                float fx = (mpos.x - g_mini_rect.x) / g_mini_rect.width;
                float fy = (mpos.y - g_mini_rect.y) / g_mini_rect.height;
                int msz = gm.map_size();
                cam.target.x = fx * msz * TILE;
                cam.target.y = fy * msz * TILE;
            }

            // ── ПКМ = open building popup (or close if already open) ──
            if (IsMouseButtonPressed(MOUSE_BUTTON_RIGHT)) {
                if (popup_open) {
                    popup_open = false;
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
                        do_build_area(env, gm, A_BUILD0 + ci, has_sel,
                                      sel_x0, sel_y0, sel_x1, sel_y1, prc_x, prc_y, status);
                        sel_action = A_BUILD0 + ci;
                        has_sel = false;
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
                    if (i == 0) { auto o = env.step(A_DAY); if (o.terminated) game_over = true; }
                    else if (i == 1) { auto o = env.step(A_WEEK); if (o.terminated) game_over = true; }
                    else advance_month(env);
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
            if (IsKeyPressed(KEY_SPACE) || IsKeyPressed(KEY_ENTER)) {
                auto out = env.step(A_DAY); if (out.terminated) game_over = true;
            }
            if (IsKeyPressed(KEY_W)) { auto out = env.step(A_WEEK); if (out.terminated) game_over = true; }
            if (over_map && cx >= 0 && cy >= 0 && cx < g.map_size() && cy < g.map_size()) {
                if (IsKeyPressed(KEY_G)) { auto r = gm.good_earth(cx, cy); if (!r.first) status = r.second; }
                if (IsKeyPressed(KEY_F)) {
                    Base* b = gm.find_slowest_base();
                    if (b) { cam.target.x = (float)b->x*TILE; cam.target.y = (float)b->y*TILE; status = "Слабейшее: " + b->data->caption; }
                    else status = "Нет изношенных построек";
                }
                if (IsKeyPressed(KEY_R)) { auto r = gm.restore(cx, cy); if (!r.ok) status = r.msg; }
                if (IsKeyPressed(KEY_A)) { auto r = gm.restore_all(); if (!r.ok) status = r.msg; }
                if (IsKeyPressed(KEY_P)) { auto r = gm.preserve(cx, cy); if (!r.first) status = r.second; }
                if (IsKeyPressed(KEY_D)) { auto r = gm.destroy(cx, cy); if (!r.first) status = r.second; }
                if (IsKeyPressed(KEY_U)) { if (!gm.undo()) status = "Нечего отменять"; }
            }
            if (IsKeyPressed(KEY_B)) { dlg_mode = 0; cur_dlg = DLG_MARKET; }
            if (IsKeyPressed(KEY_S)) { dlg_mode = 1; cur_dlg = DLG_MARKET; }
            if (IsKeyPressed(KEY_K)) cur_dlg = DLG_BANK;
            if (IsKeyPressed(KEY_N)) { auto o = env.step(A_DAY); if (o.terminated) game_over = true; }

            // ─── Menu shortcuts ───
            if (IsKeyPressed(KEY_F4)) cur_dlg = DLG_NEW;
            if (IsKeyPressed(KEY_F5)) cur_dlg = DLG_OPEN;
            if (IsKeyPressed(KEY_F2)) cur_dlg = DLG_SAVE;
            if (IsKeyPressed(KEY_F1)) {
                strcpy(mess_title, "Помощь");
                strcpy(mess_text, "Стрелки — сдвиг карты, Пробел — день, W — неделя,\nB — купить, S — продать, K — банк, G — улучшить,\nF — поиск, R — восстановить, P — блок, D — разобрать,\nU — отмена, ЛКМ по карте — зажать и выделить область, ПКМ — меню зданий (выбор ЛКМ строит).");
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
                draw_info_box(L, INFO_X, INFO_Y, INFO_W);
            }
        }

        // separator under top band
        DrawRectangle(0, MAP_Y - 6, WIN_W, 1, {70, 90, 80, 255});

        // ─── MAP ───
        {
            int vx = MAP_X, vy = MAP_Y, vw = MAP_W, vh = MAP_H;
            BeginScissorMode(vx, vy, vw, vh);
            DrawRectangle(vx, vy, vw, vh, C_LAND);
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
            // cursor = animated selection frame (only when no build menu is open)
            if (!popup_open && over_map && cx >= 0 && cy >= 0 && cx < msz && cy < msz) {
                int px = cx * TILE, py = cy * TILE;
                int f = (int)(GetTime() * 12.0) % 12;
                int8_t fl = g.earth.lot(cx, cy);
                draw_tex(fl == LT_NONE ? selNoneTex[f] : selEarthTex[f], px, py, TILE);
            }
            // area selection = every cell inside the rectangle gets an animated frame
            if (selecting || has_sel) {
                int x0 = std::min(sel_x0, sel_x1), x1 = std::max(sel_x0, sel_x1);
                int y0 = std::min(sel_y0, sel_y1), y1 = std::max(sel_y0, sel_y1);
                int f = (int)(GetTime() * 12.0) % 12;
                for (int y = y0; y <= y1; y++)
                    for (int x = x0; x <= x1; x++)
                        draw_tex(selEarthTex[f], x*TILE, y*TILE, TILE);
            }
            EndMode2D();
            EndScissorMode();
            DrawRectangleLines(vx, vy, vw, vh, {60, 80, 70, 255});
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
                sy = by + bh + 6;
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
            // status of current cell
            char st[256];
            if (over_map && cx >= 0 && cy >= 0 && cx < g.map_size() && cy < g.map_size()) {
                int8_t lot = g.earth.lot(cx, cy);
                snprintf(st, sizeof(st), "Клетка (%d,%d): %s", cx, cy, LOT_NAMES[lot < 0 || lot > 8 ? 8 : lot]);
                if (g.is_good(cx, cy)) { int l = (int)strlen(st); snprintf(st + l, sizeof(st) - l, ", улучшена"); }
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
            } else {
                snprintf(st, sizeof(st), "Ход: %s %d %s %d", month_ru(g.month), g.day, "года", g.year);
            }
            text(st, 6, by + 4, 16, C_ACCENT);

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
            int lott[] = {LT_WATER, LT_NORMAL, LT_WOOD, LT_COAL, LT_IRON, LT_OIL, LT_GOLD};
            const char* lname[] = {"вода","земля","лес","уголь","железо","нефть","золото"};
            for (int i = 0; i < 7; i++) {
                int t = 16;
                if (lott[i] == LT_NORMAL) DrawRectangle(lx, ly + 2, t, t, C_LAND);
                else draw_tex(earthTex[earth_icon_index((int8_t)lott[i], 0, 0)], lx, ly + 2, t);
                text(lname[i], lx + t + 4, ly + 2, 16, {40, 50, 30, 255});
                lx += t + 6 + (int)strlen(lname[i]) * CW;
            }
            text("ЛКМ — выделить область, ПКМ — меню зданий, колесо — зум, ср. кнопка — сдвиг", lx + 6, ly + 2, 14, {70, 70, 40, 255});
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
                default: break;
            }
        }

        g_dlg_prev_frame = (int)cur_dlg;
        EndDrawing();
    }

    CloseWindow();
    return 0;
}
