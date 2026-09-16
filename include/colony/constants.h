#pragma once

#include <array>
#include <cstdint>
#include <string>

namespace colony {

// Индексы ресурсов (порядок фиксирован, как в оригинале)
constexpr int SUNDUK_SIZE = 9;
constexpr int GOLD = 0, FOOD = 1, COAL = 2, IRON = 3, OIL = 4, STONE = 5,
              WATER = 6, WOOD = 7, ENERGY = 8;

// Старт партии
constexpr int INIT_MONEY = 82000;
constexpr int INIT_PEOPLE = 28;
constexpr int START_YEAR = 1890, START_MONTH = 3, START_DAY = 1;

// Улучшенная земля (3.47)
constexpr int BUYGOODEARTH = 32997;

// Банк
constexpr int CREDITPERCENT = 2;
constexpr int MAXCREDIT_LIGHT = 10000000;

// Рождаемость/смертность
constexpr int BIRTH_DAYS = 364 * 6;    // 2184
constexpr int DEATH_DAYS = 364 * 70;   // 25480

// События
constexpr int EVENT_ATTEMPT_DAYS = 455;

// Уровень сложности
constexpr const char* DIFFICULTY_NORMAL = "normal";
constexpr const char* DIFFICULTY_LIGHT = "light";
constexpr int LIGHT_MONEY_MULT = 2;

// Пополнение (20 июня)
constexpr int ADDPEOPLE = 28;
constexpr int ADDPEOPLERANGE = 5;

// Налоги
constexpr int NALOG_MAIN = 500000;
constexpr int NALOG_EARTH = 420;
constexpr int NALOG_BUYPERCENT = 1;
constexpr int NALOG_SALEPERCENT = 2;
constexpr int NALOG_ECOLOGY = 509;
constexpr int NALOG_SOCIAL = 1220;
constexpr int NALOG_RES = 2957;

// Рынок
inline constexpr std::array<int, SUNDUK_SIZE> BUY_SUNDUK = {1000, 2, 32, 54, 48, 4, 6, 14, 37};
inline constexpr std::array<int, SUNDUK_SIZE> SALE_SUNDUK = {1000, 1, 30, 50, 45, 4, 5, 12, 35};

// Сезоны (индексы; соответствие python: season_for_month)
enum Season { SEASON_SUMMER = 0, SEASON_AUTUMN = 1, SEASON_WINTER = 2, SEASON_SPRING = 3 };
inline constexpr const char* SEASON_NAMES[4] = {"summer", "autumn", "winter", "spring"};

// Дней в сезоне (июнь-август=92, сент-нояб=91, дек-фев=90, март-май=92)
inline constexpr int SEASON_DAYS[4] = {92, 91, 90, 92};

inline Season season_for_month(int month) {
    if (month >= 3 && month <= 5) return SEASON_SPRING;
    if (month >= 6 && month <= 8) return SEASON_SUMMER;
    if (month >= 9 && month <= 11) return SEASON_AUTUMN;
    return SEASON_WINTER;
}

// Типы земли
enum LotType {
    LT_NONE = 0, LT_NORMAL = 1, LT_WATER = 2, LT_WOOD = 3, LT_COAL = 4,
    LT_IRON = 5, LT_OIL = 6, LT_GOLD = 7, LT_LAST = 8,
    LT_EVERYWHERE = 10,
};

// Названия ресурсов и месяцев (для текстов событий/даты)
inline constexpr const char* SUNDUK_CAPTIONS[SUNDUK_SIZE] = {
    "Золото", "Продовольствие", "Уголь", "Железо", "Нефть",
    "Камень", "Вода", "Дерево", "Энергия",
};
inline constexpr const char* MONTH_NAMES[12] = {
    "января", "февраля", "марта", "апреля", "мая", "июня",
    "июля", "августа", "сентября", "октября", "ноября", "декабря",
};

// Имя главной постройки
inline constexpr const char* DEPOT_ID = "City";

// Условия конца игры
constexpr bool GAME_OVER_NO_CITY = true;
constexpr int GAME_OVER_NO_PEOPLE_DAYS = 365;

// Число «пустых слотов» при выборе события
constexpr int EVENT_BLANK_SLOTS = 20;

// Праздничные/событийные даты
constexpr int EVENT_SEASON_START_YEAR = 1891, EVENT_SEASON_START_MONTH = 3;
constexpr int POPULATION_REMINDER_MONTH = 5, POPULATION_REMINDER_DAY = 20;
constexpr int POPULATION_ARRIVAL_MONTH = 6, POPULATION_ARRIVAL_DAY = 20;

inline constexpr std::array<int, 12> DAYS_IN_MONTH = {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};

inline bool is_leap(int year) {
    return year % 4 == 0 && (year % 100 != 0 || year % 400 == 0);
}

inline int days_in_month(int year, int month) {
    if (month == 2 && is_leap(year)) return 29;
    return DAYS_IN_MONTH[month - 1];
}

inline std::string thousands(int64_t value) {
    std::string s = std::to_string(value);
    std::string out;
    int cnt = 0;
    for (auto it = s.rbegin(); it != s.rend(); ++it) {
        if (cnt && cnt % 3 == 0) out.push_back(' ');
        out.push_back(*it);
        cnt++;
    }
    std::string res(out.rbegin(), out.rend());
    return res;
}

// --- константы RL-среды (rl/env.py) ---
constexpr int A_DAY = 0;
constexpr int A_WEEK = 1;
constexpr int A_BUILD0 = 2;
constexpr int N_MANAGERS = 11;
// Directional road actions. The plain BUILD:Road action places a road at the
// BFS-first legal cell, which the agent cannot aim -- so water 7 cells away
// stayed unreachable (the WaterChannel action was masked on 0/2000 random
// steps). These four extend the road frontier towards a chosen compass
// direction instead, giving the policy a way to act on water_dx/water_dy and
// the minimap. They sit AFTER the manager block so the
// `A_BUILD0 + i <-> build_ids_[i]` invariant is untouched.
constexpr int N_ROAD_DIRS = 4;
inline constexpr int ROAD_DIR_DX[N_ROAD_DIRS] = {1, -1, 0, 0};   // E, W, S, N
inline constexpr int ROAD_DIR_DY[N_ROAD_DIRS] = {0, 0, 1, -1};
inline constexpr const char* ROAD_DIR_NAMES[N_ROAD_DIRS] = {
    "ROAD_E", "ROAD_W", "ROAD_S", "ROAD_N"};
constexpr int MAX_STEPS = 10000;
constexpr int LOT_RADIUS = 45;
inline constexpr const char* ROAD_ID = "Road";
constexpr int TAX_GRACE_DAYS = 60;

inline constexpr const char* BUILD_SUBSET[32] = {
    "WaterChannel", "Farm", "Garden", "House", "SmallHouse",
    "Sawmill", "Coalmine", "Ironmine", "Refinery", "Goldmine",
    "PowerStation", "HydroStation", "Road", "Fish", "CoalCut",
    "HuntingLand", "CowFarm", "Mushroom", "BigHouse", "BigFarm",
    "Apiary", "Torchlight", "Hothouse", "SuperHouse", "BigSawmill",
    "WaterMill", "BigRefinary", "Puerperal", "BigIronmine",
    "AirStation", "SmallAtomStation", "AtomStation"};

}  // namespace colony