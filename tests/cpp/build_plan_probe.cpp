// «Идеальная партия за 2 года»: расчёт оптимального плана застройки на живом
// движке игры (Game), без RL-обёртки — так играет человек в GUI: в один день
// можно сделать сколько угодно действий (строить, торговать, брать кредит),
// время двигает только «Пробел»/«Неделя».
//
// Зачем файл: ответить числами на вопрос «какие здания строить и когда»,
// а не на глазок. Здесь живут два режима: жадный планировщик с параметрами
// (см. Params) и проигрывание готового сценария «дата ▸ постройка» (--plan).
// Метрика — на 01.03.1892: КЭШ (деньги + склад по цене продажи − долг банку)
// и АКТИВЫ (здания по остаточной стоимости с учётом износа).
// Итоговое расписание лежит в docs/OPTIMAL_BUILD_2Y.md, перебор сценариев —
// в tests/cpp/search_plan.py.
//
// Build (из корня репозитория):
//   g++ -std=c++17 -O2 -Iinclude -Iinclude/third_party -o /tmp/plan
//       tests/cpp/build_plan_probe.cpp src/data.cpp src/resources.cpp
//       src/game.cpp src/rng.cpp src/earth.cpp
//   /tmp/plan --schedule --seed 42
//   /tmp/plan --plan docs/plan_2y.txt --schedule --trace --seed 42
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <optional>
#include <queue>
#include <string>
#include <vector>

#include "colony/data.h"
#include "colony/game.h"

using namespace colony;

namespace {

// Цены рынка (constants.h) — дублируем как int64 для арифметики.
const int64_t SALEP[SUNDUK_SIZE] = {1000, 1, 30, 50, 45, 4, 5, 12, 35};
const int64_t BUYP[SUNDUK_SIZE] = {1000, 2, 32, 54, 48, 4, 6, 14, 37};

// ─────────────────────────────────────────────────────────────── параметры
struct Params {
    int64_t seed = 42;
    int map_size = 280;
    int horizon_days = 731;      // 01.03.1890 → 01.03.1892
    int pu_target = 4;           // сколько «Домов матери и ребёнка» (родильных)
    int pu_cutoff = 400;         // позже этого дня роддома уже не окупятся
    int pu_start = 0;            // раньше этого дня роддома не строим
    bool pu_borrow = false;      // можно ли брать кредит под роддом
    int pop_cap = 250;           // потолок жилья (людей больше не кормим стройкой домов)
    int gold_max = 40;           // потолок числа золотых приисков
    int refinery_max = 0;        // нефтедобыча
    int bigsawmill_max = 0;      // лесоповал
    int torchlight_max = 0;      // факел (круглый год)
    int mushroom_max = 0;        // грибная плантация (осень)
    int secondary_max = 0;       // «мелочь» (пасека, лесопилка, карьер…) — по столько штук
    double payback = 1.0;        // минимум «заработает до конца / цена»
    int64_t credit_cap = 300000; // рабочий потолок долга (жёсткий лимит 500k)
    int borrow_window = 40;      // за сколько дней до начала работы можно занимать
    bool gold_only_credit = true;  // в долг берём только под прииск (окупается за лето)
    int64_t small_credit = 12000;   // мелкий заём (добить до цены) — можно под всё
    int64_t cash_reserve = 6000;   // подушка (плюс оборотка на воду по приискам)
    int house_margin = 20;       // запас мест над прогнозом населения
    int build_workers_slack = 0; // запас людей сверх need_workers
    std::string plan_file;       // сценарий «дата id count» вместо жадности
    std::string bases_path = "configs/bases.json";  // можно подсунуть иной баланс
    bool schedule = false;       // печатать план
    bool trace = false;          // помесячная трассировка
};

struct Event {           // строка расписания
    int day;             // индекс дня от старта
    int y, m, d;
    std::string id;
    int count;
    int64_t price;
    int64_t money_after;
};

struct PlanItem {  // строка сценария: «1890-03-01 Refinery 2»
    int y, m, d;
    std::string id;
    int count;
};

std::vector<PlanItem> load_plan(const std::string& path) {
    std::vector<PlanItem> out;
    FILE* f = fopen(path.c_str(), "r");
    if (!f) {
        fprintf(stderr, "не открыть сценарий: %s\n", path.c_str());
        exit(2);
    }
    char line[512];
    while (fgets(line, sizeof(line), f)) {
        char* hash = strchr(line, '#');
        if (hash) *hash = 0;
        PlanItem it;
        char id[128];
        int n = sscanf(line, "%d-%d-%d %127s %d", &it.y, &it.m, &it.d, id, &it.count);
        if (n < 4) continue;
        it.id = id;
        it.count = (n >= 5 && it.count > 0) ? it.count : 1;
        out.push_back(it);
    }
    fclose(f);
    return out;
}

const char* const MONTHS[12] = {"янв", "фев", "мар", "апр", "май", "июн",
                                "июл", "авг", "сен", "окт", "ноя", "дек"};

// ───────────────────────────────────────────────────── размещение построек
struct Site {
    int x = 0, y = 0;
    std::vector<std::pair<int, int>> roads;  // клетки под дорогу (по порядку)
};

// Ближайшая (по BFS от колонии) свободная клетка нужного типа земли.
// roads — клетки, которые надо замостить дорогой, чтобы клетка стала связной.
std::optional<Site> find_site(const Game& g, int need_earth, bool prefer_normal) {
    const int ms = g.map_size();
    const std::vector<int8_t>& lots = g.earth.lots();
    const std::vector<int32_t>& bidx = g.base_index_map();
    std::vector<int> dist((size_t)ms * ms, -1);
    std::vector<int> prev((size_t)ms * ms, -1);
    std::queue<int> q;
    for (const Base& b : g.bases) {
        size_t i = (size_t)b.y * ms + b.x;
        if (dist[i] < 0) {
            dist[i] = 0;
            q.push((int)i);
        }
    }
    const int dx4[4] = {1, -1, 0, 0};
    const int dy4[4] = {0, 0, 1, -1};
    int best = -1, best_d = 1 << 30;
    while (!q.empty()) {
        int cur = q.front();
        q.pop();
        if (dist[cur] >= best_d) continue;  // дальше уже не улучшим
        int cx = cur % ms, cy = cur / ms;
        for (int k = 0; k < 4; k++) {
            int nx = cx + dx4[k], ny = cy + dy4[k];
            if (nx < 0 || ny < 0 || nx >= ms || ny >= ms) continue;
            size_t ni = (size_t)ny * ms + nx;
            if (dist[ni] >= 0) continue;
            int8_t lt = lots[ni];
            if (!(lt >= LT_NORMAL && lt < LT_LAST)) continue;  // вода-«ничто» и край
            if (bidx[ni] >= 0) continue;                        // занято постройкой
            if (g.destroyed_lots[ni] > 0) continue;             // пепелище
            dist[ni] = dist[cur] + 1;
            prev[ni] = cur;
            bool match = (need_earth == LT_EVERYWHERE)
                             ? (prefer_normal ? lt == LT_NORMAL
                                              : (lt >= LT_NORMAL && lt < LT_LAST))
                             : (lt == need_earth);
            if (match && dist[ni] < best_d) {
                best_d = dist[ni];
                best = (int)ni;
            }
            q.push((int)ni);
        }
    }
    if (best < 0) return std::nullopt;
    Site s;
    s.x = best % ms;
    s.y = best / ms;
    for (int cur = prev[best]; cur >= 0 && dist[cur] > 0; cur = prev[cur])
        s.roads.push_back({cur % ms, cur / ms});
    std::reverse(s.roads.begin(), s.roads.end());  // от колонии наружу
    return s;
}

// ─────────────────────────────────────────────────────────── планировщик
class Planner {
public:
    Planner(const std::vector<BaseData>& bd, const std::vector<BaseEvent>& ed,
            const Params& p)
        : p_(p), g_(bd, ed, p.seed, p.map_size) {
        g_.set_enable_undo(false);  // undo = глубокая копия карты на каждое действие
        for (const BaseData& d : bd) data_[d.id] = &d;
        auto add = [&](const char* id, int limit) {
            auto it = data_.find(id);
            if (it != data_.end() && limit > 0) cands_.push_back({it->second, limit});
        };
        add("Goldmine", p.gold_max);
        add("Refinery", p.refinery_max);
        add("BigSawmill", p.bigsawmill_max);
        add("Torchlight", p.torchlight_max);
        add("Mushroom", p.mushroom_max);
        for (const char* id : {"Apiary", "Sawmill", "Ironmine", "HuntingLand",
                               "WaterChannel", "Hothouse", "Fish", "CowFarm",
                               "Farm", "Garden", "AirStation", "Coalmine", "CoalCut"})
            add(id, p.secondary_max);
        if (!p.plan_file.empty()) plan_ = load_plan(p.plan_file);
        // ВАЖНО: bd живёт снаружи, Game хранит свою копию — для цен/сезонов
        // достаточно указателей на внешний вектор.
    }

    struct Result {
        int64_t net_worth = 0;
        int64_t assets = 0;
        int64_t peak_credit = 0;
        int64_t revenue = 0;
        int64_t money = 0, credit = 0, stock = 0;
        int64_t people = 0, homes = 0;
        int bases = 0, roads = 0;
        int days = 0;
        bool blocked = false;
        std::string blocker;
        std::vector<Event> schedule;
        std::map<std::string, int> counts;
    };

    Result run() {
        Result r;
        for (day_ = 0; day_ < p_.horizon_days; day_++) {
            settle_tax();
            if (blocked_) break;
            market_and_bank();
            build_phase();
            if (p_.trace && g_.day == 1) trace_line();
            if (g_.credit > peak_credit_) peak_credit_ = g_.credit;
            Game::AdvanceOut out = g_.advance_day();
            if (!out.ok) {
                blocked_ = true;
                blocker_ = out.blocker.empty() ? std::string("tax") : out.blocker;
                break;
            }
            if (g_.game_over()) {
                blocked_ = true;
                blocker_ = g_.game_over()->reason;
                break;
            }
        }
        // финал: распродать склад, погасить долг
        sell_all();
        repay_max();
        r.money = g_.money;
        r.credit = g_.credit;
        r.stock = stock_value();
        r.net_worth = g_.money + r.stock - g_.credit;
        for (const Base& b : g_.bases) {  // остаточная стоимость (износ линейный)
            if (b.data->id == ROAD_ID || b.data->id == DEPOT_ID) continue;
            int64_t total = b.data->live_time_total();
            r.assets += total > 0 ? b.data->price * b.live_time / total : b.data->price;
        }
        r.people = g_.people;
        r.homes = g_.now_home_places();
        r.days = (int)g_.days_alive;
        r.peak_credit = peak_credit_;
        r.revenue = revenue_;
        r.blocked = blocked_;
        r.blocker = blocker_;
        r.schedule = schedule_;
        for (const Base& b : g_.bases) {
            r.counts[b.data->id]++;
            if (b.data->id == ROAD_ID) r.roads++;
            else r.bases++;
        }
        return r;
    }

private:
    void trace_line() const {
        // сколько зданий реально получат рабочих и сырьё завтра
        int64_t free = g_.people;
        int working = 0, idle_workers = 0;
        for (auto it = g_.bases.rbegin(); it != g_.bases.rend(); ++it) {
            const Base& b = *it;
            if (!b.can_work(g_.season)) continue;
            if (b.data->need_workers == 0) continue;
            if (free >= b.data->need_workers) {
                free -= b.data->need_workers;
                working++;
            } else {
                idle_workers++;
            }
        }
        printf("%02d.%02d.%d %-6s money=%9lld credit=%8lld ppl=%4lld home=%4lld "
               "free=%4lld work=%2d idleW=%2d gold=%3lld stone=%5lld water=%5lld\n",
               g_.day, g_.month, g_.year, SEASON_NAMES[g_.season],
               (long long)g_.money, (long long)g_.credit, (long long)g_.people,
               (long long)g_.now_home_places(), (long long)free, working,
               idle_workers, (long long)g_.sunduk[GOLD], (long long)g_.sunduk[STONE],
               (long long)g_.sunduk[WATER]);
    }

    // ── деньги/рынок ────────────────────────────────────────────────────
    int64_t stock_value() const {
        int64_t v = 0;
        for (int i = 0; i < SUNDUK_SIZE; i++) v += g_.sunduk[i] * SALEP[i];
        return v;
    }

    // Что понадобится зданиям завтра: обходим базы в том же порядке, что и
    // движок (с конца вектора), раздаём людей и складываем consume.
    Sunduk needs_tomorrow() const {
        Sunduk need;
        int64_t free = g_.people;
        for (auto it = g_.bases.rbegin(); it != g_.bases.rend(); ++it) {
            const Base& b = *it;
            if (!b.can_work(g_.season)) continue;
            if (free < b.data->need_workers) continue;
            free -= b.data->need_workers;
            need.add(b.data->consume);
        }
        return need;
    }

    void sell_all() {
        Sunduk s;
        for (int i = 0; i < SUNDUK_SIZE; i++) s[i] = g_.sunduk[i];
        if (!s.empty()) g_.market_sell(s);
    }

    void repay_max() {
        if (g_.credit > 0 && g_.money > 0)
            g_.bank_give(std::min<int64_t>(g_.credit, g_.money));
    }

    void market_and_bank() {
        Sunduk need = needs_tomorrow();
        // 1) продать всё, что не нужно на завтра
        Sunduk sell;
        for (int i = 0; i < SUNDUK_SIZE; i++) {
            int64_t extra = g_.sunduk[i] - need[i];
            if (extra > 0) sell[i] = extra;
        }
        if (!sell.empty()) {
            Game::MarketOut mo = g_.market_sell(sell);
            if (mo.ok) revenue_ += mo.total;
        }
        // 2) докупить недостающее сырьё (иначе здание встанет: need_sunduk)
        Sunduk buy;
        int64_t cost = 0;
        for (int i = 0; i < SUNDUK_SIZE; i++) {
            int64_t lack = need[i] - g_.sunduk[i];
            if (lack > 0) {
                buy[i] = lack;
                cost += lack * BUYP[i];
            }
        }
        if (cost > 0) {
            if (g_.money < cost) borrow(cost - g_.money);
            if (g_.money >= cost) g_.market_buy(buy);
        }
        // 3) гасим долг свободными деньгами (2‰ в день = 107% годовых)
        if (g_.credit > 0) {
            int64_t spare = g_.money - reserve();
            if (spare > 0) g_.bank_give(std::min(spare, g_.credit));
        }
    }

    // Занять столько, чтобы хватило на value, не пробивая рабочий потолок.
    bool borrow(int64_t value) {
        if (value <= 0) return true;
        int64_t room = p_.credit_cap - g_.credit;
        if (room <= 0) return false;
        int64_t take = std::min(value, room);
        return g_.bank_take(take).first;
    }

    void settle_tax() {
        if (!g_.annual_tax_due() && !g_.main_tax_due()) return;
        int64_t amount = g_.annual_tax_due() ? g_.annual_tax_amount()
                                             : g_.main_tax_amount();
        if (p_.trace && g_.annual_tax_due()) {  // разбор налога по статьям
            int64_t nb = g_.taxed_cells();
            printf("  НАЛОГ %02d.%02d.%d: итого %lld = клетки %lld×420=%lld"
                   " + 1%% закупок(%lld)=%lld + 2%% продаж(%lld)=%lld + сборы 4686\n",
                   g_.day, g_.month, g_.year, (long long)amount, (long long)nb,
                   (long long)(nb * 420), (long long)g_.summ_buy,
                   (long long)(g_.summ_buy / 100), (long long)g_.summ_sale,
                   (long long)(g_.summ_sale * 2 / 100));
        }
        if (g_.money < amount) sell_all();
        if (g_.money < amount) borrow(amount - g_.money);
        if (g_.annual_tax_due() && !g_.pay_annual_tax()) {
            blocked_ = true;
            blocker_ = "annual_tax_unpaid";
        }
        if (g_.main_tax_due() && !g_.pay_main_tax()) {
            blocked_ = true;
            blocker_ = "main_tax_unpaid";
        }
    }

    // ── прогнозы ────────────────────────────────────────────────────────
    int working_puerperal() const {
        int n = 0;
        for (const Base& b : g_.bases)
            if (b.data->id == "Puerperal" && b.build_days == 0 && !b.preserved) n++;
        return n;
    }

    // Население через ahead дней: рождаемость P/birth_days + завоз 20 июня.
    double project_people(int ahead) const {
        double p = (double)g_.people;
        int bd = BIRTH_DAYS >> std::min(10, working_puerperal());
        if (bd < 1) bd = 1;
        int y = g_.year, m = g_.month, d = g_.day;
        for (int i = 0; i < ahead; i++) {
            p += p / (double)bd - p / (double)DEATH_DAYS;
            d++;
            if (d > days_in_month(y, m)) {
                d = 1;
                if (++m > 12) { m = 1; y++; }
            }
            if (m == POPULATION_ARRIVAL_MONTH && d == POPULATION_ARRIVAL_DAY && y > START_YEAR)
                p += ADDPEOPLE;
        }
        // жильё режет население: прогноз выше потолка бессмысленен
        return p;
    }

    // Сколько дней здание проработает от ready до горизонта.
    int working_days_left(const BaseData& d, int ready_in) const {
        int y = g_.year, mo = g_.month, da = g_.day;
        int total = 0;
        for (int i = 0; i < p_.horizon_days - day_; i++) {
            if (i >= ready_in) {
                Season s = season_for_month(mo);
                if (d.season_works(s)) total++;
            }
            da++;
            if (da > days_in_month(y, mo)) {
                da = 1;
                if (++mo > 12) { mo = 1; y++; }
            }
        }
        return total;
    }

    int64_t daily_net(const BaseData& d) const {
        int64_t v = 0;
        for (int i = 0; i < SUNDUK_SIZE; i++)
            v += d.profit[i] * SALEP[i] - d.consume[i] * BUYP[i];
        return v;
    }

    // Рабочие, уже «расписанные» по зданиям сезона s (стройка тоже считается).
    int64_t committed_workers(Season s) const {
        int64_t w = 0;
        for (const Base& b : g_.bases)
            if (!b.preserved && b.data->season_works(s)) w += b.data->need_workers;
        return w;
    }

    // Сезон через ahead дней.
    Season season_in(int ahead) const {
        int y = g_.year, mo = g_.month, da = g_.day;
        for (int i = 0; i < ahead; i++) {
            da++;
            if (da > days_in_month(y, mo)) {
                da = 1;
                if (++mo > 12) { mo = 1; y++; }
            }
        }
        return season_for_month(mo);
    }

    int count_of(const std::string& id) const {
        int n = 0;
        for (const Base& b : g_.bases)
            if (b.data->id == id) n++;
        return n;
    }

    // ── строительство ───────────────────────────────────────────────────
    bool try_build(const std::string& id, bool allow_credit) {
        auto it = data_.find(id);
        if (it == data_.end()) return false;
        const BaseData& d = *it->second;
        std::optional<Site> site = find_site(g_, d.need_earth, true);
        if (!site && d.need_earth == LT_EVERYWHERE)
            site = find_site(g_, d.need_earth, false);
        if (!site) return false;
        int64_t road_cost = (int64_t)site->roads.size() * road_price();
        int64_t total = d.price + road_cost;
        int64_t avail = g_.money - reserve();
        if (avail < total) {
            if (!allow_credit) return false;
            if (!borrow(total - avail)) return false;
            if (g_.money - reserve() < total) return false;
        }
        for (const auto& rc : site->roads) {
            auto ok = g_.build(ROAD_ID, rc.first, rc.second);
            if (!ok.first) return false;
        }
        auto ok = g_.build(id, site->x, site->y);
        if (!ok.first) return false;
        log_build(id, d.price + road_cost);
        return true;
    }

    int64_t road_price() const {
        auto it = data_.find(ROAD_ID);
        return it == data_.end() ? 400 : it->second->price;
    }

    void log_build(const std::string& id, int64_t price) {
        Event e;
        e.day = day_;
        e.y = g_.year;
        e.m = g_.month;
        e.d = g_.day;
        e.id = id;
        e.count = ++built_[id];
        e.price = price;
        e.money_after = g_.money;
        schedule_.push_back(e);
    }

    // Жильё: держим места под прогноз населения, но не выше pop_cap.
    // С налогом «за каждую занятую клетку» (docs/PLAN_TAX_PER_CELL.md) важна не
    // только цена места, но и число построек: хижина — 295 ₽/место и 21 ₽/место
    // налога в год, дом — 234 и 7, большой дом — 332 и 2,1. Поэтому берём самое
    // крупное жильё, которое закрывает дефицит, а хижину — только на остаток.
    void housing_phase() {
        const BaseData& house = *data_.at("House");
        const BaseData& small = *data_.at("SmallHouse");
        auto big = data_.find("BigHouse");
        for (int guard = 0; guard < 8; guard++) {
            int64_t homes = g_.now_home_places();
            for (const Base& b : g_.bases)  // стройка добавит места позже
                if (b.build_days > 0) homes += b.data->home_places;
            if (homes >= p_.pop_cap) break;
            double proj = project_people((int)house.build_time + 5);
            if (proj > (double)p_.pop_cap) proj = (double)p_.pop_cap;
            double deficit = proj + p_.house_margin - (double)homes;
            if (deficit <= 0) break;
            // «горит» — людей станет больше мест раньше, чем достроится дом
            bool urgent = (double)homes < project_people((int)house.build_time);
            const char* id = "House";
            if (big != data_.end() && deficit > (double)house.home_places * 1.5 &&
                g_.money - reserve() >= big->second->price)
                id = "BigHouse";
            else if (deficit <= (double)small.home_places || urgent)
                id = "SmallHouse";
            if (!try_build(id, true)) {
                const char* alt = (id == std::string("SmallHouse")) ? "House" : "SmallHouse";
                if (!try_build(alt, true)) break;
            }
        }
    }

    // Оценка кандидата: сколько он заработает до горизонта на рубль цены.
    // <0 — строить нельзя (нет рабочих / не окупится / нет денег).
    double score(const BaseData& d, int limit, bool* out_borrow) const {
        if (limit <= 0 || count_of(d.id) >= limit) return -1.0;
        const int ready = (int)d.build_time;
        const int64_t gain = daily_net(d) * working_days_left(d, ready);
        if ((double)gain < p_.payback * (double)d.price) return -1.0;
        int first = first_working_day(d, ready);
        if (first < 0) return -1.0;
        // рабочие: сравниваем с загрузкой того сезона, в котором здание оживёт
        Season s = season_in(first);
        double proj = project_people(first);
        if (proj > (double)p_.pop_cap) proj = (double)p_.pop_cap;
        if (proj < (double)(committed_workers(s) + d.need_workers + p_.build_workers_slack))
            return -1.0;
        // кредит — только под скорый запуск (2‰/день съедают простой)
        bool cash_ok = g_.money - reserve() >= d.price;
        bool may_borrow = (first - ready) <= p_.borrow_window &&
                          (!p_.gold_only_credit || d.id == "Goldmine" ||
                           d.price - (g_.money - reserve()) <= p_.small_credit);
        if (!cash_ok && !may_borrow) return -1.0;
        *out_borrow = may_borrow;
        return (double)gain / (double)d.price;
    }

    void build_phase() {
        housing_phase();
        if (!plan_.empty()) {  // режим сценария: строим ровно то, что записано
            for (PlanItem& it : plan_) {
                if (it.count <= 0) continue;
                if (it.y > g_.year) continue;
                if (it.y == g_.year && (it.m > g_.month ||
                                        (it.m == g_.month && it.d > g_.day)))
                    continue;
                if (it.id == "BuyLand") {  // выкуп земли: снимает налог с клетки
                    while (it.count > 0 && g_.money - reserve() >= BUYGOODEARTH) {
                        const Base* b = g_.find_taxed_base();
                        if (b == nullptr) { it.count = 0; break; }
                        if (!g_.good_earth(b->x, b->y).first) { it.count = 0; break; }
                        it.count--;
                    }
                    continue;
                }
                while (it.count > 0 && try_build(it.id, true)) it.count--;
            }
            return;
        }
        // роддома: каждый вдвое ускоряет рождаемость (birth_days >> n), 3 рабочих
        if (day_ >= p_.pu_start && day_ <= p_.pu_cutoff &&
            count_of("Puerperal") < p_.pu_target) {
            const BaseData& pu = *data_.at("Puerperal");
            bool cash_ok = g_.money - reserve() >= pu.price;
            if (cash_ok || p_.pu_borrow) try_build("Puerperal", p_.pu_borrow);
        }
        // доходные здания: каждый день берём лучшее по «доход/цена»
        for (int guard = 0; guard < 12; guard++) {
            const BaseData* best = nullptr;
            double best_score = 0.0;
            bool best_borrow = false;
            for (const auto& c : cands_) {
                bool borrow_flag = false;
                double s = score(*c.first, c.second, &borrow_flag);
                if (s > best_score) {
                    best_score = s;
                    best = c.first;
                    best_borrow = borrow_flag;
                }
            }
            if (!best) break;
            if (!try_build(best->id, best_borrow)) break;
        }
    }

    // Через сколько дней здание, заложенное сегодня, впервые заработает.
    int first_working_day(const BaseData& d, int ready) const {
        int y = g_.year, mo = g_.month, da = g_.day;
        for (int i = 0; i < p_.horizon_days - day_; i++) {
            if (i >= ready && d.season_works(season_for_month(mo))) return i;
            da++;
            if (da > days_in_month(y, mo)) {
                da = 1;
                if (++mo > 12) { mo = 1; y++; }
            }
        }
        return -1;
    }

    // Подушка: налог + оборотка на воду приискам (200 ед/день каждому).
    int64_t reserve() const {
        return p_.cash_reserve + 1400 * count_of("Goldmine");
    }

    Params p_;
    Game g_;
    std::map<std::string, const BaseData*> data_;
    std::vector<std::pair<const BaseData*, int>> cands_;
    std::vector<PlanItem> plan_;
    std::map<std::string, int> built_;
    std::vector<Event> schedule_;
    int day_ = 0;
    int64_t peak_credit_ = 0;
    int64_t revenue_ = 0;
    bool blocked_ = false;
    std::string blocker_;
};

void print_schedule(const Planner::Result& r) {
    printf("\n%-6s %-14s %-22s %10s %12s\n", "день", "дата", "постройка", "цена",
           "деньги после");
    for (const Event& e : r.schedule) {
        char date[32];
        snprintf(date, sizeof(date), "%02d %s %d", e.d, MONTHS[e.m - 1], e.y);
        printf("%-6d %-14s %-18s #%-3d %10lld %12lld\n", e.day, date, e.id.c_str(),
               e.count, (long long)e.price, (long long)e.money_after);
    }
}

}  // namespace

int main(int argc, char** argv) {
    Params p;
    bool sweep = false;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto val = [&]() -> std::string { return i + 1 < argc ? argv[++i] : ""; };
        if (a == "--seed") p.seed = std::stoll(val());
        else if (a == "--map-size") p.map_size = std::stoi(val());
        else if (a == "--days") p.horizon_days = std::stoi(val());
        else if (a == "--pu") p.pu_target = std::stoi(val());
        else if (a == "--pu-cutoff") p.pu_cutoff = std::stoi(val());
        else if (a == "--pu-borrow") p.pu_borrow = true;
        else if (a == "--pu-start") p.pu_start = std::stoi(val());
        else if (a == "--secondary") p.secondary_max = std::stoi(val());
        else if (a == "--free-credit") p.gold_only_credit = false;
        else if (a == "--plan") p.plan_file = val();
        else if (a == "--bases") p.bases_path = val();
        else if (a == "--pop-cap") p.pop_cap = std::stoi(val());
        else if (a == "--borrow-window") p.borrow_window = std::stoi(val());
        else if (a == "--gold") p.gold_max = std::stoi(val());
        else if (a == "--refinery") p.refinery_max = std::stoi(val());
        else if (a == "--bigsawmill") p.bigsawmill_max = std::stoi(val());
        else if (a == "--torchlight") p.torchlight_max = std::stoi(val());
        else if (a == "--mushroom") p.mushroom_max = std::stoi(val());
        else if (a == "--payback") p.payback = std::stod(val());
        else if (a == "--credit") p.credit_cap = std::stoll(val());
        else if (a == "--reserve") p.cash_reserve = std::stoll(val());
        else if (a == "--house-margin") p.house_margin = std::stoi(val());
        else if (a == "--slack") p.build_workers_slack = std::stoi(val());
        else if (a == "--schedule") p.schedule = true;
        else if (a == "--sweep") sweep = true;
        else if (a == "--trace") p.trace = true;
        else {
            fprintf(stderr, "неизвестный флаг: %s\n", a.c_str());
            return 2;
        }
    }

    auto bd = load_base_data(p.bases_path);
    auto ed = load_events("configs/events.json");

    if (sweep) {
        // Перебор стратегий. Метрика — «кэш + остаточная стоимость построек»
        // на 01.03.1892 (кэш = деньги + склад − долг).
        printf("%3s %3s %3s %3s %8s %5s %5s | %4s %4s %4s %5s | %9s %9s %9s %8s %7s %s\n",
               "pu", "puB", "ref", "sec", "credit", "pcap", "payb", "gold", "ref",
               "pu", "людей", "КЭШ", "активы", "ИТОГО", "выручка", "пикдолг", "блок");
        for (int pu : {0, 1, 2, 3, 4, 5, 6}) {
            for (int pub : {0, 1}) {
                for (int ref : {0, 1, 2}) {
                    for (int sec : {0, 3}) {
                        for (int64_t cr : {(int64_t)150000, (int64_t)300000,
                                           (int64_t)420000}) {
                            for (int pcap : {90, 140, 220, 400}) {
                                for (double pb : {0.45, 0.8, 1.05}) {
                                    Params q = p;
                                    q.pu_target = pu;
                                    q.pu_borrow = pub != 0;
                                    q.refinery_max = ref;
                                    q.secondary_max = sec;
                                    q.credit_cap = cr;
                                    q.pop_cap = pcap;
                                    q.payback = pb;
                                    Planner pl(bd, ed, q);
                                    Planner::Result r = pl.run();
                                    printf("%3d %3d %3d %3d %8lld %5d %5.2f | %4d %4d "
                                           "%4d %5lld | %9lld %9lld %9lld %8lld %7lld %s\n",
                                           pu, pub, ref, sec, (long long)cr, pcap, pb,
                                           r.counts["Goldmine"], r.counts["Refinery"],
                                           r.counts["Puerperal"], (long long)r.people,
                                           (long long)r.net_worth, (long long)r.assets,
                                           (long long)(r.net_worth + r.assets),
                                           (long long)r.revenue,
                                           (long long)r.peak_credit,
                                           r.blocked ? r.blocker.c_str() : "-");
                                }
                            }
                        }
                    }
                }
            }
        }
        return 0;
    }

    Planner pl(bd, ed, p);
    Planner::Result r = pl.run();
    printf("seed=%lld дней=%d население=%lld жильё=%lld построек=%d дорог=%d\n",
           (long long)p.seed, r.days, (long long)r.people, (long long)r.homes,
           r.bases, r.roads);
    printf("деньги=%lld склад=%lld долг=%lld КЭШ=%lld активы=%lld ИТОГО=%lld%s%s\n",
           (long long)r.money, (long long)r.stock, (long long)r.credit,
           (long long)r.net_worth, (long long)r.assets,
           (long long)(r.net_worth + r.assets), r.blocked ? "  BLOCKED: " : "",
           r.blocked ? r.blocker.c_str() : "");
    printf("состав:");
    for (const auto& kv : r.counts) printf(" %s×%d", kv.first.c_str(), kv.second);
    printf("\n");
    if (p.schedule) print_schedule(r);
    return 0;
}
