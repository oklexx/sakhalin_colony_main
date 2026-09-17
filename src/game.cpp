#include "colony/game.h"

#include <algorithm>
#include <stdexcept>

namespace colony {

namespace {

std::string event_consequences(int64_t ly, int64_t p, const Sunduk& s) {
    std::string out = "Последствия:";
    if (ly < 0) out += "\n- Строению нанесено " + std::to_string(-ly) + " повреждений.";
    if (p < 0) out += "\n- Людей стало на " + std::to_string(-p) + " меньше.";
    if (p > 0) out += "\n- Людей стало на " + std::to_string(p) + " больше.";
    for (int i = 0; i < SUNDUK_SIZE; i++) {
        if (s[i] < 0)
            out += std::string("\n- ") + SUNDUK_CAPTIONS[i] + " - потеряно " +
                   std::to_string(-s[i]) + ".";
        if (s[i] > 0)
            out += std::string("\n- ") + SUNDUK_CAPTIONS[i] + " - получено " +
                   std::to_string(s[i]) + ".";
    }
    return out;
}

}  // namespace

// ---------------------------------------------------------------- Base
int64_t Base::begin_day(Season season, Sunduk& sunduk, int64_t free_workers) {
    need_sunduk = false;
    need_workers = false;
    if (!can_work(season)) return 0;
    if (sunduk.include(data->consume)) {
        if (free_workers >= data->need_workers) {
            sunduk.remove(data->consume);
            return data->need_workers;
        }
        need_workers = true;
    } else {
        need_sunduk = true;
    }
    return 0;
}

void Base::end_day(Game& game) {
    bool good = game.is_good(x, y);
    if (build_days) {
        build_days -= 1;
        if (build_days && good) build_days -= 1;
        return;
    }
    if (data->season_works(game.season) && !preserved && !need_sunduk && !need_workers) {
        Sunduk profit = data->profit.copy();
        profit.random_range(data->profit_range, game.rng);
        game.sunduk.add(profit);
    }
    if (!data->live_years) return;
    int blow = 1;
    if (game.season == SEASON_AUTUMN || game.season == SEASON_SPRING) {
        if (game.rng.randrange(7) == 0) blow += 1;
    } else if (game.season == SEASON_WINTER) {
        if (game.rng.randrange(3) == 0) blow += 1;
    }
    if (game.rng.randrange(14) != 0) blow += 1;
    if (good && game.rng.randrange(3) == 0) blow -= 1;
    live_time -= blow;
}

// ---------------------------------------------------------------- Game
Game::Game(const std::vector<BaseData>& base_data,
           const std::vector<BaseEvent>& events_data, int64_t seed,
           int map_size, const std::string& difficulty,
           bool no_city_game_over, int64_t no_people_days)
    : Game(std::make_shared<const std::vector<BaseData>>(base_data),
           std::make_shared<const std::vector<BaseEvent>>(events_data),
           seed, map_size, difficulty, no_city_game_over, no_people_days) {}

Game::Game(const std::shared_ptr<const std::vector<BaseData>>& base_data,
           const std::shared_ptr<const std::vector<BaseEvent>>& events_data,
           int64_t seed, int map_size, const std::string& difficulty,
           bool no_city_game_over, int64_t no_people_days)
    : earth(seed, map_size),
      rng(seed),
      rng_np(seed),
       base_data_(base_data),
       events_data_(events_data),
      map_size_(map_size),
      difficulty_(difficulty),
      no_city_game_over_(no_city_game_over),
      no_people_days_limit_(no_people_days) {
    good_lots.assign((size_t)map_size * map_size, 0);
    destroyed_lots.assign((size_t)map_size * map_size, 0);
    occupied.assign((size_t)map_size * map_size, 0);
    base_index_map_.assign((size_t)map_size * map_size, -1);
    bases.reserve(1024);
    if (light()) money = INIT_MONEY * LIGHT_MONEY_MULT;
    for (size_t i = 0; i < base_data_->size(); i++)
        data_id_to_idx_[(*base_data_)[i].id] = i;
    depot_ = find_data(DEPOT_ID);
    if (depot_ == nullptr)
        throw std::runtime_error(std::string("В bases.json отсутствует главная постройка ") + DEPOT_ID);
    Base depot(depot_, earth.init_sel_x, earth.init_sel_y, true);
    depot.uid = take_uid();
    bases.push_back(depot);
    base_index_map_[(size_t)depot.y * map_size_ + depot.x] = (int32_t)bases.size() - 1;
    refresh_occupied();
}

Game::Game(const Game& other)
    : earth(other.earth),
      year(other.year), month(other.month), day(other.day),
      season(other.season), money(other.money), credit(other.credit),
      summ_buy(other.summ_buy), summ_sale(other.summ_sale),
      people(other.people), busy_people(other.busy_people),
      days_alive(other.days_alive), days_no_people(other.days_no_people),
      sunduk(other.sunduk), bases(other.bases), rng(other.rng),
      rng_np(other.rng_np), good_lots(other.good_lots),
      destroyed_lots(other.destroyed_lots), occupied(other.occupied),
      base_data_(other.base_data_), events_data_(other.events_data_),
      map_size_(other.map_size_),
      difficulty_(other.difficulty_),
      no_city_game_over_(other.no_city_game_over_),
      no_people_days_limit_(other.no_people_days_limit_),
      tax_annual_paid_(other.tax_annual_paid_),
      tax_main_paid_(other.tax_main_paid_),
      tax_postponed_(other.tax_postponed_),
      tax_to_debt_(other.tax_to_debt_),
      next_uid_(other.next_uid_),
      enable_undo_(other.enable_undo_),
      gate_(other.gate_),
      gate_ctx_(other.gate_ctx_) {
    data_id_to_idx_ = other.data_id_to_idx_;
    depot_ = find_data(DEPOT_ID);
    if (other.undo_) undo_ = std::make_unique<Game>(*other.undo_);
    base_index_map_ = other.base_index_map_;
}

Game& Game::operator=(const Game& other) {
    if (this == &other) return *this;
    earth = other.earth;
    year = other.year; month = other.month; day = other.day;
    season = other.season;
    money = other.money; credit = other.credit;
    summ_buy = other.summ_buy; summ_sale = other.summ_sale;
    people = other.people; busy_people = other.busy_people;
    days_alive = other.days_alive; days_no_people = other.days_no_people;
    sunduk = other.sunduk;
    bases = other.bases;
    rng = other.rng;
    rng_np = other.rng_np;
    good_lots = other.good_lots;
    destroyed_lots = other.destroyed_lots;
    occupied = other.occupied;
    base_data_ = other.base_data_;
    events_data_ = other.events_data_;
    map_size_ = other.map_size_;
    difficulty_ = other.difficulty_;
    no_city_game_over_ = other.no_city_game_over_;
    no_people_days_limit_ = other.no_people_days_limit_;
    tax_annual_paid_ = other.tax_annual_paid_;
    tax_main_paid_ = other.tax_main_paid_;
    tax_postponed_ = other.tax_postponed_;
    tax_to_debt_ = other.tax_to_debt_;
    next_uid_ = other.next_uid_;
    enable_undo_ = other.enable_undo_;
    gate_ = other.gate_;
    gate_ctx_ = other.gate_ctx_;
    data_id_to_idx_ = other.data_id_to_idx_;
    depot_ = find_data(DEPOT_ID);
    undo_ = other.undo_ ? std::make_unique<Game>(*other.undo_) : nullptr;
    base_index_map_ = other.base_index_map_;
    return *this;
}

// ---------------------------------------------------------------- базовое
const BaseData* Game::find_data(const std::string& data_id) const {
    auto it = data_id_to_idx_.find(data_id);
    if (it != data_id_to_idx_.end()) return &(*base_data_)[it->second];
    return nullptr;
}

bool Game::depot_exists() const {
    for (const Base& b : bases)
        if (b.data->id == DEPOT_ID) return true;
    return false;
}

int64_t Game::max_credit() const {
    if (light()) return MAXCREDIT_LIGHT;
    for (int y = year; y <= year + 11; y++) {
        if (y <= START_YEAR) continue;
        if ((y - START_YEAR) % 10 == 0) {
            int e = (y - START_YEAR - 10) / 10;
            int64_t f = 1;
            for (int i = 0; i < e; i++) f *= 2;
            return NALOG_MAIN * f;
        }
    }
    return NALOG_MAIN;
}

Base* Game::base_in_box(int x, int y) {
    if (x < 0 || x >= map_size_ || y < 0 || y >= map_size_) return nullptr;
    int32_t idx = base_index_map_[(size_t)y * map_size_ + x];
    if (idx >= 0 && idx < (int32_t)bases.size())
        return &bases[idx];
    return nullptr;
}

const Base* Game::base_in_box(int x, int y) const {
    if (x < 0 || x >= map_size_ || y < 0 || y >= map_size_) return nullptr;
    int32_t idx = base_index_map_[(size_t)y * map_size_ + x];
    if (idx >= 0 && idx < (int32_t)bases.size())
        return &bases[idx];
    return nullptr;
}

void Game::refresh_occupied() {
    std::fill(occupied.begin(), occupied.end(), 0);
    for (const Base& b : bases) occupied[(size_t)b.y * map_size_ + b.x] = 1;
}

bool Game::cell_connected(int x, int y) const {
    if (bases.empty() || !earth.in_bounds(x, y)) return false;
    if (base_in_box(x, y)) return false;
    const int ms = map_size_;
    const int dx4[4] = {1, -1, 0, 0};
    const int dy4[4] = {0, 0, 1, -1};
    for (int i = 0; i < 4; i++) {
        const Base* nb = base_in_box(x + dx4[i], y + dy4[i]);
        if (nb != nullptr && nb->data->id != ROAD_ID) return true;
    }
    std::vector<char> visited((size_t)ms * ms, 0);
    std::vector<std::pair<int, int>> q;
    for (int i = 0; i < 4; i++) {
        int nx = x + dx4[i], ny = y + dy4[i];
        if (!earth.in_bounds(nx, ny)) continue;
        size_t nidx = (size_t)ny * ms + nx;
        if (visited[nidx]) continue;
        if (base_index_map_[nidx] >= 0 && bases[base_index_map_[nidx]].data->id == ROAD_ID) {
            visited[nidx] = 1;
            q.push_back({nx, ny});
        }
    }
    while (!q.empty()) {
        auto [cx, cy] = q.front();
        q.erase(q.begin());
        for (int i = 0; i < 4; i++) {
            int nx = cx + dx4[i], ny = cy + dy4[i];
            if (!earth.in_bounds(nx, ny)) continue;
            size_t nidx = (size_t)ny * ms + nx;
            if (visited[nidx]) continue;
            int32_t bidx = base_index_map_[nidx];
            if (bidx >= 0) {
                visited[nidx] = 1;
                const Base& nb = bases[bidx];
                if (nb.data->id == ROAD_ID) q.push_back({nx, ny});
                else return true;
            }
        }
    }
    return false;
}

int64_t Game::now_home_places() const {
    if (cached_home_places_ < 0 || cached_season_ != (int)season || cached_bases_count_ != bases.size())
        recalc_caches();
    return cached_home_places_;
}

int64_t Game::now_need_workers() const {
    if (cached_need_workers_ < 0 || cached_season_ != (int)season || cached_bases_count_ != bases.size())
        recalc_caches();
    return cached_need_workers_;
}

// ---------------------------------------------------------------- налоги
bool Game::annual_tax_due() const {
    return year > START_YEAR && month == 3 && day == 1 && !tax_annual_paid_;
}

bool Game::main_tax_due() const {
    int64_t y = year - START_YEAR;
    return !light() && y != 0 && y % 10 == 0 && month == 11 && day == 1 &&
           !tax_main_paid_;
}

std::pair<bool, std::string> Game::check_advance() const {
    if (credit > max_credit())
        return {false, "Вы должны банку больше " + thousands(max_credit()) +
                           ". Дальнейшее невозможно, пока не вернете долг."};
    // P0: в долговой политике (RL) налог НЕ останавливает календарь — остаток
    // переоформляет в credit settle_tax_with_debt() внутри шага среды. Блокировка
    // остаётся только для явной отсрочки (tax_postponed, GUI-диалог) и для
    // «диалоговой» политики (GUI: диалог налогов → «Нет» → конец игры).
    if (!tax_to_debt_ || tax_postponed_) {
        if (annual_tax_due()) return {false, "annual_tax"};
        if (main_tax_due()) return {false, "main_tax"};
    }
    return {true, ""};
}

int64_t Game::annual_tax_amount() const {
    int64_t occupied_n = 0;
    for (const Base& b : bases)
        if (!b.data->no_occupy) occupied_n++;
    return occupied_n * NALOG_EARTH +
           summ_buy * NALOG_BUYPERCENT / 100 +
           summ_sale * NALOG_SALEPERCENT / 100 +
           NALOG_ECOLOGY + NALOG_SOCIAL + NALOG_RES;
}

int64_t Game::main_tax_amount() const {
    if (light()) return 0;
    int64_t y = year - START_YEAR;
    if (y <= 0 || y % 10 != 0) return 0;
    int64_t f = 1;
    for (int i = 0; i < (y - 10) / 10; i++) f *= 2;
    return NALOG_MAIN * f;
}

bool Game::pay_annual_tax() {
    int64_t total = annual_tax_amount();
    if (money < total) return false;
    money -= total;
    summ_buy = 0;
    summ_sale = 0;
    tax_annual_paid_ = true;
    return true;
}

bool Game::pay_main_tax() {
    int64_t total = main_tax_amount();
    if (money < total) return false;
    money -= total;
    tax_main_paid_ = true;
    return true;
}

Game::TaxSettleOut Game::settle_tax_with_debt() {
    TaxSettleOut out;
    const bool annual = annual_tax_due();
    const bool main = main_tax_due();
    if (!annual && !main) return out;
    out.kind = annual ? "annual" : "main";
    const int64_t amount = std::max<int64_t>(0, annual ? annual_tax_amount()
                                                      : main_tax_amount());
    // Сколько можем — деньгами, остаток — долгом банку. Проценты по нему
    // (CREDITPERCENT/1000 в день) и штраф debt_coeff в env — цена решения.
    out.paid = std::min<int64_t>(money, amount);
    out.borrowed = amount - out.paid;
    money -= out.paid;
    credit += out.borrowed;
    if (annual) {
        summ_buy = 0;
        summ_sale = 0;
        tax_annual_paid_ = true;
    } else {
        tax_main_paid_ = true;
    }
    out.settled = true;
    return out;
}

// ---------------------------------------------------------------- день
bool Game::increment_date() {
    day += 1;
    tax_annual_paid_ = false;
    tax_main_paid_ = false;
    if (day > days_in_month(year, month)) {
        day = 1;
        month += 1;
        if (month > 12) {
            month = 1;
            year += 1;
        }
    }
    Season new_season = season_for_month(month);
    bool changed = new_season != season;
    season = new_season;
    return changed;
}

void Game::delete_base(Base& b) {
    destroyed_lots[(size_t)b.y * map_size_ + b.x] = 12;
    size_t idx = (size_t)(&b - bases.data());
    size_t last = bases.size() - 1;
    if (idx != last) {
        Base& moving = bases[last];
        std::swap(bases[idx], bases[last]);
        base_index_map_[(size_t)moving.y * map_size_ + moving.x] = (int32_t)idx;
    }
    bases.pop_back();
    base_index_map_[(size_t)b.y * map_size_ + b.x] = -1;
    invalidate_caches();
}

DayResult Game::new_day() {
    DayResult res;
    // 1. Рождение / смерть.
    int64_t n_puerp = 0;
    for (const Base& b : bases) {
        if (b.data->id == "Puerperal" && b.can_work(season) &&
            people >= b.data->need_workers)
            n_puerp++;
    }
    // Guard against undefined behaviour: shifting by >= 64 bits is UB for a
    // 64-bit integer. With many puerperal hospitals the effective birth
    // interval can only get as small as 1 day, so clamp there.
    int64_t birth_days = (n_puerp >= 64) ? 1 : (BIRTH_DAYS >> n_puerp);
    if (birth_days < 1) birth_days = 1;
    int64_t people_at_start = people;
    if (people_at_start) {
        int64_t born = rng_np.binomial(people_at_start, 1.0 / (double)birth_days);
        int64_t died = rng_np.binomial(people_at_start, 1.0 / (double)DEATH_DAYS);
        people += born - died;
        res.born += born;
        res.died += died;
    }

    // 2. Распределение рабочих (в обратном порядке строительства).
    int64_t free = people;
    for (auto it = bases.rbegin(); it != bases.rend(); ++it) {
        free -= it->begin_day(season, sunduk, free);
    }
    busy_people = people - free;

    // 3. Конец дня: производство + износ + разрушение.
    std::vector<Base*> doomed;
    for (auto it = bases.rbegin(); it != bases.rend(); ++it) {
        it->end_day(*this);
        if (it->live_time < 0) doomed.push_back(&*it);
    }
    for (Base* b : doomed) {
        delete_base(*b);
        res.base_lost += 1;
    }

    // 4. Переполнение жилья.
    int64_t to_go = people - now_home_places();
    if (to_go > 0) {
        people -= to_go;
        res.died += to_go;
        res.home_overflow = true;
    }

    // 5. Счётчики руин.
    for (int8_t& v : destroyed_lots)
        if (v > 0) v -= 1;

    // 6. Нет Города — деньги и ресурсы пропадают.
    if (!depot_exists()) {
        money = 0;
        sunduk.clear();
    }

    // 7. Проценты по кредиту.
    credit += credit * CREDITPERCENT / 1000;

    // 8. Дата и сезон.
    res.season_changed = increment_date();
    if (res.season_changed) res.season_new = season;

    days_alive += 1;

    // 9. Случайные события.
    if (year > EVENT_SEASON_START_YEAR ||
        (year == EVENT_SEASON_START_YEAR && month >= EVENT_SEASON_START_MONTH)) {
        for (auto it = bases.rbegin(); it != bases.rend(); ++it) {
            Base& b = *it;
            if (!b.can_work(season)) continue;
            if (!b.state_empty()) continue;
            if (rng.randrange(EVENT_ATTEMPT_DAYS) != 0) continue;
            const BaseEvent* ev = choice_event(*events_data_, b.data->id, rng);
            if (ev == nullptr) continue;
            BaseEvent::ExecResult r = ev->execute(rng);
            if (!sunduk.can_consume(r.s)) continue;
            if (r.ly < 0 && b.live_time < -r.ly * 364) continue;
            if (r.p < 0 && people < -r.p) continue;
            people += r.p;
            sunduk.add(r.s);
            b.live_time += r.ly * 364;
            res.events.push_back({ev->message, event_consequences(r.ly, r.p, r.s)});
        }
    }

    // 10. Пополнение 20 июня (и напоминание 20 мая) — на НОВУЮ дату.
    if (year > START_YEAR && month == POPULATION_REMINDER_MONTH &&
        day == POPULATION_REMINDER_DAY)
        res.reminder_may = true;
    if (year > START_YEAR && month == POPULATION_ARRIVAL_MONTH &&
        day == POPULATION_ARRIVAL_DAY) {
        int64_t p = ADDPEOPLE + ADDPEOPLERANGE -
                    (int64_t)rng.randrange(ADDPEOPLERANGE * 2 + 1);
        people += p;
        res.people_arrived = p;
    }

    // 11. Отслеживание конца игры.
    if (people <= 0) {
        people = 0;
        days_no_people += 1;
    } else {
        days_no_people = 0;
    }

    res.stop_week = res.notable();
    return res;
}

Game::AdvanceOut Game::advance_day() {
    AdvanceOut out;
    auto ok = check_advance();
    out.ok = ok.first;
    out.blocker = ok.second;
    if (!ok.first) return out;
    out.res = new_day();
    return out;
}

std::vector<DayResult> Game::advance_week() {
    std::vector<DayResult> results;
    for (int i = 0; i < 7; i++) {
        AdvanceOut r = advance_day();
        if (!r.ok) break;
        results.push_back(r.res);
        if (r.res.stop_week) break;
    }
    return results;
}

std::optional<Game::GameOverInfo> Game::game_over() const {
    if (light() && (year > 1950 || (year == 1950 && (month > 3 || (month == 3 && day >= 1))))) {
        return GameOverInfo{"light_end", days_alive,
                            "Наступило 1 марта 1950 года. Это максимальная дата для игры "
                            "с легким уровнем сложности.\nТеперь самое время начать обычную игру!"};
    }
    if (no_city_game_over_ && !depot_exists()) {
        return GameOverInfo{"no_city", days_alive,
                            "Город разрушен. Колония потеряла главное хранилище и прекратила существование."};
    }
    if (people == 0 && days_no_people >= no_people_days_limit_) {
        return GameOverInfo{"no_people", days_alive,
                            "Все жители колонии погибли."};
    }
    return std::nullopt;
}

// ---------------------------------------------------------------- действия
std::pair<bool, std::string> Game::build(const std::string& data_id, int x, int y) {
    // PR 2: гейт — первой строкой, до любых других проверок.
    if (gate_ && !gate_(gate_ctx_, data_id))
        return {false, "Постройка закрыта курикулумом."};
    if (!earth.in_bounds(x, y)) return {false, "Вне карты."};
    if (base_in_box(x, y)) return {false, "Это место занято."};
    const BaseData* d = find_data(data_id);
    if (d == nullptr) throw std::runtime_error("unknown base id: " + data_id);
    if (money < d->price) return {false, "Недостаточно денег."};
    if (!cell_connected(x, y))
        return {false, "Здание должно примыкать к другой постройке или быть связано с ней дорогой."};
    save_undo();
    money -= d->price;
    bases.emplace_back(d, x, y, false);
    bases.back().uid = take_uid();
    base_index_map_[(size_t)y * map_size_ + x] = (int32_t)bases.size() - 1;
    invalidate_caches();
    return {true, ""};
}

Game::RestoreOut Game::restore(int x, int y) {
    Base* b = base_in_box(x, y);
    if (b == nullptr) return {false, "Там нет постройки", 0, 0};
    int64_t max_live = b->data->live_time_total();
    int64_t price_per_day = b->data->restore_price_per_day();
    if (b->live_time >= max_live || price_per_day == 0)
        return {false, "Ремонт не требуется.", 0, 0};
    int64_t need_days = max_live - b->live_time;
    int64_t price = need_days * price_per_day;
    if (money < price) {
        need_days = money / price_per_day;
        price = need_days * price_per_day;
    }
    if (need_days <= 0) return {false, "Недостаточно денег.", 0, 0};
    save_undo();
    money -= price;
    b->live_time += need_days;
    return {true, "", price, need_days};
}

Game::RestoreOut Game::restore_all() {
    struct Item { Base* b; int64_t price; };
    std::vector<Item> need;
    for (Base& b : bases) {
        int64_t p = b.data->restore_price(b.live_time);
        if (p > 0) need.push_back({&b, p});
    }
    if (need.empty()) return {false, "Нет поврежденных строений", 0, 0};
    int64_t total = 0;
    for (const Item& it : need) total += it.price;
    if (money >= total) {
        save_undo();
        for (const Item& it : need) it.b->live_time = it.b->data->live_time_total();
        money -= total;
        return {true, "", total, (int64_t)need.size()};
    }

    // Частичный ремонт — точный алгоритм RestoreBases из оригинала.
    int64_t restored = 0;
    bool out_of_money = false;
    while (true) {
        Base* slow = find_slowest(-1);
        if (slow == nullptr) break;
        std::vector<Base*> group;
        for (Base& b : bases) {
            if (b.data->live_years && b.live_time == slow->live_time &&
                b.live_time < b.data->live_time_total())
                group.push_back(&b);
        }
        if (group.empty()) break;
        int64_t lt = group[0]->live_time;
        int64_t day_count = group[0]->data->live_time_total();
        for (Base* g : group)
            if (g->data->live_time_total() < day_count) day_count = g->data->live_time_total();
        Base* nxt = find_slowest(lt);
        if (nxt != nullptr && nxt->live_time < day_count) day_count = nxt->live_time;
        day_count -= lt;
        int64_t per_day = 0;
        for (Base* g : group) per_day += g->data->restore_price_per_day();
        if (per_day <= 0) break;
        if (money < per_day * day_count) day_count = money / per_day;
        if (day_count > 0) {
            for (Base* g : group) g->live_time += day_count;
            money -= per_day * day_count;
            restored += (int64_t)group.size();
            continue;
        }
        out_of_money = true;
        Base* cheapest = nullptr;
        for (Base* g : group)
            if (cheapest == nullptr ||
                g->data->restore_price_per_day() < cheapest->data->restore_price_per_day())
                cheapest = g;
        while (cheapest != nullptr && money >= cheapest->data->restore_price_per_day()) {
            cheapest->live_time += 1;
            money -= cheapest->data->restore_price_per_day();
            restored += 1;
            group.erase(std::remove(group.begin(), group.end(), cheapest), group.end());
            if (group.empty()) break;
            cheapest = nullptr;
            for (Base* g : group)
                if (cheapest == nullptr ||
                    g->data->restore_price_per_day() < cheapest->data->restore_price_per_day())
                    cheapest = g;
        }
        break;
    }
    if (restored == 0 && !out_of_money)
        return {false, "Нет поврежденных строений", 0, 0};
    if (restored == 0) return {false, "Недостаточно денег.", 0, 0};
    return {true, "Денег не хватило на восстановление всех построек. Часть отремонтирована.", 0, restored};
}

Base* Game::find_slowest(int64_t more_than) {
    Base* r = nullptr;
    for (Base& b : bases) {
        if (b.live_time > more_than && b.live_time < b.data->live_time_total()) {
            if (r == nullptr || b.live_time < r->live_time) r = &b;
        }
    }
    return r;
}

const Base* Game::find_slowest(int64_t more_than) const {
    const Base* r = nullptr;
    for (const Base& b : bases) {
        if (b.live_time > more_than && b.live_time < b.data->live_time_total()) {
            if (r == nullptr || b.live_time < r->live_time) r = &b;
        }
    }
    return r;
}

std::pair<bool, std::string> Game::destroy(int x, int y) {
    Base* b = base_in_box(x, y);
    if (b == nullptr) return {false, "Там нет постройки"};
    if (b->data->id == DEPOT_ID) return {false, "Нельзя разбирать Город"};
    save_undo();
    money += b->data->price / 10;
    delete_base(*b);
    return {true, ""};
}

std::pair<bool, std::string> Game::preserve(int x, int y) {
    Base* b = base_in_box(x, y);
    if (b == nullptr) return {false, "Там нет постройки"};
    if (b->data->no_preserve) return {false, "Нельзя блокировать это строение"};
    save_undo();
    b->preserved = !b->preserved;
    invalidate_caches();
    return {true, ""};
}

std::pair<bool, std::string> Game::good_earth(int x, int y) {
    if (!earth.in_bounds(x, y)) return {false, "Вне карты."};
    if (base_in_box(x, y)) return {false, "Это место занято."};
    if (is_good(x, y)) return {false, "Нельзя покупать этот участок"};
    int8_t cur = earth.lot(x, y);
    if (!(cur >= LT_NORMAL && cur < LT_LAST))
        return {false, "Нельзя улучшать этот участок"};
    if (money < BUYGOODEARTH) return {false, "Недостаточно денег."};
    save_undo();
    money -= BUYGOODEARTH;
    good_lots[(size_t)y * map_size_ + x] = 1;
    return {true, ""};
}

// ---------------------------------------------------------------- рынок/банк
Game::MarketOut Game::market_buy(const Sunduk& counts) {
    int64_t total = 0;
    for (int i = 0; i < SUNDUK_SIZE; i++) total += counts[i] * BUY_SUNDUK[i];
    if (total <= 0) return {false, "", 0};
    if (money < total) return {false, "Недостаточно денег.", 0};
    save_undo();
    money -= total;
    sunduk.add(counts);
    summ_buy += total;
    return {true, "", total};
}

Game::MarketOut Game::market_sell(const Sunduk& counts) {
    if (!sunduk.include(counts)) return {false, "Нет указанных ресурсов для продажи", 0};
    int64_t total = 0;
    for (int i = 0; i < SUNDUK_SIZE; i++) total += counts[i] * SALE_SUNDUK[i];
    if (total <= 0) return {false, "", 0};
    save_undo();
    money += total;
    sunduk.remove(counts);
    summ_sale += total;
    return {true, "", total};
}

std::pair<bool, std::string> Game::bank_take(int64_t value) {
    if (value <= 0) return {false, "Введите положительное число."};
    int64_t limit = max_credit();
    if (value + credit > limit)
        return {false, "В настоящее время долг перед Банком не должен превышать " +
                           thousands(limit) + "."};
    save_undo();
    credit += value;
    money += value;
    return {true, ""};
}

std::pair<bool, std::string> Game::bank_give(int64_t value) {
    if (value <= 0 || value > credit) return {false, "Нельзя вернуть больше долга."};
    if (money < value) return {false, "Недостаточно денег."};
    save_undo();
    credit -= value;
    money -= value;
    return {true, ""};
}

// ---------------------------------------------------------------- undo
void Game::save_undo() {
    if (!enable_undo_) return;
    if (!undo_) undo_ = std::make_unique<Game>(*this);
}

bool Game::undo() {
    if (!undo_) return false;
    Game saved = *undo_;
    undo_.reset();
    *this = saved;
    return true;
}

Game Game::snapshot() const { return *this; }

Base* Game::find_slowest_base() { return find_slowest(-1); }
const Base* Game::find_slowest_base() const { return find_slowest(-1); }

void Game::reset_milestones() {
    last_base_milestone_ = 0;
    last_people_milestone_ = 0;
    last_day_milestone_ = 0;
    year_bonus_given_ = false;
}

double Game::check_milestones(double milestone_base, double milestone_people,
                              double milestone_day, double milestone_year) {
    double bonus = 0.0;
    // Roads are infrastructure, not colony buildings. Counting them here let a
    // policy farm milestone_base (+30 every 5 bases) by spamming 400-money roads
    // that employ nobody and produce nothing: measured +1241 milestone reward for
    // 205 roads vs +41 for a working 5-building farm economy. Exclude them, the
    // same way build_bonus already does in env.cpp.
    int nbases = 0;
    for (const Base& b : bases)
        if (b.data->id != ROAD_ID) ++nbases;
    int64_t peop = people;
    int64_t d = days_alive;

    // Базовые milestone: каждые 5 баз
    if (nbases >= last_base_milestone_ + 5) {
        bonus += milestone_base;
        last_base_milestone_ = (nbases / 5) * 5;
    }

    // Людские milestone: каждые 50 человек
    if (peop >= last_people_milestone_ + 50) {
        bonus += milestone_people;
        last_people_milestone_ = (int)((peop / 50) * 50);
    }

    // Дневной milestone: каждые 100 дней
    if (d > 0 && d % 100 == 0 && d > last_day_milestone_) {
        bonus += milestone_day;
        last_day_milestone_ = d;
    }

    // Годовой milestone: первый год (365 дней)
    if (d >= 365 && !year_bonus_given_) {
        bonus += milestone_year;
        year_bonus_given_ = true;
    }

    return bonus;
}

}  // namespace colony