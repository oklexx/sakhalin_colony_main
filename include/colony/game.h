#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "colony/bases.h"
#include "colony/constants.h"
#include "colony/earth.h"
#include "colony/events.h"
#include "colony/resources.h"
#include "colony/rng.h"

namespace colony {

// Результат прохода одного дня (аналог DayResult).
struct DayResult {
    bool season_changed = false;
    Season season_new = SEASON_SUMMER;
    int64_t base_lost = 0, born = 0, died = 0, people_arrived = 0;
    bool home_overflow = false, reminder_may = false;
    int64_t stop_week = 0;
    std::vector<std::pair<std::string, std::string>> events;  // (текст, последствия)

    int64_t notable() const {
        if (season_changed) return 1;  // bool -> 1/0
        if (base_lost) return base_lost;
        if (born) return born;
        if (died) return died;
        if (home_overflow) return 1;  // bool -> 1/0
        if (!events.empty()) return 1;  // bool(events) -> 1/0
        if (people_arrived) return 1;   // bool(people_arrived) -> 1/0
        return 0;
    }
};

// Главный класс игры. Точная копия логики Game из core/game.py.
class Game {
public:
    Game(const std::vector<BaseData>& base_data,
         const std::vector<BaseEvent>& events_data, int64_t seed,
         int map_size = 280, const std::string& difficulty = DIFFICULTY_NORMAL,
         bool no_city_game_over = GAME_OVER_NO_CITY,
         int64_t no_people_days = GAME_OVER_NO_PEOPLE_DAYS);

    Game(const std::shared_ptr<const std::vector<BaseData>>& base_data,
         const std::shared_ptr<const std::vector<BaseEvent>>& events_data,
         int64_t seed, int map_size,
         const std::string& difficulty = DIFFICULTY_NORMAL,
         bool no_city_game_over = GAME_OVER_NO_CITY,
         int64_t no_people_days = GAME_OVER_NO_PEOPLE_DAYS);

    // статические данные (владение копией)
    const std::vector<BaseData>& base_data() const { return *base_data_; }
    const std::vector<BaseEvent>& events_data() const { return *events_data_; }

    int map_size() const { return map_size_; }
    bool light() const { return difficulty_ == DIFFICULTY_LIGHT; }

    // ---- состояние ----
    Earth earth;
    int year = START_YEAR, month = START_MONTH, day = START_DAY;
    Season season = SEASON_SPRING;
    int64_t money = INIT_MONEY, credit = 0, summ_buy = 0, summ_sale = 0;
    int64_t people = INIT_PEOPLE, busy_people = 0;
    int64_t days_alive = 0, days_no_people = 0;
    Sunduk sunduk;
    std::vector<Base> bases;
    PCG64 rng;
    PCG64 rng_np;

    std::vector<uint8_t> good_lots;      // size*size
    std::vector<int8_t> destroyed_lots;  // size*size
    std::vector<uint8_t> occupied;       // size*size

    // ---- базовое ----
    const BaseData* depot_data() const { return depot_; }
    bool depot_exists() const;
    // Max credit limit: NALOG_MAIN (500k) × 2^e, where e = decades since
    // START_YEAR (1890) minus 1, rounded down to the last decade boundary.
    // Light difficulty returns MAXCREDIT_LIGHT (10M).
    int64_t max_credit() const;
    const Base* base_in_box(int x, int y) const;
    Base* base_in_box(int x, int y);
    void refresh_occupied();
    // Клетка допустима для строительства: примыкает (4-связность) к зданию
    // (не дороге) или соединена с колонией цепочкой дорог.
    bool cell_connected(int x, int y) const;
    bool is_good(int x, int y) const { return good_lots[(size_t)y * map_size_ + x] != 0; }
    int64_t now_home_places() const;
    int64_t now_need_workers() const;
    int64_t free_people() const { return people - busy_people > 0 ? people - busy_people : 0; }

    // ---- налоги ----
    bool tax_postponed() const { return tax_postponed_; }
    void set_tax_postponed(bool v) { tax_postponed_ = v; }
    bool tax_postponed_ = false;  // отложен через GUI — авто-оплата в step запрещена
    bool annual_tax_due() const;
    bool main_tax_due() const;
    std::pair<bool, std::string> check_advance() const;
    int64_t annual_tax_amount() const;
    int64_t main_tax_amount() const;
    bool pay_annual_tax();
    bool pay_main_tax();

    // ---- день ----
    bool increment_date();
    DayResult new_day();
    struct AdvanceOut {
        bool ok;
        DayResult res;
        std::string blocker;  // причина блокировки (если !ok)
    };
    AdvanceOut advance_day();
    // 7 дней; прерывается при блокировке или «важном» дне.
    std::vector<DayResult> advance_week();
    struct GameOverInfo {
        std::string reason;  // "light_end" | "no_city" | "no_people"
        int64_t days;
        std::string message;
    };
    std::optional<GameOverInfo> game_over() const;

    // ---- действия ----
    // Жёсткий гейт построек (PR 2): ставит ColonyEnvCpp (ctor + set_curriculum +
    // reset) на себя; Game::build отказывает закрытым id до любых других проверок.
    // По умолчанию nullptr — песочница (colony_cpp.Game напрямую) строит что хочет.
    using BuildGate = bool (*)(const void* ctx, const std::string& id);
    void set_build_gate(BuildGate fn, const void* ctx) { gate_ = fn; gate_ctx_ = ctx; }
    std::pair<bool, std::string> build(const std::string& data_id, int x, int y);
    struct RestoreOut { bool ok; std::string msg; int64_t price, days; };
    RestoreOut restore(int x, int y);
    RestoreOut restore_all();
    std::pair<bool, std::string> destroy(int x, int y);
    std::pair<bool, std::string> preserve(int x, int y);
    std::pair<bool, std::string> good_earth(int x, int y);
    struct MarketOut { bool ok; std::string msg; int64_t total; };
    MarketOut market_buy(const Sunduk& counts);
    MarketOut market_sell(const Sunduk& counts);
    std::pair<bool, std::string> bank_take(int64_t value);
    std::pair<bool, std::string> bank_give(int64_t value);

    // ---- undo / snapshot ----
    void save_undo();
    bool undo();
    Game snapshot() const;
    void set_enable_undo(bool v) { enable_undo_ = v; }
    bool enable_undo() const { return enable_undo_; }

    // глубокая копия (undo/snapshot; статические данные разделяются)
    Game(const Game& other);
    Game& operator=(const Game& other);
    Game(Game&&) = default;
    Game& operator=(Game&&) = default;

    // ---- helpers ----
    const Base* find_slowest_base() const;
    Base* find_slowest_base();
    int64_t take_uid() { return next_uid_++; }
    const std::vector<int32_t>& base_index_map() const { return base_index_map_; }

    // ---- milestones ----
    double check_milestones(double milestone_base, double milestone_people,
                            double milestone_day, double milestone_year);
    void reset_milestones();

private:
    void delete_base(Base& b);
    Base* find_slowest(int64_t more_than);
    const Base* find_slowest(int64_t more_than) const;
    const BaseData* find_data(const std::string& data_id) const;

    std::shared_ptr<const std::vector<BaseData>> base_data_;
    std::shared_ptr<const std::vector<BaseEvent>> events_data_;
    const BaseData* depot_;
    int map_size_;
    std::string difficulty_;
    bool no_city_game_over_;
    int64_t no_people_days_limit_;
    bool tax_annual_paid_ = false, tax_main_paid_ = false;
    int64_t next_uid_ = 1;
    std::unique_ptr<Game> undo_;
    bool enable_undo_ = true;
    // Гейт курикулума (PR 2): ctx — owning ColonyEnvCpp, переживает undo/snapshot,
    // т.к. копируется вместе с игрой, а env живёт дольше (см. copy ctor/operator=).
    BuildGate gate_ = nullptr;
    const void* gate_ctx_ = nullptr;
    std::vector<int32_t> base_index_map_;  // [y * map_size_ + x], -1 = нет базы
    std::unordered_map<std::string, size_t> data_id_to_idx_;

    // ---- кэш home_places / need_workers ----
    mutable int64_t cached_home_places_ = -1;
    mutable int64_t cached_need_workers_ = -1;
    mutable int cached_season_ = -1;
    mutable size_t cached_bases_count_ = 0;
    int64_t recalc_caches() const {
        cached_home_places_ = 0;
        cached_need_workers_ = 0;
        for (const Base& b : bases) {
            if (b.can_work(season)) {
                cached_home_places_ += b.data->home_places;
                cached_need_workers_ += b.data->need_workers;
            }
        }
        cached_season_ = (int)season;
        cached_bases_count_ = bases.size();
        return 0;
    }
    void invalidate_caches() {
        cached_home_places_ = -1;
        cached_need_workers_ = -1;
    }

    // ---- milestone tracking ----
    int last_base_milestone_ = 0;
    int last_people_milestone_ = 0;
    int64_t last_day_milestone_ = 0;
    bool year_bonus_given_ = false;
};

}  // namespace colony