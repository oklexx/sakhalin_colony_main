#pragma once

#include <array>
#include <cstdint>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_set>
#include <vector>

#include "colony/bases.h"
#include "colony/constants.h"
#include "colony/game.h"
#include "colony/rng.h"
#include "colony/reward_config.h"
#include "colony/running_mean_std.h"
#include "colony/thread_pool.h"

namespace colony {

// Единый контракт курикулума (PR 1): Python считает, C++ хранит и применяет.
// «Нет ограничения» — явное all_builds = true, а не побочный эффект пустого
// списка (раньше пустой manual_unlock_ids_ при этапе 0 молча открывал всё).
struct Curriculum {
    bool all_builds = true;                    // false => restricted
    std::unordered_set<std::string> allowed_builds;
    bool all_resources = true;                 // PR 4: false => resource_weights активны
    std::array<double, SUNDUK_SIZE> resource_weights{1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    int stage_report = 0;                      // только для obs-фичи и дампов
    int obs_version = 0;  // PR 5: 0 = legacy 246-dim obs, 1 = 287-dim (frame appended)
    // Разобрать JSON вида {"all_builds":bool,"allowed_builds":[...],"stage":int}
    // (транспорт watch_champion -> GUI/main). Бросает std::runtime_error.
    static Curriculum from_json(const std::string& text);
};

// RL-среда: точная копия ColonyEnv из rl/env.py (награды и наблюдения).
class ColonyEnvCpp {
public:
    ColonyEnvCpp(const std::vector<BaseData>& base_data,
                 const std::vector<BaseEvent>& events_data, int64_t seed,
                 int map_size = 280, const Curriculum& curriculum = Curriculum(),
                 const RewardConfig& cfg = RewardConfig(),
                 const std::string& difficulty = "normal",
                 bool no_city_game_over = false,
                 int64_t no_people_days = GAME_OVER_NO_PEOPLE_DAYS);

    // Единственная точка входа курикулума: идемпотентна, пересчитывает каталог.
    void set_curriculum(const Curriculum& c);
    Curriculum curriculum() const { return curriculum_; }
    bool build_allowed(const std::string& id) const {
        return curriculum_.all_builds || curriculum_.allowed_builds.count(id) > 0;
    }
    // PR 2: Game::build — последний рубеж (туда же ходит GUI напрямую).
    // Пересаживается в ctor/set_curriculum/reset: reset() пересоздаёт game_
    // из свежего Game с gate_==nullptr, без пересадки гейт бы молча исчезал.
    static bool curriculum_gate_fn(const void* ctx, const std::string& id) {
        return static_cast<const ColonyEnvCpp*>(ctx)->build_allowed(id);
    }
    void seat_build_gate() { game_.set_build_gate(&curriculum_gate_fn, this); }
    // PR 6: отчёт о вырожденном сценарии (0 доступных построек на старте
    // эпизода) — пусто если всё в порядке, иначе текст для WARNING в reset().
    std::string degenerate_report();
    void set_rewards(const RewardConfig& cfg) {
        std::lock_guard lock(*cfg_mutex_);
        cfg_ = cfg;
    }
    RewardConfig reward_config() const {
        std::lock_guard lock(*cfg_mutex_);
        return cfg_;
    }

    void reset(int64_t seed);
    std::vector<float> obs() const;
    std::vector<float> obs(const Game& g) const;

    // Миникарта: ГЛОБАЛЬНАЯ сетка 32×32 на всю карту (не окно вокруг старта).
    // minimap_radius_ сохранён как мёртовое поле совместимости — на размер
    // вывода он не влияет, см. ColonyEnvCpp::minimap() (const int G = 32).
    // каналы:
    //   0: суша (LT_NORMAL)
    //   1: вода (LT_WATER)
    //   2: лес (LT_WOOD)
    //   3: уголь (LT_COAL)
    //   4: железо (LT_IRON)
    //   5: нефть (LT_OIL)
    //   6: золото (LT_GOLD)
    //   7: здание (occupied)
    // Возврат: [8, 2R+1, 2R+1] float32 (0/1), значения за границей карты = 0.
    int minimap_radius() const { return minimap_radius_; }
    int minimap_channels() const { return 8; }
    void set_minimap_radius(int r) { minimap_radius_ = r; }
    std::vector<float> minimap() const;

    struct EpisodeMetrics {
        int64_t total_reward = 0;
        int64_t days_survived = 0;
        int64_t total_builds = 0;
        int64_t unique_build_types = 0;
        int64_t chains_activated = 0;
        int64_t max_chain_depth = 0;
        int64_t reached_resources = 0;
        int64_t priority_reached = 0;  // PR 4: из них — с весом курикулума > 0
        int64_t deaths = 0;
        int64_t births = 0;
        int64_t base_count_peak = 0;
        int64_t net_worth = 0;
        int64_t population_peak = 0;
    };

    struct StepOut {
        std::vector<float> obs;
        double rew = 0.0;
        bool terminated = false, truncated = false;
        int64_t days = 0, people = 0, money = 0, n_bases = 0;
        int64_t seed = 0, tax_due_days = 0;
        bool tax_grace_expired = false;
        double ep_return = 0.0;
        int64_t steps = 0;
        EpisodeMetrics metrics;
    };
    StepOut step(int action);

    bool tax_grace_expired() const;
    int tax_grace_days() const;

    int n_build() const { return n_build_; }
    int n_bases() const { return (int)game_.bases.size(); }
    // +N_ROAD_DIRS: the four compass road actions appended after the managers.
    int n_actions() const { return road_dir_base() + N_ROAD_DIRS; }
    int road_dir_base() const { return A_BUILD0 + n_build_ + N_MANAGERS; }
    // PR 5: obs v1 appends the frame AFTER the v0 tail, so obs v0 is a strict
    // prefix of obs v1 (248 = 27+32+7+9+128+9+32+2+2; 289 = 248+9+32).
    int obs_size() const {
        return 27 + n_build_ + 7 + 9 + 4 * n_build_ + 9 + n_build_ + 2 + 2 +
               (curriculum_.obs_version >= 1 ? SUNDUK_SIZE + n_build_ : 0);
    }
    // Action mask: 1.0 = available, 0.0 = blocked. Size = n_actions().
    std::vector<float> action_mask();
    const std::vector<std::string>& build_ids() const { return build_ids_; }
    const std::vector<const BaseData*>& build_data() const { return build_data_; }
    const Game& game() const { return game_; }
    Game& game() { return game_; }
    double last_daily_value() const { return last_daily_value_; }
    double last_chain_daily() const { return last_chain_daily_; }
    int64_t steps() const { return steps_; }
    double ep_return() const { return ep_return_; }
    double last_reward() const { return last_reward_; }
    EpisodeMetrics metrics() const { return episode_metrics_; }

    struct Stats { int64_t days, people, bases, money; };
    Stats stats() const {
        return {game_.days_alive, game_.people, (int64_t)game_.bases.size(),
                game_.money};
    }

public:
    // Debug helpers
    std::optional<std::pair<int, int>> find_lot(int need_earth, bool no_near_base);
    // Same legality rules as find_lot, but among ALL reachable legal cells
    // returns the one furthest along (dx, dy) measured from the colony
    // centroid -- i.e. "extend the frontier that way". Returns nullopt when
    // no legal cell exists.
    std::optional<std::pair<int, int>> find_lot_dir(int need_earth, bool no_near_base,
                                                    int dx, int dy);
    bool lot_ok(int x, int y, int need_earth, bool no_near_base) const;
    double debug_net_worth() const { return net_worth(game_); }
    std::string dump_obs() const;
    void set_step_log(const std::string& path);
    bool step_log_enabled() const { return step_log_.is_open(); }

private:
    double net_worth() const;
    double net_worth(const Game& g) const;
    double year_production_value(const BaseData& d) const;
    int road_count() const;
    std::vector<Season> step_seasons(int y, int m, int d, int n_days) const;
    void compute_catalog();
    // PR 4: |extracted_ ∩ {w>0}| — сколько приоритетных ресурсов уже открыто.
    int64_t count_priority_reached() const;

    struct PairHash {
        size_t operator()(const std::pair<int, int>& p) const {
            return (static_cast<size_t>(p.first) << 32) ^ static_cast<size_t>(p.second);
        }
    };

    struct ChainKeyHash {
        size_t operator()(const std::pair<std::string, int>& k) const {
            return std::hash<std::string>()(k.first) ^ (size_t)k.second;
        }
    };

    std::shared_ptr<const std::vector<BaseData>> base_data_;
    std::shared_ptr<const std::vector<BaseEvent>> events_data_;
    mutable std::shared_ptr<std::mutex> cfg_mutex_ = std::make_shared<std::mutex>();
    RewardConfig cfg_;
    int map_size_;
    Curriculum curriculum_;
    std::string difficulty_;
    bool no_city_game_over_;
    int64_t no_people_days_;

    std::vector<std::string> build_ids_;
    std::vector<const BaseData*> build_data_;
    int n_build_;
    std::unordered_map<std::string, int> build_id_to_idx_;
    int manager_base_;  // A_BUILD0 + n_build_
    int road_build_idx_ = -1;  // index of ROAD_ID in build_ids_, -1 if absent
    // Placement hint for the directional road actions: when set, the build path
    // picks the legal cell furthest along this direction instead of the
    // BFS-first one. Reset on every step().
    std::optional<std::pair<int, int>> road_dir_hint_;

    std::vector<float> sale_prices_;
    std::vector<std::vector<float>> catalog_by_season_;  // [4][n_build*4]

    // Миникарта: фиксированный радиус окна (не зависит от map_size)
    int minimap_radius_ = 14;

    Game game_;
    int64_t steps_ = 0;
    double ep_return_ = 0.0;
    double last_reward_ = 0.0;
    double last_daily_value_ = 0.0;
    double last_chain_daily_ = 0.0;
    int64_t tax_due_days_ = 0;
    int tax_grace_days_ = 0;
    int64_t days_since_last_build_ = 0;
    bool has_ever_built_ = false;

    std::unordered_set<int64_t> produced_;      // uid построек, получивших бонус (аналог _produced по id() в референсе)
    std::unordered_set<int64_t> building_before_;
    std::unordered_set<std::pair<std::string, int>, ChainKeyHash> chain_done_;
    std::unordered_set<std::string> first_working_;
    // v3: типы ресурсов, уже «добытые» в этом эпизоде (для first_extraction_bonus)
    std::unordered_set<int> extracted_;
    // v3: вес ресурса для бонусов добычи (min(1, 0.25*n_consumer_types))
    std::vector<double> extract_weight_;
    EpisodeMetrics episode_metrics_;
    std::unordered_set<std::string> unique_build_ids_;
    mutable std::pair<int, int> cell_cache_;
    mutable std::vector<float> obs_buf_;

    // Кэш net_worth
    mutable double cached_net_worth_ = 0.0;
    mutable bool net_worth_valid_ = false;
    void invalidate_net_worth() { net_worth_valid_ = false; }

    // Step log
    std::string step_log_path_;
    std::ofstream step_log_;
};

// Batched vectorized environment — N ColonyEnvCpp instances in one process.
// Thread-pooled parallel step/reset, integrated VecNormalize, auto-reset on done.
struct StepBatchResult {
    // Flat buffers: [n_envs * obs_size]
    std::vector<float> obs;
    // [n_envs]
    std::vector<double> rewards;
    std::vector<bool> terminateds;
    std::vector<bool> trunceds;
    // Per-env info dicts (as JSON strings for pybind11)
    std::vector<std::string> infos;
};

class ColonyVecEnvCpp {
public:
    ColonyVecEnvCpp(const std::vector<BaseData>& base_data,
                    const std::vector<BaseEvent>& events_data,
                    int n_envs, int64_t base_seed, int map_size = 280,
                    const Curriculum& curriculum = Curriculum(),
                    const RewardConfig& cfg = RewardConfig(),
                    int n_threads = 0,
                    const std::string& difficulty = "normal");

    void reset_batch(const std::vector<int64_t>& seeds);
    void step_async_batch(const std::vector<int>& actions);
    StepBatchResult step_wait_batch();

    // Миникарты всех сред: [n_envs, 8, 2R+1, 2R+1]
    int minimap_radius() const { return minimap_radius_; }
    int minimap_channels() const { return 8; }
    void set_minimap_radius(int r) { minimap_radius_ = r; for (auto& e : envs_) e.set_minimap_radius(r); }
    std::vector<float> minimap_batch() const;
    // Action masks for all envs: [n_envs * n_actions]
    std::vector<float> action_masks_batch() const;

    void save_normalization(const std::string& path);
    void load_normalization(const std::string& path);

    int n_envs() const { return n_envs_; }
    int obs_size() const { return obs_size_; }
    int n_actions() const { return n_actions_; }
    int n_build() const { return envs_.empty() ? 0 : envs_[0].n_build(); }
    // Build ids in C++ action order (single source for display names).
    std::vector<std::string> build_ids() const {
        return envs_.empty() ? std::vector<std::string>{} : envs_[0].build_ids();
    }
    int n_bases() const { int total = 0; for (const auto& e : envs_) total += e.n_bases(); return total; }
    int64_t n_people() const { int64_t total = 0; for (const auto& e : envs_) total += e.game().people; return total; }
    double total_daily_value() const { double t = 0; for (const auto& e : envs_) t += e.last_daily_value(); return t; }
    double total_chain_daily() const { double t = 0; for (const auto& e : envs_) t += e.last_chain_daily(); return t; }
    double mean_net_worth() const { double t = 0; for (const auto& e : envs_) t += e.debug_net_worth(); return t / (double)n_envs_; }
    const float* obs_buffer() const { return obs_buffer_.data(); }

    void set_curriculum(const Curriculum& c) {
        curriculum_ = c;
        for (auto& env : envs_) env.set_curriculum(c);
    }
    Curriculum curriculum() const { return curriculum_; }
    void set_rewards(const RewardConfig& cfg) {
        cfg_ = cfg;
        for (auto& env : envs_) env.set_rewards(cfg);
    }
    void set_step_log(int env_idx, const std::string& path) {
        if (env_idx >= 0 && env_idx < n_envs_) envs_[env_idx].set_step_log(path);
    }
    void clear_step_log(int env_idx) {
        if (env_idx >= 0 && env_idx < n_envs_) envs_[env_idx].set_step_log("");
    }
    std::string dump_obs(int env_idx) const {
        if (env_idx >= 0 && env_idx < n_envs_) return envs_[env_idx].dump_obs();
        return "";
    }

private:
    void do_step(int i, int action);
    void do_reset(int i, int64_t seed);
    void update_and_normalize_obs(float* raw_obs);
    double normalize_reward(double raw_rew);

    std::shared_ptr<std::vector<BaseData>> base_data_;
    std::shared_ptr<std::vector<BaseEvent>> events_data_;
    RewardConfig cfg_;
    int map_size_;
    Curriculum curriculum_;
    int n_envs_;
    int obs_size_;
    int n_actions_;
    int64_t base_seed_;
    int minimap_radius_ = 14;

    std::vector<ColonyEnvCpp> envs_;
    ThreadPool pool_;

    // Pre-allocated buffers
    std::vector<float> obs_buffer_;       // [n_envs * obs_size]
    std::vector<float> old_obs_buffer_;   // for RMS update (SB3 order)
    std::vector<float> raw_obs_buf_;      // for terminal_observation (reusable)
    std::vector<double> rewards_;         // [n_envs]
    std::vector<double> old_rew_buffer_;  // for reward RMS
    std::vector<bool> terminateds_;
    std::vector<bool> trunceds_;

    // VecNormalize stats
    RunningMeanStd obs_rms_;
    RunningMeanStd rew_rms_;
    bool norm_obs_ = true;
    bool norm_reward_ = true;
    double clip_obs_ = 10.0;
    double clip_reward_ = 10.0;

    // Episode tracking per env
    std::vector<double> episode_return_;
    std::vector<int64_t> episode_length_;
};

}  // namespace colony