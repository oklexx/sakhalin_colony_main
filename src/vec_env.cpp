// P2-7 (docs/REMAINING_WORK_2026_09.md): расщепление src/env.cpp по зонам.
// Этот файл — ColonyVecEnvCpp: батчный векторизованный env (последний,
// самый рискованный шаг расщепления). Чистое перемещение из src/env.cpp —
// поведение не меняется (регрессия: ./scripts/cpp_checks.sh, в т.ч.
// gui_watch_check и reward_v4_longrun, которые живут на VecEnv).
#include "colony/env.h"

#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <json.hpp>

namespace colony {

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
    const std::string& difficulty,
    bool tax_to_debt)
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
          return (size_t)std::max(1, std::min(desired, (int)hw));
      }()) {
    if (n_envs <= 0)
        throw std::invalid_argument("ColonyVecEnvCpp: n_envs must be > 0");
    // Create N envs from shared (immutable) data
    envs_.reserve(n_envs);
    for (int i = 0; i < n_envs; ++i) {
        envs_.emplace_back(*base_data_, *events_data_, base_seed + i * 10000,
                           map_size, curriculum, cfg, difficulty, false,
                           GAME_OVER_NO_PEOPLE_DAYS, tax_to_debt);
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
    const size_t mm_per = (size_t)8 * 32 * 32;
    terminal_minimap_buf_.assign((size_t)n_envs_ * mm_per, 0.0f);
    terminal_minimap_valid_.assign((size_t)n_envs_, 0);

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
    if ((int)seeds.size() != n_envs_)
        throw std::invalid_argument("reset_batch: seeds.size() != n_envs");
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
    if ((int)actions.size() != n_envs_)
        throw std::invalid_argument("step_async_batch: actions.size() != n_envs");
    std::vector<std::future<void>> futures;
    futures.reserve(n_envs_);
    for (int i = 0; i < n_envs_; ++i) {
        int action = actions[i];
        futures.push_back(pool_.submit([this, i, action]() {
            do_step(i, action);
        }));
    }
    // Wait for all tasks to complete; worker exceptions must fail fast instead
    // of silently leaving stale observations/rewards in the batch.
    for (int i = 0; i < (int)futures.size(); ++i) {
        try {
            futures[(size_t)i].get();
        } catch (const std::exception& e) {
            throw std::runtime_error("ColonyVecEnvCpp::step_async_batch failed in env " +
                                     std::to_string(i) + ": " + e.what());
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
    std::fill(terminal_minimap_valid_.begin(), terminal_minimap_valid_.end(), 0);
    std::fill(terminal_minimap_buf_.begin(), terminal_minimap_buf_.end(), 0.0f);

    for (int i = 0; i < n_envs_; ++i) {
        if (terminateds_[i] || trunceds_[i]) {
            // terminal_observation must be RAW (unnormalized), per SB3 convention
            std::vector<float> terminal_obs(raw_obs_buf_.begin() + (size_t)i * obs_size_,
                                            raw_obs_buf_.begin() + (size_t)(i + 1) * obs_size_);
            // Normalized s_T — what the critic actually consumes (obs_buffer_
            // is already normalized, auto-reset has not run yet).
            std::vector<float> terminal_obs_norm(
                obs_buffer_.begin() + (size_t)i * obs_size_,
                obs_buffer_.begin() + (size_t)(i + 1) * obs_size_);

            const size_t mm_per = (size_t)8 * 32 * 32;
            std::vector<float> mm = envs_[(size_t)i].minimap();
            if (mm.size() == mm_per) {
                std::copy(mm.begin(), mm.end(),
                          terminal_minimap_buf_.begin() + (size_t)i * mm_per);
                terminal_minimap_valid_[(size_t)i] = 1;
            }

            // Build info JSON
            nlohmann::json info;
            info["terminal_observation"] = terminal_obs;
            info["terminal_observation_norm"] = terminal_obs_norm;
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
    const int G = 32;
    const size_t per = (size_t)8 * G * G;
    std::vector<float> out((size_t)n * per, 0.0f);
    for (int i = 0; i < n; i++) {
        std::vector<float> mm = envs_[(size_t)i].minimap();
        std::copy(mm.begin(), mm.end(), out.begin() + (size_t)i * per);
    }
    return out;
}

std::vector<float> ColonyVecEnvCpp::terminal_minimap_batch() const {
    // Миникарты s_T сред, завершившихся на последнем step_wait_batch.
    // Буфер заполняется в step_wait_batch (перед авто-ресетом) и обнуляется
    // в начале каждого шага: записи невалидных сред остаются нулями.
    return terminal_minimap_buf_;
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
