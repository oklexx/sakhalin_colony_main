#pragma once
/**
 * @file reward_config.h
 * @brief Reward coefficients — single source of truth.
 *
 * Defaults mirror configs/reward_v3.json. Python always overrides each field
 * explicitly; these defaults matter only for console/GUI binaries and tests.
 * Keep in sync with rl/config.py RewardConfig and configs/reward_v3.json.
 */
#include <cstdint>

namespace colony {

struct RewardConfig {
    // base bonuses
    double build_bonus = 2.0;
    double chain_bonus = 1.0;
    double chain_daily = 0.5;

    // v3: resource extraction
    double first_extraction_bonus = 3.0;
    double extraction_daily = 0.3;
    double need_fill_bonus = 1.5;
    double loan_penalty = 0.5;

    double novelty = 5.0;
    double daily_income = 1.0;
    double sale_bonus = 0.5;
    double tax_daily_bonus = 0.3;
    double survival_bonus = 0.0;
    double game_over_penalty = 10.0;
    double diversity_bonus = 3.0;

    // penalties / special actions
    double error_penalty = -2.0;
    double preserve_penalty = 0.3;
    double demolish_penalty = -3.0;
    double manual_tax_penalty = -0.5;
    double build_cost_penalty = 0.0001;
    double idle_build_penalty = -2.0;
    int idle_build_threshold_days = 7;
    double survival_coeff = 0.0;

    // milestones
    double milestone_base_bonus = 30.0;
    double milestone_people_bonus = 2.0;
    double milestone_day_bonus = 2.0;
    double milestone_year_bonus = 5.0;

    // spatial / clipping
    double proximity_bonus = 0.5;
    double clip_reward_min = -50.0;
    double clip_reward_max = 50.0;

    // flags (ablations)
    bool disable_net_worth = false;
    bool disable_daily_income = false;
    bool disable_provider_bonus = false;
    // PR 4: считать потребителей ресурсов (n_consumers в compute_catalog) только
    // по разрешённым постройкам. false = как раньше (по всем 32), старые прогоны
    // не меняются.
    bool priority_count_over_allowed = false;

    // formerly hardcoded C++ weights — must be exported or C++ keeps defaults
    double tax_fail_penalty = 5.0;
    double death_penalty = 20.0;
    double base_lost_penalty = 30.0;
    double born_bonus = 1.0;
    double debt_coeff = 0.1;
    double home_overflow_penalty = 2.0;
    double housing_need_bonus = 3.0;
    double food_need_bonus = 0.8;
    double water_need_bonus = 0.8;
    double buy_food_penalty = 3.0;
};

}  // namespace colony
