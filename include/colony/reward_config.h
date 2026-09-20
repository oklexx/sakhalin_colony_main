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
    // base bonuses (v4: build_bonus 2.0 -> 1.2 — балансировка стройки)
    double build_bonus = 1.2;
    double chain_bonus = 1.0;
    double chain_daily = 0.5;

    // v3: resource extraction
    double first_extraction_bonus = 3.0;
    double extraction_daily = 0.3;
    double need_fill_bonus = 1.5;
    double loan_penalty = 2.0;
    // P0 (2026-09-17): штраф за переоформление неоплаченного налога в долг банку
    // (Game::settle_tax_with_debt). Масштаб — log1p(borrowed/1000): ~-3 за 4k,
    // ~-12 за 500k. Само обслуживание долга идёт через debt_coeff (см. ниже).
    double tax_debt_penalty = 2.0;

    double novelty = 3.0;
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
    double build_cost_penalty = 0.00004;
    double idle_build_penalty = -2.0;
    int idle_build_threshold_days = 7;
    double survival_coeff = 0.0005;

    // milestones
    double milestone_base_bonus = 10.0;
    double milestone_people_bonus = 2.0;
    double milestone_day_bonus = 2.0;
    double milestone_year_bonus = 5.0;

    // spatial / clipping
    double proximity_bonus = 0.5;
    double clip_reward_min = -100.0;
    double clip_reward_max = 100.0;

    // Potential-based road shaping к воде (P2-9, 2026-09-19): раньше эти числа
    // были захардкожены в ColonyEnvCpp::step() и ::reset() (1.5 / 1.0 / +3.0 /
    // -0.1), поэтому их нельзя было ни подобрать, ни аблировать, ни увидеть в
    // профиле наград. Значения = прежний хардкод бит-в-бит, так что старые
    // прогоны не меняются. Смысл: дорога не приносит дохода, пока не дойдёт до
    // воды (7-14 клеток), поэтому за приближение даём потенциальный бонус, а за
    // дорогу «не туда» — маленький штраф.
    double road_shaping_cap = 1.5;         // потолок бонуса за одну дорогу
    double road_shaping_per_cell = 1.0;    // бонус за клетку приближения к воде
    double water_reach_bonus = 3.0;        // разовый бонус «дорога дошла до воды»
    double water_reach_radius = 1.5;       // что считать «дошли» (клеток до воды)
    double road_no_progress_penalty = 0.1; // штраф за дорогу без приближения
    double road_progress_epsilon = 0.25;   // гистерезис: считать ли приближение

    // flags (ablations)
    bool disable_net_worth = false;
    bool disable_daily_income = false;
    bool disable_provider_bonus = false;
    // P1 (2026-09-17): маскировать действия-менеджеры по ПРИМЕНИМОСТИ, а не по
    // формальной возможности. BUY_FOOD без нужды, SELL с пустым складом, REPAY
    // без долга, IMPROVE_LAND без денег и т.п. дают чистый error_penalty и
    // засоряют выбор (замер: err ~-1.6/шаг у masked-random). false = старые маски.
    bool mask_managers_by_applicability = true;
    // PR 4: считать потребителей ресурсов (n_consumers в compute_catalog) только
    // по разрешённым постройкам. false = как раньше (по всем 32), старые прогоны
    // не меняются.
    bool priority_count_over_allowed = false;
    // PR 5: zero the catalog rows (4 features per buildable) of locked
    // buildings in obs. false = catalog ignores gating (default, old runs
    // unchanged); idle_by_type/counts are NEVER filtered.
    bool obs_mask_locked_catalog = false;

    // formerly hardcoded C++ weights — must be exported or C++ keeps defaults
    double tax_fail_penalty = 5.0;
    double death_penalty = 12.0;
    double base_lost_penalty = 30.0;
    double born_bonus = 2.0;
    double debt_coeff = 0.0;
    double home_overflow_penalty = 2.0;
    double housing_need_bonus = 3.0;
    double food_need_bonus = 0.8;
    double water_need_bonus = 0.8;
    double buy_food_penalty = 3.0;

    // v4 (2026-09): цель на выживание, бонус за оплату 500k и предналоговое давление
    double goal_survival_coeff = 200.0;
    double main_tax_cash_bonus = 100.0;
    double main_tax_pressure_coeff = 0.002;
};

}  // namespace colony
