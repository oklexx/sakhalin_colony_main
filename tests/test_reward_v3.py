"""Тесты v3-наград: бонусы за добычу ресурса и стоимость действий.

Требуют собранный colony_cpp (C++-ядро). torch не нужен.

См. ANALYSIS_TRAINING_REPORT.md и configs/reward_v3.json.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "python"))

try:
    import colony_cpp
    ENV_OK = True
except Exception:
    ENV_OK = False

WATER = 6  # индекс ресурса (constants.h)
SEED = 42  # карта: вода в (133,140), старт-Город в (140,140)


def _base_data():
    return colony_cpp.load_base_data(str(ROOT / "configs" / "bases.json"))


def _events():
    return colony_cpp.load_events(str(ROOT / "configs" / "events.json"))


def zero_config(**overrides):
    """RewardConfig, где ВСЕ награды обнулены (кроме указанных в overrides).

    Даёт чистый фон: reward любого шага = только тестируемый компонент.
    """
    rc = colony_cpp.RewardConfig()
    for k in ("build_bonus", "chain_bonus", "chain_daily", "novelty", "daily_income",
              "sale_bonus", "tax_daily_bonus", "survival_bonus", "game_over_penalty",
              "diversity_bonus", "error_penalty", "preserve_penalty", "demolish_penalty",
              "manual_tax_penalty", "build_cost_penalty", "idle_build_penalty",
              "survival_coeff", "milestone_base_bonus", "milestone_people_bonus",
              "milestone_day_bonus", "milestone_year_bonus", "proximity_bonus",
              "tax_fail_penalty", "death_penalty", "base_lost_penalty", "born_bonus",
              "debt_coeff", "home_overflow_penalty", "housing_need_bonus",
              "food_need_bonus", "water_need_bonus", "loan_penalty",
              "first_extraction_bonus", "extraction_daily", "need_fill_bonus"):
        setattr(rc, k, 0.0)
    rc.disable_daily_income = True
    rc.disable_provider_bonus = True
    rc.disable_net_worth = True
    for k, v in overrides.items():
        setattr(rc, k, v)
    return rc


def make_env(rc, seed=SEED):
    env = colony_cpp.ColonyEnvCpp(_base_data(), _events(), seed, 280, {}, rc, "normal")  # PR 1: {} = unrestricted
    env.reset(seed)
    return env


def build_idx(env, name):
    return 2 + env.build_ids().index(name)


def mgr_idx(env, j):
    """Индекс manager-действия: DAY=0, WEEK=1, BUILD=2..2+n_build-1, MGR=2+n_build+j."""
    return 2 + env.n_build() + j


MGR_PRESERVE, MGR_UNPRESERVE, MGR_TAKE_LOAN = 38, 39, 42  # n_build=32 (BUILD_SUBSET)


def _direct_build(env, name, cells):
    """Прямая постройка через game API (без env-наград) — для чистоты теста."""
    g = env.game()
    for (x, y) in cells:
        ok, msg = g.build(name, x, y)
        if ok:
            return True
    return False


@pytest.mark.skipif(not ENV_OK, reason="colony_cpp not available")
def test_loan_penalty():
    """Успешный TAKE_LOAN стоит ровно loan_penalty."""
    env = make_env(zero_config(loan_penalty=1.0))
    assert env.action_mask()[MGR_TAKE_LOAN] == 1.0
    out = env.step(MGR_TAKE_LOAN)
    assert env.game().credit > 0, "кредит не выдан"
    assert out["reward"] == pytest.approx(-1.0, abs=1e-9)


@pytest.mark.skipif(not ENV_OK, reason="colony_cpp not available")
def test_preserve_penalty():
    """PRESERVE и UNPRESERVE стоят по preserve_penalty (ключ раньше не применялся)."""
    env = make_env(zero_config(preserve_penalty=0.5))
    assert _direct_build(env, "SmallHouse", [(141, 140), (139, 140), (140, 141), (140, 139)])
    out = env.step(MGR_PRESERVE)
    assert out["reward"] == pytest.approx(-0.5, abs=1e-9)
    out = env.step(MGR_UNPRESERVE)
    assert out["reward"] == pytest.approx(-0.5, abs=1e-9)


@pytest.mark.skipif(not ENV_OK, reason="colony_cpp not available")
def test_first_extraction_fires_once_per_resource():
    """first_extraction: ровно один раз за эпизод, когда ресурс реально добыт.

    Карта seed 42: вода в 7 клетках севернее старта. Строим мост из дорог
    напрямую, WaterChannel — через env-действие (чтобы сработало step()).
    """
    env = make_env(zero_config(first_extraction_bonus=3.0))
    g = env.game()
    # мост дорог: (140,139)..(140,134) — вода в (133,140)
    for y in range(139, 133, -1):
        ok, _ = g.build("Road", 140, y)
        assert ok, f"road at (140,{y}) failed"
    wc = build_idx(env, "WaterChannel")
    assert env.action_mask()[wc] == 1.0, "WC недоступен после моста"
    rewards = [env.step(wc)["reward"]]
    fired = []
    for d in range(60):
        out = env.step(0)
        rewards.append(out["reward"])
        if out["reward"] > 1e-6:
            fired.append((d, out["reward"]))
    assert len(fired) == 1, f"ожидался ровно 1 срабатывание, получено: {fired}"
    # вес воды = min(1, 0.25 * число типов-потребителей) = 1.0 (потребителей >= 4)
    assert fired[0][1] == pytest.approx(3.0, abs=1e-9)
    # дальше наград нет (разовый бонус)
    assert all(r == 0.0 for r in rewards[fired[0][0] + 2:])


@pytest.mark.skipif(not ENV_OK, reason="colony_cpp not available")
def test_extraction_daily_saturates():
    """extraction_daily: каждый день, пока идёт добыча; насыщение при 3+ производителях."""
    env = make_env(zero_config(extraction_daily=0.3))
    g = env.game()
    for y in range(139, 133, -1):
        ok, _ = g.build("Road", 140, y)
        assert ok
    wc = build_idx(env, "WaterChannel")
    env.step(wc)
    daily = []
    for _d in range(70):
        out = env.step(0)
        if out["reward"] > 1e-6:
            daily.append(out["reward"])
    # после запуска WC (день ~47) каждый день даётся 0.3 * w(вода)=1.0 * sat(1/3)=0.1
    assert len(daily) >= 20, f"слишком мало дней с добычей: {len(daily)}"
    assert all(r == pytest.approx(0.1, abs=1e-9) for r in daily[:10])


@pytest.mark.skipif(not ENV_OK, reason="colony_cpp not available")
def test_need_fill_bonus():
    """need_fill: бонус за постройку производителя ресурса, которого не хватает.

    A/B: две среды с одной картой. В контроле ферма не построена —
    reward постройки WC = только «фоновый» log2-терм build-бонуса.
    В тестовой ферма голодает по воде — разница == 2.0 * w(вода).
    """
    def run(with_farm):
        env = make_env(zero_config(need_fill_bonus=2.0))
        g = env.game()
        if with_farm:
            assert _direct_build(env, "Farm", [(141, 140), (139, 140), (140, 141), (140, 139)])
        for _ in range(45):  # в тестовой: ферма достроится (38 дн.) и сядет за нехваткой воды
            env.step(0)
        for y in range(139, 133, -1):
            ok, _ = g.build("Road", 140, y)
            assert ok
        wc = build_idx(env, "WaterChannel")
        return env, env.step(wc)["reward"]

    env_t, r_t = run(with_farm=True)
    farm = [b for b in env_t.game().bases() if b["id"] == "Farm"][0]
    assert farm["need_sunduk"], "ферма должна голодать по воде"
    env_c, r_c = run(with_farm=False)
    # разница = need_fill = 2.0 * w(вода)=1.0; фоновый log2-терм сокращается
    assert r_t - r_c == pytest.approx(2.0, abs=1e-9)


@pytest.mark.skipif(not ENV_OK, reason="colony_cpp not available")
def test_food_extraction_not_rewarded():
    """Анти-спам: еда/золото (их никто не потребляет в порту) => вес 0, наград нет.

    Теплица работает (воду подаём напрямую в sunduk), производит еду —
    first_extraction не срабатывает.
    """
    env = make_env(zero_config(first_extraction_bonus=3.0))
    g = env.game()
    assert _direct_build(env, "Hothouse", [(141, 140), (139, 140), (140, 141), (140, 139)])
    for _ in range(31):  # стройка 30 дней
        env.step(0)
    # подаём воду, чтобы теплица работала и производила еду
    sunduk = list(g.sunduk)
    sunduk[WATER] = 10000
    g.sunduk = sunduk
    for _ in range(10):
        out = env.step(0)
        assert out["reward"] == 0.0
    # еда в складе — но «добыча» наград не даёт
    assert g.sunduk[1] > 0, "теплица должна производить еду"
    # PR 4 §5.5: reached = ВСЕ достигнутые (еда добыта — трекинг есть);
    # priority смотрит на вес КУРИКУЛУМА (здесь all_resources — всё по 1.0),
    # а отсутствие бонуса — заслуга потребительского множителя w_food = 0
    # (пинится ассёртами reward == 0.0 выше).
    assert out["metrics"].reached_resources == 1
    assert out["metrics"].priority_reached == 1


@pytest.mark.skipif(not ENV_OK, reason="colony_cpp not available")
def test_v3_keys_roundtrip_python():
    """Ключи v3 экспортируются Python-конфигурацией (иначе UI/worker их потеряет).

    rl/config.py грузим напрямую: rl/__init__.py тянет torch, которого
    в тестовой среде может не быть.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "colony_rl_config", str(ROOT / "rl" / "config.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["colony_rl_config"] = mod  # dataclasses требует модуль в sys.modules
    spec.loader.exec_module(mod)
    RewardConfig = mod.RewardConfig
    rc = RewardConfig()
    d = rc.to_dict()
    for k in ("first_extraction_bonus", "extraction_daily", "need_fill_bonus", "loan_penalty"):
        assert k in d, f"{k} не в to_dict() — ключ теряется в C++"
    rc2 = RewardConfig.from_dict(d)
    assert rc2.first_extraction_bonus == rc.first_extraction_bonus
    assert rc2.loan_penalty == rc.loan_penalty
    # C++ принимает все ключи
    c = colony_cpp.RewardConfig()
    for k, _v in d.items():
        assert hasattr(c, k), f"C++ RewardConfig не имеет {k} (нужна пересборка pyd)"
