"""P0/P1 regression tests (2026-09): tax debt policy, step log, applicability mask, obs v2.

Что проверяем (см. docs/RL_DIAGNOSIS_2026_09.md):

* P0 — неоплаченный налог переоформляется в долг банку, календарь НЕ замирает;
  ``tax_to_debt=false`` возвращает «диалоговую» политику (GUI-путь).
* P0 — step-лог печатает debt/born/died/lost/overflow + total_raw/total_clip.
* P1 — маска менеджеров по применимости (короче/строже легальной).
* P1 — obs v2 (299): dx/dy к ближайшим ресурсам; v0/v1 — строгие префиксы.

Без собранного colony_cpp файл целиком скипается.
"""
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "python"))

try:
    import colony_cpp
    ENV_OK = True
except Exception:  # pragma: no cover - среда без расширения
    ENV_OK = False

pytestmark = pytest.mark.skipif(not ENV_OK, reason="colony_cpp not available")


def make_env(**kwargs):
    bd = colony_cpp.load_base_data(str(PROJECT_ROOT / "configs" / "bases.json"))
    ed = colony_cpp.load_events(str(PROJECT_ROOT / "configs" / "events.json"))
    env = colony_cpp.ColonyEnvCpp(bd, ed, 42, 200, {}, colony_cpp.RewardConfig(), "normal",
                                  **kwargs)
    env.reset(42)
    return env


def test_default_is_debt_policy():
    env = make_env()
    assert env.tax_to_debt() is True
    assert env.step(0)["tax_borrowed"] == 0


def test_unpaid_tax_goes_to_debt_and_time_moves():
    """Потратить капитал, затем 400 дней: календарь идёт, налог уходит в credit."""
    env = make_env()
    farm = 2 + list(env.build_ids()).index("Farm")
    for _ in range(4):
        env.step(farm)  # тратим 4*19814 из 82k
    g = env.game()
    assert g.money < 10000, "подготовка: капитал почти исчерпан"
    assert g.check_advance()[0], "налог не должен блокировать календарь"

    borrowed = 0
    prev_day = g.days_alive
    for _ in range(400):
        out = env.step(0)  # DAY
        assert out["days"] > prev_day, "календарь обязан двигаться каждый шаг"
        prev_day = out["days"]
        borrowed += int(out["tax_borrowed"])
        if out["terminated"]:
            break
    assert g.days_alive > 365, "эпизод не должен умирать на первой дате налога"
    assert borrowed > 0, "неоплаченный налог не переоформился в долг"
    assert g.credit > 0


def test_legacy_tax_policy_still_freezes_as_control():
    env = make_env(tax_to_debt=False)
    farm = 2 + list(env.build_ids()).index("Farm")
    for _ in range(4):
        env.step(farm)
    g = env.game()
    for _i in range(400):
        env.step(0)
        if env.tax_grace_expired():
            break
    assert g.days_alive <= 366, "в диалоговой политике время по-прежнему стоит"


def test_step_log_has_terms_and_raw_clip(tmp_path):
    path = tmp_path / "steplog.txt"
    env = make_env()
    env.set_step_log(str(path))
    farm = 2 + list(env.build_ids()).index("Farm")
    for _ in range(4):
        env.step(farm)
    for _ in range(400):
        out = env.step(0)
        if out["terminated"]:
            break
    text = path.read_text(encoding="utf-8", errors="replace")
    for needle in ("total_raw=", "total_clip=", "taxdebt=", "debt=", "born=", "died=",
                   "lost=", "overflow=", "tax_borrowed=", "TAX->DEBT:"):
        assert needle in text, f"в step-логе нет {needle}"


def test_manager_mask_is_applicability_based():
    env = make_env()
    mgr = 2 + env.n_build()
    m = env.action_mask()
    # на старте применимы ровно три менеджера: IMPROVE_LAND, BUY_FOOD, TAKE_LOAN
    assert m[mgr + 0] == 1.0, "IMPROVE_LAND применим (82k >= BUYGOODEARTH)"
    assert m[mgr + 6] == 0.0, "SELL применим только при излишке"
    assert m[mgr + 9] == 0.0, "REPAY без долга не применим"
    assert m[mgr + 10] == 0.0, "manual_tax доминируем DAY"
    g = env.game()
    sunduk = g.sunduk          # property возвращает копию — пишем обратно
    sunduk[7] = 500            # WOOD
    g.sunduk = sunduk
    assert env.action_mask()[mgr + 6] == 1.0, "SELL открывается с излишком"
    g.money = 0
    assert env.action_mask()[mgr + 0] == 0.0, "IMPROVE_LAND без денег не применим"


def test_obs_v2_is_default_and_has_directions():
    from rl.curriculum import NEAREST_LOT_ORDER, CurriculumState

    st = CurriculumState(True, (), True, (1.0,) * 9, 0, 2)
    assert st.obs_version == 2
    assert len(NEAREST_LOT_ORDER) == 5
    env = make_env()
    assert env.obs_size() == 299  # 248 (v0) + 41 (v1 frame) + 10 (v2 dirs)
    raw = list(env.obs())
    assert len(raw) == 299
    # v0/v1 — строгие префиксы: вода как была в 246/247
    assert abs(raw[246]) <= 1.0 and abs(raw[247]) <= 1.0
    assert sum(1 for v in raw[289:299] if v != 0.0) >= 3


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
