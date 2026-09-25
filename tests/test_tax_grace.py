"""Tests for the tax grace feature.

The grace feature lets the player postpone a tax payment for up to 7 days.
After the grace expires the environment reports ``tax_grace_expired = True``
so the GUI can show the "sell resources / end game" dialog.

Run:  pytest tests/test_tax_grace.py -v
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "python"))

import colony_cpp  # noqa: E402


def make_env(seed: int = 42):
    bd = colony_cpp.load_base_data(str(PROJECT_ROOT / "configs" / "bases.json"))
    ed = colony_cpp.load_events(str(PROJECT_ROOT / "configs" / "events.json"))
    return colony_cpp.ColonyEnvCpp(bd, ed, seed=seed, map_size=280,
                                   curriculum={},  # PR 1: {} = unrestricted
                                   reward=colony_cpp.RewardConfig())


def set_date(g, year, month, day):
    g.year, g.month, g.day = year, month, day


# ─── Test 1: attribute exists ─────────────────────────────────────────────

def test_tax_grace_expired_attribute_exists():
    """ColonyEnvCpp must expose a tax_grace_expired property."""
    env = make_env()
    assert hasattr(env, "tax_grace_expired"), (
        "ColonyEnvCpp has no 'tax_grace_expired' attribute"
    )
    assert env.tax_grace_expired() is False


# ─── Test 2: grace not expired before 7 days ──────────────────────────────

def test_grace_not_expired_before_7_days():
    """With tax due and tax_postponed, grace must NOT expire before 7 days."""
    env = make_env()
    g = env.game()
    set_date(g, g.year + 1, 3, 1)   # March 1 → annual tax due
    g.tax_postponed = True

    for i in range(6):
        env.step(0)
        assert not env.tax_grace_expired(), (
            f"tax_grace_expired is True after only {i+1} days (should be False)"
        )


# ─── Test 3: grace expires at 7 days ──────────────────────────────────────

def test_grace_expires_at_7_days():
    """After 7 days of postponement, tax_grace_expired must be True."""
    env = make_env()
    g = env.game()
    set_date(g, g.year + 1, 3, 1)
    g.tax_postponed = True

    for _ in range(7):
        env.step(0)

    assert env.tax_grace_expired(), (
        "tax_grace_expired is False after 7 days (should be True)"
    )


# ─── Test 4: grace resets when tax is paid ────────────────────────────────

def test_grace_resets_when_tax_paid():
    """Once the player pays the tax, tax_grace_expired must go back to False."""
    env = make_env()
    g = env.game()
    set_date(g, g.year + 1, 3, 1)
    g.money = 10_000_000  # enough to pay
    g.tax_postponed = True

    for _ in range(7):
        env.step(0)
    assert env.tax_grace_expired(), "precondition: grace should have expired"

    # pay the tax
    g.money = 10_000_000
    g.pay_annual_tax()
    g.tax_postponed = False

    env.step(0)
    assert not env.tax_grace_expired(), (
        "tax_grace_expired is still True after paying the tax"
    )


# ─── Test 5: grace does not trigger when tax is not due ───────────────────

def test_no_grace_when_tax_not_due():
    """If no tax is due, tax_grace_expired must stay False."""
    env = make_env()
    g = env.game()
    # January 1 — no tax due
    set_date(g, g.year + 1, 1, 1)
    g.tax_postponed = True

    for _ in range(10):
        env.step(0)

    assert not env.tax_grace_expired()


# ─── Test 6: main tax also uses grace ─────────────────────────────────────

def test_main_tax_grace_expires_at_7_days():
    """The same 7-day grace applies to the main tax."""
    env = make_env()
    g = env.game()
    # Main tax due on Nov 1 of a year where (year - START_YEAR) % 10 == 0
    # and year != START_YEAR.  START_YEAR = 1890.
    target_year = 1890 + 10  # 1900
    set_date(g, target_year, 11, 1)
    g.tax_postponed = True

    for _ in range(7):
        env.step(0)

    assert env.tax_grace_expired(), (
        "tax_grace_expired is False after 7 days with main tax due"
    )
