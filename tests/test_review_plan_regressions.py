"""Регрессионные тесты по плану ревью (Этап 0 — минимальный набор P0).

Фиксы уже в ядре (см. docs/REFACTOR_2026_09.md и план ревью), эти тесты
фиксируют инварианты, чтобы регрессия не вернулась незаметно:

  1. delete_base не портит spatial-индекс (нет фантомов «место занято»).
  2. Game::build запрещает стройку не на той земле / на сгоревшем участке
     / на занятой клетке (единый валидатор Game::can_build_at).
  3. market_buy/market_sell отвергают отрицательные количества.
  4. sunduk_from_list требует ровно 9 элементов.
  5. ep_return равен сумме реально возвращённых reward.
  6. ColonyVecEnvCpp: fail-fast на n_envs=0 и неправильных размерах batch.
  7. Python-обёртка CppVecEnv не затирает terminal_observation из C++-info
     и не выдумывает его, если C++ не прислал (P2-3, 2026-09-24).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

try:
    import colony_cpp
    ENV_OK = True
except Exception:
    ENV_OK = False

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# include/colony/constants.h
LT_NORMAL = 1
LT_WATER = 2
LT_COAL = 4
SUNDUK_SIZE = 9


@pytest.fixture(scope="module")
def base_data():
    return colony_cpp.load_base_data(str(PROJECT_ROOT / "configs" / "bases.json"))


@pytest.fixture(scope="module")
def events_data():
    return colony_cpp.load_events(str(PROJECT_ROOT / "configs" / "events.json"))


def make_game(base_data, events_data, seed=7, map_size=64):
    g = colony_cpp.Game(base_data, events_data, seed, map_size)
    g.money = 50_000_000  # деньги не должны быть ограничителем в этих тестах
    return g


def city_xy(g):
    for b in g.bases():
        if b["id"] == "City":
            return b["x"], b["y"]
    raise AssertionError("City не найден в bases()")


def cells_near_city(g, ring=3):
    """Свободные клетки LT_NORMAL вокруг города, отсортированные по дистанции."""
    msz = g.map_size()
    cx, cy = city_xy(g)
    lots = g.earth_lots()
    out = []
    for dy in range(-ring, ring + 1):
        for dx in range(-ring, ring + 1):
            x, y = cx + dx, cy + dy
            if not (0 <= x < msz and 0 <= y < msz):
                continue
            if (dx, dy) == (0, 0):
                continue
            if lots[y * msz + x] == LT_NORMAL:
                out.append((x, y, abs(dx) + abs(dy)))
    out.sort(key=lambda t: t[2])
    return [(x, y) for x, y, _ in out]


# ── 1+2. delete_base: spatial-индекс не ломается, правило «занято» живо ─────

@pytest.mark.skipif(not ENV_OK, reason="colony_cpp not available")
def test_delete_base_no_phantom_occupied(base_data, events_data):
    """Строим цепочку из 3 зданий, сносим среднее.

    Старый баг: после swap-erase base_index_map_ оставался со старыми
    индексами — base_in_box находил фантома и honest-клетка отзывалась
    «Это место занято» (или наоборот, стройка шла на занятую клетку).
    """
    g = make_game(base_data, events_data)
    free = cells_near_city(g)
    assert len(free) >= 3, "нет свободных LT_NORMAL клеток вокруг города"

    a, b, c = free[0], free[1], free[2]
    for (x, y) in (a, b, c):
        # соседние с городом/цепочкой клетки связаны → can_build_at проходит
        ok, msg = g.build("House", x, y)
        assert ok, f"House на ({x},{y}) должен строиться: {msg}"

    assert len(g.bases()) == 4  # City + 3

    # снос среднего
    ok, msg = g.destroy(*b)
    assert ok, f"destroy средней базы должен работать: {msg}"
    assert len(g.bases()) == 3, "после сноса должно остаться 3 здания"
    assert all((bb["x"], bb["y"]) != b for bb in g.bases()), "снесённая база осталась в списке"

    # клетка снесённой базы — сгоревший участок (не «занято», не «можно строить»)
    ok, msg = g.build("House", *b)
    assert not ok, "на сгоревшем участке строить нельзя"
    assert "занято" not in msg.lower(), (
        f"фантомная база в индексе: клетка ({b[0]},{b[1]}) отвечает "
        f"«{msg}» вместо «сгоревший участок»")

    # КЛЮЧЕВОЕ: клетка ПЕРЕМЕЩЁННОЙ базы (swap перемещает хвост в дырку)
    # обязана отвечать «место занято» — значит индекс не протух
    ok, msg = g.build("House", *c)
    assert not ok, "на клетке существующего здания стройка должна быть запрещена"
    assert "занято" in msg.lower(), (
        f"клетка переставленной базы ({c[0]},{c[1]}) потеряла владельца "
        f"(ответ: «{msg}») — base_index_map_ разсинхронизирован")

    # и свободная клетка по-прежнему свободна
    if len(free) >= 4:
        ok, msg = g.build("House", *free[3])
        assert ok, f"свободная клетка ({free[3][0]},{free[3][1]}) должна строиться: {msg}"


# ── 2. Game::can_build_at — единый валидатор стройки ─────────────────────────

@pytest.mark.skipif(not ENV_OK, reason="colony_cpp not available")
def test_build_rejects_wrong_earth(base_data, events_data):
    """WaterChannel — только на воде (LT_WATER), Coalmine — только на угле."""
    g = make_game(base_data, events_data)
    free = cells_near_city(g, ring=4)
    assert free, "нет нормальной земли вокруг города"

    money_before = g.money
    ok, msg = g.build("WaterChannel", *free[0])
    assert not ok, "WaterChannel не должен строиться не на воде"
    assert g.money == money_before, "неудавшаяся стройка не должна списывать деньги"

    ok, msg = g.build("Coalmine", *free[0])
    assert not ok, "Coalmine не должен строиться не на угле"
    assert g.money == money_before


@pytest.mark.skipif(not ENV_OK, reason="colony_cpp not available")
def test_build_ok_on_valid_cell_then_destroy_burns_lot(base_data, events_data):
    """Валидная стройка проходит и списывает цену; после сноса участок горит."""
    g = make_game(base_data, events_data)
    free = cells_near_city(g)
    assert free
    x, y = free[0]

    money_before = g.money
    ok, msg = g.build("House", x, y)
    assert ok, f"House на нормальной земле рядом с городом должен строиться: {msg}"
    assert g.money == money_before - 14038, "цена House должна списаться"

    ok, _ = g.destroy(x, y)
    assert ok
    msz = g.map_size()
    assert g.destroyed_lots()[y * msz + x] > 0, "после сноса destroyed_lots > 0"


@pytest.mark.skipif(not ENV_OK, reason="colony_cpp not available")
def test_build_rejects_occupied_cell(base_data, events_data):
    g = make_game(base_data, events_data)
    cx, cy = city_xy(g)
    ok, msg = g.build("House", cx, cy)
    assert not ok, "на клетке города строить нельзя"
    assert "занято" in msg.lower()


# ── 3. Market: отрицательные количества запрещены ────────────────────────────

@pytest.mark.skipif(not ENV_OK, reason="colony_cpp not available")
def test_market_rejects_negative_counts(base_data, events_data):
    g = make_game(base_data, events_data)
    money_before = g.money

    neg = [-1] + [0] * (SUNDUK_SIZE - 1)
    ok, msg, total = g.market_sell(neg)
    assert not ok, "market_sell с отрицательным количеством должен отказать"
    assert g.money == money_before, (
        "отрицательная продажа не должна создавать деньги (market exploit)")

    ok, msg, total = g.market_buy(neg)
    assert not ok, "market_buy с отрицательным количеством должен отказать"
    assert g.money == money_before


@pytest.mark.skipif(not ENV_OK, reason="colony_cpp not available")
def test_market_sell_positive_works(base_data, events_data):
    """Санити: легальная продажа продолжает работать (продаём дерево)."""
    g = make_game(base_data, events_data)
    sunduk = g.sunduk
    sunduk[7] = 100  # WOOD, см. тесты p0_p1
    g.sunduk = sunduk
    counts = [0] * SUNDUK_SIZE
    counts[7] = 10
    money_before = g.money
    ok, msg, total = g.market_sell(counts)
    assert ok, f"легальная продажа не должна отказывать: {msg}"
    assert total > 0 and g.money > money_before


# ── 4. Bindings: размер Sunduk-списка ────────────────────────────────────────

@pytest.mark.skipif(not ENV_OK, reason="colony_cpp not available")
def test_sunduk_list_must_have_9_items(base_data, events_data):
    g = make_game(base_data, events_data)
    with pytest.raises(RuntimeError, match="exactly 9 items"):
        g.market_sell([1, 2])
    with pytest.raises(RuntimeError, match="exactly 9 items"):
        g.market_buy([0] * (SUNDUK_SIZE + 1))


# ── 5. ep_return == сумма возвращённых reward ────────────────────────────────

@pytest.mark.skipif(not ENV_OK, reason="colony_cpp not available")
def test_ep_return_equals_sum_of_rewards(base_data, events_data):
    env = colony_cpp.ColonyEnvCpp(base_data, events_data, 42, 200, {},
                                  colony_cpp.RewardConfig(), "normal")
    env.reset(42)
    total = 0.0
    n_steps = 50
    for _ in range(n_steps):
        out = env.step(0)  # DAY
        total += float(out["reward"])
        assert float(out["ep_return"]) == pytest.approx(total, abs=1e-6), (
            "ep_return на каждом шаге равен сумме реально возвращённых reward")
    assert env.steps() == n_steps
    assert env.ep_return() == pytest.approx(total, abs=1e-6)


# ── 6. ColonyVecEnvCpp: fail-fast на неверных входах ─────────────────────────

@pytest.mark.skipif(not ENV_OK, reason="colony_cpp not available")
def test_vecenv_rejects_n_envs_zero(base_data, events_data):
    with pytest.raises(Exception, match="n_envs"):
        colony_cpp.ColonyVecEnvCpp(base_data, events_data, 0, 0, 64)


@pytest.mark.skipif(not ENV_OK, reason="colony_cpp not available")
def test_vecenv_rejects_wrong_batch_sizes(base_data, events_data):
    vec = colony_cpp.ColonyVecEnvCpp(base_data, events_data, 2, 0, 64)
    try:
        with pytest.raises(Exception, match="seeds"):
            vec.reset_batch([1])  # 1 seed на 2 среды
        vec.reset_batch([1, 2])
        with pytest.raises(Exception, match="actions"):
            vec.step_async_batch([0])  # 1 действие на 2 среды
    finally:
        vec.close() if hasattr(vec, "close") else None


# ── 7. Python-обёртка не затирает terminal_observation ───────────────────────

def test_cpp_vecenv_preserves_terminal_observation(monkeypatch):
    """C++ кладёт RAW (ненормализованную) терминальную наблюдку в info;
    обёртка обязана сохранить её, а не подменять текущей obs, и проставить
    TimeLimit.truncated. Регрессия плана: info['terminal_observation']
    перезаписывался obs[i].copy() даже при наличии значения из C++.
    """
    pytest.importorskip("colony_cpp")
    from cpp_vecenv import CppVecEnv

    env = CppVecEnv(n_envs=1, seed=0, map_size=64)
    try:
        env.reset()
        obs_size = int(env.observation_space.shape[0])

        class FakeResult:
            pass

        terminal_raw = [7.0] * obs_size

        def fake_wait_a():
            r = FakeResult()
            r.obs = np.zeros((1, obs_size), dtype=np.float32)  # текущая obs = нули
            r.rewards = [0.5]
            r.terminateds = [True]
            r.trunceds = [False]
            r.infos = [json.dumps({"terminal_observation": terminal_raw})]
            return r

        def fake_wait_b():
            r = FakeResult()
            r.obs = np.zeros((1, obs_size), dtype=np.float32)
            r.rewards = [-1.0]
            r.terminateds = [False]
            r.trunceds = [True]  # time limit: SB3 ждёт TimeLimit.truncated
            r.infos = ["{}"]
            return r

        # (a) C++ прислал terminal_observation — обёртка сохраняет ЗНАЧЕНИЕ.
        # pybind-атрибуты только для чтения — подменяем сам cpp_vec прокси.
        class _Proxy:
            def __init__(self, real, wait):
                self._real = real
                self._wait = wait

            def step_wait_batch(self):
                return self._wait()

            def __getattr__(self, name):
                return getattr(self._real, name)

        env.cpp_vec = _Proxy(env.cpp_vec, fake_wait_a)
        obs, rewards, dones, infos = env.step(np.array([0]))
        assert dones[0]
        to = infos[0]["terminal_observation"]
        assert isinstance(to, np.ndarray) and to.dtype == np.float32
        assert to.shape == (obs_size,)
        assert to[0] == 7.0, "terminal_observation из C++ затёрт текущей obs"
        assert infos[0]["TimeLimit.truncated"] is False

        # (b) C++ не прислал (legacy) — ключа НЕТ, есть флаг. До 2026-09-24
        # здесь подставлялась obs[i] — наблюдка уже НОВОГО эпизода после
        # авто-reset, и V(s_T) на усечении считался по ней (P2-3 ревью).
        env.cpp_vec._wait = fake_wait_b
        obs, rewards, dones, infos = env.step(np.array([0]))
        assert dones[0]
        assert "terminal_observation" not in infos[0]
        assert infos[0]["terminal_observation_missing"] is True
        assert infos[0]["TimeLimit.truncated"] is True
    finally:
        env.close()


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
