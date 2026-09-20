#!/usr/bin/env bash
# Компиляция и прогон C++-проб (tests/cpp) БЕЗ Python и pybind11.
#
# Зачем: ядро среды (src/*.cpp кроме bindings.cpp) не зависит от Python, поэтому
# проверки поведения можно гонять голым g++ — это самый быстрый сигнал в CI и
# единственный способ проверить среду там, где нет заголовков CPython.
#
# Три прохода:
#   1) объекты ядра (один раз, дальше только линковка);
#   2) compile-only всех проб — поломка сборки видна даже у проб-замеров;
#   3) прогон проверок с кодом возврата: curriculum_check, reward_regressions,
#      road_direction_check, water_mask_check, p0_p1_check, gui_watch_check,
#      reward_v4_longrun. Проба-замер (water_probe*, *_probe, longrun_check, ...)
#      всегда возвращает 0, поэтому здесь только компилируется.
#
# Запуск:  ./scripts/cpp_checks.sh          (из любого места: cd в корень сам)
#          JOBS=4 CXX=clang++ REWARD_STEPS=1500 ./scripts/cpp_checks.sh
set -euo pipefail

cd "$(dirname "$0")/.."

CXX=${CXX:-g++}
OUT=${OUT:-build/cpp_checks}
REWARD_STEPS=${REWARD_STEPS:-4000}
CXXFLAGS="-std=c++17 -O1 -Wall -Iinclude -Iinclude/third_party"

# Общие объекты ядра (bindings.cpp не нужен — он про pybind11).
SHARED_SRC="src/rng.cpp src/resources.cpp src/data.cpp src/earth.cpp src/game.cpp src/rewards.cpp src/env.cpp src/lot_finder.cpp src/action_mask.cpp"

# Проверки с кодом возврата (упала — CI красный).
CHECKS="curriculum_check reward_regressions road_direction_check water_mask_check p0_p1_check gui_watch_check reward_v4_longrun"

mkdir -p "$OUT"

echo "── [1/3] объекты ядра ────────────────────────────────────────────"
OBJS=""
for f in $SHARED_SRC; do
    o="$OUT/$(basename "$f" .cpp).o"
    echo "  CXX $f"
    # shellcheck disable=SC2086
    $CXX $CXXFLAGS -c "$f" -o "$o"
    OBJS="$OBJS $o"
done

echo "── [2/3] компиляция всех проб (compile-only) ─────────────────────"
for f in tests/cpp/*.cpp; do
    name=$(basename "$f" .cpp)
    echo "  CXX $f"
    # shellcheck disable=SC2086
    $CXX $CXXFLAGS "$f" $OBJS -o "$OUT/$name"
done

echo "── [3/3] прогон проверок ─────────────────────────────────────────"
failed=0
for name in $CHECKS; do
    echo ""
    echo "=== $name ==="
    if [ "$name" = "reward_v4_longrun" ]; then
        "$OUT/$name" "$REWARD_STEPS" || failed=$((failed + 1))
    else
        "$OUT/$name" || failed=$((failed + 1))
    fi
done

echo ""
if [ "$failed" -ne 0 ]; then
    echo "CPP CHECKS: $failed проверка(ок) упала"
    exit 1
fi
echo "CPP CHECKS: все проверки прошли"
