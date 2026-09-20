// Бенчмарк action_mask(): сколько стоит пересчёт маски (P2-11).
//
// action_mask() вызывается на КАЖДЫЙ шаг каждой среды: при n_envs=1024 это
// 1024 вызова на шаг обучения, а внутри — по одному BFS find_lot() на каждую из
// 32 построек. Кеш по (need_earth, no_near_base) убирает повторные BFS внутри
// одного вызова; этот файл меряет эффект.
//
// Замер в трёх состояниях: старт эпизода, 50 дорог, 200 дорог (чем больше баз,
// тем дороже BFS — тем заметнее кеш).
//
// Build (from the repo root, Linux/macOS) — одна строка, без продолжений:
//   g++ -std=c++17 -O2 -Iinclude -Iinclude/third_party -o /tmp/am_bench
//       tests/cpp/action_mask_bench.cpp src/env.cpp src/data.cpp
//       src/resources.cpp src/game.cpp src/rewards.cpp src/rng.cpp src/earth.cpp
//   /tmp/am_bench [calls_per_state]
#include "colony/env.h"
#include "colony/data.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace colony;

static int find_idx(ColonyEnvCpp& e, const std::string& id) {
    for (int i = 0; i < e.n_build(); ++i)
        if (e.build_ids()[(size_t)i] == id) return i;
    return -1;
}

static int road_count(const ColonyEnvCpp& e) {
    int n = 0;
    for (const Base& b : e.game().bases)
        if (b.data->id == ROAD_ID) ++n;
    return n;
}

// Достроить дорог до target (обычный BUILD:Road — клетку выбирает среда).
static void build_roads(ColonyEnvCpp& e, int target) {
    const int road = find_idx(e, "Road");
    int guard = 0;
    while (road_count(e) < target && guard++ < 5000) {
        auto m = e.action_mask();
        if (m[A_BUILD0 + road] == 0.0f) { e.step(A_DAY); continue; }
        auto out = e.step(A_BUILD0 + road);
        if (out.terminated || out.truncated) break;
    }
}

static double bench_mask_us(ColonyEnvCpp& e, int calls) {
    // прогрев
    for (int i = 0; i < 5; ++i) (void)e.action_mask();
    const auto t0 = std::chrono::steady_clock::now();
    volatile size_t sink = 0;
    for (int i = 0; i < calls; ++i) {
        auto m = e.action_mask();
        sink += m.size();
    }
    const auto t1 = std::chrono::steady_clock::now();
    (void)sink;
    return std::chrono::duration<double, std::micro>(t1 - t0).count() / calls;
}

int main(int argc, char** argv) {
    const int calls = argc > 1 ? std::atoi(argv[1]) : 2000;
    auto bd = load_base_data("configs/bases.json");
    auto ed = load_events("configs/events.json");

    printf("action_mask(): %d вызовов на состояние, -O2\n", calls);
    printf("%-18s %-8s %-12s %-14s %-14s\n",
           "state", "roads", "us/call", "calls/sec", "us/step@1024envs");
    double total = 0.0;
    int n = 0;
    struct State { int roads; bool rich; const char* label; };
    // «200 дорог без денег» — вырожденное состояние: все BUILD отсекаются
    // проверкой денег до BFS, остаётся стоимость четырёх find_lot_dir().
    // «200 дорог + деньги» — реалистичный поздний эпизод: BFS выполняется для
    // каждой доступной постройки, и кеш по (need_earth, no_near_base) решает.
    for (State st : {State{0, false, "reset"}, State{50, true, "50 roads"},
                     State{200, false, "200 roads broke"},
                     State{200, true, "200 roads + money"}}) {
        ColonyEnvCpp e(bd, ed, 42, 200);
        e.reset(42);
        if (st.roads > 0) build_roads(e, st.roads);
        if (st.rich) e.game().money = 5000000;
        double us = bench_mask_us(e, calls);
        total += us;
        n++;
        int open_builds = 0;
        auto m = e.action_mask();
        for (int i = 0; i < e.n_build(); ++i)
            if (m[A_BUILD0 + i] != 0.0f) ++open_builds;
        printf("%-18s %-8d %-12.2f %-14.0f %-14.1f  build-действий открыто: %d\n",
               st.label, road_count(e), us, 1e6 / us, us * 1024.0 / 1000.0, open_builds);
    }
    printf("среднее: %.2f мкс/вызов\n", total / n);
    return 0;
}
