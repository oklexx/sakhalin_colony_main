// Regression checks for defects found during the October 2026 full review.
// This probe deliberately exercises Game copies directly, without Python.

#include "colony/data.h"
#include "colony/game.h"

#include <cstdio>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

using namespace colony;

static int failed = 0;
static int passed = 0;

static void check(const char* name, bool ok, const std::string& detail = {}) {
    if (ok) {
        ++passed;
        std::printf("[PASS] %s\n", name);
    } else {
        ++failed;
        std::printf("[FAIL] %s%s\n", name,
                    detail.empty() ? "" : (" — " + detail).c_str());
    }
}

static const BaseData& data_for(const Game& game, const std::string& id) {
    for (const BaseData& data : game.base_data()) {
        if (data.id == id) return data;
    }
    throw std::runtime_error("missing test building " + id);
}

static Base make_finished(const BaseData& data, int x, int y) {
    Base base(&data, x, y, true);
    base.live_time = data.live_time_total();
    return base;
}

int main() {
    const auto base_data = load_base_data("configs/bases.json");
    const auto events_data = load_events("configs/events.json");

    std::printf("==== copy/undo state preservation ====\n");
    {
        Game original(base_data, events_data, 17, 200);
        // Prime the target's derived cache, then assign a state with an extra
        // finished House. operator= must invalidate the target cache.
        const int64_t before = original.now_home_places();
        Game target(base_data, events_data, 18, 200);
        (void)target.now_home_places();
        Game source = original;
        source.bases.push_back(make_finished(data_for(source, "House"), 1, 1));
        target = source;
        check("Game::operator= invalidates capacity cache",
              target.now_home_places() == before + 60,
              "expected " + std::to_string(before + 60) +
                  ", got " + std::to_string(target.now_home_places()));
    }

    {
        Game game(base_data, events_data, 19, 200);
        const BaseData& house = data_for(game, "House");
        // City + four houses crosses the first five-building milestone.
        for (int i = 0; i < 4; ++i)
            game.bases.push_back(make_finished(house, i + 1, 1));
        check("first base milestone is paid once", game.check_milestones(10.0, 0.0, 0.0, 0.0) == 10.0);

        Game snapshot = game;
        Game assigned(base_data, events_data, 20, 200);
        assigned = game;
        check("copy constructor preserves milestone progress",
              snapshot.check_milestones(10.0, 0.0, 0.0, 0.0) == 0.0);
        check("copy assignment preserves milestone progress",
              assigned.check_milestones(10.0, 0.0, 0.0, 0.0) == 0.0);
    }

    std::printf("==== partial restore accounting ====\n");
    {
        Game game(base_data, events_data, 21, 200);
        const BaseData& farm = data_for(game, "Farm");
        Base damaged = make_finished(farm, 1, 1);
        damaged.live_time = farm.live_time_total() - 3;
        game.bases.push_back(damaged);
        game.money = 1;  // Farm repair costs 1 per day; only one day is affordable.

        const Game::RestoreOut result = game.restore_all();
        check("partial restore succeeds when one repair day is affordable", result.ok);
        check("partial restore reports the actual amount spent", result.price == 1,
              "price=" + std::to_string(result.price));
        check("partial restore reports repaired building count", result.days == 1,
              "days=" + std::to_string(result.days));
        check("partial restore debits exactly the reported amount", game.money == 0);
    }

    std::printf("==== %d passed, %d failed ====\n", passed, failed);
    return failed == 0 ? 0 : 1;
}
