#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "colony/resources.h"
#include "colony/rng.h"

namespace colony {

// Случайное событие (аналог TBaseEvent)
class BaseEvent {
public:
    std::string id;
    std::string target;
    std::string message;
    int64_t live_years = 0;
    int64_t live_years_range = 0;
    int64_t people = 0;
    int64_t people_range = 0;
    Sunduk sunduk;
    Sunduk sunduk_range;

    // (live_years_delta, people_delta, sunduk_delta)
    struct ExecResult {
        int64_t ly, p;
        Sunduk s;
    };
    ExecResult execute(PCG64& rng) const {
        ExecResult r;
        r.ly = random_range_values(live_years, live_years_range, rng);
        r.p = random_range_values(people, people_range, rng);
        r.s = sunduk.copy();
        r.s.random_range(sunduk_range, rng);
        return r;
    }
};

// Выбор события для постройки: N + 20 пустых слотов (как в оригинале).
inline const BaseEvent* choice_event(const std::vector<BaseEvent>& events,
                                     const std::string& base_id, PCG64& rng) {
    int slots = 0;
    for (const BaseEvent& e : events)
        if (e.target == base_id) slots++;
    if (slots < EVENT_BLANK_SLOTS) slots = EVENT_BLANK_SLOTS;
    uint64_t i = rng.randrange((uint64_t)slots);
    if (i < static_cast<uint64_t>(slots)) {
        int idx = 0;
        for (const BaseEvent& e : events) {
            if (e.target == base_id) {
                if ((uint64_t)idx == i) return &e;
                idx++;
            }
        }
    }
    return nullptr;
}

}  // namespace colony