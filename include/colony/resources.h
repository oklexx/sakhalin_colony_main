#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "colony/constants.h"
#include "colony/rng.h"

namespace colony {

// value ± range: value + range - random(2*range + 1)
// (при rng_range == 0 результат == value — семантика идентична Python)
inline int64_t random_range_values(int64_t value, int64_t rng_range, PCG64& rng) {
    if (rng_range == 0) return value;
    return value + rng_range - (int64_t)rng.randrange((uint64_t)(rng_range * 2 + 1));
}

// 9 ресурсов (аналог TSunduk)
class Sunduk {
public:
    Sunduk() { items_.fill(0); }

    static Sunduk from_dict(const std::unordered_map<std::string, int64_t>& d);

    // from python: list of 9 ints
    static Sunduk from_items(const std::array<int64_t, SUNDUK_SIZE>& items) {
        Sunduk s;
        s.items_ = items;
        return s;
    }

    std::array<int64_t, SUNDUK_SIZE> items() const { return items_; }
    int64_t& operator[](int i) { return items_[i]; }
    int64_t operator[](int i) const { return items_[i]; }

    Sunduk copy() const { return *this; }

    void add(const Sunduk& other) {
        for (int i = 0; i < SUNDUK_SIZE; i++) items_[i] += other.items_[i];
    }
    void remove(const Sunduk& other) {
        for (int i = 0; i < SUNDUK_SIZE; i++) items_[i] -= other.items_[i];
    }
    void mul(int64_t value) {
        for (int i = 0; i < SUNDUK_SIZE; i++) items_[i] *= value;
    }
    bool include(const Sunduk& other) const {
        for (int i = 0; i < SUNDUK_SIZE; i++)
            if (other.items_[i] > items_[i]) return false;
        return true;
    }
    bool non_negative() const {
        for (int i = 0; i < SUNDUK_SIZE; i++)
            if (items_[i] < 0) return false;
        return true;
    }
    bool can_consume(const Sunduk& other) const {
        for (int i = 0; i < SUNDUK_SIZE; i++)
            if (other.items_[i] < 0 && -other.items_[i] > items_[i]) return false;
        return true;
    }
    void random_range(const Sunduk& other, PCG64& rng) {
        for (int i = 0; i < SUNDUK_SIZE; i++) {
            items_[i] = random_range_values(items_[i], other.items_[i], rng);
        }
    }
    void clear() { items_.fill(0); }
    bool empty() const {
        for (int i = 0; i < SUNDUK_SIZE; i++)
            if (items_[i] != 0) return false;
        return true;
    }
    int64_t total_value_sale() const {
        int64_t v = 0;
        for (int i = 0; i < SUNDUK_SIZE; i++) v += items_[i] * SALE_SUNDUK[i];
        return v;
    }
    bool operator==(const Sunduk& o) const { return items_ == o.items_; }

    // PR 4: каноническое имя ресурса по индексу сундука (0..8).
    // Порядок обязан совпадать с rl/curriculum.py RESOURCE_NAMES и
    // train_ui2/constants.py RESOURCE_IDS (регресс-тест в Python).
    static const char* resource_name(int idx);

private:
    std::array<int64_t, SUNDUK_SIZE> items_;
};

}  // namespace colony