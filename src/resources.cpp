#include "colony/resources.h"

namespace colony {

const char* Sunduk::resource_name(int idx) {
    static const char* names[SUNDUK_SIZE] = {
        "gold", "food", "coal", "iron", "oil", "stone", "water", "wood", "energy"};
    return (idx >= 0 && idx < SUNDUK_SIZE) ? names[idx] : "?";
}

Sunduk Sunduk::from_dict(const std::unordered_map<std::string, int64_t>& d) {
    Sunduk s;
    for (int i = 0; i < SUNDUK_SIZE; i++) {
        auto it = d.find(resource_name(i));
        s.items_[i] = (it != d.end()) ? it->second : 0;
    }
    return s;
}

}  // namespace colony