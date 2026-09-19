#include "colony/earth.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace colony {

namespace {

// np.linspace(0, n-1, size)
std::vector<double> linspace(int n, int size) {
    std::vector<double> xs(size);
    if (size == 1) {
        xs[0] = 0.0;
        return xs;
    }
    double step = (double)(n - 1) / (double)(size - 1);
    for (int i = 0; i < size; i++) xs[i] = (double)i * step;
    return xs;
}

// Знаковый шум: случайная сетка + билинейная интерполяция со сглаживанием.
// Точный порядок float-операций как в numpy-версии.
std::vector<double> value_noise(int size, MtRandom& rng, int coarse) {
    int n = size / coarse + 2;
    std::vector<double> grid((size_t)n * n);
    for (int i = 0; i < n * n; i++) grid[i] = rng.random();

    std::vector<double> xs = linspace(n, size);
    std::vector<int> gx(size);
    std::vector<double> fx(size);
    for (int i = 0; i < size; i++) {
        gx[i] = (int)std::floor(xs[i]);
        if (gx[i] < 0) gx[i] = 0;
        if (gx[i] > n - 2) gx[i] = n - 2;
        fx[i] = xs[i] - (double)gx[i];
        fx[i] = fx[i] * fx[i] * (3 - 2 * fx[i]);
    }

    std::vector<double> out((size_t)size * size);
    for (int y = 0; y < size; y++) {
        int gy = (size > 1) ? (int)((double)(y * (n - 1)) / (double)(size - 1)) : 0;
        if (gy < 0) gy = 0;
        if (gy > n - 2) gy = n - 2;
        double fy = (double)(y * (n - 1)) / (double)(size - 1) - (double)gy;
        fy = fy * fy * (3 - 2 * fy);
        double* row = &out[(size_t)y * size];
        for (int x = 0; x < size; x++) {
            double a = grid[(size_t)gy * n + gx[x]] * (1 - fx[x]) +
                       grid[(size_t)gy * n + gx[x] + 1] * fx[x];
            double b = grid[(size_t)(gy + 1) * n + gx[x]] * (1 - fx[x]) +
                       grid[(size_t)(gy + 1) * n + gx[x] + 1] * fx[x];
            row[x] = a * (1 - fy) + b * fy;
        }
    }
    return out;
}

}  // namespace

Earth::Earth(uint64_t seed, int size) : size_(size), seed_(seed) {
    if (size < 40)
        throw std::invalid_argument("Earth: map size must be >= 40 for current generator");
    lots_.assign((size_t)size * size, 0);
    subtype_.assign((size_t)size * size, 0);
    init_sel_x = size / 2;
    init_sel_y = size / 2;
    generate();
}

void Earth::generate() {
    MtRandom rng((int64_t)seed_);
    int size = size_;
    int coarse = std::max(8, size / 16);
    std::vector<double> noise = value_noise(size, rng, coarse);

    // Остров: радиальное затухание от центра + шум.
    double cy = size / 2.0;
    double cx = size / 2.0;
    std::vector<double> dist((size_t)size * size);
    std::vector<double> score((size_t)size * size);
    double denom = size * 0.62;
    for (int y = 0; y < size; y++) {
        for (int x = 0; x < size; x++) {
            double ddx = (double)x - cx;
            double ddy = (double)y - cy;
            // numpy (xx-cx)**2 использует pow(); на MSVC pow(x,2) может
            // отличаться от x*x на 1 ulp — важно для бит-в-бит эквивалентности
            double d = std::sqrt(std::pow(ddx, 2.0) + std::pow(ddy, 2.0)) / denom;
            dist[(size_t)y * size + x] = d;
            score[(size_t)y * size + x] = noise[(size_t)y * size + x] * (1.0 - 0.55 * d);
        }
    }
    double threshold = 0.30;
    std::vector<char> land((size_t)size * size);
    for (int iter = 0; iter < 5; iter++) {
        long long cnt = 0;
        for (int i = 0; i < size * size; i++) {
            if (score[i] > threshold) {
                land[i] = 1;
                cnt++;
            } else {
                land[i] = 0;
            }
        }
        if ((double)cnt / (double)(size * size) >= 0.35) break;
        threshold -= 0.03;
    }
    for (int i = 0; i < size * size; i++) lots_[i] = land[i] ? LT_NORMAL : LT_NONE;

    // Озёра.
    int lakes = (int)rng.randint(2, 5);  // randint(2,5) -> 2..5
    for (int i = 0; i < lakes; i++) {
        int px = (int)rng.randint(10, size - 11);
        int py = (int)rng.randint(10, size - 11);
        if (lots_[(size_t)py * size + px] != LT_NORMAL) continue;
        carve(px, py, (int)rng.randint(6, 16), LT_WATER, rng);
    }

    // Полезные ископаемые: леса, уголь, железо, нефть, золото.
    struct Deposit { int type, count, rmin, rmax; };
    static const Deposit deposits[] = {
        {LT_WOOD, 10, 15, 35}, {LT_COAL, 8, 8, 18}, {LT_IRON, 6, 6, 14},
        {LT_OIL, 4, 4, 9}, {LT_GOLD, 3, 2, 5},
    };
    for (const Deposit& dep : deposits) {
        for (int c = 0; c < dep.count; c++) {
            int px = (int)rng.randint(5, size - 6);
            int py = (int)rng.randint(5, size - 6);
            carve_soft(px, py, (int)rng.randint(dep.rmin, dep.rmax), dep.type, rng);
        }
    }

    // Гарантированные ресурсы возле старта.
    // ВАЖНО: порядок вычисления аргументов функции в C++ не определён (MSVC —
    // справа налево), поэтому рисуем случайные смещения в именованные переменные
    // в нужном порядке (x раньше y, как в Python-эталоне слева направо).
    int cx0 = size / 2, cy0 = size / 2;
    int wx = cx0 + (int)rng.randint(-12, 12);
    int wy = cy0 + (int)rng.randint(-12, 12);
    carve(wx, wy, 6, LT_WATER, rng);
    int wdx = cx0 + (int)rng.randint(-14, 14);
    int wdy = cy0 + (int)rng.randint(-14, 14);
    carve_soft(wdx, wdy, 10, LT_WOOD, rng);
    int cdx = cx0 + (int)rng.randint(-14, 14);
    int cdy = cy0 + (int)rng.randint(-14, 14);
    carve_soft(cdx, cdy, 6, LT_COAL, rng);
    // Добавляем железо, нефть и золото возле старта (перезаписываем существующие ресурсы)
    int idx = cx0 + (int)rng.randint(-16, 16);
    int idy = cy0 + (int)rng.randint(-16, 16);
    carve_resource(idx, idy, 8, LT_IRON, rng);
    int oix = cx0 + (int)rng.randint(-16, 16);
    int oiy = cy0 + (int)rng.randint(-16, 16);
    carve_resource(oix, oiy, 6, LT_OIL, rng);
    int gdx = cx0 + (int)rng.randint(-18, 18);
    int gdy = cy0 + (int)rng.randint(-18, 18);
    carve_resource(gdx, gdy, 4, LT_GOLD, rng);

    // Гарантированно суша в центре (после всех вырезаний).
    // Python: lots[size/2-6 : size/2+7, ...] -> строки/столбцы 134..146 (13 шт)
    const int clear_y0 = std::max(0, size / 2 - 6);
    const int clear_y1 = std::min(size, size / 2 + 7);
    const int clear_x0 = std::max(0, size / 2 - 6);
    const int clear_x1 = std::min(size, size / 2 + 7);
    for (int y = clear_y0; y < clear_y1; y++) {
        for (int x = clear_x0; x < clear_x1; x++) {
            lots_[(size_t)y * size + x] = LT_NORMAL;
        }
    }

    // Подтипы (варианты текстур) для графики.
    for (int y = 0; y < size; y++) {
        for (int x = 0; x < size; x++) {
            if (lots_[(size_t)y * size + x] != LT_NONE) {
                subtype_[(size_t)y * size + x] = (int8_t)rng.randint(0, 3);
            }
        }
    }
}

void Earth::carve(int cx0, int cy0, int radius, int lot_type, MtRandom& rngr) {
    int size = size_;
    int y0 = std::max(0, cy0 - radius), y1 = std::min(size, cy0 + radius + 1);
    int x0 = std::max(0, cx0 - radius), x1 = std::min(size, cx0 + radius + 1);
    for (int y = y0; y < y1; y++) {
        for (int x = x0; x < x1; x++) {
            double u = rngr.uniform(0.6, 1.0);
            // python u ** 2 — это pow(u, 2.0), на MSVC может отличаться от u*u
            double r2 = (double)(radius * radius) * std::pow(u, 2.0);
            double dx = (double)x - cx0, dy = (double)y - cy0;
            if (dx * dx + dy * dy <= r2) {
                if (lots_[(size_t)y * size + x] == LT_NORMAL) {
                    lots_[(size_t)y * size + x] = (int8_t)lot_type;
                }
            }
        }
    }
}

void Earth::carve_soft(int cx0, int cy0, int radius, int lot_type, MtRandom& rngr) {
    int size = size_;
    int y0 = std::max(0, cy0 - radius), y1 = std::min(size, cy0 + radius + 1);
    int x0 = std::max(0, cx0 - radius), x1 = std::min(size, cx0 + radius + 1);
    int r2 = std::max(1, radius * radius);
    for (int y = y0; y < y1; y++) {
        for (int x = x0; x < x1; x++) {
            double dx = (double)x - cx0, dy = (double)y - cy0;
            double d2 = (dx * dx + dy * dy) / (double)r2;
            if (rngr.random() > d2 * 0.85) {
                if (lots_[(size_t)y * size + x] == LT_NORMAL) {
                    lots_[(size_t)y * size + x] = (int8_t)lot_type;
                }
            }
        }
    }
}

void Earth::carve_resource(int cx0, int cy0, int radius, int lot_type, MtRandom& rngr) {
    int size = size_;
    int y0 = std::max(0, cy0 - radius), y1 = std::min(size, cy0 + radius + 1);
    int x0 = std::max(0, cx0 - radius), x1 = std::min(size, cx0 + radius + 1);
    int r2 = std::max(1, radius * radius);
    for (int y = y0; y < y1; y++) {
        for (int x = x0; x < x1; x++) {
            double dx = (double)x - cx0, dy = (double)y - cy0;
            double d2 = (dx * dx + dy * dy) / (double)r2;
            if (rngr.random() > d2 * 0.85) {
                int8_t cur = lots_[(size_t)y * size + x];
                // Allow overwriting any resource except WATER and NONE
                if (cur != LT_WATER && cur != LT_NONE) {
                    lots_[(size_t)y * size + x] = (int8_t)lot_type;
                }
            }
        }
    }
}

}  // namespace colony