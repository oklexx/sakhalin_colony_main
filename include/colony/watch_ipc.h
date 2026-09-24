// IPC наблюдения (watch_champion.py ↔ src/gui.cpp --headless-ai): чтение actions.txt.
//
// Вынесено из gui.cpp (P3-4 ревью 2026-09-24), чтобы логику можно было
// проверить без raylib: tests/cpp/watch_ipc_check.cpp в scripts/cpp_checks.sh.
//
// Три исхода чтения — и у каждого своя реакция окна:
//   * NONE     — файла нет или он пустой/недописанный (число не разобралось).
//                Файл НЕ удаляем: Python пишет атомарно (tmp + rename), и
//                следующий кадр прочитает его целиком. Удаление здесь значило
//                бы «прочитано» и вечное ожидание с обеих сторон (2026-09-20).
//   * OK       — действие в 0..n_actions-1; файл удалён (= «забрал»).
//   * INVALID  — число разобралось, но вне диапазона (-1 из отладки, чужой
//                процесс, остаток старого прогона). Раньше такой файл не
//                удалялся, окно каждый кадр видело −1, драйвер вечно ждал
//                state.json — «тихое зависание». Теперь файл считается
//                потреблённым, а окно отвечает state.json с полем "error".
#pragma once

#include <filesystem>
#include <fstream>
#include <string>
#include <system_error>

namespace colony {

enum class ActionRead { NONE, OK, INVALID };

struct ActionReadResult {
    ActionRead kind = ActionRead::NONE;
    int action = -1;  // разобранное значение (для INVALID — что именно пришло)
};

inline ActionReadResult read_action_file(const std::string& path, int n_actions) {
    ActionReadResult r;
    if (path.empty()) return r;
    {
        std::ifstream f(path);
        if (!f.is_open()) return r;
        int action = -1;
        if (!(f >> action)) return r;  // пустой/недописанный — ждём следующий кадр
        r.action = action;
    }
    r.kind = (r.action >= 0 && r.action < n_actions) ? ActionRead::OK : ActionRead::INVALID;
    // Удаляем и OK, и INVALID: для драйвера это «окно забрало действие».
    // remove(), а не усечение: Python может успеть записать новое действие
    // между close() и открытием на запись. error_code-версия: на Windows
    // remove бросает, если файл сейчас открыт другим процессом.
    std::error_code ec;
    std::filesystem::remove(path, ec);
    return r;
}

}  // namespace colony
