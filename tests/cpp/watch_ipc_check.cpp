// P3-4 ревью 2026-09-24: чтение actions.txt окном наблюдения (colony/watch_ipc.h).
//
// Регрессия «тихого зависания»: файл с -1 (или числом ≥ n_actions) раньше не
// удалялся — окно каждый кадр видело невалидное действие, драйвер вечно ждал
// state.json. Теперь такой файл потребляется (INVALID), а недописанный —
// по-прежнему остаётся до следующего кадра (NONE).
//
// Build/run: ./scripts/cpp_checks.sh (или вручную:
//   g++ -std=c++17 -Iinclude -o /tmp/watch_ipc_check tests/cpp/watch_ipc_check.cpp)

#include "colony/watch_ipc.h"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>

using namespace colony;

static int g_fail = 0, g_pass = 0;

static void check(const std::string& name, bool ok) {
    (ok ? g_pass : g_fail)++;
    printf("[%s] %s\n", ok ? "PASS" : "FAIL", name.c_str());
}

static void write(const std::string& path, const std::string& text) {
    std::ofstream f(path, std::ios::trunc);
    f << text;
}

int main() {
    namespace fs = std::filesystem;
    const fs::path dir = fs::temp_directory_path() / "colony_watch_ipc_check";
    fs::create_directories(dir);
    const std::string p = (dir / "actions.txt").string();
    const int N = 49;

    fs::remove(p);
    check("A. нет файла → NONE", read_action_file(p, N).kind == ActionRead::NONE);

    write(p, "");
    check("B. пустой файл → NONE", read_action_file(p, N).kind == ActionRead::NONE);
    check("B. пустой файл НЕ удалён (Python допишет)", fs::exists(p));

    write(p, "17\n");
    auto r = read_action_file(p, N);
    check("C. валидное → OK 17", r.kind == ActionRead::OK && r.action == 17);
    check("C. валидное → файл удалён", !fs::exists(p));

    write(p, "-1");
    r = read_action_file(p, N);
    check("D. -1 → INVALID", r.kind == ActionRead::INVALID && r.action == -1);
    check("D. -1 → файл потреблён (нет вечного ожидания)", !fs::exists(p));

    write(p, "49");
    r = read_action_file(p, N);
    check("E. 49 при n_actions=49 → INVALID", r.kind == ActionRead::INVALID && r.action == 49);
    check("E. файл потреблён", !fs::exists(p));

    write(p, "48");
    check("F. граница 48 → OK", read_action_file(p, N).kind == ActionRead::OK);

    check("G. пустой путь → NONE", read_action_file("", N).kind == ActionRead::NONE);

    fs::remove_all(dir);
    printf("\nwatch_ipc_check: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
