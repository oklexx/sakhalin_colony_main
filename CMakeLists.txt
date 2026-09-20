cmake_minimum_required(VERSION 3.20)
project(colony_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# путь к pybind11: предпочитаем pip-пакет (python -m pybind11 --includes)
execute_process(
    COMMAND python -c "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
list(APPEND CMAKE_PREFIX_PATH "${PYBIND11_CMAKE_DIR}")
find_package(pybind11 REQUIRED)

add_library(colony_cpp MODULE
    src/rng.cpp
    src/resources.cpp
    src/data.cpp
    src/earth.cpp
    src/game.cpp
    src/rewards.cpp
    src/env.cpp
    src/lot_finder.cpp
    src/action_mask.cpp
    src/observation.cpp
    src/vec_env.cpp
    src/bindings.cpp
)

target_include_directories(colony_cpp PRIVATE include include/third_party)
target_link_libraries(colony_cpp PRIVATE pybind11::module)

target_compile_definitions(colony_cpp PRIVATE VERSION_INFO="1.0")

if(MSVC)
    target_compile_options(colony_cpp PRIVATE /O2 /EHsc /W3)
else()
    target_compile_options(colony_cpp PRIVATE -O2 -Wall)
endif()

# Git SHA вшивается в расширение, чтобы handshake (python/colony_cpp_api.py)
# мог показать, из каких исходников собран бинарь. Вне git-дерева — "unknown".
execute_process(
    COMMAND git rev-parse --short HEAD
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE COLONY_GIT_SHA
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)
if(NOT COLONY_GIT_SHA)
    set(COLONY_GIT_SHA "unknown")
endif()
target_compile_definitions(colony_cpp PRIVATE "COLONY_GIT_SHA=\"${COLONY_GIT_SHA}\"")

# выходной файл .pyd кладём в python/ (рядом с обёрткой).
# Мультиконфиг-генераторы (Visual Studio) дописывают подкаталог $(Configuration)
# к ОБЩЕМУ свойству — поэтому задаём и per-config варианты: тогда выход всегда
# один (python/colony_cpp.pyd), каталога python/Release/ не возникает, и
# build_pyext.bat нечего копировать.
set_target_properties(colony_cpp PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/python"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/python"
)
foreach(_cfg DEBUG RELEASE MINSIZEREL RELWITHDEBINFO)
    set_target_properties(colony_cpp PROPERTIES
        "LIBRARY_OUTPUT_DIRECTORY_${_cfg}" "${CMAKE_SOURCE_DIR}/python"
        "RUNTIME_OUTPUT_DIRECTORY_${_cfg}" "${CMAKE_SOURCE_DIR}/python"
    )
endforeach()

# на Windows pybind11-модуль должен иметь расширение .pyd
if(WIN32)
    set_target_properties(colony_cpp PROPERTIES SUFFIX ".pyd")
else()
    # На Linux/macOS MODULE по умолчанию даёт libcolony_cpp.so, который
    # `import colony_cpp` не находит. Убираем префикс и ставим суффикс
    # интерпретатора: colony_cpp.cpython-3XX-....so — имя импортируемое.
    execute_process(
        COMMAND python -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))"
        OUTPUT_VARIABLE PYTHON_EXT_SUFFIX
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(PYTHON_EXT_SUFFIX)
        set_target_properties(colony_cpp PROPERTIES PREFIX "" SUFFIX "${PYTHON_EXT_SUFFIX}")
    else()
        set_target_properties(colony_cpp PROPERTIES PREFIX "")
    endif()
endif()