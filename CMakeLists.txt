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

# выходной файл .pyd кладём в python/ (рядом с обёрткой)
set_target_properties(colony_cpp PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/python"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/python"
)

# на Windows pybind11-модуль должен иметь расширение .pyd
if(WIN32)
    set_target_properties(colony_cpp PROPERTIES SUFFIX ".pyd")
endif()