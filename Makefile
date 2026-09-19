# Sakhalin Colony — команды проекта.
# Кроссплатформенная обёртка; на Windows канонические пути — *.bat
# (build_pyext.bat и т.д.), здесь — то же самое через make.
#
#   make build    — собрать colony_cpp.pyd (CMake, C++17, pybind11)
#   make test     — полный прогон pytest (нужны torch + собранный colony_cpp.pyd)
#   make gui      — PySide6-дашборд обучения
#   make watch    — посмотреть чемпиона: make watch MODEL_DIR=~/colony_runs/models/my_run
#   make tb       — TensorBoard по логам обучения
#   make clean    — почистить build-артефакты и кэши

PY      ?= python
MODEL_DIR ?= ~/colony_runs/models/my_run
LOG_DIR ?= ~/colony_runs/logs

.PHONY: build test test-fast gui watch tb clean

build: ## cmake-сборка C++-среды -> python/colony_cpp.pyd
	$(PY) -m cmake -B build -DCMAKE_BUILD_TYPE=Release
	$(PY) -m cmake --build build --config Release --parallel

test: ## полный прогон тестов
	$(PY) -m pytest

test-fast: ## быстрый дымовой проход (без torch-тяжёлого)
	$(PY) -m pytest tests/test_gae.py tests/test_reward_v3.py tests/test_reward_clip.py -x -q

gui: ## дашборд обучения (PySide6)
	$(PY) run_train_ui2.py

watch: ## визуальный прогон чемпиона
	$(PY) watch_champion.py --model-dir $(MODEL_DIR)

tb: ## TensorBoard
	tensorboard --logdir $(LOG_DIR)

clean: ## build + кэши (не трогаем git-файлы)
	rm -rf build .pytest_cache .mypy_cache .ruff_cache
	find . -maxdepth 2 -name __pycache__ -type d -exec rm -rf {} +
