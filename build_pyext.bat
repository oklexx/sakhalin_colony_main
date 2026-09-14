@echo off
setlocal

rem Project root = folder of this script (was hardcoded to one user's path,
rem which broke the build on every other checkout).
set "ROOT=%~dp0"
if "%ROOT:~-1%"=="\" set "ROOT=%ROOT:~0,-1%"

rem Python: COLONY_PY or PY env var wins, else `python` from PATH
rem (was hardcoded to C:\Python314\python.exe).
if not defined PY set "PY=python"
if defined COLONY_PY set "PY=%COLONY_PY%"

rem CMake generator: override with COLONY_GENERATOR if your VS version differs.
if not defined COLONY_GENERATOR set "COLONY_GENERATOR=Visual Studio 18 2026"

rem Needs the pip `cmake` package (see requirements.txt) and pybind11.
if not exist "%ROOT%\build\py" mkdir "%ROOT%\build\py"
cd /d "%ROOT%\build\py"

echo === CMake Configure ===
%PY% -m cmake "%ROOT%" -G "%COLONY_GENERATOR%" -A x64
if errorlevel 1 goto :fail

echo === CMake Build ===
%PY% -m cmake --build . --config Release
if errorlevel 1 goto :fail

rem CMakeLists emits directly to python\ (no python\Release\ subdir), so there
rem is nothing to copy — just verify the module exists and answers the handshake.
echo === Verify ===
if not exist "%ROOT%\python\colony_cpp.pyd" (
    echo ERROR: "%ROOT%\python\colony_cpp.pyd" was not produced - see the build log above.
    goto :fail
)
%PY% -c "import sys; sys.path.insert(0, r'%ROOT%\python'); import colony_cpp; print(colony_cpp.extension_info())"
if errorlevel 1 goto :fail

echo BUILD OK
exit /b 0

:fail
echo BUILD FAILED
exit /b 1
