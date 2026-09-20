@echo off
setlocal enabledelayedexpansion

rem Project root = folder of this script (was hardcoded to one user's path).
set "ROOT=%~dp0"
if "%ROOT:~-1%"=="\" set "ROOT=%ROOT:~0,-1%"
rem VCVARS may be overridden from the environment (cf. build_gui.bat probing).
if not defined VCVARS set "VCVARS=C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvarsall.bat"
set "OBJDIR=%ROOT%\build\obj"

call "%VCVARS%" x64
if errorlevel 1 (
    echo ERROR: vcvarsall failed
    exit /b 1
)

echo Working dir: %CD%

if not exist "%ROOT%\src\main.cpp" (
    echo ERROR: main.cpp not found
    exit /b 1
)
if not exist "%OBJDIR%" mkdir "%OBJDIR%"

echo === Compiling sources ===
cl.exe /O2 /EHsc /std:c++17 /nologo /c ^
    /I"%ROOT%\include" /I"%ROOT%\include\third_party" ^
    "%ROOT%\src\rng.cpp" ^
    /Fo"%OBJDIR%\rng.obj"
if errorlevel 1 goto :fail

cl.exe /O2 /EHsc /std:c++17 /nologo /c ^
    /I"%ROOT%\include" /I"%ROOT%\include\third_party" ^
    "%ROOT%\src\resources.cpp" ^
    /Fo"%OBJDIR%\resources.obj"
if errorlevel 1 goto :fail

cl.exe /O2 /EHsc /std:c++17 /nologo /c ^
    /I"%ROOT%\include" /I"%ROOT%\include\third_party" ^
    "%ROOT%\src\data.cpp" ^
    /Fo"%OBJDIR%\data.obj"
if errorlevel 1 goto :fail

cl.exe /O2 /EHsc /std:c++17 /nologo /c ^
    /I"%ROOT%\include" /I"%ROOT%\include\third_party" ^
    "%ROOT%\src\earth.cpp" ^
    /Fo"%OBJDIR%\earth.obj"
if errorlevel 1 goto :fail

cl.exe /O2 /EHsc /std:c++17 /nologo /c ^
    /I"%ROOT%\include" /I"%ROOT%\include\third_party" ^
    "%ROOT%\src\game.cpp" ^
    /Fo"%OBJDIR%\game.obj"
if errorlevel 1 goto :fail

cl.exe /O2 /EHsc /std:c++17 /nologo /c ^
    /I"%ROOT%\include" /I"%ROOT%\include\third_party" ^
    "%ROOT%\src\env.cpp" ^
    /Fo"%OBJDIR%\env.obj"
if errorlevel 1 goto :fail

cl.exe /O2 /EHsc /std:c++17 /nologo /c ^
    /I"%ROOT%\include" /I"%ROOT%\include\third_party" ^
    "%ROOT%\src\lot_finder.cpp" ^
    /Fo"%OBJDIR%\lot_finder.obj"
if errorlevel 1 goto :fail

cl.exe /O2 /EHsc /std:c++17 /nologo /c ^
    /I"%ROOT%\include" /I"%ROOT%\include\third_party" ^
    "%ROOT%\src\action_mask.cpp" ^
    /Fo"%OBJDIR%\action_mask.obj"
if errorlevel 1 goto :fail

cl.exe /O2 /EHsc /std:c++17 /nologo /c ^
    /I"%ROOT%\include" /I"%ROOT%\include\third_party" ^
    "%ROOT%\src\main.cpp" ^
    /Fo"%OBJDIR%\main.obj"
if errorlevel 1 goto :fail

echo === Linking ===
cl.exe /nologo ^
    "%OBJDIR%\main.obj" ^
    "%OBJDIR%\rng.obj" ^
    "%OBJDIR%\resources.obj" ^
    "%OBJDIR%\data.obj" ^
    "%OBJDIR%\earth.obj" ^
    "%OBJDIR%\game.obj" ^
    "%OBJDIR%\env.obj" ^
    "%OBJDIR%\lot_finder.obj" ^
    "%OBJDIR%\action_mask.obj" ^
    /Fe"%ROOT%\sakhalin_colony.exe" ^
    /link /SUBSYSTEM:CONSOLE
if errorlevel 1 goto :fail

echo Build OK: %ROOT%\sakhalin_colony.exe
exit /b 0

:fail
echo BUILD FAILED
exit /b 1
