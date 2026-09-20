@echo off
setlocal EnableDelayedExpansion

rem Project root = folder of this script (was hardcoded to one user's path,
rem which broke the build on every other checkout).
set "ROOT=%~dp0"
if "%ROOT:~-1%"=="\" set "ROOT=%ROOT:~0,-1%"
set "OBJDIR=%ROOT%\build\gui_obj"

rem --- raylib: env var COLONY_RAYLIB_DIR wins, else <ROOT>\raylib\raylib-6.0_win64_msvc16 ---
if not defined COLONY_RAYLIB_DIR set "COLONY_RAYLIB_DIR=%ROOT%\raylib\raylib-6.0_win64_msvc16"
set "RLDIR=%COLONY_RAYLIB_DIR%"
if not exist "%RLDIR%\lib\raylib.lib" (
    echo ERROR: raylib not found at "%RLDIR%"
    echo   Download https://github.com/raysan5/raylib/releases ^(raylib-6.0_win64_msvc16.zip^)
    echo   and unpack into "%ROOT%\raylib\", or set COLONY_RAYLIB_DIR.
    exit /b 1
)

rem --- Visual Studio: use VCVARS env var, else probe common locations via vswhere ---
if not defined VCVARS (
    set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
    if exist "!VSWHERE!" (
        for /f "usebackq tokens=*" %%i in (`"!VSWHERE!" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VSPATH=%%i"
    )
    if defined VSPATH set "VCVARS=!VSPATH!\VC\Auxiliary\Build\vcvarsall.bat"
)
if not defined VCVARS (
    for %%V in (
        "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvarsall.bat"
        "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat"
        "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvarsall.bat"
        "C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat"
        "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvarsall.bat"
        "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvarsall.bat"
    ) do if exist %%V set "VCVARS=%%~V"
)
if not defined VCVARS (
    echo ERROR: vcvarsall.bat not found. Install Visual Studio Build Tools
    echo   or set VCVARS=C:\path\to\vcvarsall.bat
    exit /b 1
)
echo Using vcvars: %VCVARS%

call "%VCVARS%" x64
if errorlevel 1 (
    echo ERROR: vcvarsall failed
    exit /b 1
)

echo Working dir: %CD%
if not exist "%ROOT%\src\gui.cpp" (
    echo ERROR: gui.cpp not found
    exit /b 1
)
if not exist "%OBJDIR%" mkdir "%OBJDIR%"

echo === Compiling game sources ===
cl.exe /O2 /EHsc /std:c++17 /utf-8 /nologo /c ^
    /I"%ROOT%\include" /I"%ROOT%\include\third_party" ^
    "%ROOT%\src\rng.cpp" ^
    /Fo"%OBJDIR%\rng.obj"
if errorlevel 1 goto :fail

cl.exe /O2 /EHsc /std:c++17 /utf-8 /nologo /c ^
    /I"%ROOT%\include" /I"%ROOT%\include\third_party" ^
    "%ROOT%\src\resources.cpp" ^
    /Fo"%OBJDIR%\resources.obj"
if errorlevel 1 goto :fail

cl.exe /O2 /EHsc /std:c++17 /utf-8 /nologo /c ^
    /I"%ROOT%\include" /I"%ROOT%\include\third_party" ^
    "%ROOT%\src\data.cpp" ^
    /Fo"%OBJDIR%\data.obj"
if errorlevel 1 goto :fail

cl.exe /O2 /EHsc /std:c++17 /utf-8 /nologo /c ^
    /I"%ROOT%\include" /I"%ROOT%\include\third_party" ^
    "%ROOT%\src\earth.cpp" ^
    /Fo"%OBJDIR%\earth.obj"
if errorlevel 1 goto :fail

cl.exe /O2 /EHsc /std:c++17 /utf-8 /nologo /c ^
    /I"%ROOT%\include" /I"%ROOT%\include\third_party" ^
    "%ROOT%\src\game.cpp" ^
    /Fo"%OBJDIR%\game.obj"
if errorlevel 1 goto :fail

cl.exe /O2 /EHsc /std:c++17 /utf-8 /nologo /c ^
    /I"%ROOT%\include" /I"%ROOT%\include\third_party" ^
    "%ROOT%\src\rewards.cpp" ^
    /Fo"%OBJDIR%\rewards.obj"
if errorlevel 1 goto :fail

cl.exe /O2 /EHsc /std:c++17 /utf-8 /nologo /c ^
    /I"%ROOT%\include" /I"%ROOT%\include\third_party" ^
    "%ROOT%\src\env.cpp" ^
    /Fo"%OBJDIR%\env.obj"
if errorlevel 1 goto :fail

cl.exe /O2 /EHsc /std:c++17 /utf-8 /nologo /c ^
    /I"%ROOT%\include" /I"%ROOT%\include\third_party" ^
    "%ROOT%\src\lot_finder.cpp" ^
    /Fo"%OBJDIR%\lot_finder.obj"
if errorlevel 1 goto :fail

cl.exe /O2 /EHsc /std:c++17 /utf-8 /nologo /c ^
    /I"%ROOT%\include" /I"%ROOT%\include\third_party" ^
    "%ROOT%\src\action_mask.cpp" ^
    /Fo"%OBJDIR%\action_mask.obj"
if errorlevel 1 goto :fail

cl.exe /O2 /EHsc /std:c++17 /utf-8 /nologo /c ^
    /I"%ROOT%\include" /I"%ROOT%\include\third_party" ^
    "%ROOT%\src\observation.cpp" ^
    /Fo"%OBJDIR%\observation.obj"
if errorlevel 1 goto :fail

echo === Compiling GUI ===
cl.exe /O2 /EHsc /std:c++17 /utf-8 /nologo /c ^
    /I"%ROOT%\include" /I"%ROOT%\include\third_party" /I"%RLDIR%\include" ^
    "%ROOT%\src\gui.cpp" ^
    /Fo"%OBJDIR%\gui.obj"
if errorlevel 1 goto :fail

echo === Linking ===
cl.exe /nologo ^
    "%OBJDIR%\gui.obj" ^
    "%OBJDIR%\rng.obj" ^
    "%OBJDIR%\resources.obj" ^
    "%OBJDIR%\data.obj" ^
    "%OBJDIR%\earth.obj" ^
    "%OBJDIR%\game.obj" ^
    "%OBJDIR%\env.obj" ^
    "%OBJDIR%\lot_finder.obj" ^
    "%OBJDIR%\action_mask.obj" ^
    "%OBJDIR%\observation.obj" ^
    "%OBJDIR%\rewards.obj" ^
    /Fe"%ROOT%\sakhalin_colony_gui.exe" ^
    /link /SUBSYSTEM:CONSOLE /NODEFAULTLIB:libcmt.lib ^
    "%RLDIR%\lib\raylib.lib" opengl32.lib winmm.lib gdi32.lib user32.lib shell32.lib ^
    ucrt.lib vcruntime.lib msvcrt.lib advapi32.lib ole32.lib oleaut32.lib
if errorlevel 1 goto :fail

echo Build OK: %ROOT%\sakhalin_colony_gui.exe
echo Copying raylib.dll...
copy "%RLDIR%\lib\raylib.dll" "%ROOT%\raylib.dll" >nul
echo Copying original icons (ui/assets -> assets)...
if not exist "%ROOT%\assets" mkdir "%ROOT%\assets"
xcopy /Y /I "%ROOT%\ui\assets\*.png" "%ROOT%\assets" >nul
exit /b 0

:fail
echo BUILD FAILED
exit /b 1
