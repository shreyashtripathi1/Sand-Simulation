@echo off
rem ---------------------------------------------------------------------------
rem  Runs the 3D sand cube through every benchmark configuration and collects the
rem  results in benchmark_results.csv. Works from any terminal.
rem
rem    run_benchmarks.bat            full sweep (3 runs per configuration)
rem    run_benchmarks.bat quick      small sweep, 1 run each (about a minute)
rem    run_benchmarks.bat nobuild    skip compiling, use the existing sim3d.exe
rem
rem  Sweep 1: grid size   (64 96 128 192 256 320) with the default block 32x4x2
rem  Sweep 2: block shape (32x4x2 32x8x1 16x4x4 16x16x1 8x8x4 8x8x8) at grid 256
rem
rem  Each run opens the window for a few seconds, drops sand, slowly orbits the
rem  camera, prints one result line and closes. Do not touch the window while it runs.
rem ---------------------------------------------------------------------------
setlocal EnableDelayedExpansion
cd /d "%~dp0"

set "SIZES=64 96 128 192 256 320"
set "BLOCKS=32x4x2 32x8x1 16x4x4 16x16x1 8x8x4 8x8x8"
set "DEFAULT_BLOCK=32x4x2"
set "BLOCK_SWEEP_GRID=256"
set "REPEATS=3"

set "QUICK="
set "NOBUILD="
for %%A in (%*) do (
    if /i "%%A"=="quick" set "QUICK=1"
    if /i "%%A"=="nobuild" set "NOBUILD=1"
)
if defined QUICK (
    set "SIZES=64 128 192"
    set "BLOCKS=32x4x2 16x16x1"
    set "BLOCK_SWEEP_GRID=128"
    set "REPEATS=1"
)

if defined NOBUILD goto run

rem ---- build -----------------------------------------------------------------
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" (
    echo Could not find Visual Studio ^(vswhere.exe^). Install "Desktop development with C++".
    exit /b 1
)
set "VSDIR="
for /f "usebackq delims=" %%i in (`"%VSWHERE%" -latest -products * -property installationPath`) do set "VSDIR=%%i"
if not defined VSDIR (
    echo No Visual Studio installation found.
    exit /b 1
)
call "%VSDIR%\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 exit /b 1

nvcc main.cu -o sim3d -allow-unsupported-compiler -Xcompiler "/MD" -I"glfw\include" -L"glfw\lib-vc2022" -lglfw3 -lopengl32 -lgdi32 -luser32 -lshell32
if errorlevel 1 (
    echo Build failed.
    exit /b 1
)

:run
if not exist sim3d.exe (
    echo sim3d.exe not found. Run without "nobuild" first.
    exit /b 1
)

rem keep the previous results instead of mixing old and new runs
if exist benchmark_results.csv move /y benchmark_results.csv benchmark_results_prev.csv >nul

echo.
echo ===== Sweep 1: grid size, block %DEFAULT_BLOCK% =====
for %%N in (%SIZES%) do (
    for /l %%R in (1,1,%REPEATS%) do (
        echo [grid %%N  block %DEFAULT_BLOCK%  run %%R of %REPEATS%]
        sim3d.exe %%N block=%DEFAULT_BLOCK% bench
        if errorlevel 1 echo   FAILED - grid %%N may need more GPU memory than is available
    )
)

echo.
echo ===== Sweep 2: block shape, grid %BLOCK_SWEEP_GRID% =====
for %%B in (%BLOCKS%) do (
    for /l %%R in (1,1,%REPEATS%) do (
        echo [grid %BLOCK_SWEEP_GRID%  block %%B  run %%R of %REPEATS%]
        sim3d.exe %BLOCK_SWEEP_GRID% block=%%B bench
        if errorlevel 1 echo   FAILED
    )
)

echo.
echo ===== Done. Raw results: %cd%\benchmark_results.csv =====
where python >nul 2>nul
if not errorlevel 1 (
    if exist summarize_benchmarks.py python summarize_benchmarks.py
) else (
    echo Python not found - open benchmark_results.csv in a spreadsheet instead.
)
endlocal
