@echo off
rem ---------------------------------------------------------------------------
rem  Benchmarks the 2D sand simulation (main_2d.cu) and collects the results in
rem  benchmark_2d_results.csv. Works from any terminal.
rem
rem    run_2d_benchmarks.bat            full sweep (3 runs per configuration)
rem    run_2d_benchmarks.bat quick      small sweep, 1 run each
rem    run_2d_benchmarks.bat nobuild    skip compiling, use the existing sim2d.exe
rem
rem  Sweep 1: resolution   (640x360 1280x720 1920x1080 2560x1440 3840x2160), block 8x8
rem  Sweep 2: block shape  (8x8 16x16 32x8 32x4 16x8 64x4) at 1920x1080
rem  Sweep 3: pinned vs pageable host memory for the colour-buffer copy, at three resolutions
rem
rem  Each run opens a window for a few seconds with a fixed scripted scene (sand spawned
rem  while the cursor sweeps, periodic blasts), prints one result line and closes.
rem  Do not touch the window while it runs.
rem ---------------------------------------------------------------------------
setlocal EnableDelayedExpansion
cd /d "%~dp0"

set "SIZES=640x360 1280x720 1920x1080 2560x1440 3840x2160"
set "BLOCKS=8x8 16x16 32x8 32x4 16x8 64x4"
set "BLOCK_SIZE=1920x1080"
set "PIN_SIZES=1280x720 1920x1080 3840x2160"
set "REPEATS=3"

set "NOBUILD="
for %%A in (%*) do (
    if /i "%%A"=="nobuild" set "NOBUILD=1"
    if /i "%%A"=="quick" (
        set "SIZES=1280x720 1920x1080"
        set "BLOCKS=8x8 32x8"
        set "PIN_SIZES=1920x1080"
        set "REPEATS=1"
    )
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

nvcc main_2d.cu -o sim2d -allow-unsupported-compiler -Xcompiler "/MD" -I"glfw\include" -L"glfw\lib-vc2022" -lglfw3 -lopengl32 -lgdi32 -luser32 -lshell32
if errorlevel 1 (
    echo Build failed.
    exit /b 1
)

:run
if not exist sim2d.exe (
    echo sim2d.exe not found. Run without "nobuild" first.
    exit /b 1
)

rem keep the previous results instead of mixing old and new runs
if exist benchmark_2d_results.csv move /y benchmark_2d_results.csv benchmark_2d_results_prev.csv >nul

echo.
echo ===== Sweep 1: resolution, block 8x8 =====
for %%S in (%SIZES%) do (
    for /l %%R in (1,1,%REPEATS%) do (
        echo [size %%S  block 8x8  run %%R of %REPEATS%]
        sim2d.exe size=%%S block=8x8 bench
        if errorlevel 1 echo   FAILED
    )
)

echo.
echo ===== Sweep 2: block shape at %BLOCK_SIZE% =====
for %%B in (%BLOCKS%) do (
    for /l %%R in (1,1,%REPEATS%) do (
        echo [size %BLOCK_SIZE%  block %%B  run %%R of %REPEATS%]
        sim2d.exe size=%BLOCK_SIZE% block=%%B bench
        if errorlevel 1 echo   FAILED
    )
)

echo.
echo ===== Sweep 3: pinned host memory =====
for %%S in (%PIN_SIZES%) do (
    for /l %%R in (1,1,%REPEATS%) do (
        echo [size %%S  pinned  run %%R of %REPEATS%]
        sim2d.exe size=%%S block=8x8 pinned bench
        if errorlevel 1 echo   FAILED
    )
)

echo.
echo ===== Done. Raw results: %cd%\benchmark_2d_results.csv =====
where python >nul 2>nul
if not errorlevel 1 (
    if exist summarize_2d_benchmarks.py python summarize_2d_benchmarks.py
) else (
    echo Python not found - open benchmark_2d_results.csv in a spreadsheet instead.
)
endlocal
