@echo off
rem ---------------------------------------------------------------------------
rem  Benchmarks the CPU (OpenMP) version of the sand cube: the same simulation and
rem  culling as the CUDA version, run on the processor instead of the GPU.
rem  Results go to benchmark_cpu_results.csv.
rem
rem    run_cpu_benchmarks.bat                full sweep
rem    run_cpu_benchmarks.bat quick          small sweep (a couple of minutes)
rem    run_cpu_benchmarks.bat battery        label the results "battery" (or: ac)
rem    run_cpu_benchmarks.bat nobuild        reuse the existing cpu_sim.exe
rem    (options can be combined, e.g.  run_cpu_benchmarks.bat battery quick)
rem
rem  Sweep 1: grid size 64..320 with 1 thread and with all logical cores
rem  Sweep 2: thread scaling at grid 192 (1 2 4 6 8 and all cores)
rem
rem  Power state is detected automatically when possible; pass battery or ac to
rem  set it yourself. Leave the laptop alone while this runs.
rem ---------------------------------------------------------------------------
setlocal EnableDelayedExpansion
cd /d "%~dp0"

set "SIZES=64 96 128 192 256 320"
set "REPEATS=2"
set "FRAMES=120"
set "MAXT=%NUMBER_OF_PROCESSORS%"
set "THREAD_LIST=1 2 4 6 8 %MAXT%"
set "POWER="
set "NOBUILD="

for %%A in (%*) do (
    if /i "%%A"=="quick"   ( set "SIZES=64 128 192" & set "REPEATS=1" & set "FRAMES=60" & set "THREAD_LIST=1 4 %MAXT%" )
    if /i "%%A"=="nobuild" set "NOBUILD=1"
    if /i "%%A"=="battery" set "POWER=battery"
    if /i "%%A"=="ac"      set "POWER=ac"
)

if not defined POWER (
    set "POWER=unknown"
    for /f "delims=" %%P in ('powershell -NoProfile -Command "$b=Get-CimInstance Win32_Battery; if ^($b -and $b.BatteryStatus -eq 1^) {Write-Output battery} else {Write-Output ac}" 2^>nul') do set "POWER=%%P"
)
echo Power state: %POWER%    Logical cores: %MAXT%

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

cl /nologo /O2 /openmp /std:c++17 /EHsc /D_CRT_SECURE_NO_WARNINGS cpu_sim.cpp /Fe:cpu_sim.exe
if errorlevel 1 (
    echo Build failed.
    exit /b 1
)

:run
if not exist cpu_sim.exe (
    echo cpu_sim.exe not found. Run without "nobuild" first.
    exit /b 1
)
cpu_sim.exe selftest
if errorlevel 1 (
    echo Self-test failed, stopping.
    exit /b 1
)

if exist benchmark_cpu_results.csv move /y benchmark_cpu_results.csv benchmark_cpu_results_prev.csv >nul

echo.
echo ===== Sweep 1: grid size, 1 thread and %MAXT% threads =====
for %%N in (%SIZES%) do (
    for /l %%R in (1,1,%REPEATS%) do (
        echo [grid %%N  1 thread  run %%R of %REPEATS%]
        cpu_sim.exe %%N threads=1 frames=%FRAMES% power=%POWER%
        echo [grid %%N  %MAXT% threads  run %%R of %REPEATS%]
        cpu_sim.exe %%N threads=%MAXT% frames=%FRAMES% power=%POWER%
    )
)

echo.
echo ===== Sweep 2: thread scaling at grid 192 =====
for %%T in (%THREAD_LIST%) do (
    for /l %%R in (1,1,%REPEATS%) do (
        echo [grid 192  %%T threads  run %%R of %REPEATS%]
        cpu_sim.exe 192 threads=%%T frames=%FRAMES% power=%POWER%
    )
)

echo.
echo ===== Done. Raw results: %cd%\benchmark_cpu_results.csv =====
where python >nul 2>nul
if not errorlevel 1 (
    if exist compare_cpu_gpu.py python compare_cpu_gpu.py
) else (
    echo Python not found - open benchmark_cpu_results.csv in a spreadsheet instead.
)
endlocal
