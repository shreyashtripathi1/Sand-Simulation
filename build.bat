@echo off
rem Builds and runs the 3D sand cube from ANY terminal (cmd, PowerShell, VS Code).
rem Usage:  build.bat [gridSize]      e.g.  build.bat 96
setlocal
cd /d "%~dp0"

set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" (
    echo Could not find Visual Studio ^(vswhere.exe^). Install "Desktop development with C++" from Visual Studio.
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

echo Build OK - starting sim3d.exe
sim3d.exe %*
