@echo off
setlocal enabledelayedexpansion

:: Rapid local Windows build + verification runner for TokenWing.
:: Usage:
::   tokenwing-tools\testing\check_windows.cmd path\to\small_moe.gguf [extra check.py flags...]

if "%~1"=="" (
    echo Usage: %~nx0 ^<path-to-routed-moe.gguf^> [extra check.py options]
    exit /b 1
)

set "SCRIPT_DIR=%~dp0"
set "MODEL=%~f1"
shift
set "EXTRA="
:collect
if "%~1"=="" goto :collected
set "EXTRA=!EXTRA! "%~1""
shift
goto :collect
:collected

for %%D in ("%SCRIPT_DIR%..\..") do set "REPO_ROOT=%%~fD"
pushd "%REPO_ROOT%" || exit /b 1

call :ensure_msvc
where cl.exe >nul 2>nul
if errorlevel 1 (
    echo Error: MSVC cl.exe not found. Open an x64 Native Tools Command Prompt or install VS C++ build tools.
    popd
    exit /b 1
)

where ninja.exe >nul 2>nul
if errorlevel 1 (
    for /f "delims=" %%P in ('python -c "import os, sysconfig; print(sysconfig.get_path('scripts'))" 2^>nul') do (
        if exist "%%P\ninja.exe" set "PATH=%%P;!PATH!"
    )
)

if not defined BUILD_DIR set "BUILD_DIR=build-win"
if not defined WORK_DIR set "WORK_DIR=%BUILD_DIR%\tw-check"

if not exist "%BUILD_DIR%\CMakeCache.txt" (
    echo Configuring %BUILD_DIR% ...
    where ninja.exe >nul 2>nul
    if not errorlevel 1 (
        cmake -B "%BUILD_DIR%" -G Ninja -DCMAKE_BUILD_TYPE=Release
    ) else (
        cmake -B "%BUILD_DIR%"
    )
    if errorlevel 1 (
        popd
        exit /b 1
    )
)

python tokenwing-tools\testing\check.py --model "%MODEL%" --build "%BUILD_DIR%" --work "%WORK_DIR%" !EXTRA!
set "RC=%ERRORLEVEL%"
popd
exit /b %RC%

:ensure_msvc
where cl.exe >nul 2>nul
if not errorlevel 1 exit /b 0
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if exist "%VSWHERE%" (
    for /f "delims=" %%I in ('"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath') do (
        if exist "%%I\VC\Auxiliary\Build\vcvars64.bat" (
            call "%%I\VC\Auxiliary\Build\vcvars64.bat" >nul
            exit /b 0
        )
    )
)
for %%E in (Enterprise Professional Community BuildTools) do (
    for %%Y in (2022 18) do (
        if exist "%ProgramFiles%\Microsoft Visual Studio\%%Y\%%E\VC\Auxiliary\Build\vcvars64.bat" (
            call "%ProgramFiles%\Microsoft Visual Studio\%%Y\%%E\VC\Auxiliary\Build\vcvars64.bat" >nul
            exit /b 0
        )
    )
)
exit /b 1
