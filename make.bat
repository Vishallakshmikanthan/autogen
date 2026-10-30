@echo off
setlocal

set PYTHON=python

if "%1"=="check" goto check
if "%1"=="test" goto test
if "%1"=="lint" goto lint
if "%1"=="format" goto format
if "%1"=="clean" goto clean
if "%1"=="" goto check

echo Usage: make [check^|test^|lint^|format^|clean]
exit /b 1

:check
call :lint
if errorlevel 1 exit /b %errorlevel%
call :test
if errorlevel 1 exit /b %errorlevel%
echo All checks passed!
exit /b 0

:test
echo [make] Running pytest...
%PYTHON% -m pytest -q
exit /b %errorlevel%

:lint
echo [make] Running ruff check...
%PYTHON% -m ruff check packages apps tests
if errorlevel 1 exit /b %errorlevel%

echo [make] Running ruff format --check...
%PYTHON% -m ruff format --check packages apps tests
if errorlevel 1 exit /b %errorlevel%

echo [make] Running mypy...
%PYTHON% -m mypy packages apps tests
exit /b %errorlevel%

:format
echo [make] Running ruff format...
%PYTHON% -m ruff format packages apps tests
%PYTHON% -m ruff check --fix packages apps tests
exit /b %errorlevel%

:clean
echo [make] Cleaning cache files...
if exist .pytest_cache rd /s /q .pytest_cache
if exist .ruff_cache rd /s /q .ruff_cache
if exist .mypy_cache rd /s /q .mypy_cache
exit /b 0
