@echo off
rem Strata quick tour: self-tests, demo renders, Nuke import.
cd /d "%~dp0"
set OFX_PLUGIN_PATH=%CD%\ofx-test-plugins

if exist bin\strata_tests.exe (
  echo === Engine self-tests
  bin\strata_tests.exe | findstr /C:"checks"
  echo.
  echo === OFX host self-tests
  bin\strata_ofx_tests.exe ofx-test-plugins | findstr /C:"checks"
  echo.
)
echo === Render the demo comp (frames 1-4)
bin\strata.exe render examples\demo.strata -f 1-4 -v
echo.
echo === Render the OFX demo comp
bin\strata.exe render examples\ofx_demo.strata
echo.
echo === Translate and render the sample Nuke script (frame 12)
set PY=
if exist python\python.exe set PY=python\python.exe
if not defined PY where py >nul 2>nul && set PY=py
if not defined PY where python >nul 2>nul && set PY=python
if defined PY (
  %PY% tools\nk2strata.py examples\nuke\shot010_comp_v003.nk
) else (
  echo Python not found - using the pre-translated examples\nuke\shot010_comp_v003.strata
)
bin\strata.exe render examples\nuke\shot010_comp_v003.strata -f 12
echo.
echo Done. Opening the output folder...
start "" "%CD%\examples\out"
pause
