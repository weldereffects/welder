@echo off
rem Welder quick tour: self-tests, demo renders, .nk import.
cd /d "%~dp0"
set OFX_PLUGIN_PATH=%CD%\ofx-test-plugins

if exist bin\welder_tests.exe (
  echo === Engine self-tests
  bin\welder_tests.exe | findstr /C:"checks"
  echo.
  echo === OFX host self-tests
  bin\welder_ofx_tests.exe ofx-test-plugins | findstr /C:"checks"
  echo.
)
echo === Render the demo comp (frames 1-4)
bin\welder.exe render examples\demo.weld -f 1-4 -v
echo.
echo === Render the OFX demo comp
bin\welder.exe render examples\ofx_demo.weld
echo.
echo === Translate and render the sample .nk script (frame 12)
set PY=
if exist python\python.exe set PY=python\python.exe
if not defined PY where py >nul 2>nul && set PY=py
if not defined PY where python >nul 2>nul && set PY=python
if defined PY (
  %PY% tools\nk2strata.py examples\nk\shot010_comp_v003.nk
) else (
  echo Python not found - using the pre-translated examples\nk\shot010_comp_v003.weld
)
bin\welder.exe render examples\nk\shot010_comp_v003.weld -f 12
echo.
echo Done. Opening the output folder...
start "" "%CD%\examples\out"
pause
