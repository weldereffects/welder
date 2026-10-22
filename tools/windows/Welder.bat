@echo off
rem Launch the Welder editor. Drag a .weld or .nk file onto this file to open it.
cd /d "%~dp0"
set OFX_PLUGIN_PATH=%CD%\ofx-test-plugins;%OFX_PLUGIN_PATH%
start "" "%~dp0bin\welder_ui.exe" %*
