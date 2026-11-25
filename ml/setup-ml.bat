@echo off
rem Welder ML setup: the Python environments the ML worker runs in.
rem   venv        Python 3.12, PyTorch 2.8 (CUDA 12.8): LTX-2.5, Alpha Gen, MapAnything, NeRF
rem   venv-splat  Python 3.10, PyTorch 2.4 (CUDA 12.4) with gsplat's prebuilt kernels: Gaussian splats
rem   venv-corridorkey  Python 3.12, PyTorch 2.9.1 (CUDA 12.8): CorridorKey (it pins numpy < 2, transformers < 5)
rem   venv-directml  AMD Radeon only: Python 3.12, PyTorch 2.4.1 + torch-directml: the AI masking nodes
rem The PyTorch builds follow the GPU: CUDA for NVIDIA, XPU for Intel Arc, DirectML for AMD Radeon (the main
rem environment then gets PyTorch's CPU build), the CPU build when there's no GPU PyTorch can use.
rem Usage: setup-ml.bat [ml home] [gpu]    (Welder's "Set up ML" button runs this for you)
rem   gpu: nvidia, intel, amd or cpu.
rem   The GPU is found by itself; the second argument, or WELDER_ML_GPU, picks one instead.
setlocal EnableExtensions
set "MLHOME=%~1"
if "%MLHOME%"=="" if not exist "%LOCALAPPDATA%\Welder\ml" if exist "%LOCALAPPDATA%\Strata\ml" set "MLHOME=%LOCALAPPDATA%\Strata\ml"
if "%MLHOME%"=="" set "MLHOME=%LOCALAPPDATA%\Welder\ml"
set "HERE=%~dp0"
set "UV_LINK_MODE=copy"

rem which GPU to install PyTorch for
set "GPUKIND=%~2"
if "%GPUKIND%"=="" set "GPUKIND=%WELDER_ML_GPU%"
set "GPUNAMES="
for /f "usebackq tokens=1* delims=|" %%A in (`powershell -NoProfile -ExecutionPolicy Bypass -File "%HERE%detect-gpu.ps1"`) do (
  set "FOUNDKIND=%%A"
  set "GPUNAMES=%%B"
)
if "%GPUKIND%"=="" set "GPUKIND=%FOUNDKIND%"
if /i not "%GPUKIND%"=="nvidia" if /i not "%GPUKIND%"=="intel" if /i not "%GPUKIND%"=="amd" if /i not "%GPUKIND%"=="cpu" set "GPUKIND=nvidia"

echo.
echo  Welder ML setup
echo  ---------------
echo  Installs into: %MLHOME%
echo  GPU: %GPUNAMES%
if /i "%GPUKIND%"=="nvidia" echo  Setting up for NVIDIA (CUDA). Needs a current driver and about 20 GB of disk.
if /i "%GPUKIND%"=="intel" echo  Setting up for Intel Arc (XPU). Needs a current Intel graphics driver and about 15 GB of disk.
if /i "%GPUKIND%"=="amd" echo  Setting up for AMD Radeon (DirectML): the AI masking nodes (TrackMask, CorridorKey) run on the GPU,
if /i "%GPUKIND%"=="amd" echo  the other ML nodes on the CPU. Needs a current AMD driver and about 12 GB of disk.
if /i "%GPUKIND%"=="cpu" echo  No GPU PyTorch can use was found: setting up for the CPU (works, but slowly).
echo  Another GPU: run setup-ml.bat "%MLHOME%" nvidia^|intel^|amd^|cpu
echo  The models themselves are downloaded later, from each node's panel.
echo.
if not exist "%MLHOME%" mkdir "%MLHOME%"

rem Welder asks a running worker to stop first (its python.exe and DLLs would be in use): wait for it
set WAITS=0
:waitworker
if not exist "%MLHOME%\stop" goto :workerstopped
if %WAITS%==0 echo Waiting for the ML worker to stop ...
set /a WAITS+=1
if %WAITS% GEQ 30 goto :workerstopped
timeout /t 1 /nobreak >nul
goto :waitworker
:workerstopped
if exist "%MLHOME%\stop" del "%MLHOME%\stop"
rem give the worker process a moment to exit after it saw the request
if not %WAITS%==0 timeout /t 2 /nobreak >nul

call uv --version >nul 2>nul
if not errorlevel 1 goto :haveuv
if exist "%USERPROFILE%\.local\bin\uv.exe" goto :uvpath
echo Installing uv, a fast Python package installer (https://astral.sh/uv) ...
powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"
if errorlevel 1 goto :fail
:uvpath
set "PATH=%USERPROFILE%\.local\bin;%PATH%"
:haveuv
rem Python itself goes in the ML home too (python\), not in your user profile: then the whole ML home can be
rem copied to another machine, for example an offline one, and Welder points the environments at it there.
set "UV_PYTHON_INSTALL_DIR=%MLHOME%\python"
set "UV_PYTHON_PREFERENCE=only-managed"

echo.
echo [1/7] Python 3.12 environment
call :venv venv 3.12
if errorlevel 1 goto :fail
set "PY=%MLHOME%\venv\Scripts\python.exe"

rem (the builds are pinned with their +local tag, so a change of GPU swaps the build in place)
set "TORCHBUILD=cu128"
if /i "%GPUKIND%"=="intel" set "TORCHBUILD=xpu"
if /i "%GPUKIND%"=="amd" set "TORCHBUILD=cpu"
if /i "%GPUKIND%"=="cpu" set "TORCHBUILD=cpu"
echo %GPUKIND%> "%MLHOME%\gpu.txt"

echo.
echo [2/7] PyTorch 2.8 (%TORCHBUILD%, up to 3 GB)
call :torch "%PY%" "torch==2.8.0+%TORCHBUILD% torchvision==0.23.0+%TORCHBUILD% torchaudio==2.8.0+%TORCHBUILD% --index-url https://download.pytorch.org/whl/%TORCHBUILD%"
if errorlevel 1 goto :fail

echo.
echo [3/7] LTX-2.5 pipelines, MapAnything and the rest
call uv pip install --python "%PY%" --no-sources -r "%HERE%requirements.txt"
if errorlevel 1 goto :fail
rem what was installed: Welder compares it with a newer requirements.txt and offers "Update ML"
copy /y "%HERE%requirements.txt" "%MLHOME%\requirements-installed.txt" >nul

echo.
echo [4/7] Gaussian splat environment: Python 3.10 + PyTorch 2.4 (CUDA 12.4)
if /i not "%GPUKIND%"=="nvidia" (
  echo  skipped: gsplat's kernels are CUDA only. GaussianSplat trains with the slower PyTorch rasteriser.
  goto :corridorkey
)
call :venv venv-splat 3.10
if errorlevel 1 goto :splatfail
set "SPY=%MLHOME%\venv-splat\Scripts\python.exe"
rem 2.4.1: 2.4.0's fbgemm.dll needs libomp140.x86_64.dll, which Windows doesn't have
call uv pip install --python "%SPY%" torch==2.4.1 --index-url https://download.pytorch.org/whl/cu124
if errorlevel 1 goto :splatfail
call uv pip install --python "%SPY%" "numpy<2" opencv-python-headless pillow ninja jaxtyping rich packaging setuptools
if errorlevel 1 goto :splatfail

echo.
echo [5/7] gsplat (prebuilt CUDA kernels)
call uv pip install --python "%SPY%" --no-deps "gsplat==1.5.3+pt24cu124" --index-url https://docs.gsplat.studio/whl
if errorlevel 1 goto :splatfail

echo.
echo Checking the splat environment loads ...
"%SPY%" -c "import sys; sys.path.insert(0, r'%HERE%.'); from strata_ml.winfix import fix_torch_openmp; print(fix_torch_openmp() or 'ok'); import torch, gsplat; print('torch', torch.__version__, 'CUDA', torch.cuda.is_available(), 'gsplat', gsplat.__version__)"
if errorlevel 1 goto :splatfail
goto :corridorkey

:splatfail
echo.
echo  The Gaussian splat environment could not be set up. GaussianSplat will still train,
echo  with a much slower PyTorch rasteriser. Everything else is ready.

:corridorkey
echo.
if /i "%GPUKIND%"=="amd" (
  echo [6/7] CorridorKey environment: Python 3.12 + PyTorch 2.4.1 with DirectML
) else (
  echo [6/7] CorridorKey environment: Python 3.12 + PyTorch 2.9.1 ^(%TORCHBUILD%^)
)
call :venv venv-corridorkey 3.12
if errorlevel 1 goto :ckfail
set "CPY=%MLHOME%\venv-corridorkey\Scripts\python.exe"
if /i "%GPUKIND%"=="amd" goto :ckdml
call :torch "%CPY%" "torch==2.9.1+%TORCHBUILD% torchvision==0.24.1+%TORCHBUILD% --index-url https://download.pytorch.org/whl/%TORCHBUILD%"
if errorlevel 1 goto :ckfail
goto :ckreqs
:ckdml
call :torch "%CPY%" "torch==2.4.1 torchvision==0.19.1 torch-directml==0.2.5.dev240914"
if errorlevel 1 goto :ckfail
:ckreqs
call uv pip install --python "%CPY%" -r "%HERE%requirements-corridorkey.txt"
if errorlevel 1 goto :ckfail
goto :directml

:ckfail
echo.
echo  The CorridorKey environment could not be set up: the CorridorKey node won't run.
echo  Everything else is ready. Run "Set up ML" again to retry.

:directml
echo.
echo [7/7] DirectML environment (AMD Radeon): Python 3.12 + PyTorch 2.4.1 + torch-directml
if /i not "%GPUKIND%"=="amd" (
  echo  skipped: only needed for AMD GPUs
  goto :done
)
call :venv venv-directml 3.12
if errorlevel 1 goto :dmlfail
set "DPY=%MLHOME%\venv-directml\Scripts\python.exe"
call :torch "%DPY%" "torch==2.4.1 torchvision==0.19.1 torch-directml==0.2.5.dev240914"
if errorlevel 1 goto :dmlfail
call uv pip install --python "%DPY%" -r "%HERE%requirements-directml.txt"
if errorlevel 1 goto :dmlfail
echo Checking DirectML sees the GPU ...
"%DPY%" -c "import torch, torch_directml as d; n = d.device_count(); print('torch', torch.__version__, '- DirectML adapters:', ', '.join(d.device_name(i) for i in range(n)) or 'none')"
if errorlevel 1 goto :dmlfail
goto :done

:dmlfail
echo.
echo  The DirectML environment could not be set up: the AI masking nodes run on the CPU (slower).
echo  Everything else is ready. Run "Set up ML" again to retry.

:done
echo ok> "%MLHOME%\setup-ok.txt"
echo.
echo  Done. Welder starts the ML worker by itself when a node needs it.
echo  For gated models (LTX), accept the licence on Hugging Face and paste a Read token
echo  into the HF token field of a node's ML panel.
echo.
pause
exit /b 0

:torch
rem %1 = an environment's python, %2 = the PyTorch packages to install (and where from). An install or update
rem that was cut short (or a file Windows kept locked) can leave a torch folder without its __init__.py: PyTorch
rem then "has no attribute 'cuda'". uv counts it as installed, so check it loads, and reinstall it if not.
call uv pip install --python "%~1" %~2
if errorlevel 1 exit /b 1
"%~1" -c "import sys, torch; sys.exit(0 if getattr(torch, '__file__', None) and hasattr(torch, 'cuda') else 1)" >nul 2>nul
if not errorlevel 1 exit /b 0
echo PyTorch there is incomplete, an earlier install was cut short: reinstalling it ...
call uv pip install --python "%~1" --reinstall-package torch --reinstall-package torchvision %~2
if errorlevel 1 exit /b 1
"%~1" -c "import torch; print('torch', torch.__version__, 'loads')"
exit /b %errorlevel%

:venv
rem %1 = the environment's folder, %2 = its Python version. One that's there is kept (packages and all).
rem One from before Python lived in the ML home is moved onto the ML home's Python in place: its packages stay.
if not exist "%MLHOME%\%~1\Scripts\python.exe" goto :venvnew
if exist "%MLHOME%\%~1\.welder-portable" (
  echo already there: kept
  exit /b 0
)
echo already there: moving it onto the ML home's own Python, so the ML home can be copied
call uv venv --allow-existing --python %~2 "%MLHOME%\%~1"
if errorlevel 1 (
  echo couldn't move it: kept as it is
  exit /b 0
)
echo portable> "%MLHOME%\%~1\.welder-portable"
exit /b 0
:venvnew
call uv venv --python %~2 "%MLHOME%\%~1"
if errorlevel 1 exit /b 1
echo portable> "%MLHOME%\%~1\.welder-portable"
exit /b 0

:fail
echo.
echo  Setup failed (see above). Fix the problem and run "Set up ML" again: finished steps are kept.
echo.
pause
exit /b 1
