@echo off
rem Strata ML setup: the Python environments the ML worker runs in.
rem   venv        Python 3.12, PyTorch 2.8 (CUDA 12.8): LTX-2.5, Alpha Gen, MapAnything, NeRF
rem   venv-splat  Python 3.10, PyTorch 2.4 (CUDA 12.4) with gsplat's prebuilt kernels: Gaussian splats
rem Usage: setup-ml.bat [ml home]    (Strata's "Set up ML" button runs this for you)
setlocal EnableExtensions
set "MLHOME=%~1"
if "%MLHOME%"=="" set "MLHOME=%LOCALAPPDATA%\Strata\ml"
set "HERE=%~dp0"
set "UV_LINK_MODE=copy"
echo.
echo  Strata ML setup
echo  ---------------
echo  Installs into: %MLHOME%
echo  Needs an NVIDIA GPU with a current driver and about 15 GB of disk.
echo  The models themselves are downloaded later, from each node's panel.
echo.
if not exist "%MLHOME%" mkdir "%MLHOME%"

rem Strata asks a running worker to stop first (its python.exe and DLLs would be in use): wait for it
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

echo.
echo [1/5] Python 3.12 environment
rem an environment that's there is kept: re-creating it replaces python.exe, which a running worker holds
if exist "%MLHOME%\venv\Scripts\python.exe" (
  echo already there: kept
) else (
  call uv venv --python 3.12 "%MLHOME%\venv"
  if errorlevel 1 goto :fail
)
set "PY=%MLHOME%\venv\Scripts\python.exe"

echo.
echo [2/5] PyTorch 2.8 with CUDA 12.8 (about 3 GB)
call uv pip install --python "%PY%" torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
if errorlevel 1 goto :fail

echo.
echo [3/5] LTX-2.5 pipelines, MapAnything and the rest
call uv pip install --python "%PY%" --no-sources -r "%HERE%requirements.txt"
if errorlevel 1 goto :fail

echo.
echo [4/5] Gaussian splat environment: Python 3.10 + PyTorch 2.4 (CUDA 12.4)
if exist "%MLHOME%\venv-splat\Scripts\python.exe" (
  echo already there: kept
) else (
  call uv venv --python 3.10 "%MLHOME%\venv-splat"
  if errorlevel 1 goto :splatfail
)
set "SPY=%MLHOME%\venv-splat\Scripts\python.exe"
rem 2.4.1: 2.4.0's fbgemm.dll needs libomp140.x86_64.dll, which Windows doesn't have
call uv pip install --python "%SPY%" torch==2.4.1 --index-url https://download.pytorch.org/whl/cu124
if errorlevel 1 goto :splatfail
call uv pip install --python "%SPY%" "numpy<2" opencv-python-headless pillow ninja jaxtyping rich packaging setuptools
if errorlevel 1 goto :splatfail

echo.
echo [5/5] gsplat (prebuilt CUDA kernels)
call uv pip install --python "%SPY%" --no-deps "gsplat==1.5.3+pt24cu124" --index-url https://docs.gsplat.studio/whl
if errorlevel 1 goto :splatfail

echo.
echo Checking the splat environment loads ...
"%SPY%" -c "import sys; sys.path.insert(0, r'%HERE%.'); from strata_ml.winfix import fix_torch_openmp; print(fix_torch_openmp() or 'ok'); import torch, gsplat; print('torch', torch.__version__, 'CUDA', torch.cuda.is_available(), 'gsplat', gsplat.__version__)"
if errorlevel 1 goto :splatfail
goto :done

:splatfail
echo.
echo  The Gaussian splat environment could not be set up. GaussianSplat will still train,
echo  with a much slower PyTorch rasteriser. Everything else is ready.

:done
echo ok> "%MLHOME%\setup-ok.txt"
echo.
echo  Done. Strata starts the ML worker by itself when a node needs it.
echo  For gated models (LTX), accept the licence on Hugging Face and paste a Read token
echo  into the HF token field of a node's ML panel.
echo.
pause
exit /b 0

:fail
echo.
echo  Setup failed (see above). Fix the problem and run "Set up ML" again: finished steps are kept.
echo.
pause
exit /b 1
