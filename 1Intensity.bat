@echo off
setlocal
pushd "%~dp0"

:: 가상환경 Python 경로 설정
set "PYTHON_CMD=venv\Scripts\python.exe"

:: 가상환경 확인 (Setup 실행 여부 체크)
if not exist "%PYTHON_CMD%" (
    echo [ERROR] Virtual environment not found.
    echo Please run '0_Setup.bat' first.
    pause
    goto end
)

:lang_menu
cls
echo Select language / 언어 선택:
echo ==============================
echo  1. Korean
echo  2. English
echo ==============================
set "lang_arg="
set "lang_choice="
set /p "lang_choice=Choose language (1-2): "
if /I "%lang_choice%"=="1" set "lang_arg=" & goto menu
if /I "%lang_choice%"=="2" set "lang_arg=-mode EN" & goto menu
echo Invalid selection.
echo Try again.
pause
goto lang_menu

:menu
cls
echo Program Set for Fluorescence Intensity Analysis
echo ==============================
echo  1. ROI Drawing
echo  2. Morphology Analysis
echo  3. Fluorescence Intensity Analysis
echo  4. Image Cropper by ROI
echo  Q. Quit
echo ==============================
set "choice="
set /p "choice=Choose (1-4, Q=Quit): "

echo.
if /I "%choice%"=="1" goto run1
if /I "%choice%"=="2" goto run2
if /I "%choice%"=="3" goto run3
if /I "%choice%"=="4" goto run4
if /I "%choice%"=="Q" goto end
if /I "%choice%"=="q" goto end

echo Invalid selection.
echo Try again.
pause
goto menu

:run1
cls
echo Select ROI Drawing Mode:
echo ==============================
echo  1. Manual (roi_manual_drawer.py)
echo  2. Auto (ROI_auto_drawer.py)
echo ==============================
set "roi_choice="
set /p "roi_choice=Choose (1-2): "
if /I "%roi_choice%"=="1" (
    call :run "src\roi_manual_drawer.py"
    goto menu
)
if /I "%roi_choice%"=="2" (
    call :run "src\ROI_auto_drawer.py"
    goto menu
)
echo Invalid selection.
pause
goto run1

:run2
call :run "src\MOR_by_ROI.py"
goto menu

:run3
call :run "src\INT\Fluor_INT.py"
goto menu

:run4
call :run "src\roi_channel_cropper.py"
goto menu

:run
cls
echo Running: %~1
:: 가상환경의 Python으로 스크립트 실행
"%PYTHON_CMD%" "%~1" %lang_arg%
call :ask_again
exit /b

:ask_again
set "again="
set /p "again=Run another program? (Y/N): "
if /I "%again%"=="Y" goto menu
if /I "%again%"=="N" goto end
if /I "%again%"=="Q" goto end
if /I "%again%"=="q" goto end
echo Invalid input.
echo Enter Y or N.
goto ask_again

:end
echo Bye.
pause >nul
popd
endlocal