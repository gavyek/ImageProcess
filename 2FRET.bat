@echo off
setlocal
pushd "%~dp0"

:menu
cls
echo ==============================
echo  1. ROI BND drawer (roi_manual_drawer.py)
echo  2. Morphology analysis (MOR_by_ROI.py)
echo  3. General FRET analysis (fret_ratio_builder.py)
echo  4. Nesprin2_FRET analysis (Nesprin2_FRET_Builder.py)
echo  5. Image Cropper by ROI (roi_channel_cropper.py)
echo  Q. Quit
echo ==============================
set "choice="
set /p "choice=Choose (1-5, Q=Quit): "

echo.
if /I "%choice%"=="1" goto run1
if /I "%choice%"=="2" goto run2
if /I "%choice%"=="3" goto run3
if /I "%choice%"=="4" goto run4
if /I "%choice%"=="5" goto run5
if /I "%choice%"=="Q" goto end
if /I "%choice%"=="q" goto end

echo Invalid selection. Try again.
pause
goto menu

:run1
call :run "src\roi_manual_drawer.py"
goto menu

:run2
call :run "src\MOR_by_ROI.py"
goto menu

:run3
call :run "src\FRET\fret_ratio_builder.py"
goto menu

:run4
call :run "src\FRET\Nesprin2_FRET_Builder.py"
goto menu

:run5
call :run "src\roi_channel_cropper.py"
goto menu

:run
cls
echo Running: %~1
python "%~1"
call :ask_again
exit /b

:ask_again
set "again="
set /p "again=Run another program? (Y/N): "
if /I "%again%"=="Y" goto menu
if /I "%again%"=="N" goto end
if /I "%again%"=="Q" goto end
if /I "%again%"=="q" goto end
echo Invalid input. Enter Y or N.
goto ask_again

:end
echo Bye.
pause >nul
popd
endlocal