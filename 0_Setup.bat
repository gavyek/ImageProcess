@echo off
setlocal

:: Update pip
python -m pip install --upgrade pip

:: Install dependencies
python -m pip install ^
    numpy ^
    pandas ^
    matplotlib ^
    Pillow ^
    tifffile ^
    scipy ^
    scikit-image ^
    python-pptx

:: Print messages in Korean via PowerShell (no encoding issues)
powershell -NoProfile -EncodedCommand VwByAGkAdABlAC0ASABvAHMAdAAgACcAJMFYziAARMbMuC4AIABcuPitfLkgAFXWeMdY1TjBlMYuACcA
powershell -NoProfile -EncodedCommand VwByAGkAdABlAC0ASABvAHMAdAAgACcAxKyNwVjVJLh0uiAARMU0uyAApNCYsCAABLJ0uTjBlMYuAC4ALgAnAA==
pause >nul
endlocal