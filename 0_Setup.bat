@echo off
setlocal

:: 가상환경 폴더명 설정
set "VENV_NAME=venv"

:: 1. 가상환경 존재 여부 확인 및 생성
if not exist "%VENV_NAME%" (
    echo Creating virtual environment...
    python -m venv %VENV_NAME%
)

:: 2. 가상환경 활성화 및 패키지 설치
:: 가상환경 내의 pip를 직접 호출하여 활성화 단계를 생략하고 오류 가능성 차단
echo Installing dependencies from requirements.txt...
"%VENV_NAME%\Scripts\python.exe" -m pip install --upgrade pip
"%VENV_NAME%\Scripts\pip.exe" install -r requirements.txt

:: 3. 설치 완료 메시지 (기존 PowerShell 인코딩 유지)
powershell -NoProfile -EncodedCommand VwByAGkAdABlAC0ASABvAHMAdAAgACcAJMFYziAARMbMuC4AIABcuPitfLkgAFXWeMdY1TjBlMYuACcA
powershell -NoProfile -EncodedCommand VwByAGkAdABlAC0ASABvAHMAdAAgACcAxKyNwVjVJLh0uiAARMU0uyAApNCYsCAABLJ0uTjBlMYuAC4ALgAnAA==

pause >nul
endlocal