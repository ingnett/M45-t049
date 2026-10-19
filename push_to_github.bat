@echo off
chcp 65001 >nul
echo ======================================================================
echo   M45: PUSH TO GITHUB AUTOMATION SCRIPT
echo ======================================================================
echo.
set /p REPO_URL="Nhập đường dẫn GitHub Repository (ví dụ https://github.com/username/m45.git): "

if "%REPO_URL%"=="" (
    echo [!] Ban chua nhap duong dan repository. Huy thao tac.
    pause
    exit /b 1
)

echo.
echo [*] Dang cau hinh git remote origin...
git remote remove origin 2>nul
git remote add origin %REPO_URL%

echo [*] Dang kiem tra nhanh master...
git branch -M master

echo [*] Dang day code len GitHub...
git push -u origin master

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ======================================================================
    echo   [V] DAY CODE LEN GITHUB THANH CONG!
    echo   Ban co the truy cap share.streamlit.io de deploy len Streamlit Cloud.
    echo ======================================================================
) else (
    echo.
    echo [!] Co loi khi day code. Vui long kiem tra lai quyen truy cap hoac Token.
)

echo.
pause
