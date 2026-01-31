@echo off
echo ===================================================
echo 🚀 AUTOMATED GITHUB PUSH SCRIPT
echo ===================================================

echo.
echo 1. Kiểm tra Git...
where git >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Git chua duoc cai dat!
    echo Vui long tai va cai dat Git tai: https://git-scm.com/download/win
    pause
    exit
)

echo.
echo 2. Khoi tao Repository...
git init
git add .
git commit -m "Initial Logic: Silver Price Prediction with Ridge Regression & Realtime Data"

echo.
echo 3. Cau hinh Remote...
git branch -M main
:: Xoa remote cu neu co de tranh loi
git remote remove origin 2>nul
git remote add origin https://github.com/chien273145/Silver-Price-Prediction.git

echo.
echo 4. Day code len GitHub...
echo [INFO] Neu day la lan dau, ban se duoc yeu cau dang nhap trong cua so moi...
git push -u origin main

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Co loi xay ra khi push code.
    echo Vui long kiem tra lai ket noi hoac quyen truy cap.
) else (
    echo.
    echo [SUCCESS] Da day code len GitHub thanh cong!
    echo Link: https://github.com/chien273145/Silver-Price-Prediction
)

echo.
pause
