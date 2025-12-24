@echo off
chcp 65001
cls
echo ========================================================
echo          Github 自動同步與上傳工具 (完整版)
echo ========================================================
echo.

echo [1/4] 正在從雲端拉取最新資料 (Git Pull)...
echo --------------------------------------------------------
git pull origin main
if %errorlevel% neq 0 (
    echo.
    echo [嚴重錯誤] 拉取失敗！
    echo 可能原因：
    echo 1. 本地檔案與雲端檔案有衝突 (Conflict)
    echo 2. 網路連線問題
    echo.
    echo 請手動解決衝突後再執行，或聯絡開發人員。
    pause
    exit /b
)
echo.
echo [成功] 資料同步完成，目前版本已是最新。
echo --------------------------------------------------------
echo.

echo [2/4] 準備上傳...
set /p msg=請輸入本次更新說明 (例如: fix bugs): 
if "%msg%"=="" set msg=Auto update

echo.
echo [3/4] 加入檔案與建立版本 (Git Commit)...
git add .
git commit -m "%msg%"

echo.
echo [4/4] 推送至雲端 (Git Push)...
git push origin main

echo.
echo ========================================================
echo          全部完成！視窗將在 10 秒後關閉
echo ========================================================
timeout /t 10