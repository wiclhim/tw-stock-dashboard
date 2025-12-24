@echo off
chcp 65001
set /p msg=請輸入 commit 訊息:
git add .
git commit -m "%msg%"
git push origin main
pause