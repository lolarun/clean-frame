@echo off
rem Clean Frame installer (Windows)
cd /d %~dp0
python -m venv .venv || goto :err
.venv\Scripts\python -m pip install -U pip -i https://mirrors.aliyun.com/pypi/simple/
.venv\Scripts\python -m pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ || goto :err
.venv\Scripts\python -c "import onnxruntime as o; o.preload_dlls(); print(o.get_available_providers())"
echo Install OK
goto :eof
:err
echo Install FAILED
exit /b 1
