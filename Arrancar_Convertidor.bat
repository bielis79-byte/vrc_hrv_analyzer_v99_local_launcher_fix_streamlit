@echo off
title VRC HRV Analyzer - Arranque local y convertidor PNG

echo ============================================================
echo  VRC / HRV RRi Analyzer Pro
echo  Arranque local + capturador PNG
echo ============================================================
echo.
echo Carpeta actual:
echo %~dp0
echo.

cd /d "%~dp0"

echo Comprobando Python...
python --version
if errorlevel 1 (
    echo.
    echo ERROR: Python no esta disponible en PATH.
    echo Instala Python o abre este BAT desde un entorno donde Python funcione.
    pause
    exit /b 1
)

echo.
echo Instalando/actualizando dependencias necesarias...
python -m pip install -r "%~dp0requirements.txt"

echo.
echo Arrancando Streamlit en una ventana nueva...
start "VRC HRV Streamlit" cmd /k "cd /d "%~dp0" && python -m streamlit run app.py --server.port 8501 --server.address localhost"

echo.
echo Esperando 12 segundos a que Streamlit arranque...
timeout /t 12 >nul

echo.
echo Abriendo la app en el navegador...
start "" "http://localhost:8501/"

echo.
echo Esperando a que cargue la pagina...
timeout /t 6 >nul

echo.
echo Generando captura PNG de la app local...
python "%~dp0capture_streamlit_localhost_png.py" "http://localhost:8501/" "%~dp0captura_streamlit.png"

echo.
echo Intentando convertir los HTML exportados a PNG si existen...
python "%~dp0convert_html_to_png.py"

echo.
echo ============================================================
echo  Proceso terminado.
echo.
echo  App local:
echo  http://localhost:8501/
echo.
echo  Captura principal:
echo  %~dp0captura_streamlit.png
echo.
echo  PNG desde HTML, si existen:
echo  %~dp0graficos\png_from_html
echo ============================================================
echo.
pause
