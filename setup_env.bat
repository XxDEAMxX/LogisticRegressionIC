@echo off
echo Creando entorno virtual para el proyecto...
python -m venv venv

echo.
echo Activando entorno virtual...
call venv\Scripts\activate.bat

echo.
echo Instalando dependencias...
pip install -r requirements.txt

echo.
echo ¡Entorno virtual configurado correctamente!
echo Para activar el entorno en futuras sesiones, ejecuta: venv\Scripts\activate.bat
echo Para ejecutar el proyecto, usa: python main.py
pause
