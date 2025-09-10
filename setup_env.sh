#!/bin/bash
echo "Creando entorno virtual para el proyecto..."
python3 -m venv venv

echo ""
echo "Activando entorno virtual..."
source venv/bin/activate

echo ""
echo "Instalando dependencias..."
pip install -r requirements.txt

echo ""
echo "¡Entorno virtual configurado correctamente!"
echo "Para activar el entorno en futuras sesiones, ejecuta: source venv/bin/activate"
echo "Para ejecutar el proyecto, usa: python main.py"
