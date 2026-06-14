#!/bin/bash
set -e

echo ">>> Actualizando paquetes del sistema..."
export DEBIAN_FRONTEND=noninteractive
apt-get update

echo ">>> Instalando LaTeX (texlive-full)..."
apt-get install -y --no-install-recommends texlive-full

echo ">>> Limpiando cache de apt..."
apt-get clean
rm -rf /var/lib/apt/lists/*

echo ">>> Instalando paquetes de Python para análisis de datos e IA..."
pip install --upgrade pip
pip install \
    ipykernel \
    numpy \
    pandas \
    matplotlib \
    seaborn \
    scipy \
    scikit-learn \
    statsmodels \
    plotly \
    jupyter \
    jupyterlab \
    notebook \
    ipykernel \
    tensorflow \
    torch \
    openpyxl \
    xlsxwriter

echo ">>> Verificando instalación de FEniCSx (dolfinx)..."
python3 -c "import dolfinx; print('dolfinx OK, version:', dolfinx.__version__)"

echo ">>> Verificando LaTeX..."
pdflatex --version | head -n 1

echo ">>> ¡Listo! El entorno está configurado."
