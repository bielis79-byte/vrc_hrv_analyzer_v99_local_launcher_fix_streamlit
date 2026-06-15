# VRC / HRV RRi Analyzer Pro v9.9

## Cambios v8.6
- Añadida interpretación avanzada de HVG/grafos.
- Añadida clasificación compacto local / equilibrado / disperso global.
- Añadido índice de compactación HVG en escala -2 a +2.
- Añadidas definiciones y valores orientativos de métricas HVG.
- Añadidas definiciones y valores orientativos de dominios:
  - Amplitud
  - Vagal
  - Complejidad
  - MSE 1-20
  - Recurrencia
- Conserva correcciones v8.5:
  - SampEn y MSE con la misma señal y tolerancia.
  - MSE1 = SampEn.
  - Entropías con λ=500.


## v8.7
- Añadida exportación completa de gráficos desde la pestaña Exportar.
- El ZIP incluye una carpeta `graficos/` con archivos HTML interactivos.
- Se exportan gráficos de HRV, dominios, MSE, Dashboard, Poincaré, RRi superpuesto y HVG/grafos cuando están disponibles.
- Opción experimental para intentar exportar PNG si el servidor tiene kaleido disponible.


## v8.8
- Exportación de gráficos en PNG, SVG y/o HTML interactivo desde la pestaña Exportar.
- Añadido `kaleido` a requirements.txt para generar PNG/SVG directamente desde Plotly.
- El ZIP incluye un script `convert_html_to_png.py` para convertir los HTML a PNG localmente si Streamlit Cloud no permite PNG automático.
- Carpeta de salida:
  - `graficos/html`
  - `graficos/png`
  - `graficos/svg`
  - `graficos/indice_graficos_exportados.csv`


## v8.9
- Corrige el NameError de `CONVERT_HTML_TO_PNG_SCRIPT`.
- El script `convert_html_to_png.py` queda definido antes de la pestaña Exportar.
- Añadida protección para que un fallo al escribir el script no bloquee la app.


## v9.0
- Corrige el KeyError en Poincaré/Grafos cuando `long_df` no contiene columna `Fase`.
- La sección HVG ahora muestra un aviso si no hay métricas disponibles en vez de romper la app.


## v9.1
- Corrige la exportación de gráficos para conservar colores.
- Añadida paleta fija `EXPORT_COLORWAY`.
- PNG, SVG y HTML se preparan con plantilla `plotly_dark`, fondo oscuro y colores explícitos.
- Evita que los traces sin color explícito salgan en gris al exportar con Kaleido.


## v9.2
- Convierte los gráficos principales a formato columnas verticales + líneas de tendencia suavizadas.
- Actualiza comparativas individuales, dominios y MSE 1-20.
- Mantiene colores fijos en pantalla y exportación.


## v9.3
- Corrige el orden cronológico de registros con nombres truncados tipo `026-04-14`, interpretándolo como 2026.
- Las líneas de tendencia con 3 puntos ahora se suavizan mediante ajuste cuadrático.
- Añadido botón independiente: `Descargar sólo gráficos PNG`.


## v9.4
- Dashboard: leyendas manuales con color en el margen derecho de cada subplot.
- Mayor separación entre paneles para evitar solapamiento de leyendas con columnas o con otros gráficos.
- Fases ampliadas: Basal2-Basal5 y R1-R6 para poder seleccionar varias ventanas basales y más de dos ventanas de recuperación.
- Botones rápidos en la barra lateral para activar todas las ventanas basales o todas las de recuperación.


## v9.5
- Añadido script `capture_streamlit_localhost_png.py`.
- Permite capturar `http://localhost:8501/` como PNG cuando ejecutas la app localmente.
- Se mantiene `convert_html_to_png.py` para convertir los HTML exportados a PNG.


## v9.6
- Corrige la posición de las leyendas del dashboard.
- Las leyendas quedan dentro de la esquina superior derecha de cada subplot.
- Se elimina el desplazamiento de leyendas hacia el margen inferior.


## v9.7
- Añadido `Arrancar_Convertidor.bat`.
- Usa `%~dp0`, por lo que funciona desde cualquier carpeta o equipo Windows.
- Abre `http://localhost:8501/`.
- Ejecuta `capture_streamlit_localhost_png.py`.
- También intenta ejecutar `convert_html_to_png.py` para convertir HTML exportados a PNG.


## v9.8
- Corrige el NameError de `ARRANCAR_CONVERTIDOR_BAT`.
- La constante queda definida antes de usarse.
- Añadida protección por si la constante no estuviera disponible en tiempo de ejecución.


## v9.9
- Corrige el problema del enlace `http://localhost:8501/`.
- El BAT ahora no sólo abre el enlace: primero arranca Streamlit localmente con `python -m streamlit run app.py --server.port 8501`.
- Usa `%~dp0` y `cd /d "%~dp0"`, por lo que funciona desde cualquier carpeta.
- Añade Playwright a requirements para el capturador local.
