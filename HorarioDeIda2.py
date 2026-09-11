import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
import threading
import os
from datetime import datetime, time

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

archivo_actual = None
tabla_conteos_global = None
tabla_porcentajes_global = None


# ==========================================
# UTILIDADES
# ==========================================

def obtener_franja_desde_hora(hora_entera):
    if hora_entera is None:
        return None

    if 0 <= hora_entera < 6:
        return "00:00 - 06:00"
    elif 6 <= hora_entera < 12:
        return "06:00 - 12:00"
    elif 12 <= hora_entera < 18:
        return "12:00 - 18:00"
    elif 18 <= hora_entera < 24:
        return "18:00 - 24:00"
    return None


def extraer_hora(valor):
    """
    Intenta extraer la hora desde distintos formatos:
    - datetime
    - time
    - string tipo '21:30' o '2025-01-10 21:30:00'
    - números de Excel (fracción del día)
    """

    if pd.isna(valor):
        return None

    # datetime.time
    if isinstance(valor, time):
        return valor.hour

    # datetime o Timestamp
    if isinstance(valor, (datetime, pd.Timestamp)):
        return valor.hour

    # Si es numérico y parece una fracción de día de Excel
    if isinstance(valor, (int, float)):
        if 0 <= valor < 1:
            return int(valor * 24)
        if 0 <= valor <= 23:
            return int(valor)

    # Intentar parsear texto
    texto = str(valor).strip()
    if not texto:
        return None

    dt = pd.to_datetime(texto, errors="coerce")
    if pd.notna(dt):
        return int(dt.hour)

    return None


def encontrar_columnas(df):
    """
    Busca automáticamente la columna de triaje y la de hora/fecha.
    """

    columnas = [str(c).strip() for c in df.columns]

    posibles_triaje = [
        "Nivel de triaje",
        "nivel de triaje",
        "Triaje",
        "triaje",
        "Nivel triaje"
    ]

    posibles_hora = [
        "Hora",
        "hora"
    ]

    posibles_fecha = [
        "Fecha de atención",
        "fecha de atención",
        "Fecha",
        "fecha",
        "Fecha de Atencion",
        "fecha de atencion"
    ]

    col_triaje = None
    col_tiempo = None

    for col in posibles_triaje:
        if col in columnas:
            col_triaje = col
            break

    for col in posibles_hora:
        if col in columnas:
            col_tiempo = col
            break

    if col_tiempo is None:
        for col in posibles_fecha:
            if col in columnas:
                col_tiempo = col
                break

    return col_triaje, col_tiempo


# ==========================================
# ACCIONES DE INTERFAZ
# ==========================================

def seleccionar_archivo():
    global archivo_actual

    ruta = filedialog.askopenfilename(
        title="Seleccionar dataset",
        filetypes=[
            ("Excel", "*.xlsx *.xls"),
            ("CSV", "*.csv"),
            ("Todos los archivos", "*.*")
        ]
    )

    if not ruta:
        return

    archivo_actual = ruta
    lbl_archivo.config(text=os.path.basename(ruta))
    lbl_estado.config(text="Archivo seleccionado. Presiona 'Analizar'.")


def analizar():
    if archivo_actual is None:
        messagebox.showwarning("Aviso", "Primero selecciona un archivo.")
        return

    btn_analizar.config(state="disabled")
    btn_archivo.config(state="disabled")
    btn_guardar.config(state="disabled")

    barra.start()
    lbl_estado.config(text="Analizando horario y triaje...")

    hilo = threading.Thread(target=procesar_archivo, daemon=True)
    hilo.start()


def procesar_archivo():
    try:
        ruta = archivo_actual

        # Leer solo encabezados primero
        if ruta.lower().endswith(".csv"):
            encabezado = pd.read_csv(ruta, nrows=5, low_memory=False)
        else:
            encabezado = pd.read_excel(ruta, nrows=5, engine="openpyxl")

        col_triaje, col_tiempo = encontrar_columnas(encabezado)

        if col_triaje is None:
            raise ValueError("No se encontró la columna de 'Nivel de triaje'.")

        if col_tiempo is None:
            raise ValueError("No se encontró una columna de 'Hora' o 'Fecha de atención'.")

        # Leer solo las columnas necesarias
        if ruta.lower().endswith(".csv"):
            df = pd.read_csv(
                ruta,
                usecols=[col_tiempo, col_triaje],
                low_memory=False
            )
        else:
            df = pd.read_excel(
                ruta,
                usecols=[col_tiempo, col_triaje],
                engine="openpyxl"
            )

        # Limpiar triaje
        df[col_triaje] = pd.to_numeric(df[col_triaje], errors="coerce")
        df = df[df[col_triaje].isin([1, 2, 3, 4, 5])]

        # Extraer hora
        df["Hora_extraida"] = df[col_tiempo].apply(extraer_hora)
        df = df.dropna(subset=["Hora_extraida"])

        # Crear franja
        df["Franja horaria"] = df["Hora_extraida"].apply(obtener_franja_desde_hora)
        df = df.dropna(subset=["Franja horaria"])

        orden_franjas = [
            "00:00 - 06:00",
            "06:00 - 12:00",
            "12:00 - 18:00",
            "18:00 - 24:00"
        ]

        # Tabla de conteos
        tabla_conteos = pd.crosstab(df["Franja horaria"], df[col_triaje])

        for nivel in [1, 2, 3, 4, 5]:
            if nivel not in tabla_conteos.columns:
                tabla_conteos[nivel] = 0

        tabla_conteos = tabla_conteos[[1, 2, 3, 4, 5]]
        tabla_conteos = tabla_conteos.reindex(orden_franjas, fill_value=0)
        tabla_conteos["Total"] = tabla_conteos.sum(axis=1)

        total_general = int(tabla_conteos["Total"].sum())

        # % del total general por franja
        tabla_conteos["% del total"] = (
            tabla_conteos["Total"] / total_general * 100
        ).round(2)

        # Tabla de porcentajes internos por franja
        tabla_porcentajes = tabla_conteos[[1, 2, 3, 4, 5]].div(
            tabla_conteos["Total"].replace(0, pd.NA), axis=0
        ) * 100

        tabla_porcentajes = tabla_porcentajes.fillna(0).round(2)

        # Franja con más pacientes
        franja_maxima = tabla_conteos["Total"].idxmax()
        cantidad_maxima = int(tabla_conteos.loc[franja_maxima, "Total"])
        porcentaje_maximo = float(tabla_conteos.loc[franja_maxima, "% del total"])

        # Triaje dominante por franja
        resumen = []
        for franja in orden_franjas:
            fila_cant = tabla_conteos.loc[franja, [1, 2, 3, 4, 5]]
            fila_pct = tabla_porcentajes.loc[franja, [1, 2, 3, 4, 5]]

            nivel_dom = int(fila_cant.idxmax())
            cant_dom = int(fila_cant.max())
            pct_dom = float(fila_pct.loc[nivel_dom])

            resumen.append((franja, nivel_dom, cant_dom, pct_dom))

        ventana.after(
            0,
            mostrar_resultados,
            tabla_conteos,
            tabla_porcentajes,
            total_general,
            franja_maxima,
            cantidad_maxima,
            porcentaje_maximo,
            resumen
        )

    except Exception as e:
        ventana.after(0, mostrar_error, str(e))


def mostrar_resultados(
    tabla_conteos,
    tabla_porcentajes,
    total_general,
    franja_maxima,
    cantidad_maxima,
    porcentaje_maximo,
    resumen
):
    global tabla_conteos_global, tabla_porcentajes_global

    tabla_conteos_global = tabla_conteos.copy()
    tabla_porcentajes_global = tabla_porcentajes.copy()

    barra.stop()

    btn_analizar.config(state="normal")
    btn_archivo.config(state="normal")
    btn_guardar.config(state="normal")

    # Limpiar tabla
    for item in tabla.get_children():
        tabla.delete(item)

    # Insertar datos
    for franja, fila in tabla_conteos.iterrows():
        tabla.insert(
            "",
            "end",
            values=(
                franja,
                f"{int(fila[1]):,}",
                f"{int(fila[2]):,}",
                f"{int(fila[3]):,}",
                f"{int(fila[4]):,}",
                f"{int(fila[5]):,}",
                f"{int(fila['Total']):,}",
                f"{float(fila['% del total']):.2f}%"
            )
        )

    lbl_estado.config(
        text=f"Análisis terminado. Registros válidos: {total_general:,}"
    )

    lbl_resumen_principal.config(
        text=(
            f"Franja con mayor cantidad de pacientes: {franja_maxima}\n"
            f"Total: {cantidad_maxima:,} pacientes ({porcentaje_maximo:.2f}% del total)"
        )
    )

    texto_detalle = ""
    for franja, nivel, cantidad, porcentaje in resumen:
        texto_detalle += (
            f"{franja}: Triaje dominante = {nivel} "
            f"({cantidad:,} pacientes, {porcentaje:.2f}% dentro de la franja)\n"
        )

    lbl_resumen_secundario.config(text=texto_detalle)

    dibujar_grafico_totales(tabla_conteos)
    dibujar_grafico_porcentajes(tabla_porcentajes)


def mostrar_error(error):
    barra.stop()

    btn_analizar.config(state="normal")
    btn_archivo.config(state="normal")

    lbl_estado.config(text="Error durante el análisis.")
    messagebox.showerror("Error", error)


def guardar_reporte():
    if tabla_conteos_global is None or tabla_porcentajes_global is None:
        messagebox.showwarning("Aviso", "Primero debes realizar el análisis.")
        return

    ruta = filedialog.asksaveasfilename(
        defaultextension=".xlsx",
        filetypes=[("Excel", "*.xlsx")],
        title="Guardar reporte"
    )

    if not ruta:
        return

    try:
        with pd.ExcelWriter(ruta, engine="openpyxl") as writer:
            tabla_exportar_1 = tabla_conteos_global.reset_index()
            tabla_exportar_1.to_excel(writer, sheet_name="Conteos", index=False)

            tabla_exportar_2 = tabla_porcentajes_global.reset_index()
            tabla_exportar_2.to_excel(writer, sheet_name="Porcentajes", index=False)

        messagebox.showinfo("Éxito", "Reporte guardado correctamente.")
    except Exception as e:
        messagebox.showerror("Error", str(e))


# ==========================================
# GRÁFICOS
# ==========================================

def limpiar_frame(frame):
    for widget in frame.winfo_children():
        widget.destroy()


def dibujar_grafico_totales(tabla_conteos):
    limpiar_frame(frame_grafico_1)

    figura = plt.Figure(figsize=(7, 4), dpi=100)
    ax = figura.add_subplot(111)

    x = list(tabla_conteos.index)
    y = tabla_conteos["Total"].tolist()

    ax.bar(x, y)
    ax.set_title("Total de pacientes por franja horaria")
    ax.set_xlabel("Franja horaria")
    ax.set_ylabel("Cantidad de pacientes")
    ax.grid(axis="y", alpha=0.3)

    for i, valor in enumerate(y):
        ax.text(i, valor, f"{valor:,}", ha="center", va="bottom", fontsize=9)

    figura.tight_layout()

    canvas = FigureCanvasTkAgg(figura, master=frame_grafico_1)
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True)


def dibujar_grafico_porcentajes(tabla_porcentajes):
    limpiar_frame(frame_grafico_2)

    figura = plt.Figure(figsize=(7, 4), dpi=100)
    ax = figura.add_subplot(111)

    x = list(tabla_porcentajes.index)

    base = [0] * len(x)
    niveles = [1, 2, 3, 4, 5]

    for nivel in niveles:
        valores = tabla_porcentajes[nivel].tolist()
        ax.bar(x, valores, bottom=base, label=f"Triaje {nivel}")
        base = [base[i] + valores[i] for i in range(len(base))]

    ax.set_title("Distribución porcentual del triaje por franja")
    ax.set_xlabel("Franja horaria")
    ax.set_ylabel("Porcentaje dentro de la franja")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    figura.tight_layout()

    canvas = FigureCanvasTkAgg(figura, master=frame_grafico_2)
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True)


# ==========================================
# INTERFAZ
# ==========================================

ventana = tk.Tk()
ventana.title("Análisis de Horarios por Nivel de Triaje")
ventana.geometry("1250x850")

titulo = tk.Label(
    ventana,
    text="Análisis de Horarios por Nivel de Triaje",
    font=("Arial", 22, "bold")
)
titulo.pack(pady=15)

frame_superior = tk.Frame(ventana)
frame_superior.pack(pady=5)

btn_archivo = tk.Button(
    frame_superior,
    text="Seleccionar dataset",
    width=22,
    command=seleccionar_archivo
)
btn_archivo.grid(row=0, column=0, padx=8)

btn_analizar = tk.Button(
    frame_superior,
    text="Analizar",
    width=22,
    command=analizar
)
btn_analizar.grid(row=0, column=1, padx=8)

btn_guardar = tk.Button(
    frame_superior,
    text="Guardar reporte",
    width=22,
    command=guardar_reporte,
    state="disabled"
)
btn_guardar.grid(row=0, column=2, padx=8)

lbl_archivo = tk.Label(
    ventana,
    text="Ningún archivo seleccionado",
    font=("Arial", 10)
)
lbl_archivo.pack(pady=4)

barra = ttk.Progressbar(
    ventana,
    mode="indeterminate",
    length=600
)
barra.pack(pady=10)

lbl_estado = tk.Label(
    ventana,
    text="Esperando archivo...",
    font=("Arial", 10)
)
lbl_estado.pack(pady=5)

lbl_resumen_principal = tk.Label(
    ventana,
    text="",
    font=("Arial", 15, "bold"),
    justify="center"
)
lbl_resumen_principal.pack(pady=8)

lbl_resumen_secundario = tk.Label(
    ventana,
    text="",
    font=("Arial", 11),
    justify="center"
)
lbl_resumen_secundario.pack(pady=5)

# Notebook con pestañas
notebook = ttk.Notebook(ventana)
notebook.pack(fill="both", expand=True, padx=15, pady=15)

# Pestaña tabla
tab_tabla = tk.Frame(notebook)
notebook.add(tab_tabla, text="Tabla de resultados")

frame_tabla = tk.Frame(tab_tabla)
frame_tabla.pack(fill="both", expand=True, padx=10, pady=10)

columnas = (
    "franja", "t1", "t2", "t3", "t4", "t5", "total", "pct_total"
)

tabla = ttk.Treeview(
    frame_tabla,
    columns=columnas,
    show="headings",
    height=12
)

tabla.heading("franja", text="Franja horaria")
tabla.heading("t1", text="Triaje 1")
tabla.heading("t2", text="Triaje 2")
tabla.heading("t3", text="Triaje 3")
tabla.heading("t4", text="Triaje 4")
tabla.heading("t5", text="Triaje 5")
tabla.heading("total", text="Total")
tabla.heading("pct_total", text="% del total")

tabla.column("franja", width=170, anchor="center")
tabla.column("t1", width=90, anchor="center")
tabla.column("t2", width=90, anchor="center")
tabla.column("t3", width=90, anchor="center")
tabla.column("t4", width=90, anchor="center")
tabla.column("t5", width=90, anchor="center")
tabla.column("total", width=110, anchor="center")
tabla.column("pct_total", width=110, anchor="center")

scroll_y = ttk.Scrollbar(frame_tabla, orient="vertical", command=tabla.yview)
tabla.configure(yscrollcommand=scroll_y.set)

tabla.pack(side="left", fill="both", expand=True)
scroll_y.pack(side="right", fill="y")

# Pestaña gráficos
tab_graficos = tk.Frame(notebook)
notebook.add(tab_graficos, text="Gráficos")

frame_grafico_1 = tk.LabelFrame(
    tab_graficos,
    text="Gráfico 1: Total por franja horaria",
    padx=10,
    pady=10
)
frame_grafico_1.pack(fill="both", expand=True, padx=10, pady=10)

frame_grafico_2 = tk.LabelFrame(
    tab_graficos,
    text="Gráfico 2: Distribución porcentual de triaje por franja",
    padx=10,
    pady=10
)
frame_grafico_2.pack(fill="both", expand=True, padx=10, pady=10)

ventana.mainloop()