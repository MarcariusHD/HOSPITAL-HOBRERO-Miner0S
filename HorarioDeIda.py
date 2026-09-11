import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
import threading
import os

archivo_actual = None


# ================================
# SELECCIONAR ARCHIVO
# ================================

def seleccionar_archivo():
    global archivo_actual

    ruta = filedialog.askopenfilename(
        title="Seleccionar dataset",
        filetypes=[
            ("Excel", "*.xlsx"),
            ("CSV", "*.csv"),
            ("Todos", "*.*")
        ]
    )

    if not ruta:
        return

    archivo_actual = ruta

    lbl_archivo.config(
        text=os.path.basename(ruta)
    )

    lbl_estado.config(
        text="Archivo seleccionado. Presiona Analizar."
    )


# ================================
# ANALIZAR
# ================================

def analizar():

    if archivo_actual is None:

        messagebox.showwarning(
            "Aviso",
            "Primero selecciona un archivo."
        )

        return

    btn_analizar.config(state="disabled")
    btn_archivo.config(state="disabled")

    barra.start()

    lbl_estado.config(
        text="Analizando horarios y triaje..."
    )

    hilo = threading.Thread(
        target=procesar_archivo,
        daemon=True
    )

    hilo.start()


# ================================
# CLASIFICAR HORARIO
# ================================

def obtener_franja(hora):

    if pd.isna(hora):
        return None

    h = hora.hour

    if 0 <= h < 6:
        return "00:00 - 06:00"

    elif 6 <= h < 12:
        return "06:00 - 12:00"

    elif 12 <= h < 18:
        return "12:00 - 18:00"

    else:
        return "18:00 - 24:00"


# ================================
# PROCESAR ARCHIVO
# ================================

def procesar_archivo():

    try:

        ruta = archivo_actual

        # --------------------------------
        # LEER SOLO LAS COLUMNAS NECESARIAS
        # --------------------------------

        if ruta.lower().endswith(".csv"):

            df = pd.read_csv(
                ruta,
                usecols=["Hora", "Nivel de triaje"],
                low_memory=False
            )

        else:

            df = pd.read_excel(
                ruta,
                usecols=["Hora", "Nivel de triaje"],
                engine="openpyxl"
            )

        # --------------------------------
        # LIMPIAR TRIAJE
        # --------------------------------

        df["Nivel de triaje"] = pd.to_numeric(
            df["Nivel de triaje"],
            errors="coerce"
        )

        df = df[
            df["Nivel de triaje"].isin([1, 2, 3, 4, 5])
        ]

        # --------------------------------
        # CONVERTIR HORA
        # --------------------------------

        df["Hora"] = pd.to_datetime(
            df["Hora"],
            errors="coerce"
        )

        # Eliminar horas inválidas
        df = df.dropna(
            subset=["Hora"]
        )

        # --------------------------------
        # CREAR FRANJA HORARIA
        # --------------------------------

        df["Franja horaria"] = df["Hora"].apply(
            obtener_franja
        )

        # --------------------------------
        # TABLA CRUZADA
        # --------------------------------

        tabla_resultado = pd.crosstab(
            df["Franja horaria"],
            df["Nivel de triaje"]
        )

        # Asegurar que siempre aparezcan
        # los niveles del 1 al 5

        for nivel in [1, 2, 3, 4, 5]:

            if nivel not in tabla_resultado.columns:
                tabla_resultado[nivel] = 0

        tabla_resultado = tabla_resultado[
            [1, 2, 3, 4, 5]
        ]

        # Orden correcto de horarios

        orden = [
            "00:00 - 06:00",
            "06:00 - 12:00",
            "12:00 - 18:00",
            "18:00 - 24:00"
        ]

        tabla_resultado = tabla_resultado.reindex(
            orden,
            fill_value=0
        )

        # Total por franja

        tabla_resultado["Total"] = (
            tabla_resultado.sum(axis=1)
        )

        # --------------------------------
        # FRANJA CON MÁS PACIENTES
        # --------------------------------

        franja_maxima = (
            tabla_resultado["Total"].idxmax()
        )

        cantidad_maxima = int(
            tabla_resultado.loc[
                franja_maxima,
                "Total"
            ]
        )

        # --------------------------------
        # TRIAJE MÁS COMÚN POR FRANJA
        # --------------------------------

        triaje_mas_comun = {}

        for franja in orden:

            fila = tabla_resultado.loc[
                franja,
                [1, 2, 3, 4, 5]
            ]

            nivel = int(
                fila.idxmax()
            )

            cantidad = int(
                fila.max()
            )

            triaje_mas_comun[franja] = (
                nivel,
                cantidad
            )

        # Mostrar resultados

        ventana.after(
            0,
            mostrar_resultados,
            tabla_resultado,
            franja_maxima,
            cantidad_maxima,
            triaje_mas_comun
        )

    except Exception as e:

        ventana.after(
            0,
            mostrar_error,
            str(e)
        )


# ================================
# MOSTRAR RESULTADOS
# ================================

def mostrar_resultados(
    resultados,
    franja_maxima,
    cantidad_maxima,
    triaje_comun
):

    barra.stop()

    btn_analizar.config(state="normal")
    btn_archivo.config(state="normal")

    # Limpiar tabla

    for item in tabla.get_children():
        tabla.delete(item)

    # Insertar resultados

    for franja, fila in resultados.iterrows():

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
                f"{int(fila['Total']):,}"
            )
        )

    # Resultado principal

    texto = (
        f"Franja con mayor cantidad de pacientes:\n"
        f"{franja_maxima}\n"
        f"Total: {cantidad_maxima:,} pacientes"
    )

    lbl_resultado.config(
        text=texto
    )

    # Mostrar triaje más común por cada franja

    resumen = ""

    for franja, datos in triaje_comun.items():

        nivel, cantidad = datos

        resumen += (
            f"{franja}: "
            f"Triaje {nivel} "
            f"({cantidad:,} pacientes)\n"
        )

    lbl_resumen.config(
        text=resumen
    )

    lbl_estado.config(
        text="Análisis terminado correctamente."
    )


# ================================
# ERROR
# ================================

def mostrar_error(error):

    barra.stop()

    btn_analizar.config(state="normal")
    btn_archivo.config(state="normal")

    lbl_estado.config(
        text="Error durante el análisis."
    )

    messagebox.showerror(
        "Error",
        error
    )


# ================================
# INTERFAZ
# ================================

ventana = tk.Tk()

ventana.title(
    "Análisis de Horarios y Triaje"
)

ventana.geometry(
    "950x650"
)


titulo = tk.Label(
    ventana,
    text="Análisis de Horarios por Nivel de Triaje",
    font=("Arial", 18, "bold")
)

titulo.pack(
    pady=20
)


btn_archivo = tk.Button(
    ventana,
    text="Seleccionar dataset",
    command=seleccionar_archivo,
    width=25
)

btn_archivo.pack(
    pady=5
)


lbl_archivo = tk.Label(
    ventana,
    text="Ningún archivo seleccionado"
)

lbl_archivo.pack(
    pady=5
)


btn_analizar = tk.Button(
    ventana,
    text="Analizar",
    command=analizar,
    width=25
)

btn_analizar.pack(
    pady=10
)


barra = ttk.Progressbar(
    ventana,
    mode="indeterminate",
    length=500
)

barra.pack(
    pady=10
)


lbl_estado = tk.Label(
    ventana,
    text="Esperando archivo..."
)

lbl_estado.pack(
    pady=5
)


# ================================
# TABLA
# ================================

columnas = (
    "franja",
    "t1",
    "t2",
    "t3",
    "t4",
    "t5",
    "total"
)

tabla = ttk.Treeview(
    ventana,
    columns=columnas,
    show="headings",
    height=6
)


tabla.heading(
    "franja",
    text="Franja horaria"
)

tabla.heading(
    "t1",
    text="Triaje 1"
)

tabla.heading(
    "t2",
    text="Triaje 2"
)

tabla.heading(
    "t3",
    text="Triaje 3"
)

tabla.heading(
    "t4",
    text="Triaje 4"
)

tabla.heading(
    "t5",
    text="Triaje 5"
)

tabla.heading(
    "total",
    text="Total"
)


tabla.column(
    "franja",
    width=150,
    anchor="center"
)

for col in [
    "t1",
    "t2",
    "t3",
    "t4",
    "t5",
    "total"
]:

    tabla.column(
        col,
        width=110,
        anchor="center"
    )


tabla.pack(
    pady=20
)


# ================================
# RESULTADOS
# ================================

lbl_resultado = tk.Label(
    ventana,
    text="",
    font=("Arial", 14, "bold")
)

lbl_resultado.pack(
    pady=10
)


lbl_resumen = tk.Label(
    ventana,
    text="",
    font=("Arial", 11),
    justify="left"
)

lbl_resumen.pack(
    pady=5
)


ventana.mainloop()