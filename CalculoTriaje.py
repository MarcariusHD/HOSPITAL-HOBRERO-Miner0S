import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
import threading
import os

archivo_actual = None


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


def analizar():
    if archivo_actual is None:
        messagebox.showwarning(
            "Aviso",
            "Primero selecciona un archivo."
        )
        return

    btn_analizar.config(state="disabled")
    btn_archivo.config(state="disabled")

    lbl_estado.config(
        text="Procesando dataset... puede tardar unos minutos."
    )

    barra.start()

    hilo = threading.Thread(
        target=procesar_archivo,
        daemon=True
    )

    hilo.start()


def procesar_archivo():

    try:

        ruta = archivo_actual

        # ============================
        # CSV
        # ============================

        if ruta.lower().endswith(".csv"):

            conteo_total = {}

            total_validos = 0

            # Procesamos por bloques
            for chunk in pd.read_csv(
                ruta,
                usecols=["Nivel de triaje"],
                chunksize=100000,
                low_memory=False
            ):

                columna = pd.to_numeric(
                    chunk["Nivel de triaje"],
                    errors="coerce"
                )

                columna = columna[
                    columna.isin([1, 2, 3, 4, 5])
                ]

                conteo = columna.value_counts()

                for nivel, cantidad in conteo.items():

                    conteo_total[nivel] = (
                        conteo_total.get(nivel, 0)
                        + cantidad
                    )

                total_validos += len(columna)

        # ============================
        # EXCEL
        # ============================

        else:

            # Leer solamente la columna necesaria
            df = pd.read_excel(
                ruta,
                usecols=["Nivel de triaje"],
                engine="openpyxl"
            )

            columna = pd.to_numeric(
                df["Nivel de triaje"],
                errors="coerce"
            )

            columna = columna[
                columna.isin([1, 2, 3, 4, 5])
            ]

            conteo_total = (
                columna
                .value_counts()
                .to_dict()
            )

            total_validos = len(columna)

        # ============================
        # RESULTADOS
        # ============================

        resultados = []

        for nivel in range(1, 6):

            cantidad = int(
                conteo_total.get(nivel, 0)
            )

            if total_validos > 0:

                porcentaje = (
                    cantidad / total_validos
                ) * 100

            else:
                porcentaje = 0

            resultados.append(
                (
                    nivel,
                    cantidad,
                    porcentaje
                )
            )

        ventana.after(
            0,
            mostrar_resultados,
            resultados,
            total_validos
        )

    except Exception as e:

        ventana.after(
            0,
            mostrar_error,
            str(e)
        )


def mostrar_resultados(resultados, total):

    barra.stop()

    btn_analizar.config(state="normal")
    btn_archivo.config(state="normal")

    for item in tabla.get_children():
        tabla.delete(item)

    porcentaje_45 = 0

    for nivel, cantidad, porcentaje in resultados:

        tabla.insert(
            "",
            "end",
            values=(
                nivel,
                f"{cantidad:,}",
                f"{porcentaje:.2f}%"
            )
        )

        if nivel in [4, 5]:
            porcentaje_45 += porcentaje

    lbl_estado.config(
        text=f"Análisis terminado. Registros válidos: {total:,}"
    )

    lbl_resultado.config(
        text=(
            f"Triaje 4 + 5: "
            f"{porcentaje_45:.2f}%"
        )
    )


def mostrar_error(error):

    barra.stop()

    btn_analizar.config(state="normal")
    btn_archivo.config(state="normal")

    lbl_estado.config(
        text="Error durante el procesamiento."
    )

    messagebox.showerror(
        "Error",
        error
    )


# =================================
# INTERFAZ
# =================================

ventana = tk.Tk()

ventana.title(
    "Minería de Datos - Triaje"
)

ventana.geometry(
    "700x500"
)


titulo = tk.Label(
    ventana,
    text="Análisis de Nivel de Triaje",
    font=("Arial", 18, "bold")
)

titulo.pack(pady=20)


btn_archivo = tk.Button(
    ventana,
    text="Seleccionar dataset",
    command=seleccionar_archivo,
    width=25
)

btn_archivo.pack(pady=5)


lbl_archivo = tk.Label(
    ventana,
    text="Ningún archivo seleccionado"
)

lbl_archivo.pack(pady=5)


btn_analizar = tk.Button(
    ventana,
    text="Analizar",
    command=analizar,
    width=25
)

btn_analizar.pack(pady=10)


barra = ttk.Progressbar(
    ventana,
    mode="indeterminate",
    length=400
)

barra.pack(pady=10)


lbl_estado = tk.Label(
    ventana,
    text="Esperando archivo..."
)

lbl_estado.pack(pady=5)


columnas = (
    "nivel",
    "cantidad",
    "porcentaje"
)

tabla = ttk.Treeview(
    ventana,
    columns=columnas,
    show="headings",
    height=6
)

tabla.heading(
    "nivel",
    text="Nivel de triaje"
)

tabla.heading(
    "cantidad",
    text="Cantidad"
)

tabla.heading(
    "porcentaje",
    text="Porcentaje"
)

tabla.column(
    "nivel",
    width=130,
    anchor="center"
)

tabla.column(
    "cantidad",
    width=180,
    anchor="center"
)

tabla.column(
    "porcentaje",
    width=180,
    anchor="center"
)

tabla.pack(pady=20)


lbl_resultado = tk.Label(
    ventana,
    text="",
    font=("Arial", 14, "bold")
)

lbl_resultado.pack(pady=10)


ventana.mainloop()