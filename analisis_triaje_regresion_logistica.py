import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import os
import threading
import unicodedata

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

archivo_actual = None

# Para evitar que el modelo sea demasiado pesado con datasets
# de más de 1 millón de registros.
MAX_FILAS_MODELO = 200000


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def normalizar_texto(texto):
    """
    Convierte un texto a una forma sencilla para comparar nombres
    de columnas sin preocuparnos por mayúsculas o tildes.

    Ejemplo:
        "Nivel de Triaje" -> "nivel de triaje"
        "Día"             -> "dia"
    """
    texto = str(texto).strip().lower()
    texto = unicodedata.normalize("NFKD", texto)
    texto = "".join(
        caracter
        for caracter in texto
        if not unicodedata.combining(caracter)
    )
    return texto


def buscar_columna(columnas, posibles_nombres):
    """
    Busca una columna usando varios nombres posibles.

    Retorna el nombre REAL de la columna encontrada.
    """
    mapa = {
        normalizar_texto(columna): columna
        for columna in columnas
    }

    for nombre in posibles_nombres:
        clave = normalizar_texto(nombre)

        if clave in mapa:
            return mapa[clave]

    return None


def convertir_hora(valor):
    """
    Convierte distintos formatos de hora a un número de 0 a 23.

    Ejemplos:
        "14:30" -> 14
        "08:15:00" -> 8
        17 -> 17
    """
    if pd.isna(valor):
        return None

    # Si ya es número
    if isinstance(valor, (int, float)):
        hora = int(valor)

        if 0 <= hora <= 23:
            return hora

    texto = str(valor).strip()

    # Intentamos convertir como hora/fecha
    convertido = pd.to_datetime(
        texto,
        errors="coerce"
    )

    if not pd.isna(convertido):
        return convertido.hour

    # Intento simple para valores como "14:30"
    try:
        hora = int(texto.split(":")[0])

        if 0 <= hora <= 23:
            return hora
    except Exception:
        pass

    return None


# ============================================================
# SELECCIÓN DE ARCHIVO
# ============================================================

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


# ============================================================
# INICIO DEL ANÁLISIS
# ============================================================

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
        text="Procesando dataset y entrenando modelo..."
    )

    barra.start()

    hilo = threading.Thread(
        target=procesar_archivo,
        daemon=True
    )

    hilo.start()


# ============================================================
# LECTURA DEL DATASET
# ============================================================

def leer_dataset(ruta):
    """
    Lee únicamente las columnas necesarias.

    Las columnas que intentaremos utilizar son:
    - Nivel de triaje
    - Edad
    - Sexo
    - Hora
    - Día

    Primero leemos solo los encabezados para conocer
    exactamente cómo se llaman las columnas.
    """

    if ruta.lower().endswith(".csv"):
        encabezado = pd.read_csv(
            ruta,
            nrows=0,
            low_memory=False
        )
    else:
        encabezado = pd.read_excel(
            ruta,
            nrows=0,
            engine="openpyxl"
        )

    columnas = encabezado.columns.tolist()

    col_triaje = buscar_columna(
        columnas,
        [
            "Nivel de triaje",
            "Triaje",
            "Nivel Triaje",
            "Nivel de triage",
            "Triage"
        ]
    )

    col_edad = buscar_columna(
        columnas,
        ["Edad"]
    )

    col_sexo = buscar_columna(
        columnas,
        [
            "Sexo",
            "Genero",
            "Género"
        ]
    )

    col_hora = buscar_columna(
        columnas,
        ["Hora"]
    )

    col_dia = buscar_columna(
        columnas,
        [
            "Día",
            "Dia"
        ]
    )

    if col_triaje is None:
        raise ValueError(
            "No se encontró una columna de triaje.\n\n"
            "El programa buscó nombres como:\n"
            "'Nivel de triaje', 'Triaje' o 'Triage'."
        )

    columnas_utiles = [
        columna
        for columna in [
            col_triaje,
            col_edad,
            col_sexo,
            col_hora,
            col_dia
        ]
        if columna is not None
    ]

    if ruta.lower().endswith(".csv"):
        df = pd.read_csv(
            ruta,
            usecols=columnas_utiles,
            low_memory=False
        )
    else:
        df = pd.read_excel(
            ruta,
            usecols=columnas_utiles,
            engine="openpyxl"
        )

    return (
        df,
        col_triaje,
        col_edad,
        col_sexo,
        col_hora,
        col_dia
    )


# ============================================================
# PROCESAMIENTO PRINCIPAL
# ============================================================

def procesar_archivo():
    try:
        ruta = archivo_actual

        (
            df,
            col_triaje,
            col_edad,
            col_sexo,
            col_hora,
            col_dia
        ) = leer_dataset(ruta)

        # ====================================================
        # 1. ANÁLISIS DESCRIPTIVO DE TRIAJE
        # ====================================================

        columna_triaje = pd.to_numeric(
            df[col_triaje],
            errors="coerce"
        )

        columna_triaje = columna_triaje[
            columna_triaje.isin([1, 2, 3, 4, 5])
        ]

        conteo_total = (
            columna_triaje
            .value_counts()
            .to_dict()
        )

        total_validos = len(columna_triaje)

        resultados_triaje = []

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

            resultados_triaje.append(
                (
                    nivel,
                    cantidad,
                    porcentaje
                )
            )

        # ====================================================
        # 2. PREPARAR DATOS PARA REGRESIÓN LOGÍSTICA
        # ====================================================

        modelo_df = df.copy()

        modelo_df["TriajeNumerico"] = pd.to_numeric(
            modelo_df[col_triaje],
            errors="coerce"
        )

        modelo_df = modelo_df[
            modelo_df["TriajeNumerico"].isin(
                [1, 2, 3, 4, 5]
            )
        ].copy()

        # ----------------------------------------------------
        # VARIABLE OBJETIVO
        # ----------------------------------------------------
        # 0 = Triaje 1, 2 o 3
        # 1 = Triaje 4 o 5
        #
        # Esta es la variable que el modelo intentará predecir.
        modelo_df["Objetivo_Triaje_45"] = (
            modelo_df["TriajeNumerico"]
            .isin([4, 5])
            .astype(int)
        )

        # ====================================================
        # 3. PREPARAR VARIABLES DE ENTRADA (X)
        # ====================================================

        variables_modelo = []

        # EDAD
        if col_edad is not None:
            modelo_df["Edad_modelo"] = pd.to_numeric(
                modelo_df[col_edad],
                errors="coerce"
            )
            variables_modelo.append(
                "Edad_modelo"
            )

        # HORA
        if col_hora is not None:
            modelo_df["Hora_modelo"] = (
                modelo_df[col_hora]
                .apply(convertir_hora)
            )

            variables_modelo.append(
                "Hora_modelo"
            )

        # SEXO
        if col_sexo is not None:
            modelo_df["Sexo_modelo"] = (
                modelo_df[col_sexo]
                .astype(str)
                .str.strip()
            )

            variables_modelo.append(
                "Sexo_modelo"
            )

        # DÍA
        if col_dia is not None:
            modelo_df["Dia_modelo"] = (
                modelo_df[col_dia]
                .astype(str)
                .str.strip()
            )

            variables_modelo.append(
                "Dia_modelo"
            )

        if len(variables_modelo) == 0:
            raise ValueError(
                "Se encontró la columna de triaje, "
                "pero no se encontraron variables para entrenar "
                "el modelo.\n\n"
                "Se recomienda tener columnas como:\n"
                "Edad, Sexo, Hora o Día."
            )

        # ====================================================
        # 4. ELIMINAR DATOS INVÁLIDOS
        # ====================================================

        columnas_necesarias = (
            variables_modelo
            + ["Objetivo_Triaje_45"]
        )

        modelo_df = modelo_df[
            columnas_necesarias
        ].copy()

        # Reemplazamos cadenas vacías por valores faltantes.
        modelo_df = modelo_df.replace(
            ["", "nan", "None", "NaN"],
            pd.NA
        )

        modelo_df = modelo_df.dropna()

        if len(modelo_df) < 50:
            raise ValueError(
                "Después de limpiar los datos quedaron muy pocos "
                "registros para entrenar un modelo confiable."
            )

        # ====================================================
        # 5. MUESTRA PARA EVITAR SOBRECARGAR LA COMPUTADORA
        # ====================================================

        filas_originales_modelo = len(modelo_df)

        if len(modelo_df) > MAX_FILAS_MODELO:
            modelo_df = modelo_df.sample(
                n=MAX_FILAS_MODELO,
                random_state=42
            )

        # ====================================================
        # 6. SEPARAR X e y
        # ====================================================

        X = modelo_df[
            variables_modelo
        ].copy()

        y = modelo_df[
            "Objetivo_Triaje_45"
        ]

        # ====================================================
        # 7. CONVERTIR VARIABLES CATEGÓRICAS A NÚMEROS
        # ====================================================
        #
        # Regresión logística necesita números.
        #
        # Ejemplo:
        # Sexo:
        #   Hombre
        #   Mujer
        #
        # pandas puede convertirlo a columnas:
        # Sexo_modelo_Hombre
        # Sexo_modelo_Mujer

        columnas_categoricas = []

        if "Sexo_modelo" in X.columns:
            columnas_categoricas.append(
                "Sexo_modelo"
            )

        if "Dia_modelo" in X.columns:
            columnas_categoricas.append(
                "Dia_modelo"
            )

        if columnas_categoricas:
            X = pd.get_dummies(
                X,
                columns=columnas_categoricas,
                drop_first=True,
                dtype=int
            )

        # Asegurarnos de que todo sea numérico.
        X = X.apply(
            pd.to_numeric,
            errors="coerce"
        )

        validos = ~X.isna().any(axis=1)

        X = X.loc[validos]
        y = y.loc[validos]

        # Deben existir las dos clases.
        if y.nunique() < 2:
            raise ValueError(
                "La variable objetivo tiene una sola clase.\n\n"
                "Para regresión logística deben existir registros "
                "tanto de triaje 1-3 como de triaje 4-5."
            )

        # ====================================================
        # 8. TRAIN / TEST
        # ====================================================
        #
        # 80% entrenamiento
        # 20% prueba
        #
        # stratify=y mantiene una proporción parecida de clases
        # en entrenamiento y prueba.

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y
        )

        # ====================================================
        # 9. ENTRENAR REGRESIÓN LOGÍSTICA
        # ====================================================

        modelo = LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        )

        modelo.fit(
            X_train,
            y_train
        )

        # ====================================================
        # 10. HACER PREDICCIONES
        # ====================================================

        predicciones = modelo.predict(
            X_test
        )

        # ====================================================
        # 11. MÉTRICAS
        # ====================================================

        accuracy = accuracy_score(
            y_test,
            predicciones
        )

        precision = precision_score(
            y_test,
            predicciones,
            zero_division=0
        )

        recall = recall_score(
            y_test,
            predicciones,
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            predicciones,
            zero_division=0
        )

        matriz = confusion_matrix(
            y_test,
            predicciones
        )

        tn, fp, fn, tp = matriz.ravel()

        resultados_modelo = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
            "train": len(X_train),
            "test": len(X_test),
            "variables": list(X.columns),
            "filas_originales_modelo": filas_originales_modelo,
            "filas_usadas_modelo": len(X)
        }

        # Enviar resultados al hilo principal de Tkinter
        ventana.after(
            0,
            mostrar_resultados,
            resultados_triaje,
            total_validos,
            resultados_modelo
        )

    except Exception as e:
        ventana.after(
            0,
            mostrar_error,
            str(e)
        )


# ============================================================
# MOSTRAR RESULTADOS
# ============================================================

def mostrar_resultados(
    resultados,
    total,
    resultados_modelo
):
    barra.stop()

    btn_analizar.config(state="normal")
    btn_archivo.config(state="normal")

    # Limpiar tabla
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
        text=(
            f"Análisis terminado. "
            f"Registros válidos: {total:,}"
        )
    )

    lbl_resultado.config(
        text=(
            f"Triaje 4 + 5: "
            f"{porcentaje_45:.2f}%"
        )
    )

    # ========================================================
    # RESULTADOS DEL MODELO
    # ========================================================

    lbl_accuracy.config(
        text=(
            f"Accuracy: "
            f"{resultados_modelo['accuracy'] * 100:.2f}%"
        )
    )

    lbl_precision.config(
        text=(
            f"Precision: "
            f"{resultados_modelo['precision'] * 100:.2f}%"
        )
    )

    lbl_recall.config(
        text=(
            f"Recall: "
            f"{resultados_modelo['recall'] * 100:.2f}%"
        )
    )

    lbl_f1.config(
        text=(
            f"F1-Score: "
            f"{resultados_modelo['f1'] * 100:.2f}%"
        )
    )

    lbl_division.config(
        text=(
            f"Entrenamiento: "
            f"{resultados_modelo['train']:,} registros | "
            f"Prueba: {resultados_modelo['test']:,} registros"
        )
    )

    lbl_matriz.config(
        text=(
            "Matriz de confusión\n\n"
            "                    Predicción\n"
            "                  0             1\n"
            f"Real 0       "
            f"{resultados_modelo['tn']:<10}"
            f"{resultados_modelo['fp']}\n"
            f"Real 1       "
            f"{resultados_modelo['fn']:<10}"
            f"{resultados_modelo['tp']}\n\n"
            "0 = Triaje 1, 2 o 3\n"
            "1 = Triaje 4 o 5"
        )
    )

    # Mostrar si se usó muestra
    original = resultados_modelo[
        "filas_originales_modelo"
    ]

    usadas = resultados_modelo[
        "filas_usadas_modelo"
    ]

    if original > usadas:
        lbl_muestra.config(
            text=(
                f"Modelo entrenado con una muestra de "
                f"{usadas:,} de {original:,} registros "
                f"para mejorar el rendimiento."
            )
        )
    else:
        lbl_muestra.config(
            text=(
                f"Modelo entrenado con "
                f"{usadas:,} registros válidos."
            )
        )


# ============================================================
# ERROR
# ============================================================

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


# ============================================================
# INTERFAZ
# ============================================================

ventana = tk.Tk()

ventana.title(
    "Minería de Datos - Triaje y Regresión Logística"
)

ventana.geometry(
    "820x850"
)

# Canvas para permitir scroll
canvas = tk.Canvas(
    ventana
)

scrollbar = ttk.Scrollbar(
    ventana,
    orient="vertical",
    command=canvas.yview
)

contenedor = tk.Frame(
    canvas
)

contenedor.bind(
    "<Configure>",
    lambda evento: canvas.configure(
        scrollregion=canvas.bbox("all")
    )
)

canvas.create_window(
    (0, 0),
    window=contenedor,
    anchor="nw"
)

canvas.configure(
    yscrollcommand=scrollbar.set
)

canvas.pack(
    side="left",
    fill="both",
    expand=True
)

scrollbar.pack(
    side="right",
    fill="y"
)

# ============================================================
# TÍTULO
# ============================================================

titulo = tk.Label(
    contenedor,
    text="Análisis de Nivel de Triaje",
    font=("Arial", 18, "bold")
)

titulo.pack(
    pady=20
)

# ============================================================
# ARCHIVO
# ============================================================

btn_archivo = tk.Button(
    contenedor,
    text="Seleccionar dataset",
    command=seleccionar_archivo,
    width=25
)

btn_archivo.pack(
    pady=5
)

lbl_archivo = tk.Label(
    contenedor,
    text="Ningún archivo seleccionado"
)

lbl_archivo.pack(
    pady=5
)

# ============================================================
# BOTÓN ANALIZAR
# ============================================================

btn_analizar = tk.Button(
    contenedor,
    text="Analizar",
    command=analizar,
    width=25
)

btn_analizar.pack(
    pady=10
)

barra = ttk.Progressbar(
    contenedor,
    mode="indeterminate",
    length=400
)

barra.pack(
    pady=10
)

lbl_estado = tk.Label(
    contenedor,
    text="Esperando archivo..."
)

lbl_estado.pack(
    pady=5
)

# ============================================================
# TABLA DE TRIAJE
# ============================================================

columnas = (
    "nivel",
    "cantidad",
    "porcentaje"
)

tabla = ttk.Treeview(
    contenedor,
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
    width=150,
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

tabla.pack(
    pady=20
)

lbl_resultado = tk.Label(
    contenedor,
    text="",
    font=("Arial", 14, "bold")
)

lbl_resultado.pack(
    pady=10
)

# ============================================================
# SECCIÓN REGRESIÓN LOGÍSTICA
# ============================================================

separador = ttk.Separator(
    contenedor,
    orient="horizontal"
)

separador.pack(
    fill="x",
    padx=50,
    pady=15
)

titulo_modelo = tk.Label(
    contenedor,
    text="Regresión Logística",
    font=("Arial", 18, "bold")
)

titulo_modelo.pack(
    pady=10
)

descripcion_modelo = tk.Label(
    contenedor,
    text=(
        "Objetivo: predecir si un registro pertenece "
        "a Triaje 4 o 5.\n"
        "Clase 0 = Triaje 1, 2 o 3 | "
        "Clase 1 = Triaje 4 o 5"
    ),
    font=("Arial", 10)
)

descripcion_modelo.pack(
    pady=5
)

lbl_division = tk.Label(
    contenedor,
    text="Entrenamiento: -- | Prueba: --",
    font=("Arial", 10, "bold")
)

lbl_division.pack(
    pady=8
)

lbl_accuracy = tk.Label(
    contenedor,
    text="Accuracy: --",
    font=("Arial", 12)
)

lbl_accuracy.pack(
    pady=3
)

lbl_precision = tk.Label(
    contenedor,
    text="Precision: --",
    font=("Arial", 12)
)

lbl_precision.pack(
    pady=3
)

lbl_recall = tk.Label(
    contenedor,
    text="Recall: --",
    font=("Arial", 12)
)

lbl_recall.pack(
    pady=3
)

lbl_f1 = tk.Label(
    contenedor,
    text="F1-Score: --",
    font=("Arial", 12, "bold")
)

lbl_f1.pack(
    pady=3
)

lbl_matriz = tk.Label(
    contenedor,
    text="Matriz de confusión: --",
    font=("Courier New", 11),
    justify="left"
)

lbl_matriz.pack(
    pady=15
)

lbl_muestra = tk.Label(
    contenedor,
    text="",
    font=("Arial", 9),
    wraplength=700
)

lbl_muestra.pack(
    pady=10
)

ventana.mainloop()
