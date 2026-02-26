# Plataforma Interactiva de Aprendizaje Supervisado 🤖📊

Esta aplicación interactiva en Streamlit está diseñada para la enseñanza de modelos de Aprendizaje Supervisado (Clasificación y Regresión) a nivel de posgrado. Permite explorar de forma dinámica todas las etapas de un pipeline de Machine Learning: desde el preprocesamiento y extracción de características (PCA, Clustering), hasta el entrenamiento, validación cruzada y evaluación de múltiples algoritmos.

**Autor:** Jorge Iván Padilla Buriticá, Profesor de IA - Universidad EAFIT.
**Contexto:** Desarrollado para la Maestría en Ciencia de Datos.

## Características Principales
* **Tareas soportadas:** Clasificación y Regresión.
* **Datasets integrados:** Digits, Breast Cancer, Wine, Diabetes, California Housing, y datos sintéticos.
* **Modelos:** Naive Bayes, Regresión Lineal/Logística, KNN, Árboles de Decisión, Random Forest, SVM, Ensambles y Redes Neuronales (MLP).
* **Validación:** Partición personalizable (Train/Test Split), K-Fold, Leave-One-Out (LOOCV).
* **Visualización:** Fronteras de decisión, reducción de dimensionalidad con PCA, matrices de confusión y curvas de regresión.

## Instrucciones de Instalación
1. Clona este repositorio o descarga los archivos.
2. Crea un entorno virtual (opcional pero recomendado): `python -m venv env`
3. Activa el entorno virtual.
4. Instala las dependencias: `pip install -r requirements.txt`
5. Ejecuta la aplicación: `streamlit run main_app.py`
