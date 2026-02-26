import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-Learn modules
from sklearn import datasets
from sklearn.model_selection import train_test_split, KFold, LeaveOneOut, cross_val_score
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, f1_score, confusion_matrix,
                             mean_squared_error, mean_absolute_error, r2_score)

# Models
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="IA EAFIT - Aprendizaje Supervisado", layout="wide")

# --- ENCABEZADO ---
st.title("Plataforma Interactiva de Aprendizaje Supervisado")
st.markdown("""
**Autor:** Jorge Iván Padilla Buriticá, Profesor de IA - Universidad EAFIT  
**Contexto:** Desarrollado para la Maestría en Ciencia de Datos.
""")
st.write("Explora pipelines completos de Machine Learning: Preprocesamiento, Selección de Modelos, Validación y Evaluación.")

# --- BARRA LATERAL: CONFIGURACIÓN GENERAL ---
st.sidebar.header("1. Configuración del Problema")
task_type = st.sidebar.selectbox("¿Qué tarea deseas realizar?", ("Clasificación", "Regresión"))

# --- CARGA DE DATOS ---
def get_dataset(dataset_name, task):
    if task == "Clasificación":
        if dataset_name == "Iris": data = datasets.load_iris()
        elif dataset_name == "Wine": data = datasets.load_wine()
        elif dataset_name == "Breast Cancer": data = datasets.load_breast_cancer()
        elif dataset_name == "Digits": data = datasets.load_digits()
        else: data = datasets.make_moons(n_samples=500, noise=0.3, random_state=42); return data[0], data[1], ["F1", "F2"], ["Clase 0", "Clase 1"]
    else:
        if dataset_name == "Diabetes": data = datasets.load_diabetes()
        elif dataset_name == "California Housing": data = datasets.fetch_california_housing()
        else:
            # Synthetic regression data
            X, y = datasets.make_regression(n_samples=500, n_features=1, noise=20, random_state=42)
            return X, y, ["Feature 1"], ["Target"]
    
    return data.data, data.target, data.feature_names, getattr(data, 'target_names', None)

st.sidebar.header("2. Selección de Datos")
if task_type == "Clasificación":
    dataset_name = st.sidebar.selectbox("Selecciona un Dataset", ("Iris", "Wine", "Breast Cancer", "Digits", "Moons (Sintético)"))
else:
    dataset_name = st.sidebar.selectbox("Selecciona un Dataset", ("Diabetes", "California Housing", "Regresión Sintética"))

X, y, feature_names, target_names = get_dataset(dataset_name, task_type)

st.write(f"### Dataset seleccionado: {dataset_name}")
st.write(f"Forma del dataset: `X: {X.shape}`, `y: {y.shape}`")

# --- PREPROCESAMIENTO Y EXTRACCIÓN (PCA / Clustering) ---
st.sidebar.header("3. Preprocesamiento")
use_pca = st.sidebar.checkbox("Aplicar PCA (Reducir a 2D)", value=True)
use_scaling = st.sidebar.checkbox("Estandarizar datos (StandardScaler)", value=True)
add_clustering = st.sidebar.checkbox("Agregar Clustering (K-Means) como feature", value=False)

if use_scaling:
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

if add_clustering:
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X).reshape(-1, 1)
    X = np.hstack((X, clusters))
    feature_names.append("Cluster_ID")

if use_pca and X.shape[1] > 2:
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    st.write(f"Varianza explicada por PCA: `{sum(pca.explained_variance_ratio_)*100:.2f}%`")
else:
    X_pca = X[:, :2] if X.shape[1] > 2 else X # Tomamos 2 features para graficar fronteras

# --- ESTRATEGIA DE VALIDACIÓN ---
st.sidebar.header("4. Estrategia de Validación")
val_method = st.sidebar.selectbox("Método de Validación", ("Train/Test Split", "K-Fold Cross Validation", "Leave-One-Out (LOOCV)"))
if val_method == "Train/Test Split":
    test_size = st.sidebar.slider("Porcentaje de Prueba (Test Size)", 0.1, 0.9, 0.3)
elif val_method == "K-Fold Cross Validation":
    k_folds = st.sidebar.slider("Número de Folds (k)", 2, 10, 5)

# --- SELECCIÓN Y CONFIGURACIÓN DEL MODELO ---
st.sidebar.header("5. Selección del Modelo")
def get_model(task_type):
    if task_type == "Clasificación":
        model_name = st.sidebar.selectbox("Algoritmo", 
            ("Naive Bayes", "Regresión Logística", "KNN", "SVM", "Árbol de Decisión", "Random Forest", "Ensambles (Gradient Boosting)", "Red Neuronal (MLP)"))
        
        # Hyperparameters
        if model_name == "KNN":
            k = st.sidebar.slider("K (Vecinos)", 1, 15, 5)
            return model_name, KNeighborsClassifier(n_neighbors=k)
        elif model_name == "SVM":
            c = st.sidebar.slider("C (Regularización)", 0.01, 10.0, 1.0)
            kernel = st.sidebar.selectbox("Kernel", ("linear", "rbf", "poly"))
            return model_name, SVC(C=c, kernel=kernel)
        elif model_name == "Random Forest":
            n_estimators = st.sidebar.slider("Número de árboles", 10, 200, 50)
            max_depth = st.sidebar.slider("Profundidad máxima", 1, 20, 5)
            return model_name, RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        elif model_name == "Red Neuronal (MLP)":
            hidden_layers = st.sidebar.slider("Neuronas capa oculta", 10, 200, 100)
            return model_name, MLPClassifier(hidden_layer_sizes=(hidden_layers,), max_iter=1000, random_state=42)
        elif model_name == "Ensambles (Gradient Boosting)":
            return model_name, GradientBoostingClassifier(n_estimators=100, random_state=42)
        elif model_name == "Árbol de Decisión":
            depth = st.sidebar.slider("Profundidad máxima", 1, 20, 5)
            return model_name, DecisionTreeClassifier(max_depth=depth, random_state=42)
        elif model_name == "Regresión Logística":
            return model_name, LogisticRegression(max_iter=1000)
        else:
            return model_name, GaussianNB()
            
    else: # Regresión
        model_name = st.sidebar.selectbox("Algoritmo", 
            ("Regresión Lineal", "KNN Regressor", "SVR", "Árbol de Decisión Regressor", "Random Forest Regressor", "Red Neuronal (MLP) Regressor"))
        
        if model_name == "KNN Regressor":
            k = st.sidebar.slider("K (Vecinos)", 1, 15, 5)
            return model_name, KNeighborsRegressor(n_neighbors=k)
        elif model_name == "SVR":
            c = st.sidebar.slider("C (Regularización)", 0.01, 10.0, 1.0)
            return model_name, SVR(C=c)
        elif model_name == "Random Forest Regressor":
            n_estimators = st.sidebar.slider("Número de árboles", 10, 200, 50)
            return model_name, RandomForestRegressor(n_estimators=n_estimators, random_state=42)
        elif model_name == "Árbol de Decisión Regressor":
            depth = st.sidebar.slider("Profundidad", 1, 20, 5)
            return model_name, DecisionTreeRegressor(max_depth=depth)
        elif model_name == "Red Neuronal (MLP) Regressor":
            hidden_layers = st.sidebar.slider("Neuronas capa oculta", 10, 200, 100)
            return model_name, MLPRegressor(hidden_layer_sizes=(hidden_layers,), max_iter=1000, random_state=42)
        else:
            return model_name, LinearRegression()

model_name, model = get_model(task_type)

# --- ENTRENAMIENTO Y EVALUACIÓN ---
st.write(f"### Entrenamiento y Resultados: {model_name}")

if val_method == "Train/Test Split":
    X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=test_size, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Métricas de Evaluación")
        if task_type == "Clasificación":
            st.write(f"**Accuracy:** {accuracy_score(y_test, y_pred):.4f}")
            st.write(f"**Precision (macro):** {precision_score(y_test, y_pred, average='macro', zero_division=0):.4f}")
            st.write(f"**F1 Score (macro):** {f1_score(y_test, y_pred, average='macro', zero_division=0):.4f}")
            
            fig, ax = plt.subplots(figsize=(4,3))
            sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_title("Matriz de Confusión")
            st.pyplot(fig)
        else:
            st.write(f"**MSE (Mean Squared Error):** {mean_squared_error(y_test, y_pred):.4f}")
            st.write(f"**MAE (Mean Absolute Error):** {mean_absolute_error(y_test, y_pred):.4f}")
            st.write(f"**R2 Score:** {r2_score(y_test, y_pred):.4f}")
            
            fig, ax = plt.subplots(figsize=(4,3))
            ax.scatter(y_test, y_pred, alpha=0.6)
            ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
            ax.set_xlabel("Valores Reales")
            ax.set_ylabel("Predicciones")
            ax.set_title("Real vs Predicción")
            st.pyplot(fig)

    with col2:
        st.subheader("Frontera de Decisión (Espacio 2D)")
        # Para graficar fronteras de decisión se requiere un grid 2D
        x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
        y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05),
                             np.arange(y_min, y_max, 0.05))
        
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        fig, ax = plt.subplots(figsize=(5, 4))
        if task_type == "Clasificación":
            ax.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
            scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y, s=20, edgecolor='k', cmap='viridis')
        else:
            ax.contourf(xx, yy, Z, alpha=0.5, cmap='coolwarm')
            scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y, s=20, edgecolor='k', cmap='coolwarm')
        
        ax.set_xlabel("Componente / Feature 1")
        ax.set_ylabel("Componente / Feature 2")
        st.pyplot(fig)

else:
    # Cross Validation setup
    st.write(f"Calculando {val_method}...")
    cv = KFold(n_splits=k_folds, shuffle=True, random_state=42) if "K-Fold" in val_method else LeaveOneOut()
    scoring = 'accuracy' if task_type == "Clasificación" else 'r2'
    scores = cross_val_score(model, X_pca, y, cv=cv, scoring=scoring)
    
    st.success(f"**Score promedio ({scoring}):** {np.mean(scores):.4f} (+/- {np.std(scores):.4f})")
    st.info("Nota: Las visualizaciones gráficas detalladas aplican mejor bajo partición simple. Intenta Train/Test split para ver fronteras de decisión.")

# --- SECCIÓN PEDAGÓGICA Y DESPLIEGUE ---
st.markdown("---")
st.header("📚 Conceptos Fundamentales y Despliegue en Producción")

st.markdown("""
### ¿Qué deben dominar en esta etapa de la Maestría?
Como futuros Magísteres en Ciencia de Datos, es imperativo que comprendan que entrenar un modelo es solo el comienzo. Deben interiorizar:
1. **Bias-Variance Tradeoff:** Entender cómo modelos complejos (ej. Redes Neuronales profundas o Árboles sin podar) tienden al sobreajuste (Alta varianza), mientras que modelos muy simples (Regresión Lineal pura) pueden no capturar el patrón (Alto sesgo).
2. **Validación Cruzada:** La partición 70-30 es útil, pero estrategias como *K-Fold* o *Leave-One-Subject-Out* son críticas en ambientes donde los datos son escasos o existen agrupaciones naturales (ej. medidas repetidas de un mismo paciente o sensor).
3. **Maldición de la Dimensionalidad:** Por qué herramientas como PCA o selección de características son vitales antes de aplicar algoritmos basados en distancias como KNN.

### 🚀 Recomendaciones para Despliegue (Producción Masiva)
Una vez que el modelo es validado (como hemos hecho arriba), llevarlo a producción requiere ingeniería de software sólida:
* **Serialización:** Guardar el modelo pre-entrenado y los objetos de preprocesamiento (como el `StandardScaler` o `PCA`) usando `joblib` o formato `ONNX` para interoperabilidad.
* **APIs RESTful:** Exponer el modelo a través de frameworks como **FastAPI** o **Flask**. Esto permite que cualquier aplicación front-end consuma las predicciones enviando un archivo JSON.
* **Contenedores (Docker):** Empaquetar el entorno completo (Python, dependencias, modelo) en una imagen Docker para garantizar que funcione idénticamente en tu laptop, en los servidores de la universidad, o en la nube (AWS, GCP, Azure).
* **Monitoreo (MLOps):** Los modelos en producción sufren de *Data Drift* (los datos cambian con el tiempo). Implementar herramientas de monitoreo garantiza que si el *accuracy* cae por debajo de un umbral, se dispare una alerta para reentrenar el modelo con datos frescos.
""")
