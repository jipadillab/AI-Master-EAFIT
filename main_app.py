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
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import (accuracy_score, precision_score, f1_score, confusion_matrix,
                             mean_squared_error, mean_absolute_error, r2_score, roc_curve, auc)

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

st.title("Plataforma Interactiva de Aprendizaje Supervisado")
st.markdown("**Autor:** Jorge Iván Padilla Buriticá, Profesor de IA - Universidad EAFIT")

# --- 1. CONFIGURACIÓN Y CARGA DE DATOS ---
st.sidebar.header("1. Configuración del Problema")
task_type = st.sidebar.selectbox("¿Qué tarea deseas realizar?", ("Clasificación", "Regresión"))

@st.cache_data
def get_dataset(dataset_name, task):
    if task == "Clasificación":
        if dataset_name == "Iris": data = datasets.load_iris()
        elif dataset_name == "Wine": data = datasets.load_wine()
        elif dataset_name == "Breast Cancer": data = datasets.load_breast_cancer()
        elif dataset_name == "Digits": data = datasets.load_digits()
        else: 
            X, y = datasets.make_moons(n_samples=500, noise=0.3, random_state=42)
            return X, y, ["F1", "F2"], ["Clase 0", "Clase 1"]
    else:
        if dataset_name == "Diabetes": data = datasets.load_diabetes()
        elif dataset_name == "California Housing": 
            data = datasets.fetch_california_housing()
            # Reducir tamaño para la demo web para no bloquear Streamlit
            X, y = data.data[:1000], data.target[:1000]
            return X, y, data.feature_names, None
        else:
            X, y = datasets.make_regression(n_samples=500, n_features=2, noise=15, random_state=42)
            return X, y, ["Feature 1", "Feature 2"], ["Target"]
    
    return data.data, data.target, data.feature_names, getattr(data, 'target_names', None)

st.sidebar.header("2. Selección de Datos")
if task_type == "Clasificación":
    dataset_name = st.sidebar.selectbox("Dataset", ("Iris", "Wine", "Breast Cancer", "Digits", "Moons (Sintético)"))
else:
    dataset_name = st.sidebar.selectbox("Dataset", ("Diabetes", "California Housing", "Regresión Sintética"))

X, y, feature_names, target_names = get_dataset(dataset_name, task_type)

# --- 2. PREPROCESAMIENTO ---
st.sidebar.header("3. Preprocesamiento")
use_pca = st.sidebar.checkbox("Aplicar PCA (Reducir a 2D para visualización)", value=True)
use_scaling = st.sidebar.checkbox("Estandarizar datos", value=True)

if use_scaling:
    X = StandardScaler().fit_transform(X)

# Siempre necesitamos 2D para graficar fronteras cómodamente
if use_pca and X.shape[1] > 2:
    pca = PCA(n_components=2)
    X_viz = pca.fit_transform(X)
else:
    X_viz = X[:, :2] if X.shape[1] >= 2 else np.hstack((X, np.zeros_like(X)))

# --- VISUALIZACIÓN INICIAL DE LA BASE DE DATOS ---
with st.expander("👁️ Ver Exploración de la Base de Datos (EDA)", expanded=False):
    st.write(f"**Dataset:** {dataset_name} | **Muestras:** {X.shape[0]} | **Características originales:** {X.shape[1]}")
    fig_eda, ax_eda = plt.subplots(figsize=(6, 4))
    if task_type == "Clasificación":
        scatter = ax_eda.scatter(X_viz[:, 0], X_viz[:, 1], c=y, cmap='viridis', edgecolor='k', alpha=0.7)
        legend1 = ax_eda.legend(*scatter.legend_elements(), title="Clases")
        ax_eda.add_artist(legend1)
    else:
        scatter = ax_eda.scatter(X_viz[:, 0], X_viz[:, 1], c=y, cmap='coolwarm', edgecolor='k', alpha=0.7)
        plt.colorbar(scatter, ax=ax_eda, label='Valor Target')
    ax_eda.set_title("Distribución de los Datos (Espacio 2D)")
    ax_eda.set_xlabel("Componente 1")
    ax_eda.set_ylabel("Componente 2")
    st.pyplot(fig_eda)

# --- 3. VALIDACIÓN Y MODELO ---
st.sidebar.header("4. Estrategia de Validación")
test_size = st.sidebar.slider("Porcentaje de Prueba (Test Size)", 0.1, 0.9, 0.3, step=0.05)

st.sidebar.header("5. Selección del Modelo")
if task_type == "Clasificación":
    model_name = st.sidebar.selectbox("Algoritmo", ("Naive Bayes", "Regresión Logística", "KNN", "SVM", "Árbol de Decisión", "Random Forest", "Red Neuronal (MLP)"))
    if model_name == "SVM": model = SVC(probability=True, kernel='rbf', C=1.0) # probability=True es vital para ROC
    elif model_name == "KNN": model = KNeighborsClassifier(n_neighbors=5)
    elif model_name == "Random Forest": model = RandomForestClassifier(max_depth=5, random_state=42)
    elif model_name == "Red Neuronal (MLP)": model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
    elif model_name == "Árbol de Decisión": model = DecisionTreeClassifier(max_depth=5)
    elif model_name == "Regresión Logística": model = LogisticRegression()
    else: model = GaussianNB()
else:
    model_name = st.sidebar.selectbox("Algoritmo", ("Regresión Lineal", "KNN Regressor", "SVR", "Random Forest Regressor", "Red Neuronal (MLP) Regressor"))
    if model_name == "SVR": model = SVR(C=1.0)
    elif model_name == "KNN Regressor": model = KNeighborsRegressor(n_neighbors=5)
    elif model_name == "Random Forest Regressor": model = RandomForestRegressor(max_depth=5, random_state=42)
    elif model_name == "Red Neuronal (MLP) Regressor": model = MLPRegressor(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
    else: model = LinearRegression()

# --- ENTRENAMIENTO ---
X_train, X_test, y_train, y_test = train_test_split(X_viz, y, test_size=test_size, random_state=42)
model.fit(X_train, y_train)

y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

st.markdown("---")
st.subheader(f"Resultados del Modelo: {model_name}")

col1, col2 = st.columns([1, 1.2])

with col1:
    st.markdown("#### Métricas y Gráficas de Desempeño")
    if task_type == "Clasificación":
        st.write(f"**Accuracy (Train):** {accuracy_score(y_train, y_pred_train):.3f} | **(Test):** {accuracy_score(y_test, y_pred_test):.3f}")
        
        # Gráfica ROC (Train vs Test)
        fig_roc, ax_roc = plt.subplots(figsize=(5, 4))
        if hasattr(model, "predict_proba") and len(np.unique(y)) == 2: # Binario
            y_prob_train = model.predict_proba(X_train)[:, 1]
            y_prob_test = model.predict_proba(X_test)[:, 1]
            
            fpr_tr, tpr_tr, _ = roc_curve(y_train, y_prob_train)
            fpr_te, tpr_te, _ = roc_curve(y_test, y_prob_test)
            
            ax_roc.plot(fpr_tr, tpr_tr, label=f'Train AUC = {auc(fpr_tr, tpr_tr):.2f}', color='blue')
            ax_roc.plot(fpr_te, tpr_te, label=f'Test AUC = {auc(fpr_te, tpr_te):.2f}', color='red', linestyle='--')
            ax_roc.plot([0, 1], [0, 1], 'k--')
            ax_roc.set_xlabel('Falsos Positivos')
            ax_roc.set_ylabel('Verdaderos Positivos')
            ax_roc.set_title('Curva ROC: Train vs Validation')
            ax_roc.legend(loc='lower right')
            st.pyplot(fig_roc)
        else:
            st.info("Curva ROC omitida (El dataset es multiclase o el modelo no soporta probabilidades directas). Se muestra la matriz de confusión del Test.")
            fig_cm, ax_cm = plt.subplots(figsize=(4, 3))
            sns.heatmap(confusion_matrix(y_test, y_pred_test), annot=True, fmt='d', cmap='Blues', ax=ax_cm)
            ax_cm.set_title("Matriz de Confusión (Test)")
            st.pyplot(fig_cm)

    else: # Regresión
        st.write(f"**R² (Train):** {r2_score(y_train, y_pred_train):.3f} | **(Test):** {r2_score(y_test, y_pred_test):.3f}")
        st.write(f"**MSE (Train):** {mean_squared_error(y_train, y_pred_train):.3f} | **(Test):** {mean_squared_error(y_test, y_pred_test):.3f}")
        
        # Gráfica Real vs Predicción (Train vs Test)
        fig_reg, ax_reg = plt.subplots(figsize=(5, 4))
        ax_reg.scatter(y_train, y_pred_train, alpha=0.5, label='Train', color='blue', marker='o')
        ax_reg.scatter(y_test, y_pred_test, alpha=0.7, label='Test', color='red', marker='x')
        
        min_val = min(min(y), min(y_pred_train), min(y_pred_test))
        max_val = max(max(y), max(y_pred_train), max(y_pred_test))
        ax_reg.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label='Ideal')
        
        ax_reg.set_xlabel("Valores Reales")
        ax_reg.set_ylabel("Predicciones")
        ax_reg.set_title("Real vs Predicción (Train vs Validation)")
        ax_reg.legend()
        st.pyplot(fig_reg)

with col2:
    st.markdown("#### Frontera de Decisión (Espacio 2D)")
    st.caption("🟢/🔵 Círculos = Datos de Entrenamiento | ❌ Cruces = Datos de Prueba (Validación)")
    
    x_min, x_max = X_viz[:, 0].min() - 1, X_viz[:, 0].max() + 1
    y_min, y_max = X_viz[:, 1].min() - 1, X_viz[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    fig_bound, ax_bound = plt.subplots(figsize=(6, 5))
    
    if task_type == "Clasificación":
        ax_bound.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
        # Train points
        ax_bound.scatter(X_train[:, 0], X_train[:, 1], c=y_train, marker='o', s=40, edgecolor='k', cmap='viridis', label='Train')
        # Test points
        ax_bound.scatter(X_test[:, 0], X_test[:, 1], c=y_test, marker='x', s=60, cmap='viridis', label='Test')
    else:
        contour = ax_bound.contourf(xx, yy, Z, alpha=0.4, cmap='coolwarm')
        plt.colorbar(contour, ax=ax_bound, label='Predicción Continua')
        # Train points
        ax_bound.scatter(X_train[:, 0], X_train[:, 1], c=y_train, marker='o', s=40, edgecolor='k', cmap='coolwarm', label='Train')
        # Test points
        ax_bound.scatter(X_test[:, 0], X_test[:, 1], c=y_test, marker='x', s=60, cmap='coolwarm', label='Test', linewidths=2)
    
    ax_bound.set_xlabel("Componente Principal 1")
    ax_bound.set_ylabel("Componente Principal 2")
    ax_bound.legend(loc='best')
    st.pyplot(fig_bound)
