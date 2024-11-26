import streamlit as st
import pickle
import numpy as np

# Configuración de la página
st.set_page_config(page_title="Predicción de Supervivencia en el Titanic", layout="centered")

# Cargar los modelos entrenados
with open('random_forest_model.pkl', 'rb') as rf_file:
    rf_model = pickle.load(rf_file)

with open('svm_model.pkl', 'rb') as svm_file:
    svm_model = pickle.load(svm_file)

# Función para hacer predicciones
def predict_survival(model, features):
    prediction = model.predict([features])
    return "Sobrevivió" if prediction[0] == 1 else "No sobrevivió"

# Título de la aplicación
st.title("Predicción de Supervivencia en el Titanic")
st.write("Ingrese los datos del pasajero para predecir si habría sobrevivido o no.")

# Formularios para ingresar datos del pasajero
pclass = st.selectbox("Clase del pasajero (Pclass)", options=[1, 2, 3], index=2)
sex = st.selectbox("Sexo", options=["Male", "Female"], index=0)
age = st.number_input("Edad", min_value=0.0, max_value=100.0, value=30.0, step=0.1)
sibsp = st.number_input("Número de hermanos/esposos a bordo (SibSp)", min_value=0, max_value=10, value=0, step=1)
parch = st.number_input("Número de padres/hijos a bordo (Parch)", min_value=0, max_value=10, value=0, step=1)
fare = st.number_input("Tarifa pagada (Fare)", min_value=0.0, max_value=1000.0, value=32.0, step=0.1)

# Convertir sexo a variable numérica
sex_encoded = 1 if sex == "Female" else 0

# Botón para realizar predicciones
if st.button("Predecir"):
    # Crear el conjunto de características
    features = [pclass, sex_encoded, age, sibsp, parch, fare]

    # Predicción con Random Forest
    rf_prediction = predict_survival(rf_model, features)

    # Predicción con SVM
    svm_prediction = predict_survival(svm_model, features)

    # Mostrar resultados
    st.subheader("Resultados de la predicción")
    st.write(f"**Random Forest:** {rf_prediction}")
    st.write(f"**SVM:** {svm_prediction}")
