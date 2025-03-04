import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression


def main():
    


    st.title("Revisa tu salud")
    st.subheader("Rellena el siguiente formulario para averiguar la probabilidad de riesgo en el corazón")

    with st.form("risk"):
        col1,col2,col3 = st.columns(3)
        with col1:
            dolorPecho = st.radio(label="¿Sueles tener dolor en el pecho?",  options=["Si", "No"])
            faltaAire = st.radio(label="¿Sueles notar que te falta el aire?",  options=["Si", "No"])
            fatiga = st.radio(label="¿Sueles tener fatiga sin razón aparente?",  options=["Si", "No"])
            palp = st.radio(label="¿Sueles notar las palpitaciones del corazón?",  options=["Si", "No"])
            mareos = st.radio(label="¿Sueles tener mareos?",  options=["Si", "No"])
            hinchazones = st.radio(label="¿Notas hinchazones inusuales?",  options=["Si", "No"])
        with col2:
            dolorAJB = st.radio(label="¿Notas dolor en brazos, mandibula o espalda?",  options=["Si", "No"])
            nausea = st.radio(label="¿Tienes nauseas o sudores fríos?",  options=["Si", "No"])
            altasPulsaciones = st.radio(label="¿Tienes un ritmo cardiaco elevado?",  options=["Si", "No"])
            altoColesterol = st.radio(label="¿Tienes el nivel de colesterol alto?",  options=["Si", "No"])
            diabetes = st.radio(label="¿Tienes diabetes?",  options=["Si", "No"])
            fumar = st.radio(label="¿Fumas?",  options=["Si", "No"])
        with col3:
            obesidad = st.radio(label="¿Sufres de obesidad?",  options=["Si", "No"])
            vidaSedentaria = st.radio(label="¿Haces ejercicio regularmente?",  options=["Si", "No"])
            familia = st.radio(label="¿Conoces algún caso en tu familia de problemas en el corazón?",  options=["Si", "No"])
            estresCronico = st.radio(label="¿Sufres de estrés?",  options=["Si", "No"])
            genero = st.radio(label="¿Cuál es tu genero?",  options=["Femenino", "Masculino"])
            edad = st.number_input(label="Introduce tu edad", format="%0.1f")

       

        submit = st.form_submit_button(label="enviar")
        if submit:
            values = [dolorPecho, faltaAire, fatiga, palp, mareos, hinchazones, dolorAJB,nausea, altasPulsaciones, altoColesterol, diabetes, fumar, obesidad, vidaSedentaria, familia, estresCronico, genero, edad]


            for i in range(len(values)):
                if (values[i] == "Si") or (values[i] == "Masculino"):
                     values[i] = 1.0
                elif (values[i] == "No") or (values[i] == "Femenino"):
                     values[i] = 0.0
            
            genDataset(values)



def genDataset(values):

    

    # st.write(values)

    # Lista de nombres de columnas
    columns = [
        'Chest_Pain', 'Shortness_of_Breath', 'Fatigue', 'Palpitations', 'Dizziness',
        'Swelling', 'Pain_Arms_Jaw_Back', 'Cold_Sweats_Nausea', 'High_BP', 'High_Cholesterol',
        'Diabetes', 'Smoking', 'Obesity', 'Sedentary_Lifestyle', 'Family_History',
        'Chronic_Stress', 'Gender', 'Age'
    ]

    # Crear el DataFrame usando comprensión de diccionario
    new_data = pd.DataFrame({col: [values[idx]] for idx, col in enumerate(columns)})
    predict(new_data)

data = pd.read_csv("hds.csv")
X = data.drop('Heart_Risk', axis=1)
optimal_k = 3
# Aplicar KMeans con el número óptimo de clusters
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
data['Cluster'] = kmeans.fit_predict(X)

models = {}
for cluster in data['Cluster'].unique():
    cluster_data = data[data['Cluster'] == cluster]
    X_cluster = cluster_data.drop(['Heart_Risk', 'Cluster'], axis=1)
    y_cluster = cluster_data['Heart_Risk']
    
    X_train, X_test, y_train, y_test = train_test_split(X_cluster, y_cluster, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    models[cluster] = model
    

def predict(new_data):
# Predecir el cluster para los nuevos datos
    cluster = kmeans.predict(new_data)
    
    # Seleccionar el modelo correspondiente al cluster predicho
    model = models[cluster[0]]
    
    # Predecir usando el modelo seleccionado
    prediction = model.predict(new_data)[0]

    risk = True if prediction > 0.4 else False
    prediction = 1.0 if prediction > 1.0 else abs(prediction)
    prediction = round(prediction, 2)

    warn = "⚠️" if risk else "😄"
    @st.dialog(f"Resultados: {warn}")
    def resultGood():
        st.subheader(f"La probabilidad de tener riesgo de corazon es del {prediction * 100}%")

    resultGood()


if __name__ == "__main__":
    main() 