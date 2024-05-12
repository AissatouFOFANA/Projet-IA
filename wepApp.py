import pandas as pd  # type: ignore
from sklearn.metrics import accuracy_score  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore
from sklearn.ensemble import RandomForestClassifier  # type: ignore
from PIL import Image  # type: ignore
import streamlit as st # type: ignore

# Créons un titre et un sous-titre
st.write("""
# Diabetes Detection
   Détecter si quelqu'un a le diabète en utilisant le machine learning et le Python!      
""")
# Open and display an image
image = Image.open('D:\licence GLSI\2e SEMESTRE\Veille technologique\Projet IA\Projet-IA\diabete_1.jpg')
st.image(image, caption='ML', use_column_width=True)

# Récupérons les données
# Assurez-vous de spécifier le chemin correct du fichier contenant les données
df = pd.read_csv('#chemin du fichier contenant les donnees')

# Sous-titre
st.subheader('Data Information:')
# Affichons les données sous forme de table
st.dataframe(df)
# Show statistics on the data
st.write(df.describe())
# Show the data as a chart
chart = st.bar_chart(df)

# Split the data into independent 'X' and dependent 'Y' variables
x = df.iloc[:, 0:8].values
y = df.iloc[:, -1].values
# Split the data into 75% training and 25% testing
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.25, random_state=0)

# Get the feature input from the user
def get_user_input():
    pregnancies = st.sidebar.slider('pregnancies', 0, 17, 3)
    glucose = st.sidebar.slider('glucose', 0, 199, 117)
    blood_pressure = st.sidebar.slider('blood_pressure', 0, 122, 72)
    skin_thickness = st.sidebar.slider('skin+thickness', 0, 99, 23)
    insulin = st.sidebar.slider('insuline', 0.0, 846.0, 30.5)
    BMI = st.sidebar.slider('BMI', 0.0, 67.1, 32.0)
    DPF = st.sidebar.slider('DPF', 0.078, 2.42, 0.3725)
    age = st.sidebar.slider('age', 21, 81, 29)

    # Store a dictionary into a variable
    user_data = {'pregnancies' : pregnancies,
                 'glucose': glucose,
                 'blood_pressure': blood_pressure,
                 'skin_thickness': skin_thickness,
                 'insulin':insulin,
                 'BMI':BMI,
                 'DPF':DPF,
                 'age':age
                 }
    # Transform the data into a data frame
    features = pd.DataFrame(user_data, index=[0])
    return features

# Store the user input into a variable
user_input = get_user_input()

# Set a subheader and display the users input
st.subheader('User Input:')
st.write(user_input)

# Create and train the model
RandomForestClassifier = RandomForestClassifier()
RandomForestClassifier.fit(X_train, Y_train)

# Show the models metrics
st.subheader('Model Test Accuracy Score:')
st.write(str(accuracy_score(Y_test, RandomForestClassifier.predict(X_test)) *100)+'%')

# Store the models predictions in a variable
prediction = RandomForestClassifier.predict(user_input)

# Set a subheader and display the classification
st.subheader('Classification:')
st.write(prediction)
