from flask import Flask, request, jsonify
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pandas as pd
from io import StringIO

app = Flask(__name__)

# Variáveis globais para armazenar o modelo treinado e dados
model = None
X_train = None
y_train = None

# Função para carregar o conjunto de dados do CSV
def load_data():
    csv_data = """
    Diagnostico;Eritracitos;Hemoglobina;Hematacrito;Eritracitos;VGM;CHGM;Metarrubracitos;Proteina Plasmatica;Leucacitos;Leucograma;Segmentados;Bastonetes;Segmentados;Metamielacitos;Mielacitos;Linfacitos;Monacitos;Eosinafilos;Basafilos;Plaquetas;
    DRC;6,5;14,7;44;100;67,69;33,4;100;9,9;28.500;100;23.085;1.140;100;0;0;1.710;1.425;1.140;0;Agregadas
    Hipercolesterolemia;6,47;15,1;45;100;69,55;33,55;1;8;8.500;100;4.420;0;100;0;0;3.655;340;85;0;245.000
    Anemia;5,9;13,3;39;100;66,1;34,1;0;7,5;9.800;100;6.762;0;100;0;0;1.960;490;588;0;368.000
    Lesao hepatica;7,23;19;55;100;76,38;34,54;0;7,8;7.300;100;4.088;0;100;0;0;1.752;730;730;0;432.000
    LesÃ£o hepÃ¡tica,"7,23",19,55,,"76,38","34,54",0,"7,8",7.300,,4.088,0,,0,0,1.752,730,730,0,432.000



    """

    # Lendo o CSV a partir da string
    df = pd.read_csv(StringIO(csv_data), delimiter=";")

    # Convertendo as colunas para os tipos corretos
    for column in df.columns:
        if df[column].dtype == object:
            df[column] = df[column].str.replace(',', '.').astype(float)

    return df

# Rota para treinar o modelo
@app.route('/train_model', methods=['POST'])
def train_model():
    global model, X_train, y_train

    # Carregando os dados
    data = load_data()

    # Separando os dados em conjuntos de treinamento e teste
    features = data.drop('Diagnostico', axis=1)
    labels = data['Diagnostico']
    X_train, _, y_train, _ = train_test_split(features, labels, test_size=0.2, random_state=42)

    # Treinando o modelo de regressão linear
    model = LinearRegression()
    model.fit(X_train, y_train)

    return jsonify({'message': 'Model trained successfully'})

# Rota para realizar predição
@app.route('/predict_diagnosis', methods=['POST'])
def predict_diagnosis():
    global model

    # Carregando os dados
    data = request.get_json()
    features = [data['Eritracitos'], data['Hemoglobina'], data['Hematacrito'], data['Eritracitos'],
                data['VGM'], data['CHGM'], data['Metarrubracitos'], data['Proteina Plasmatica'],
                data['Leucacitos'], data['Leucograma'], data['Segmentados'], data['Bastonetes'],
                data['Segmentados'], data['Metamielacitos'], data['Mielacitos'], data['Linfacitos'],
                data['Monacitos'], data['Eosinafilos'], data['Basafilos'], data['Plaquetas']]

    # Realizando a predição usando o modelo linear
    prediction = model.predict([features])
    diagnosis = data.columns[prediction.argmax()]

    return jsonify({'diagnosis': diagnosis})

if __name__ == '__main__':
    app.run(debug=True)
