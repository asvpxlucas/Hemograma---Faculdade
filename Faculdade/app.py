from flask import Flask, request, jsonify
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

app = Flask(__name__)

# Carregue os dados do arquivo CSV
df = pd.read_csv('imoveis_hogwarts.csv', delimiter=';')

# Separe os recursos (X) e os rótulos (y)
X = df[['QUARTO', 'Banheiro', 'Garagens', 'M', 'NDE', 'QuadraQ']]
y = df[['Venda', 'Aluguel']]


# Divida os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crie e ajuste o modelo de regressão linear
linearRegression = LinearRegression()
model = linearRegression.fit(X_train, y_train)

# Salve o modelo treinado no diretório atual
#model_file = "modelo_imoveis.pkl"
#with open(model_file, 'wb') as model_f:
#    pickle.dump(model, model_f)

@app.route('/sugerir_valor', methods=['POST'])
def sugerir_valor():
    try:
        data = request.json

        # Faça previsões com base nos dados fornecidos pelo usuário
        features = [data['QUARTO'], data['Banheiro'], data['Garagens'], data['M'], data['NDE'], data['QuadraQ']]
        predicted_values = model.predict([features])[0]

        return jsonify({'Aluguel_sugerido': predicted_values[0], 'Venda_sugerida': predicted_values[1]})

    except Exception as e:
        return jsonify({'erro': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
