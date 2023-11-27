import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from flask import Flask, request, jsonify
import logging

app = Flask(__name__)

def carregar_dataset():
    dataset = pd.read_csv("hemograma.csv", delimiter=";")

    
    for coluna in dataset.columns:
        if coluna != "Diagnostico" and dataset[coluna].dtype == 'O': 
            dataset[coluna] = pd.to_numeric(dataset[coluna].str.replace(",", "."), errors="coerce")

    return dataset




dataset_hemograma = carregar_dataset()


for coluna in dataset_hemograma.columns:
    if coluna != "Diagnostico" and dataset_hemograma[coluna].dtype == 'O': 
        dataset_hemograma[coluna] = pd.to_numeric(dataset_hemograma[coluna].str.replace(",", "."), errors="coerce")



y = dataset_hemograma["Diagnostico"]


X = dataset_hemograma.drop(["Diagnostico"], axis=1)


imputer = SimpleImputer(strategy="mean")
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)


model = LogisticRegression()
model.fit(X_scaled, y)

@app.route("/diagnostico", methods=["POST"])
def sugerir_diagnostico():
    try:
        data = request.get_json(force=True)

        required_fields = [
            "Eritracitos", "Hemoglobina", "Hematacrito", "Eritracitos", "VGM", "CHGM", "Metarrubracitos",
            "Proteina Plasmatica", "Leucacitos", "Leucograma", "Segmentados", "Bastonetes", "Segmentados",
            "Metamielacitos", "Mielacitos", "Linfacitos", "Monacitos", "Eosinafilos",
            "Basafilos", "Plaquetas"
        ]

        for field in required_fields:
            if field not in data:
                logging.error(f"Campo obrigatório ausente: {field}")
                return jsonify({"erro": f"O campo {field} é obrigatório."}), 400

        features = [float(data[field]) for field in required_fields]
        features_imputed = imputer.transform([features])
        features_scaled = scaler.transform(features_imputed)

        diagnostico_predito = model.predict(features_scaled)[0]

        logging.debug(f"Diagnóstico previsto: {diagnostico_predito}")

        matching_rows = dataset_hemograma[dataset_hemograma['Diagnostico'] == diagnostico_predito]

        if not matching_rows.empty:
            
            id_correspondente = str(matching_rows.iloc[0]['Diagnostico'])
            diagnostico_predito = int(diagnostico_predito)

            return jsonify({"id": id_correspondente, "diagnostico_predito": diagnostico_predito})
        else:
            return jsonify({"erro": f"Não foi possível encontrar um ID correspondente para o diagnóstico {diagnostico_predito}."}), 404

    except Exception as e:
        logging.error(f"Erro inesperado: {str(e)}")
        return jsonify({"erro": "Ocorreu um erro inesperado."}), 500


if __name__ == "__main__":
    app.run(debug=True)
