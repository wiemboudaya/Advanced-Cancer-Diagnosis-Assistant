from flask import Flask, request, render_template
import joblib
from sklearn.preprocessing import StandardScaler
import numpy as np

# Initialize the Flask application
app = Flask(__name__)

# Load the trained model
model_path = r'C:\Users\21698\OneDrive\Bureau\projects\breast cancer wisconsin\Advanced-Cancer-Diagnosis-Assistant\cancer_diagnosis_rf_model.joblib'
model = joblib.load(model_path)

# Recreate the scaler (use this only if you didn't save the original scaler)
# Ideally, you should load the saved scaler used during model training
scaler_path= r'C:\Users\21698\OneDrive\Bureau\projects\breast cancer wisconsin\Advanced-Cancer-Diagnosis-Assistant\scaler.joblib'
scaler = joblib.load(scaler_path)

# Feature explanations
feature_explanations = {
    #  feature explanations as previously defined...
}

@app.route('/', methods=['GET'])
def home():
    # Only render the form template on GET request
    return render_template('index.html', feature_explanations=feature_explanations)



@app.route('/predict', methods=['POST'])
def predict():
    prediction_text = ''
    try:
        
        # Extract and scale the input features
        input_features = [
            float(request.form.get('clump_thickness', 0)),
            float(request.form.get('uniformity_cell_size', 0)),
            float(request.form.get('uniformity_cell_shape', 0)),
            float(request.form.get('marginal_adhesion', 0)),
            float(request.form.get('single_epithelial_cell_size', 0)),
            float(request.form.get('bare_nuclei', 0)),
            float(request.form.get('bland_chromatin', 0)),
            float(request.form.get('normal_nucleoli', 0)),
            float(request.form.get('mitoses', 0))
            ]
        # Scale the input features
        input_features_scaled = scaler.transform([input_features])

        # Make a prediction using the scaled features
        prediction = model.predict(input_features_scaled)
        output = 'Malignant' if prediction[0] == 1 else 'Benign'
        prediction_text = f'Tumor Class: {output}'

    except Exception as e:
        print("An error occurred: ", e)
        prediction_text = "Error in prediction"

    # Render the same template with the prediction result
    return render_template('index.html', prediction_text=prediction_text, feature_explanations=feature_explanations)



if __name__ == '__main__':
    app.run(debug=True)
