import pickle
from flask import Flask, request, render_template
from src.pipeline.predict_pipeline import CustomData, PredictPipeline
import numpy as np
import pandas as pd

application = Flask(__name__)

app = application

# Route for home page
@app.route('/')
def index():
    return render_template('index.html')

# Route for prediction
@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        # Collect the input data from the form (excluding volume)
        sma_20 = float(request.form.get('sma_20'))
        ema_20 = float(request.form.get('ema_20'))
        rsi = float(request.form.get('rsi'))

        # Create a CustomData object with the inputs
        data = CustomData(
            sma_20=sma_20,
            ema_20=ema_20,
            rsi=rsi
        )

        # Convert the input data into a DataFrame for prediction
        pred_df = data.get_data_as_data_frame()

        # Create an instance of the prediction pipeline and make the prediction
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)

        # Return the result back to the user via the home.html template
        return render_template('home.html', results=results[0])


# Run the Flask app
if __name__ == "__main__":
    app.run(host="0.0.0.0")

