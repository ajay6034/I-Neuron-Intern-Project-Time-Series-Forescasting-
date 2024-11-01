from flask import Flask, request, render_template
from src.pipeline.predict_pipeline import CustomData, PredictPipeline
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

application = Flask(__name__)
app = application

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        # Collect input data from the form
        sma_20 = float(request.form.get('sma_20'))
        ema_20 = float(request.form.get('ema_20'))
        rsi = float(request.form.get('rsi'))
        volume = float(request.form.get('volume'))
        start_date = request.form.get('start_date')
        forecast_days = int(request.form.get('forecast_days'))

        # Create a CustomData object with the inputs
        data = CustomData(
            sma_20=sma_20,
            ema_20=ema_20,
            rsi=rsi,
            volume=volume,
            start_date=start_date,
            forecast_days=forecast_days
        )

        # Convert input data into a DataFrame for prediction
        pred_df = data.get_data_as_data_frame()

        # Create an instance of the prediction pipeline and make the prediction
        predict_pipeline = PredictPipeline()
        predicted_prices = predict_pipeline.predict(pred_df, forecast_days)

        # Debugging step: Check the length of predicted_prices
        print("Length of predicted_prices:", len(predicted_prices))
        print("Forecast days requested:", forecast_days)

        # Ensure predicted_prices has enough elements
        if len(predicted_prices) < forecast_days:
            return "Error: Not enough predictions returned by the model to match forecast days.", 500

        # Generate future dates for each prediction
        start_date_obj = datetime.strptime(start_date, "%Y-%m-%d")
        prediction_results = [
            (start_date_obj + timedelta(days=i), predicted_prices[i])
            for i in range(forecast_days)
        ]

        # Ensure results are iterable for template display
        results = [(date.strftime("%Y-%m-%d"), price) for date, price in prediction_results]

        # Render the result back to the user via the home.html template
        return render_template('home.html', results=results)

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
