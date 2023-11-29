# Import necessary libraries
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

# Create a Flask app
app = Flask(__name__)

# Load the TensorFlow SavedModel
model = load_model('best_model.h5')

# Define a route for the home page with the form
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get user input from the form
        food_product = request.form['food_product']
        main_ingredient = request.form['main_ingredient']
        sweetener = request.form['sweetener']
        fat_oil = request.form['fat_oil']
        seasoning = request.form['seasoning']

        # Combine the input data
        vectorizer = CountVectorizer()
        combined_input = vectorizer.transform([f"{food_product} {main_ingredient} {sweetener} {fat_oil} {seasoning}"])

        # Make predictions using the loaded model
        pred_probabilities = model.predict(combined_input.toarray())[0]

        # Create a dictionary of allergen probabilities
        allergen_probabilities = {f'Allergen_{i}': prob for i, prob in enumerate(pred_probabilities)}

        # Render a new page with the prediction results
        # return render_template('result.html', allergen_probabilities=allergen_probabilities)
        return allergen_probabilities

    # Render the initial form page
    return render_template('index.html')

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
