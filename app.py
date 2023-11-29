from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load the best trained model, vectorizer, and multi-label binarizer
trained_model = load_model('best_model.h5')
vectorizer = CountVectorizer()
mlb = MultiLabelBinarizer()

data = pd.read_csv("Allergen_Status_of_Food_Products.csv")

data = data.fillna('')

X = vectorizer.fit_transform(data['Food Product'] + " " + data['Main Ingredient'] + " " +
                             data['Sweetener'] + " " + data['Fat/Oil'] + " " + data['Seasoning'])
y = mlb.fit_transform(data['Allergens'].str.split(', '))

@app.route('/')
def home():
    return render_template('index.html')

# Prediction and result display
@app.route('/', methods=['POST'])
def predict():
    # Get user input
    food_product = request.form['food_product']
    main_ingredient = request.form['main_ingredient']
    sweetener = request.form['sweetener']
    fat_oil = request.form['fat_oil']
    seasoning = request.form['seasoning']

    # Handle NaN values in user input
    food_product = '' if pd.isna(food_product) else food_product
    main_ingredient = '' if pd.isna(main_ingredient) else main_ingredient
    sweetener = '' if pd.isna(sweetener) else sweetener
    fat_oil = '' if pd.isna(fat_oil) else fat_oil
    seasoning = '' if pd.isna(seasoning) else seasoning



    # Transform the input using the vectorizer
    input_data = vectorizer.transform([f"{food_product} {main_ingredient} {sweetener} {fat_oil} {seasoning}"])
    input_data_dense = input_data.toarray()


    # Get model predictions
    predictions = trained_model.predict(input_data_dense)

    allergen_probabilities = {allergen: f"{prob:.2f}%" for allergen, prob in zip(mlb.classes_, 100 * predictions[0])}

    filtered_allergens = {allergen: prob for allergen, prob in allergen_probabilities.items() if float(prob[:-1]) >= 80}

    # Display the result on result.html
    return render_template('result.html', allergen_probabilities=filtered_allergens)

if __name__ == '__main__':
    app.run(debug=True)
