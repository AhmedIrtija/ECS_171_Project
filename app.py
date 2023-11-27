import numpy as np
import pickle
import pandas as pd
from flask import Flask, request, render_template

# Create Flask app
app = Flask(__name__)

# Load the pickles
model = pickle.load(open("Allergen.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/prediction", methods=["POST"])
def prediction():
    


if __name__ == "__main__":
    app.run(debug=True)
