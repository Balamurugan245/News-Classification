from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load the trained model (which includes the TfidfVectorizer)
with open('clf.pickle', 'rb') as f:
    clf = pickle.load(f)

# News category mapping (numerical label to category name)
news_categories = {
    0: 'Entertainment',
    1: 'Politics',
    2: 'Science',
    3: 'Sports',
    4: 'Technology',
    5: 'World'
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the news content from the form
        news_content = request.form['news_content']
        
        # Use the model pipeline to predict the category
        prediction = clf.predict([news_content])[0]
        news_category = news_categories[prediction]
        
        # Render the result back to the template
        return render_template('index.html', prediction=news_category)

if __name__ == '__main__':
    app.run(debug=True)
