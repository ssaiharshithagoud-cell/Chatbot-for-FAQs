from flask import Flask, render_template, request
from faq_data import faqs

import nltk
import string

from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt')

app = Flask(__name__)

questions = list(faqs.keys())
answers = list(faqs.values())

# Text preprocessing
def preprocess(text):
    text = text.lower()

    tokens = word_tokenize(text)

    tokens = [
        word for word in tokens
        if word not in string.punctuation
    ]

    return " ".join(tokens)

processed_questions = [
    preprocess(q) for q in questions
]

vectorizer = TfidfVectorizer()

question_vectors = vectorizer.fit_transform(
    processed_questions
)

# Chatbot function
def get_response(user_input):

    processed_input = preprocess(user_input)

    input_vector = vectorizer.transform(
        [processed_input]
    )

    similarity = cosine_similarity(
        input_vector,
        question_vectors
    )

    best_match = similarity.argmax()

    score = similarity[0][best_match]

    if score < 0.2:
        return "Sorry, I don't understand."

    return answers[best_match]

@app.route("/", methods=["GET", "POST"])
def home():

    response = ""

    if request.method == "POST":

        user_message = request.form["message"]

        response = get_response(user_message)

    return render_template(
        "index.html",
        response=response
    )

if __name__ == "__main__":
    app.run(debug=True)
