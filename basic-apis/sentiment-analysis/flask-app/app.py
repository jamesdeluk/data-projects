from flask import Flask, request, jsonify
from textblob import TextBlob

app = Flask(__name__)

@app.route("/")
def home():
    return {"health_check": "OK"}

@app.route('/analyse', methods=['POST'])
def analyse():
    data = request.json
    text = data.get('text', '')

    if not text:
        return jsonify({'error': 'No text provided'}), 400

    blob = TextBlob(text)
    sentiment_score = blob.sentiment.polarity

    return jsonify({
        'input': text,
        'sentiment_score': sentiment_score,
        'sentiment': (
            'positive' if sentiment_score > 0 else
            'negative' if sentiment_score < 0 else
            'neutral'
        )
    })

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=False)