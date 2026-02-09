
from flask import Flask, request, jsonify, render_template
from langdetect import detect, LangDetectException
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import json
from urllib import request as urlrequest
from urllib import error as urlerror

# Initialize Flask app
app = Flask(__name__)

# Download the VADER lexicon for sentiment analysis
nltk.download('vader_lexicon')

# Initialize VADER sentiment analyzer
sid = SentimentIntensityAnalyzer()

# Language code to full name mapping
language_mapping = {
    'en': 'English',
    'zh-cn':'Chinese',
    'ja': 'Japanese',
    'ar': 'Arabic',
    'hi': 'Hindi',
    'bn': 'Bengali',
    'te': 'Telugu',
    'mr': 'Marathi',
    'ta': 'Tamil',
    'ur': 'Urdu',
    'gu': 'Gujarati',
    'ml': 'Malayalam',
    'kn': 'Kannada',
    'or': 'Odia',
    'pa': 'Punjabi',
    'as': 'Assamese',
    'mai': 'Maithili',
    'sa': 'Sanskrit',
    'doi': 'Dogri',
    'mni': 'Manipuri',
    'ks': 'Kashmiri',
    'sat': 'Santali',
    'sd': 'Sindhi',
    # Add more languages as needed
}

# Define the main route for the web app
@app.route('/')
def home():
    return render_template('index.html')

# Define the prediction route that interacts with the frontend
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json  # Get the input data from the frontend
    text = data['text']  # Extract the text field
    sentiment_scores = sid.polarity_scores(text)  # Get sentiment scores
    
    # Determine sentiment based on compound score
    if sentiment_scores['compound'] >= 0.05:
        sentiment = "Positive"
    elif sentiment_scores['compound'] <= -0.05:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"
    
    # Calculate character and word count
    char_count = len(text)
    word_count = len(text.split())
    
    # Detect language
    try:
        language_code = detect(text)
        # Get full language name from mapping
        language = language_mapping.get(language_code, "Unknown Language")
    except LangDetectException:
        language = "Unknown Language"
    
    return jsonify({
        'sentiment': sentiment,
        'char_count': char_count,
        'word_count': word_count,
        'language': language
    })

def call_gemini(api_key, prompt):
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"gemini-pro:generateContent?key={api_key}"
    )
    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": prompt
                    }
                ]
            }
        ]
    }
    data = json.dumps(payload).encode("utf-8")
    req = urlrequest.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"}
    )
    with urlrequest.urlopen(req, timeout=30) as response:
        response_data = json.loads(response.read().decode("utf-8"))
    return response_data["candidates"][0]["content"]["parts"][0]["text"]

def call_groq(api_key, prompt):
    url = "https://api.groq.com/openai/v1/chat/completions"
    payload = {
        "model": "llama3-8b-8192",
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.7
    }
    data = json.dumps(payload).encode("utf-8")
    req = urlrequest.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
    )
    with urlrequest.urlopen(req, timeout=30) as response:
        response_data = json.loads(response.read().decode("utf-8"))
    return response_data["choices"][0]["message"]["content"]

@app.route('/agent', methods=['POST'])
def agent():
    data = request.json or {}
    provider = (data.get('provider') or '').strip().lower()
    api_key = (data.get('api_key') or '').strip()
    prompt = (data.get('prompt') or '').strip()

    if not provider or not api_key or not prompt:
        return jsonify({'error': 'Provider, API key, and prompt are required.'}), 400

    try:
        if provider == 'gemini':
            response_text = call_gemini(api_key, prompt)
        elif provider == 'groq':
            response_text = call_groq(api_key, prompt)
        else:
            return jsonify({'error': 'Unsupported provider selected.'}), 400
    except (urlerror.HTTPError, urlerror.URLError, KeyError, IndexError, ValueError) as exc:
        return jsonify({'error': f'Failed to reach {provider} API: {exc}'}), 502

    return jsonify({
        'provider': provider,
        'response': response_text
    })

if __name__ == '__main__':
    app.run(debug=True)
