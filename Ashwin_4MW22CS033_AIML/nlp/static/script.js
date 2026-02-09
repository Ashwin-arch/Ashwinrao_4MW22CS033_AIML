document.getElementById('sentimentForm').addEventListener('submit', function (e) {
    e.preventDefault();  // Prevent form from reloading the page
    
    // Get the input text from the form
    const text = document.getElementById('text').value;
    
    // Send a POST request to the Flask backend
    fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ text: text })
    })
    .then(response => response.json())
    .then(data => {
        // Display the sentiment result along with character and word counts, and language detection
        document.getElementById('result').innerHTML = 
            `Sentiment: ${data.sentiment}<br>
            Character Count: ${data.char_count}<br>
            Word Count: ${data.word_count}<br>
            Language: ${data.language}`;
    })
    .catch(error => {
        console.error('Error:', error);
    });    
});

document.getElementById('agentForm').addEventListener('submit', function (e) {
    e.preventDefault();

    const provider = document.getElementById('provider').value;
    const apiKey = document.getElementById('apiKey').value;
    const prompt = document.getElementById('prompt').value;
    const resultBox = document.getElementById('agentResult');

    resultBox.textContent = 'Thinking...';

    fetch('/agent', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            provider: provider,
            api_key: apiKey,
            prompt: prompt
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            resultBox.textContent = `Error: ${data.error}`;
            return;
        }
        resultBox.textContent = data.response;
    })
    .catch(error => {
        console.error('Error:', error);
        resultBox.textContent = 'Error: Unable to reach the agent service.';
    });
});
