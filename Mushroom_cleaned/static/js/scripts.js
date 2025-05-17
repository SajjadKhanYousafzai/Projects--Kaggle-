document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('predict-form');
    const resultsDiv = document.getElementById('results');
    
    if (form) {
        form.addEventListener('submit', async (e) => {
            e.preventDefault();
            resultsDiv.style.display = 'none';
            
            const formData = new FormData(form);
            
            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    body: formData
                });
                const data = await response.json();
                
                if (data.error) {
                    alert('Error: ' + data.error);
                    return;
                }
                
                // Update Random Forest results
                const rfResult = document.getElementById('rf-result');
                rfResult.textContent = data.rf_prediction;
                rfResult.className = `result-text ${data.rf_prediction === 'Edible' ? 'edible' : 'poisonous'}`;
                document.getElementById('rf-confidence').textContent = `Confidence: ${(data.rf_confidence * 100).toFixed(2)}%`;
                document.getElementById('rf-gauge').src = `data:image/png;base64,${data.rf_gauge}`;
                
                // Update Logistic Regression results
                const lrResult = document.getElementById('lr-result');
                lrResult.textContent = data.lr_prediction;
                lrResult.className = `result-text ${data.lr_prediction === 'Edible' ? 'edible' : 'poisonous'}`;
                document.getElementById('lr-confidence').textContent = `Confidence: ${(data.lr_confidence * 100).toFixed(2)}%`;
                document.getElementById('lr-gauge').src = `data:image/png;base64,${data.lr_gauge}`;
                
                resultsDiv.style.display = 'block';
                resultsDiv.scrollIntoView({ behavior: 'smooth' });
            } catch (error) {
                alert('Error making prediction: ' + error.message);
            }
        });
    }
});