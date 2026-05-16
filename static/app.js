// ═══════════════════════════════════════════
// NeuraType — Frontend Logic
// ═══════════════════════════════════════════

const inputText = document.getElementById('inputText');
const predictBtn = document.getElementById('predictBtn');
const btnLoader = document.getElementById('btnLoader');
const wordCount = document.getElementById('wordCount');
const resultsSection = document.getElementById('resultsSection');
const predictionsGrid = document.getElementById('predictionsGrid');
const generatedText = document.getElementById('generatedText');
const errorCard = document.getElementById('errorCard');
const errorText = document.getElementById('errorText');

// ── Word count update ──
inputText.addEventListener('input', () => {
    const words = inputText.value.trim().split(/\s+/).filter(w => w.length > 0);
    wordCount.textContent = `${words.length} word${words.length !== 1 ? 's' : ''}`;
});

// ── Keyboard shortcut ──
inputText.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        predict();
    }
});

// ── Slider update ──
function updateSlider(sliderId, valueId) {
    const slider = document.getElementById(sliderId);
    const display = document.getElementById(valueId);
    display.textContent = slider.value;
}

// ── Main predict function ──
async function predict() {
    const text = inputText.value.trim();
    if (!text) {
        showError('Please enter some text first!');
        inputText.focus();
        return;
    }

    const temperature = parseFloat(document.getElementById('temperature').value);
    const numWords = parseInt(document.getElementById('numWords').value);

    // UI: loading state
    predictBtn.classList.add('loading');
    predictBtn.disabled = true;
    hideError();

    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text, temperature, num_words: numWords })
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || 'Prediction failed');
        }

        displayResults(data);
    } catch (err) {
        showError(err.message || 'Something went wrong. Is the server running?');
    } finally {
        predictBtn.classList.remove('loading');
        predictBtn.disabled = false;
    }
}

// ── Display results ──
function displayResults(data) {
    resultsSection.style.display = 'flex';

    // Top predictions
    predictionsGrid.innerHTML = '';
    if (data.top_predictions && data.top_predictions.length > 0) {
        data.top_predictions.forEach((pred, i) => {
            const card = document.createElement('div');
            card.className = `prediction-item${i === 0 ? ' top' : ''}`;
            card.innerHTML = `
                <span class="rank">#${i + 1}</span>
                <div class="word">${escapeHtml(pred.word)}</div>
                <div class="prob-bar">
                    <div class="prob-fill" style="width: 0%"></div>
                </div>
                <div class="prob-text">${pred.probability}%</div>
            `;
            card.addEventListener('click', () => {
                inputText.value = data.input_text + ' ' + pred.word;
                inputText.dispatchEvent(new Event('input'));
                inputText.focus();
            });
            predictionsGrid.appendChild(card);

            // Animate probability bar
            requestAnimationFrame(() => {
                setTimeout(() => {
                    const maxProb = data.top_predictions[0].probability;
                    const fill = card.querySelector('.prob-fill');
                    fill.style.width = `${(pred.probability / maxProb) * 100}%`;
                }, 50 + i * 80);
            });
        });
    }

    // Generated text with typewriter effect
    const inputPart = escapeHtml(data.input_text);
    const words = data.generated_words || [];

    let html = `<span class="input-part">${inputPart}</span>`;
    if (words.length > 0) {
        html += ' <span class="predicted-part">';
        words.forEach((word, i) => {
            html += `<span class="predicted-word" style="animation-delay: ${i * 0.12}s">${escapeHtml(word)}</span> `;
        });
        html += '</span>';
    }
    generatedText.innerHTML = html;

    // Smooth scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// ── Copy result ──
function copyResult() {
    const text = generatedText.textContent;
    navigator.clipboard.writeText(text).then(() => {
        const btn = document.getElementById('copyBtn');
        btn.classList.add('copied');
        btn.querySelector('span').textContent = 'Copied!';
        setTimeout(() => {
            btn.classList.remove('copied');
            btn.querySelector('span').textContent = 'Copy';
        }, 2000);
    });
}

// ── Error handling ──
function showError(msg) {
    errorText.textContent = msg;
    errorCard.style.display = 'flex';
    errorCard.style.animation = 'fadeUp 0.3s ease-out';
}

function hideError() {
    errorCard.style.display = 'none';
}

// ── Utility ──
function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}
