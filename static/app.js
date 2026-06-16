document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const chatForm = document.getElementById('chat-form');
    const userInput = document.getElementById('user-input');
    const sendBtn = document.getElementById('send-btn');
    const chatMessages = document.getElementById('chat-messages');
    const welcomeScreen = document.getElementById('welcome-screen');
    const clearChatBtn = document.getElementById('clear-chat');
    const refreshDbBtn = document.getElementById('refresh-db');
    const modelSelect = document.getElementById('model-select');
    const activeModelBadge = document.getElementById('active-model-badge');

    // Sidebar DB Stats
    const dbChunks = document.getElementById('db-chunks');
    const dbFiles = document.getElementById('db-files');

    // Configuration Sliders
    const tempSlider = document.getElementById('temp-slider');
    const tempVal = document.getElementById('temp-val');
    const topkSlider = document.getElementById('topk-slider');
    const topkVal = document.getElementById('topk-val');
    const thresholdSlider = document.getElementById('threshold-slider');
    const thresholdVal = document.getElementById('threshold-val');

    // Setup slider values synchronization
    tempSlider.addEventListener('input', (e) => { tempVal.textContent = parseFloat(e.target.value).toFixed(2); });
    topkSlider.addEventListener('input', (e) => { topkVal.textContent = e.target.value; });
    thresholdSlider.addEventListener('input', (e) => { thresholdVal.textContent = parseFloat(e.target.value).toFixed(2); });

    // Sync active model badge
    modelSelect.addEventListener('change', (e) => {
        activeModelBadge.textContent = e.target.value;
    });

    // Auto-resize textarea height as user types
    userInput.addEventListener('input', function() {
        this.style.height = 'auto';
        this.style.height = (this.scrollHeight - 4) + 'px';
        
        // Enable/Disable send button based on text
        sendBtn.disabled = this.value.trim() === '';
    });

    // Submit query on Enter keypress (without Shift)
    userInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            chatForm.requestSubmit();
        }
    });

    // Load DB Status on init
    loadDatabaseStatus();

    // Event listener for Suggestion Chips
    document.querySelectorAll('.suggestion-chip').forEach(chip => {
        chip.addEventListener('click', () => {
            const promptText = chip.getAttribute('data-prompt');
            userInput.value = promptText;
            userInput.dispatchEvent(new Event('input')); // trigger auto-resize & button activation
            submitMessage(promptText);
        });
    });

    // Form submission
    chatForm.addEventListener('submit', (e) => {
        e.preventDefault();
        const text = userInput.value.trim();
        if (!text) return;
        submitMessage(text);
    });

    // Clear chat history
    clearChatBtn.addEventListener('click', () => {
        chatMessages.innerHTML = '';
        chatMessages.appendChild(welcomeScreen);
        welcomeScreen.style.display = 'flex';
    });

    // Refresh DB stats button
    refreshDbBtn.addEventListener('click', () => {
        loadDatabaseStatus();
    });

    // Helper: HTML Escaping to prevent XSS
    function escapeHtml(text) {
        if (!text) return '';
        const map = {
            '&': '&amp;',
            '<': '&lt;',
            '>': '&gt;',
            '"': '&quot;',
            "'": '&#039;'
        };
        return String(text).replace(/[&<>"']/g, m => map[m]);
    }

    // Toggle sources display (Global function for inline onclick handler)
    window.toggleSources = function(button) {
        button.classList.toggle('active');
        const content = button.nextElementSibling;
        const icon = button.querySelector('i');
        
        if (content.style.display === 'flex') {
            content.style.display = 'none';
            icon.className = 'fa-solid fa-chevron-down';
        } else {
            content.style.display = 'flex';
            icon.className = 'fa-solid fa-chevron-up';
        }
    };

    // Fetch Database status from API
    async function loadDatabaseStatus() {
        dbFiles.innerHTML = '<li class="loading-item">Loading database stats...</li>';
        dbChunks.textContent = '-';
        
        try {
            const response = await fetch('/api/status');
            if (!response.ok) throw new Error('Failed to fetch status');
            const data = await response.json();
            
            dbChunks.textContent = data.document_count;
            dbFiles.innerHTML = '';
            
            if (data.sources && data.sources.length > 0) {
                data.sources.forEach(src => {
                    const li = document.createElement('li');
                    li.textContent = src;
                    dbFiles.appendChild(li);
                });
            } else {
                dbFiles.innerHTML = '<li class="loading-item">No policies indexed.</li>';
            }
        } catch (error) {
            console.error('Error loading DB status:', error);
            dbFiles.innerHTML = '<li class="loading-item" style="color: var(--error-color) !important;"><i class="fa-solid fa-triangle-exclamation"></i> Error loading stats</li>';
        }
    }

    // Submit message to API
    async function submitMessage(queryText) {
        // Hide welcome screen on first message
        if (welcomeScreen.style.display !== 'none') {
            welcomeScreen.style.display = 'none';
        }

        // 1. Add User Bubble
        appendUserMessage(queryText);

        // Clear input area
        userInput.value = '';
        userInput.style.height = 'auto';
        sendBtn.disabled = true;

        // 2. Add Typing Indicator Bubble
        const typingIndicator = appendTypingIndicator();
        scrollToBottom();

        // 3. Prepare payload settings
        const payload = {
            message: queryText,
            model: modelSelect.value,
            temperature: parseFloat(tempSlider.value),
            top_k: parseInt(topkSlider.value),
            score_threshold: parseFloat(thresholdSlider.value)
        };

        try {
            // Fetch response from server
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Failed to fetch response');
            }

            const data = await response.json();
            
            // Remove typing indicator
            typingIndicator.remove();

            // 4. Add Assistant Bubble with sources
            appendAssistantMessage(data.answer, data.sources);

        } catch (error) {
            console.error('API Error:', error);
            typingIndicator.remove();
            appendErrorMessage(error.message);
        }

        scrollToBottom();
    }

    function appendUserMessage(text) {
        const msg = document.createElement('div');
        msg.className = 'message user';
        msg.innerHTML = `
            <div class="message-avatar"><i class="fa-solid fa-user"></i></div>
            <div class="message-bubble">${escapeHtml(text)}</div>
        `;
        chatMessages.appendChild(msg);
    }

    function appendTypingIndicator() {
        const msg = document.createElement('div');
        msg.className = 'message assistant';
        msg.innerHTML = `
            <div class="message-avatar"><i class="fa-solid fa-robot"></i></div>
            <div class="message-bubble">
                <div class="typing-indicator">
                    <span class="typing-dot"></span>
                    <span class="typing-dot"></span>
                    <span class="typing-dot"></span>
                </div>
            </div>
        `;
        chatMessages.appendChild(msg);
        return msg;
    }

    function appendErrorMessage(errText) {
        const msg = document.createElement('div');
        msg.className = 'message assistant';
        msg.innerHTML = `
            <div class="message-avatar"><i class="fa-solid fa-triangle-exclamation" style="color: var(--error-color)"></i></div>
            <div class="message-bubble" style="border-color: rgba(255, 71, 87, 0.3); background-color: rgba(255, 71, 87, 0.02); color: var(--error-color)">
                <strong>Error generating response:</strong> ${escapeHtml(errText)}
            </div>
        `;
        chatMessages.appendChild(msg);
    }

    function appendAssistantMessage(answer, sources) {
        const msg = document.createElement('div');
        msg.className = 'message assistant';
        
        // Process markdown inside answer
        const htmlContent = marked.parse(answer);

        // Process sources
        let sourcesHtml = '';
        if (sources && sources.length > 0) {
            sourcesHtml = `
                <div class="sources-container">
                    <button class="sources-toggle" onclick="toggleSources(this)">
                        <i class="fa-solid fa-chevron-down"></i> View Sources (${sources.length})
                    </button>
                    <div class="sources-content">
            `;
            
            sources.forEach(src => {
                const score = src.similarity_score;
                // High matches (>=0.6) get success-color green, otherwise warning orange
                const scoreClass = score >= 0.6 ? 'score-high' : 'score-medium';
                
                let sourceFile = src.metadata.source || src.metadata.file_path || 'Unknown';
                if (sourceFile.includes('/')) {
                    sourceFile = sourceFile.substring(sourceFile.lastIndexOf('/') + 1);
                }
                
                const pageNumber = src.metadata.page !== undefined ? ` - Page ${src.metadata.page + 1}` : '';
                
                sourcesHtml += `
                    <div class="source-card">
                        <div class="source-card-header">
                            <span class="source-title"><i class="fa-solid fa-file-pdf"></i> ${sourceFile}${pageNumber}</span>
                            <span class="source-score ${scoreClass}">Score: ${score.toFixed(4)}</span>
                        </div>
                        <div class="source-snippet">${escapeHtml(src.content)}</div>
                    </div>
                `;
            });
            
            sourcesHtml += `
                    </div>
                </div>
            `;
        }

        msg.innerHTML = `
            <div class="message-avatar"><i class="fa-solid fa-robot"></i></div>
            <div class="message-bubble">
                <div class="message-content">${htmlContent}</div>
                ${sourcesHtml}
            </div>
        `;
        chatMessages.appendChild(msg);
    }

    function scrollToBottom() {
        chatMessages.scrollTo({
            top: chatMessages.scrollHeight,
            behavior: 'smooth'
        });
    }
});
