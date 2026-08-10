document.addEventListener('DOMContentLoaded', () => {
    const askBtn = document.getElementById('askBtn');
    const messageInput = document.getElementById('messageInput');
    const chatHistory = document.getElementById('chatHistory');
    const errorBanner = document.getElementById('errorBanner');
    const healthIndicator = document.getElementById('healthIndicator');
    const statusDot = healthIndicator.querySelector('.status-dot');
    const statusText = healthIndicator.querySelector('.status-text');
    
    let currentSessionId = null;

    // Health check
    async function checkHealth() {
        try {
            const res = await fetch('/health');
            if (res.ok) {
                statusDot.className = 'status-dot online';
                statusText.textContent = 'Agent online';
            } else {
                throw new Error('Not OK');
            }
        } catch (e) {
            statusDot.className = 'status-dot offline';
            statusText.textContent = 'Agent unavailable';
        }
    }
    
    // Initial health check and periodic ping
    checkHealth();
    setInterval(checkHealth, 30000);

    // Secure HTML escaper
    function escapeHtml(unsafe) {
        if (!unsafe) return '';
        return unsafe
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }

    function appendUserMessage(text) {
        const msgDiv = document.createElement('div');
        msgDiv.className = 'message user';
        
        const label = document.createElement('div');
        label.className = 'message-label';
        label.textContent = 'You';
        
        const bubble = document.createElement('div');
        bubble.className = 'message-bubble';
        bubble.textContent = text; // Safe assignment
        
        msgDiv.appendChild(label);
        msgDiv.appendChild(bubble);
        chatHistory.appendChild(msgDiv);
        chatHistory.scrollTop = chatHistory.scrollHeight;
    }

    function createLoadingMessage() {
        const msgDiv = document.createElement('div');
        msgDiv.className = 'message agent';
        
        const label = document.createElement('div');
        label.className = 'message-label';
        label.textContent = 'AI Research Agent';
        
        const bubble = document.createElement('div');
        bubble.className = 'message-bubble';
        
        const loader = document.createElement('div');
        loader.className = 'typing-indicator';
        loader.innerHTML = '<div class="typing-dot"></div><div class="typing-dot"></div><div class="typing-dot"></div>';
        
        bubble.appendChild(loader);
        msgDiv.appendChild(label);
        msgDiv.appendChild(bubble);
        chatHistory.appendChild(msgDiv);
        chatHistory.scrollTop = chatHistory.scrollHeight;
        
        return { msgDiv, bubble };
    }

    function renderAgentResponse(bubble, data) {
        bubble.innerHTML = ''; // Clear loader
        
        // 1. Answer (escaping raw HTML to prevent XSS)
        const answerDiv = document.createElement('div');
        answerDiv.className = 'answer-content';
        answerDiv.textContent = data.answer; // Safely sets text and ignores HTML tags
        bubble.appendChild(answerDiv);
        
        // 2. Sources
        if (data.sources && data.sources.length > 0) {
            const srcTitle = document.createElement('div');
            srcTitle.className = 'section-title';
            srcTitle.textContent = 'SOURCES';
            bubble.appendChild(srcTitle);
            
            const srcList = document.createElement('ul');
            srcList.className = 'sources-list';
            
            data.sources.forEach(src => {
                const li = document.createElement('li');
                li.className = 'source-item';
                li.textContent = `📄 ${src.source} (Chunk ${src.chunk_index})`;
                srcList.appendChild(li);
            });
            bubble.appendChild(srcList);
        }
        
        // 3. Trace
        if (data.trace && data.trace.length > 0) {
            const traceTitle = document.createElement('div');
            traceTitle.className = 'section-title';
            traceTitle.textContent = 'AGENT EXECUTION';
            bubble.appendChild(traceTitle);
            
            const traceList = document.createElement('div');
            traceList.className = 'trace-list';
            
            data.trace.forEach(event => {
                const item = document.createElement('div');
                item.className = 'trace-item';
                let desc = event.event_type;
                if (event.event_type === 'tool_call') {
                    desc = `Called tool: ${event.details.tool_name || 'unknown'}`;
                } else if (event.event_type === 'tool_result') {
                    desc = `Tool completed.`;
                } else if (event.event_type === 'reflection') {
                    desc = `Reflection completed (${event.details.status || 'unknown'}).`;
                } else if (event.event_type === 'final_answer') {
                    desc = `Final answer generated.`;
                } else if (event.event_type === 'agent_error') {
                    desc = `Error: ${event.details.error}`;
                }
                item.textContent = `↓ ${desc}`;
                traceList.appendChild(item);
            });
            bubble.appendChild(traceList);
        }
    }

    async function submitMessage() {
        const text = messageInput.value.trim();
        if (!text) return;

        errorBanner.className = 'error-banner hidden';
        errorBanner.textContent = '';
        
        askBtn.disabled = true;
        askBtn.textContent = 'Researching...';
        messageInput.disabled = true;
        
        appendUserMessage(text);
        messageInput.value = '';
        
        const loadingBox = createLoadingMessage();
        
        try {
            const payload = { message: text };
            if (currentSessionId) {
                payload.session_id = currentSessionId;
            }
            
            const res = await fetch('/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            
            if (res.ok) {
                const data = await res.json();
                if (data.session_id) {
                    currentSessionId = data.session_id;
                }
                renderAgentResponse(loadingBox.bubble, data);
            } else if (res.status === 400) {
                const errorData = await res.json();
                errorBanner.textContent = errorData.detail || 'Invalid request.';
                errorBanner.className = 'error-banner';
                loadingBox.msgDiv.remove();
            } else {
                errorBanner.textContent = 'The agent encountered an error. Please try again.';
                errorBanner.className = 'error-banner';
                loadingBox.msgDiv.remove();
            }
        } catch (err) {
            errorBanner.textContent = 'Unable to connect to the research agent.';
            errorBanner.className = 'error-banner';
            loadingBox.msgDiv.remove();
        } finally {
            askBtn.disabled = false;
            askBtn.textContent = 'Ask Agent';
            messageInput.disabled = false;
            messageInput.focus();
            chatHistory.scrollTop = chatHistory.scrollHeight;
        }
    }

    askBtn.addEventListener('click', submitMessage);

    messageInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            submitMessage();
        }
    });
});
