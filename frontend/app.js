document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const askBtn = document.getElementById('askBtn');
    const messageInput = document.getElementById('messageInput');
    const chatHistory = document.getElementById('chatHistory');
    const errorBanner = document.getElementById('errorBanner');
    const healthIndicator = document.getElementById('healthIndicator');
    const statusDot = healthIndicator.querySelector('.status-dot');
    const statusText = healthIndicator.querySelector('.status-text');
    const newChatBtn = document.getElementById('newChatBtn');
    const historyList = document.getElementById('historyList');
    
    let currentSessionId = null;
    let conversations = [];
    let currentConversationId = null;

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

    // --- LocalStorage Management ---
    function loadConversations() {
        const data = localStorage.getItem('chat_history');
        if (data) {
            try {
                conversations = JSON.parse(data);
            } catch (e) {
                conversations = [];
            }
        } else {
            conversations = [];
        }
    }

    function saveConversations() {
        localStorage.setItem('chat_history', JSON.stringify(conversations));
    }

    function generateId() {
        return Date.now().toString(36) + Math.random().toString(36).substring(2);
    }

    function startNewChat() {
        currentSessionId = null;
        currentConversationId = null;
        chatHistory.innerHTML = '';
        errorBanner.className = 'error-banner hidden';
        errorBanner.textContent = '';
        renderSidebar();
    }

    function ensureConversationExists(firstMessage) {
        if (!currentConversationId) {
            currentConversationId = generateId();
            const title = firstMessage.length > 45 ? firstMessage.substring(0, 45) + '...' : firstMessage;
            const newConv = {
                id: currentConversationId,
                title: title,
                createdAt: new Date().toISOString(),
                updatedAt: new Date().toISOString(),
                sessionId: currentSessionId,
                messages: []
            };
            conversations.unshift(newConv);
            saveConversations();
            renderSidebar();
        } else {
            const conv = conversations.find(c => c.id === currentConversationId);
            if (conv) {
                conv.updatedAt = new Date().toISOString();
                if (currentSessionId) conv.sessionId = currentSessionId;
                saveConversations();
            }
        }
    }

    function pushMessageToHistory(role, data) {
        if (!currentConversationId) return;
        const conv = conversations.find(c => c.id === currentConversationId);
        if (conv) {
            conv.messages.push({ role, data });
            conv.updatedAt = new Date().toISOString();
            if (currentSessionId) conv.sessionId = currentSessionId;
            saveConversations();
        }
    }

    // --- Sidebar Rendering ---
    function renderSidebar() {
        if (!historyList) return; // For safety if running without updated HTML
        historyList.innerHTML = '';
        if (conversations.length === 0) {
            return;
        }

        // Sort by updatedAt descending
        conversations.sort((a, b) => new Date(b.updatedAt) - new Date(a.updatedAt));

        const now = new Date();
        const today = new Date(now.getFullYear(), now.getMonth(), now.getDate()).getTime();
        const yesterday = today - 86400000;

        const groups = {
            'TODAY': [],
            'YESTERDAY': [],
            'OLDER': []
        };

        conversations.forEach(conv => {
            const time = new Date(conv.updatedAt).getTime();
            if (time >= today) {
                groups['TODAY'].push(conv);
            } else if (time >= yesterday) {
                groups['YESTERDAY'].push(conv);
            } else {
                groups['OLDER'].push(conv);
            }
        });

        for (const [groupName, convs] of Object.entries(groups)) {
            if (convs.length > 0) {
                const groupTitle = document.createElement('div');
                groupTitle.className = 'history-group-title';
                groupTitle.textContent = groupName;
                historyList.appendChild(groupTitle);

                convs.forEach(conv => {
                    const item = document.createElement('div');
                    item.className = `history-item ${conv.id === currentConversationId ? 'active' : ''}`;
                    item.onclick = () => loadConversation(conv.id);

                    const titleSpan = document.createElement('span');
                    titleSpan.className = 'history-item-title';
                    titleSpan.textContent = conv.title;
                    
                    const delBtn = document.createElement('button');
                    delBtn.className = 'history-item-delete';
                    delBtn.innerHTML = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 6h18"></path><path d="M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6"></path><path d="M8 6V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2"></path></svg>';
                    delBtn.onclick = (e) => {
                        e.stopPropagation();
                        deleteConversation(conv.id);
                    };

                    item.appendChild(titleSpan);
                    item.appendChild(delBtn);
                    historyList.appendChild(item);
                });
            }
        }
    }

    function deleteConversation(id) {
        if (confirm("Are you sure you want to delete this chat?")) {
            conversations = conversations.filter(c => c.id !== id);
            saveConversations();
            if (currentConversationId === id) {
                startNewChat();
            } else {
                renderSidebar();
            }
        }
    }

    function loadConversation(id) {
        const conv = conversations.find(c => c.id === id);
        if (!conv) return;

        currentConversationId = conv.id;
        currentSessionId = conv.sessionId;
        
        chatHistory.innerHTML = '';
        errorBanner.className = 'error-banner hidden';
        errorBanner.textContent = '';

        conv.messages.forEach(msg => {
            if (msg.role === 'user') {
                renderUserMessageToDOM(msg.data.text);
            } else {
                renderAgentResponseToDOM(msg.data);
            }
        });

        renderSidebar();
        chatHistory.scrollTop = chatHistory.scrollHeight;
    }

    // --- DOM Rendering functions ---
    function renderUserMessageToDOM(text) {
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

    function renderAgentResponseToDOM(data, targetBubble = null) {
        let bubble = targetBubble;
        let msgDiv = null;
        if (!bubble) {
            msgDiv = document.createElement('div');
            msgDiv.className = 'message agent';
            const label = document.createElement('div');
            label.className = 'message-label';
            label.textContent = 'AI Research Agent';
            bubble = document.createElement('div');
            bubble.className = 'message-bubble';
            msgDiv.appendChild(label);
            msgDiv.appendChild(bubble);
            chatHistory.appendChild(msgDiv);
        } else {
            bubble.innerHTML = ''; // Clear loader
        }
        
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
        chatHistory.scrollTop = chatHistory.scrollHeight;
    }

    async function submitMessage() {
        const text = messageInput.value.trim();
        if (!text) return;

        errorBanner.className = 'error-banner hidden';
        errorBanner.textContent = '';
        
        askBtn.disabled = true;
        askBtn.textContent = 'Researching...';
        messageInput.disabled = true;
        
        // Render user message & save state
        renderUserMessageToDOM(text);
        ensureConversationExists(text);
        pushMessageToHistory('user', { text });

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
                    ensureConversationExists(text); // update session ID if it changed
                }
                renderAgentResponseToDOM(data, loadingBox.bubble);
                pushMessageToHistory('agent', data);
            } else if (res.status === 400) {
                const errorData = await res.json();
                errorBanner.textContent = errorData.detail || 'Invalid request.';
                errorBanner.className = 'error-banner';
                loadingBox.msgDiv.remove();
            } else if (res.status === 429) {
                errorBanner.textContent = 'AI provider quota reached. Please try again later or switch to another provider.';
                errorBanner.className = 'error-banner';
                loadingBox.msgDiv.remove();
            } else if (res.status === 502) {
                errorBanner.textContent = 'The AI provider returned an upstream error. Please try again.';
                errorBanner.className = 'error-banner';
                loadingBox.msgDiv.remove();
            } else if (res.status === 503) {
                errorBanner.textContent = 'The research service is temporarily unavailable. Please try again later.';
                errorBanner.className = 'error-banner';
                loadingBox.msgDiv.remove();
            } else {
                errorBanner.textContent = 'The agent encountered an error. Please try again.';
                errorBanner.className = 'error-banner';
                loadingBox.msgDiv.remove();
            }
        } catch (err) {
            errorBanner.textContent = 'Unable to connect to the research server. Please make sure the backend is running.';
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

    if (newChatBtn) {
        newChatBtn.addEventListener('click', startNewChat);
    }

    // Initialization
    loadConversations();
    if (conversations.length > 0) {
        conversations.sort((a, b) => new Date(b.updatedAt) - new Date(a.updatedAt));
        loadConversation(conversations[0].id);
    }
    renderSidebar();
});
