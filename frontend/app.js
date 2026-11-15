document.addEventListener('DOMContentLoaded', () => {
    // --- DOM Elements ---
    const askBtn = document.getElementById('askBtn');
    const messageInput = document.getElementById('messageInput');
    const chatHistory = document.getElementById('chatHistory');
    const errorBanner = document.getElementById('errorBanner');
    const healthIndicator = document.getElementById('healthIndicator');
    const statusDot = healthIndicator.querySelector('.status-dot');
    const statusText = healthIndicator.querySelector('.status-text');
    const newChatBtn = document.getElementById('newChatBtn');
    const historyList = document.getElementById('historyList');
    const sidebar = document.getElementById('sidebar');
    const sidebarToggle = document.getElementById('sidebarToggle');
    const sidebarBackdrop = document.getElementById('sidebarBackdrop');
    const emptyState = document.getElementById('emptyState');
    const chatScroll = document.querySelector('.chat-scroll');

    function scrollToBottom() {
        chatScroll.scrollTop = chatScroll.scrollHeight;
    }

    const STORAGE_KEY = 'chat_history';
    const MAX_STORED_CONVERSATIONS = 50;

    let currentSessionId = null;
    let conversations = [];
    let currentConversationId = null;
    let isSubmitting = false;

    // --- Health check ---
    async function checkHealth() {
        try {
            const res = await fetch('/health');
            if (!res.ok) throw new Error('Not OK');
            statusDot.className = 'status-dot online';
            statusText.textContent = 'Agent online';
        } catch (e) {
            statusDot.className = 'status-dot offline';
            statusText.textContent = 'Agent unavailable';
        }
    }
    checkHealth();
    setInterval(checkHealth, 30000);

    // --- Markdown rendering (sanitised) ---
    // marked + DOMPurify are loaded from a CDN. If either is unavailable the answer
    // falls back to plain text, so LLM output is never inserted as unsanitised HTML.
    const CITATION_PATTERN = /\[(WEB|LOCAL):\s*([^\]]+)\]/g;

    function escapeAttr(value) {
        return value.replace(/&/g, '&amp;').replace(/"/g, '&quot;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    }

    // Turns a <cite data-kind data-label> placeholder into a chip. Web citations of the
    // form "Title (https://url)" become links to the source.
    function renderCitationChip(cite) {
        const kind = cite.dataset.kind === 'web' ? 'web' : 'local';
        const label = cite.dataset.label || '';
        let chip;
        const urlMatch = kind === 'web' ? label.match(/^(.*)\((https?:\/\/[^\s)]+)\)\s*$/) : null;
        if (urlMatch && isSafeHttpUrl(urlMatch[2])) {
            chip = document.createElement('a');
            chip.href = urlMatch[2];
            chip.textContent = urlMatch[1].trim() || urlMatch[2];
        } else {
            chip = document.createElement('span');
            chip.textContent = label;
        }
        chip.className = `citation citation-${kind}`;
        chip.title = `${kind === 'web' ? 'Web' : 'Local'} source: ${label}`;
        cite.replaceWith(chip);
    }

    function renderMarkdownInto(element, text) {
        const markdown = text || '';
        if (window.marked && window.DOMPurify) {
            try {
                // Swap citation tags for placeholders BEFORE markdown parsing; otherwise the
                // URL inside "[WEB: Title (https://...)]" is auto-linked and the tag is split.
                const withPlaceholders = markdown.replace(CITATION_PATTERN, (match, kind, label) =>
                    `<cite data-kind="${kind.toLowerCase()}" data-label="${escapeAttr(label.trim())}"></cite>`);
                const html = window.marked.parse(withPlaceholders, { gfm: true, breaks: true });
                element.innerHTML = window.DOMPurify.sanitize(html, { USE_PROFILES: { html: true } });
                element.querySelectorAll('cite[data-kind]').forEach(renderCitationChip);
                element.querySelectorAll('a').forEach(a => {
                    a.setAttribute('target', '_blank');
                    a.setAttribute('rel', 'noopener noreferrer');
                });
                highlightCitations(element);
                element.classList.add('markdown');
                return;
            } catch (e) {
                // fall through to plain text
            }
        }
        element.textContent = markdown;
        element.classList.add('plain');
    }

    // Wraps [WEB: ...] / [LOCAL: ...] citation tags in styled chips (text nodes only).
    function highlightCitations(root) {
        const pattern = /\[(WEB|LOCAL):\s*([^\]]+)\]/g;
        const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
        const textNodes = [];
        while (walker.nextNode()) {
            if (pattern.test(walker.currentNode.nodeValue)) textNodes.push(walker.currentNode);
            pattern.lastIndex = 0;
        }
        textNodes.forEach(node => {
            const frag = document.createDocumentFragment();
            let last = 0;
            const value = node.nodeValue;
            value.replace(pattern, (match, kind, label, offset) => {
                if (offset > last) frag.appendChild(document.createTextNode(value.slice(last, offset)));
                const chip = document.createElement('span');
                chip.className = `citation citation-${kind.toLowerCase()}`;
                chip.textContent = `${kind === 'WEB' ? 'Web' : 'Local'}: ${label.trim()}`;
                chip.title = match;
                frag.appendChild(chip);
                last = offset + match.length;
                return match;
            });
            if (last < value.length) frag.appendChild(document.createTextNode(value.slice(last)));
            node.parentNode.replaceChild(frag, node);
        });
    }

    function isSafeHttpUrl(url) {
        try {
            const parsed = new URL(url);
            return parsed.protocol === 'http:' || parsed.protocol === 'https:';
        } catch (e) {
            return false;
        }
    }

    // --- LocalStorage Management (never let storage errors break the chat) ---
    function loadConversations() {
        try {
            const data = localStorage.getItem(STORAGE_KEY);
            const parsed = data ? JSON.parse(data) : [];
            conversations = Array.isArray(parsed) ? parsed : [];
        } catch (e) {
            conversations = [];
        }
    }

    function saveConversations() {
        conversations = conversations.slice(0, MAX_STORED_CONVERSATIONS);
        try {
            localStorage.setItem(STORAGE_KEY, JSON.stringify(conversations));
        } catch (e) {
            // Quota exceeded or storage disabled: drop the oldest chats and retry once.
            try {
                conversations = conversations.slice(0, Math.max(1, Math.floor(conversations.length / 2)));
                localStorage.setItem(STORAGE_KEY, JSON.stringify(conversations));
            } catch (e2) { /* storage unavailable; history just won't persist */ }
        }
    }

    function generateId() {
        return Date.now().toString(36) + Math.random().toString(36).substring(2);
    }

    function updateEmptyState() {
        const hasMessages = chatHistory.children.length > 0;
        emptyState.classList.toggle('hidden', hasMessages);
    }

    function hideError() {
        errorBanner.className = 'error-banner hidden';
        errorBanner.textContent = '';
    }

    function showError(message) {
        errorBanner.textContent = message;
        errorBanner.className = 'error-banner';
    }

    function startNewChat() {
        currentSessionId = null;
        currentConversationId = null;
        chatHistory.innerHTML = '';
        hideError();
        renderSidebar();
        updateEmptyState();
        closeSidebar();
        messageInput.focus();
    }

    function ensureConversationExists(firstMessage) {
        if (!currentConversationId) {
            currentConversationId = generateId();
            const title = firstMessage.length > 45 ? firstMessage.substring(0, 45) + '...' : firstMessage;
            conversations.unshift({
                id: currentConversationId,
                title,
                createdAt: new Date().toISOString(),
                updatedAt: new Date().toISOString(),
                sessionId: currentSessionId,
                messages: []
            });
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

    // --- Sidebar ---
    function openSidebar() {
        sidebar.classList.add('open');
        sidebarBackdrop.classList.add('visible');
    }

    function closeSidebar() {
        sidebar.classList.remove('open');
        sidebarBackdrop.classList.remove('visible');
    }

    function renderSidebar() {
        historyList.innerHTML = '';
        if (conversations.length === 0) {
            const empty = document.createElement('div');
            empty.className = 'history-empty';
            empty.textContent = 'No conversations yet.';
            historyList.appendChild(empty);
            return;
        }

        conversations.sort((a, b) => new Date(b.updatedAt) - new Date(a.updatedAt));

        const now = new Date();
        const today = new Date(now.getFullYear(), now.getMonth(), now.getDate()).getTime();
        const yesterday = today - 86400000;
        const groups = { 'Today': [], 'Yesterday': [], 'Older': [] };

        conversations.forEach(conv => {
            const time = new Date(conv.updatedAt).getTime();
            if (time >= today) groups['Today'].push(conv);
            else if (time >= yesterday) groups['Yesterday'].push(conv);
            else groups['Older'].push(conv);
        });

        for (const [groupName, convs] of Object.entries(groups)) {
            if (convs.length === 0) continue;
            const groupTitle = document.createElement('div');
            groupTitle.className = 'history-group-title';
            groupTitle.textContent = groupName;
            historyList.appendChild(groupTitle);

            convs.forEach(conv => {
                const item = document.createElement('div');
                item.className = `history-item ${conv.id === currentConversationId ? 'active' : ''}`;
                item.setAttribute('role', 'button');
                item.tabIndex = 0;
                item.onclick = () => loadConversation(conv.id);
                item.onkeydown = (e) => { if (e.key === 'Enter') loadConversation(conv.id); };

                const titleSpan = document.createElement('span');
                titleSpan.className = 'history-item-title';
                titleSpan.textContent = conv.title;

                const delBtn = document.createElement('button');
                delBtn.className = 'history-item-delete';
                delBtn.setAttribute('aria-label', 'Delete conversation');
                delBtn.title = 'Delete conversation';
                delBtn.innerHTML = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true"><path d="M3 6h18"></path><path d="M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6"></path><path d="M8 6V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2"></path></svg>';
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

    function deleteConversation(id) {
        if (isSubmitting && id === currentConversationId) return;
        if (!confirm('Delete this chat?')) return;
        const conv = conversations.find(c => c.id === id);
        if (conv && conv.sessionId) {
            // Free the server-side session too (best effort; it may already be gone).
            fetch(`/sessions/${encodeURIComponent(conv.sessionId)}`, { method: 'DELETE' }).catch(() => {});
        }
        conversations = conversations.filter(c => c.id !== id);
        saveConversations();
        if (currentConversationId === id) startNewChat();
        else renderSidebar();
    }

    function loadConversation(id) {
        if (isSubmitting) return;
        const conv = conversations.find(c => c.id === id);
        if (!conv) return;

        currentConversationId = conv.id;
        currentSessionId = conv.sessionId;
        chatHistory.innerHTML = '';
        hideError();

        conv.messages.forEach(msg => {
            if (msg.role === 'user') renderUserMessageToDOM(msg.data.text);
            else renderAgentResponseToDOM(msg.data);
        });

        renderSidebar();
        updateEmptyState();
        closeSidebar();
        scrollToBottom();
    }

    // --- Message rendering ---
    function createMessageShell(role) {
        const msgDiv = document.createElement('div');
        msgDiv.className = `message ${role}`;
        const label = document.createElement('div');
        label.className = 'message-label';
        label.textContent = role === 'user' ? 'You' : 'AI Research Agent';
        const bubble = document.createElement('div');
        bubble.className = 'message-bubble';
        msgDiv.appendChild(label);
        msgDiv.appendChild(bubble);
        chatHistory.appendChild(msgDiv);
        updateEmptyState();
        return { msgDiv, bubble };
    }

    function renderUserMessageToDOM(text) {
        const { bubble } = createMessageShell('user');
        bubble.textContent = text;
        scrollToBottom();
    }

    function createLoadingMessage() {
        const { msgDiv, bubble } = createMessageShell('agent');
        const row = document.createElement('div');
        row.className = 'loading-row';
        const loader = document.createElement('div');
        loader.className = 'typing-indicator';
        loader.innerHTML = '<div class="typing-dot"></div><div class="typing-dot"></div><div class="typing-dot"></div>';
        const status = document.createElement('span');
        status.className = 'loading-status';
        status.textContent = 'Researching…';
        row.appendChild(loader);
        row.appendChild(status);
        bubble.appendChild(row);
        scrollToBottom();

        // Research can take a while (multiple searches + verification); show progress.
        const started = Date.now();
        const timer = setInterval(() => {
            const secs = Math.round((Date.now() - started) / 1000);
            let phase = 'Researching';
            if (secs > 8) phase = 'Gathering and checking evidence';
            if (secs > 25) phase = 'Verifying claims and citations';
            if (secs > 60) phase = 'Still working — complex questions can take a couple of minutes';
            status.textContent = `${phase}… ${secs}s`;
        }, 1000);

        return { msgDiv, bubble, stop: () => clearInterval(timer) };
    }

    const TRACE_LABELS = {
        graph_start: 'Request received',
        classified_as_simple_query: 'Classified as a simple question (fast path)',
        fast_llm_path: 'Answered directly without tools',
        agent_start: 'Research started',
        iteration_start: 'Agent reasoning step',
        research_source_selected: null,
        tool_result: null,
        evidence_synthesis: null,
        claim_assessment: null,
        final_answer: 'Final answer ready',
        graph_end: null,
        force_synthesis: 'Synthesised an answer from the collected evidence',
        reflection_limit_reached: 'Reflection limit reached',
        iteration_limit_reached: 'Step limit reached — using collected evidence',
        refinement_requested: 'Asked the agent to revise unsupported claims',
        grounding_gate_triggered: 'Rewrote sections with unsupported claims',
    };

    const TOOL_NAMES = {
        search_web: 'Web search',
        search_local_knowledge: 'Local knowledge search',
        calculate_product: 'Calculator',
    };

    function describeTraceEvent(event) {
        const d = event.details || {};
        switch (event.event_type) {
            case 'research_plan':
                return `Plan: ${String(d.intent || 'general').replace(/_/g, ' ')}`;
            case 'tool_call': {
                const name = TOOL_NAMES[d.tool_name] || d.tool_name || 'tool';
                const query = d.arguments && d.arguments.query ? ` — “${d.arguments.query}”` : '';
                return `${name}${query}`;
            }
            case 'evidence_collected':
                return `Evidence collected (${d.count || 0} item${d.count === 1 ? '' : 's'})`;
            case 'reflection':
                return `Evidence check: ${d.status || 'unknown'}${d.feedback ? ` — ${d.feedback}` : ''}`;
            case 'reflection_error':
                return 'Evidence check unavailable — continued with collected evidence';
            case 'claim_extraction':
                return `Extracted ${d.count || 0} claim${d.count === 1 ? '' : 's'} to verify`;
            case 'grounding_skipped':
                return 'No sources retrieved — verification skipped';
            case 'grounding_check': {
                const score = typeof d.grounding_score === 'number' ? Math.round(d.grounding_score * 100) : null;
                return `Grounding: ${d.supported_claims ?? 0}/${d.claim_count ?? 0} claims supported${score !== null ? ` (${score}%)` : ''}`;
            }
            case 'citation_validation': {
                const invalid = (d.invalid_citations || []).length;
                return invalid ? `Citations: ${invalid} issue${invalid === 1 ? '' : 's'} found` : `Citations: ${d.citations_found || 0} checked, all valid`;
            }
            case 'agent_error':
                return `Error: ${d.error}`;
            default:
                return Object.prototype.hasOwnProperty.call(TRACE_LABELS, event.event_type)
                    ? TRACE_LABELS[event.event_type]
                    : event.event_type.replace(/_/g, ' ');
        }
    }

    function renderSources(bubble, sources) {
        if (!sources || sources.length === 0) return;
        const title = document.createElement('div');
        title.className = 'section-title';
        title.textContent = `Sources (${sources.length})`;
        bubble.appendChild(title);

        const list = document.createElement('ul');
        list.className = 'sources-list';
        sources.forEach(src => {
            const li = document.createElement('li');
            li.className = 'source-item';
            const badge = document.createElement('span');
            const isWeb = src.type === 'web' || (!!src.url && src.type !== 'local');
            badge.className = `source-badge ${isWeb ? 'web' : 'local'}`;
            badge.textContent = isWeb ? 'WEB' : 'LOCAL';
            li.appendChild(badge);

            if (isWeb && src.url && isSafeHttpUrl(src.url)) {
                const a = document.createElement('a');
                a.href = src.url;
                a.target = '_blank';
                a.rel = 'noopener noreferrer';
                a.textContent = src.source || src.url;
                a.title = src.url;
                li.appendChild(a);
            } else {
                const span = document.createElement('span');
                const chunk = src.chunk_index !== undefined && src.chunk_index !== '' ? ` · chunk ${src.chunk_index}` : '';
                span.textContent = `${src.source}${chunk}`;
                li.appendChild(span);
            }
            list.appendChild(li);
        });
        bubble.appendChild(list);
    }

    function renderTrace(bubble, data) {
        if (!data.trace || data.trace.length === 0) return;
        const details = document.createElement('details');
        details.className = 'trace-details';
        const summary = document.createElement('summary');
        const toolCount = data.tool_calls || 0;
        summary.textContent = `How this answer was produced · ${data.iterations || 0} step${data.iterations === 1 ? '' : 's'}, ${toolCount} tool call${toolCount === 1 ? '' : 's'}`;
        details.appendChild(summary);

        const list = document.createElement('ol');
        list.className = 'trace-list';
        data.trace.forEach(event => {
            const desc = describeTraceEvent(event);
            if (!desc) return;
            const item = document.createElement('li');
            item.className = `trace-item trace-${event.event_type}`;
            item.textContent = desc;
            list.appendChild(item);
        });
        details.appendChild(list);
        bubble.appendChild(details);
    }

    function renderAgentResponseToDOM(data, targetBubble = null) {
        const bubble = targetBubble || createMessageShell('agent').bubble;
        bubble.innerHTML = '';

        const answerDiv = document.createElement('div');
        answerDiv.className = 'answer-content';
        renderMarkdownInto(answerDiv, data.answer);
        bubble.appendChild(answerDiv);

        // Be explicit when an answer did not go through evidence verification.
        const trace = data.trace || [];
        const unverified = trace.some(e => e.event_type === 'grounding_skipped' || e.event_type === 'classified_as_simple_query');
        if (unverified && !(data.sources && data.sources.length)) {
            const note = document.createElement('div');
            note.className = 'unverified-note';
            note.textContent = 'Answered from the model’s own knowledge. No sources were retrieved, so this answer was not verified.';
            bubble.appendChild(note);
        }

        renderSources(bubble, data.sources);
        renderTrace(bubble, data);
        scrollToBottom();
    }

    // --- Networking ---
    async function postChat(payload) {
        return fetch('/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
    }

    async function readErrorDetail(res) {
        try {
            const body = await res.json();
            if (typeof body.detail === 'string') return body.detail;
            if (Array.isArray(body.detail) && body.detail[0] && body.detail[0].msg) return body.detail[0].msg;
        } catch (e) { /* non-JSON error */ }
        return null;
    }

    function errorMessageFor(status, detail) {
        switch (status) {
            case 400: return detail || 'Invalid request.';
            case 422: return detail ? `Invalid request: ${detail}` : 'Invalid request.';
            case 429: return 'AI provider quota reached. Please try again later.';
            case 502: return 'The AI provider returned an upstream error. Please try again.';
            case 503: return 'The research service is temporarily unavailable. Please try again later.';
            default: return 'The agent encountered an error. Please try again.';
        }
    }

    function handleSuccess(data, text, loadingBox) {
        if (data.session_id) {
            currentSessionId = data.session_id;
            ensureConversationExists(text);
        }
        renderAgentResponseToDOM(data, loadingBox.bubble);
        pushMessageToHistory('agent', data);
    }

    async function submitMessage() {
        const text = messageInput.value.trim();
        if (!text || isSubmitting) return;
        isSubmitting = true;
        hideError();

        askBtn.disabled = true;
        askBtn.textContent = 'Researching…';
        messageInput.disabled = true;

        renderUserMessageToDOM(text);
        ensureConversationExists(text);
        pushMessageToHistory('user', { text });
        messageInput.value = '';
        autoResize();

        const loadingBox = createLoadingMessage();

        try {
            const payload = { message: text };
            if (currentSessionId) payload.session_id = currentSessionId;

            let res = await postChat(payload);

            if (res.status === 404 && payload.session_id) {
                // The server no longer knows this session (e.g. it restarted and its
                // in-memory store was wiped). Clear it and retry ONCE with a fresh session.
                currentSessionId = null;
                const staleConv = conversations.find(c => c.id === currentConversationId);
                if (staleConv) { staleConv.sessionId = null; saveConversations(); }
                res = await postChat({ message: text });
            }

            loadingBox.stop();
            if (res.ok) {
                handleSuccess(await res.json(), text, loadingBox);
            } else {
                const detail = await readErrorDetail(res);
                showError(errorMessageFor(res.status, detail));
                loadingBox.msgDiv.remove();
            }
        } catch (err) {
            loadingBox.stop();
            showError('Unable to connect to the research server. Please make sure the backend is running.');
            loadingBox.msgDiv.remove();
        } finally {
            isSubmitting = false;
            askBtn.disabled = false;
            askBtn.textContent = 'Ask Agent';
            messageInput.disabled = false;
            messageInput.focus();
            updateEmptyState();
            scrollToBottom();
        }
    }

    function autoResize() {
        messageInput.style.height = 'auto';
        messageInput.style.height = Math.min(messageInput.scrollHeight, 200) + 'px';
    }

    // --- Events ---
    askBtn.addEventListener('click', submitMessage);
    messageInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) {
            e.preventDefault();
            submitMessage();
        }
    });
    messageInput.addEventListener('input', autoResize);
    newChatBtn.addEventListener('click', startNewChat);
    sidebarToggle.addEventListener('click', () => {
        if (sidebar.classList.contains('open')) closeSidebar();
        else openSidebar();
    });
    sidebarBackdrop.addEventListener('click', closeSidebar);
    document.querySelectorAll('.example-chip').forEach(chip => {
        chip.addEventListener('click', () => {
            messageInput.value = chip.dataset.prompt || chip.textContent;
            autoResize();
            messageInput.focus();
        });
    });

    // --- Initialization ---
    loadConversations();
    if (conversations.length > 0) {
        conversations.sort((a, b) => new Date(b.updatedAt) - new Date(a.updatedAt));
        loadConversation(conversations[0].id);
    } else {
        renderSidebar();
    }
    updateEmptyState();
});
