/**
 * Gemini 增强脚本
 * 功能：WebSocket 联动 + Fetch 劫持 + 自动化操作
 * 适配：Google Gemini 专用
 */
(function () {
    console.log('🚀 Gemini 增强脚本已注入！');

    const SCRIPT_ID = 'gemini-tools';
    const WS_URL = "wss://d.810114.xyz/ws/browser"; 

    let ws = null;
    let uiRegistered = false;

    // ==========================================
    // 1. 核心自动化操作
    // ==========================================

    // 新建对话
    function doNewChat() {
        // Gemini 的新建对话通常是一个带有 "New chat" aria-label 的按钮或链接
        const newChatBtn = document.querySelector('a[href="/app"]') || 
                           document.querySelector('button[aria-label*="New chat"]') ||
                           document.querySelector('.new-chat-button');
        
        if (newChatBtn) {
            newChatBtn.click();
            console.log("✅ Gemini 新建对话已点击");
        } else {
            console.error("❌ 未找到 Gemini 新建对话按钮");
        }
    }

    // 输入内容并发送
    function doSendPrompt(text) {
        // Gemini 使用的是 contenteditable div
        const editor = document.querySelector('div[contenteditable="true"][role="textbox"]');
        if (!editor) return console.error("❌ 未找到 Gemini 输入框");

        editor.focus();
        // 清空并填入内容
        editor.innerText = text;
        
        // 触发 input 事件让编辑器状态更新
        editor.dispatchEvent(new Event('input', { bubbles: true }));

        // 延时点击发送
        setTimeout(() => {
            // Gemini 的发送按钮通常带有 "Send message" aria-label
            const sendBtn = document.querySelector('button[aria-label*="Send message"]') ||
                            document.querySelector('button.send-button');
            
            if (sendBtn) {
                sendBtn.click();
                console.log("✅ Gemini 消息已发送: " + text);
            } else {
                console.warn("⚠️ 未找到发送按钮，尝试回车...");
                editor.dispatchEvent(new KeyboardEvent('keydown', {
                    key: 'Enter', code: 'Enter', keyCode: 13, which: 13, bubbles: true, ctrlKey: false
                }));
            }
        }, 300);
    }

    // 删除首个消息元素
    function doRemoveFirstElement() {
        // Gemini 的消息容器通常是 chat-history 里的元素
        const messages = document.querySelectorAll('message-content, .message-container');
        if (messages.length > 0) {
            messages[0].remove();
            console.log("✅ 已删除 Gemini 首个消息");
        } else {
            console.warn("⚠️ 未能定位消息元素");
        }
    }

    // ==========================================
    // 2. WebSocket 通信
    // ==========================================
    function connectWS() {
        if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) return;

        ws = new WebSocket(WS_URL);

        ws.onopen = () => {
            console.log("🟢 Gemini 已连接到控制中心");
            sendToServer("register", null, { model: "gemini" });
            updateUI();
        };

        ws.onclose = () => {
            ws = null;
            updateUI();
            setTimeout(connectWS, 5000);
        };

        ws.onmessage = (event) => {
            try {
                const msg = JSON.parse(event.data);
                if (msg.type === "command") handleCommand(msg);
            } catch (e) {}
        };
    }

    function sendToServer(type, content = null, extra = {}) {
        if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ type, content, ...extra }));
        }
    }

    function handleCommand(msg) {
        switch (msg.command) {
            case "new_chat": doNewChat(); break;
            case "send_prompt": doSendPrompt(msg.params?.prompt || ""); break;
            case "remove_msg": doRemoveFirstElement(); break;
        }
    }

    // ==========================================
    // 3. Fetch 劫持 (Gemini 主要使用 Fetch 而非 XHR)
    // ==========================================
    const originalFetch = window.fetch;
    window.fetch = async (...args) => {
        const response = await originalFetch(...args);
        const url = typeof args[0] === 'string' ? args[0] : args[0].url;

        // Gemini 的对话接口通常包含 /BardChatUi
        if (url.includes('BardChatUi')) {
            const clone = response.clone();
            const reader = clone.body.getReader();
            
            // 读取流数据
            (async function readStream() {
                while (true) {
                    const { done, value } = await reader.read();
                    if (done) {
                        sendToServer("token", { done: true });
                        break;
                    }
                    const chunk = new TextDecoder().decode(value);
                    // Gemini 的返回格式比较复杂，是嵌套的数组字符串，这里将其作为原始 token 传回
                    sendToServer("token", { raw: chunk });
                }
            })();
        }
        return response;
    };

    // ==========================================
    // 4. 悬浮球 UI 注册
    // ==========================================
    function registerUI() {
        if (window.FloatingBallAPI) {
            uiRegistered = true;
            updateUI();
        } else {
            setTimeout(registerUI, 500);
        }
    }

    function updateUI() {
        if (!uiRegistered || !window.FloatingBallAPI) return;

        const isConnected = ws && ws.readyState === WebSocket.OPEN;

        window.FloatingBallAPI.registerAction(
            SCRIPT_ID,
            'gemini_reconnect',
            isConnected ? '🟢 Gemini 在线' : '🔴 重连 Gemini',
            () => { connectWS(); }
        );

        window.FloatingBallAPI.registerAction(
            SCRIPT_ID,
            'gemini_del',
            '🗑️ 删除消息',
            () => { doRemoveFirstElement(); }
        );

        window.FloatingBallAPI.registerAction(
            SCRIPT_ID,
            'gemini_new',
            '➕ 新对话',
            () => { doNewChat(); }
        );

        window.FloatingBallAPI.registerAction(
            SCRIPT_ID,
            'gemini_hw',
            '🚀 Hello Gemini',
            () => { doSendPrompt('helloworld'); }
        );
    }

    connectWS();
    registerUI();

})();