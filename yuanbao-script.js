/**
 * 腾讯元宝 (Yuanbao) 增强脚本
 * 功能：WebSocket 联动 + XHR 劫持 + 自动化操作
 */
(function () {
    console.log('🚀 腾讯元宝增强脚本已注入！');

    const SCRIPT_ID = 'yuanbao-tools';
    const WS_URL = "wss://d.810114.xyz/ws/browser"; 

    let ws = null;
    let uiRegistered = false;

    // ==========================================
    // 1. 核心自动化操作
    // ==========================================

    // 新建对话
    function doNewChat() {
        const icon = document.querySelector('.icon-yb-ic_newchat_20');
        if (icon) {
            // 尝试点击图标本身或其最近的按钮容器
            const btn = icon.closest('div') || icon;
            btn.click();
            console.log("✅ 新建对话按钮已点击");
        } else {
            console.error("❌ 未找到新建对话按钮 (.icon-yb-ic_newchat_20)");
        }
    }

    // 输入内容并发送
    function doSendPrompt(text) {
        // 元宝使用的是 Quill 富文本编辑器而非普通 textarea
        const editor = document.querySelector('.ql-editor');
        if (!editor) return console.error("❌ 未找到富文本输入框 (.ql-editor)");

        editor.focus();
        // 清空并填入新内容，元宝需要包裹在 <p> 标签内
        editor.innerHTML = `<p>${text}</p>`;
        
        // 触发 input 事件让框架感知内容变化，解除发送按钮禁用
        editor.dispatchEvent(new Event('input', { bubbles: true }));

        // 延时点击发送
        setTimeout(() => {
            const sendBtn = document.querySelector('#yuanbao-send-btn');
            if (sendBtn) {
                sendBtn.click();
                console.log("✅ 消息已发送: " + text);
            } else {
                console.error("❌ 未找到发送按钮 (#yuanbao-send-btn)");
            }
        }, 300);
    }

    // 删除首个元素（通常是对话列表中的第一条消息）
    function doRemoveFirstElement() {
        // 元宝的消息容器通常包含在特定类名的 div 中
        // 这里寻找对话流中的第一个消息块
        const messageItem = document.querySelector('.agent-instance-item, .chat-message-item');
        if (messageItem) {
            messageItem.remove();
            console.log("✅ 成功删除首个消息元素");
        } else {
            console.warn("⚠️ 未能定位到消息元素");
        }
    }

    // ==========================================
    // 2. WebSocket 通信
    // ==========================================
    function connectWS() {
        if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) return;

        ws = new WebSocket(WS_URL);

        ws.onopen = () => {
            console.log("🟢 成功连接到远程 Server");
            // 注册身份为 yuanbao
            sendToServer("register", null, { model: "yuanbao" });
            updateUI();
        };

        ws.onclose = () => {
            console.log("🔴 连接断开，尝试重连...");
            ws = null;
            updateUI();
            setTimeout(connectWS, 5000);
        };

        ws.onmessage = (event) => {
            try {
                const msg = JSON.parse(event.data);
                if (msg.type === "command") handleCommand(msg);
            } catch (e) { console.error("解析指令失败", e); }
        };
    }

    function sendToServer(type, content = null, extra = {}) {
        if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ type, content, ...extra }));
        }
    }

    function handleCommand(msg) {
        console.log("📥 收到指令:", msg.command);
        switch (msg.command) {
            case "new_chat": doNewChat(); break;
            case "send_prompt": doSendPrompt(msg.params?.prompt || ""); break;
            case "remove_msg": doRemoveFirstElement(); break;
        }
    }

    // ==========================================
    // 3. XHR 劫持 (监听元宝对话 API)
    // ==========================================
    const originalOpen = window.XMLHttpRequest.prototype.open;
    const originalSend = window.XMLHttpRequest.prototype.send;

    window.XMLHttpRequest.prototype.open = function (method, url) {
        this._interceptUrl = typeof url === 'string' ? url : (url?.toString() || '');
        return originalOpen.apply(this, arguments);
    };

    window.XMLHttpRequest.prototype.send = function () {
        let lastLength = 0;
        this.addEventListener('readystatechange', function () {
            // 元宝的对话 API 包含 /api/chat/
            if (this._interceptUrl && this._interceptUrl.includes('/api/chat/')) {
                if (this.readyState === 3 || this.readyState === 4) {
                    const currentText = this.responseText;
                    const newData = currentText.substring(lastLength);
                    lastLength = currentText.length;

                    if (!newData) return;

                    // 处理 SSE 流数据
                    const lines = newData.split('\n');
                    for (const line of lines) {
                        if (line.trim().startsWith('data:')) {
                            const jsonStr = line.replace('data:', '').trim();
                            if (jsonStr === '[DONE]') {
                                sendToServer("token", { done: true });
                            } else {
                                try {
                                    const parsed = JSON.parse(jsonStr);
                                    sendToServer("token", parsed);
                                } catch (e) {}
                            }
                        }
                    }
                }
            }
        });
        return originalSend.apply(this, arguments);
    };

    // ==========================================
    // 4. 悬浮球按钮导出
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

        // 1. 重连按钮
        window.FloatingBallAPI.registerAction(
            SCRIPT_ID,
            'yb_reconnect',
            isConnected ? '🟢 已连接元宝' : '🔴 重连服务器',
            () => { connectWS(); }
        );

        // 2. 删除元素
        window.FloatingBallAPI.registerAction(
            SCRIPT_ID,
            'yb_del',
            '🗑️ 删除消息',
            () => { doRemoveFirstElement(); }
        );

        // 3. 新的对话
        window.FloatingBallAPI.registerAction(
            SCRIPT_ID,
            'yb_new',
            '➕ 新建对话',
            () => { doNewChat(); }
        );

        // 4. 输入helloworld并发送
        window.FloatingBallAPI.registerAction(
            SCRIPT_ID,
            'yb_hw',
            '🚀 发送Hello',
            () => { doSendPrompt('helloworld'); }
        );
    }

    // 初始化
    connectWS();
    registerUI();

})();