/**
 * 通义千问 (Qwen) 增强脚本
 * 功能：WebSocket 联动 + XHR 劫持 + 自动化操作
 */
(function () {
    console.log('🚀 通义千问增强脚本已注入！');

    const SCRIPT_ID = 'qwen-tools';
    const WS_URL = "wss://d.810114.xyz/ws/browser"; 

    let ws = null;
    let uiRegistered = false;

    // ==========================================
    // 1. 核心自动化操作
    // ==========================================

    // 新建对话
    function doNewChat() {
        // 匹配包含 "New Chat" 文字的侧边栏按钮
        const sidebarItems = document.querySelectorAll('.sidebar-entry-list-content');
        let target = null;
        sidebarItems.forEach(item => {
            if (item.innerText.includes('New Chat')) {
                target = item;
            }
        });

        if (target) {
            target.click();
            console.log("✅ 新建对话按钮已点击");
        } else {
            // 兜底方案：尝试通过 SVG ID 查找
            const plusIcon = document.querySelector('use[*|href="#icon-line-plus-01"]');
            if (plusIcon) {
                plusIcon.closest('.sidebar-entry-list-content')?.click();
                console.log("✅ 通过图标找到并点击了新建对话");
            } else {
                console.error("❌ 未找到新建对话按钮");
            }
        }
    }

    // 输入内容并发送
    function doSendPrompt(text) {
        const textarea = document.querySelector('.message-input-textarea');
        if (!textarea) return console.error("❌ 未找到输入框 (.message-input-textarea)");

        // 聚焦并填入内容
        textarea.focus();
        
        // 针对 React 框架的特殊赋值方式
        const nativeValueSetter = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, "value").set;
        nativeValueSetter.call(textarea, text);
        
        // 必须触发 input 事件，否则发送按钮不会解除禁用
        textarea.dispatchEvent(new Event('input', { bubbles: true }));

        // 延时点击发送
        setTimeout(() => {
            const sendBtn = document.querySelector('.send-button');
            if (sendBtn && !sendBtn.disabled) {
                sendBtn.click();
                console.log("✅ 消息已发送: " + text);
            } else {
                console.warn("⚠️ 发送按钮不可用，尝试回车发送");
                textarea.dispatchEvent(new KeyboardEvent('keydown', {
                    key: 'Enter', code: 'Enter', keyCode: 13, which: 13, bubbles: true
                }));
            }
        }, 300);
    }

    // 删除首个消息元素
    function doRemoveFirstElement() {
        // 通义千问的消息通常在 message-item 或类似的容器中
        const messageItem = document.querySelector('.qwen-message-item, .message-item');
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
            sendToServer("register", null, { model: "qwen" });
            updateUI();
        };

        ws.onclose = () => {
            console.log("🔴 WS连接断开");
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
        console.log("📥 收到远程指令:", msg.command);
        switch (msg.command) {
            case "new_chat": doNewChat(); break;
            case "send_prompt": doSendPrompt(msg.params?.prompt || ""); break;
            case "remove_msg": doRemoveFirstElement(); break;
        }
    }

    // ==========================================
    // 3. XHR/Fetch 劫持 (监听千问流式回复)
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
            // 通义千问的对话 API 通常包含 /api/chat 或类似的特征
            if (this._interceptUrl && this._interceptUrl.includes('/api/chat')) {
                if (this.readyState === 3 || this.readyState === 4) {
                    const currentText = this.responseText;
                    const newData = currentText.substring(lastLength);
                    lastLength = currentText.length;

                    if (!newData) return;

                    // 处理 SSE 数据流
                    const lines = newData.split('\n');
                    for (const line of lines) {
                        const trimmed = line.trim();
                        if (trimmed.startsWith('data:')) {
                            const jsonStr = trimmed.replace('data:', '').trim();
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

        // 1. 重连
        window.FloatingBallAPI.registerAction(
            SCRIPT_ID,
            'qwen_reconnect',
            isConnected ? '🟢 Qwen 已连接' : '🔴 重连服务器',
            () => { connectWS(); }
        );

        // 2. 删除元素
        window.FloatingBallAPI.registerAction(
            SCRIPT_ID,
            'qwen_del',
            '🗑️ 删除消息',
            () => { doRemoveFirstElement(); }
        );

        // 3. 新的对话
        window.FloatingBallAPI.registerAction(
            SCRIPT_ID,
            'qwen_new',
            '➕ 新建对话',
            () => { doNewChat(); }
        );

        // 4. 输入helloworld并发送
        window.FloatingBallAPI.registerAction(
            SCRIPT_ID,
            'qwen_hw',
            '🚀 发送 Hello',
            () => { doSendPrompt('helloworld'); }
        );
    }

    // 初始化启动
    connectWS();
    registerUI();

})();