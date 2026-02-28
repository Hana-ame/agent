/**
 * DeepSeek 增强脚本 (整合 WebSocket + XHR 劫持版)
 * 修改版：优化按钮匹配逻辑 (SVG Path) & 修改端口 8765
 */
(function () {
    console.log('🚀 DeepSeek 增强脚本已注入 (XHR + WS 路径识别版)！');

    const SCRIPT_ID = 'deepseek-tools';
    const WS_URL = "ws://localhost:8765/ws/browser"; // 修改端口为 8765

    let ws = null;
    let uiRegistered = false;

    // ==========================================
    // 工具函数：通过特定的 SVG Path 找到并点击按钮
    // ==========================================
    function findAndClickByPath(pathD) {
        const buttons = Array.from(document.querySelectorAll('div[role="button"], button'));
        const targetBtn = buttons.find(btn => btn.innerHTML.includes(pathD));

        if (targetBtn) {
            // 检查是否处于禁用状态 (aria-disabled="true")
            if (targetBtn.getAttribute('aria-disabled') === 'true') {
                console.warn("⚠️ 按钮当前处于禁用状态，跳过点击");
                return false;
            }
            targetBtn.click();
            return true;
        }
        return false;
    }

    function doNewChat() {
        // 新建对话按钮的 Path
        const path = 'M10 1.22943C5.15604';
        if (findAndClickByPath(path)) {
            console.log("✅ 新建对话按钮已点击");
        } else {
            console.error("❌ 未找到新建对话按钮");
        }
    }

    function doSendPrompt(text) {
        const textarea = document.querySelector('#chat-input') || document.querySelector('textarea');
        if (!textarea) return console.error("❌ 未找到输入框");

        // 强制填入值
        textarea.focus();
        const nativeSetter = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, "value").set;
        nativeSetter.call(textarea, text);
        textarea.dispatchEvent(new Event('input', { bubbles: true }));

        // 延时点击发送，等待 DOM 响应 input 事件解除按钮禁用
        setTimeout(() => {
            // 发送按钮 Path
            const sendPath = 'M8.3125 0.981587C8.66767 1.0545 8.97902 1.20558 9.2627 1.43374';
            if (findAndClickByPath(sendPath)) {
                console.log("✅ 消息已发送");
            } else {
                console.log("⚠️ 未找到发送按钮或按钮禁用，尝试模拟回车...");
                textarea.dispatchEvent(new KeyboardEvent('keydown', {
                    key: 'Enter', code: 'Enter', keyCode: 13, which: 13, bubbles: true
                }));
            }
        }, 300);
    }

    function doRemoveFirstElement() {
        // 通过 unique style 变量定位到消息列表项
        const sampleItem = document.querySelector('[style*="--assistant-last-margin-bottom: 32px"]');
        if (!sampleItem) return console.error("❌ 未能定位消息列表");

        const container = sampleItem.parentElement.parentElement;
        const firstChild = container.firstElementChild;

        if (firstChild) {
            // 检查是否包含输入框，防止误删底栏
            if (!firstChild.querySelector('input, textarea')) {
                firstChild.remove();
                console.log("✅ 成功删除首个消息元素");
            } else {
                console.warn("⚠️ 首个元素包含输入区域，已取消操作");
            }
        }
    }

    // ==========================================
    // 1. WebSocket 客户端与 Golang 通信
    // ==========================================
    function connectWS() {
        if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) {
            return;
        }

        ws = new WebSocket(WS_URL);

        ws.onopen = () => {
            console.log("🟢 成功连接到 Golang Server (8765)");
            // 🔥 【哆啦A梦补丁】：必须在这里告诉服务器，我是 deepseek 模型
            sendToServer("register", null, { model: "deepseek" });

            updateUI();
        };

        ws.onclose = () => {
            console.log("🔴 连接断开，5秒后尝试重连...");
            ws = null;
            updateUI();
            setTimeout(connectWS, 5000);
        };

        ws.onmessage = (event) => {
            try {
                const msg = JSON.parse(event.data);
                handleCommand(msg);
            } catch (e) {
                console.error("解析指令失败", e);
            }
        };
    }

    function sendToServer(type, content = null, extra = {}) {
        if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ type, content, ...extra }));
        }
    }

    // ==========================================
    // 2. 指令处理中枢 (响应远程请求)
    // ==========================================
    function handleCommand(msg) {
        if (msg.type !== "command") return;
        console.log("📥 收到远程指令:", msg.command, msg.params);

        switch (msg.command) {
            case "match":
                // 验证当前网页是否是目标模型
                const target = msg.params?.target || "";
                const currentHost = location.hostname;
                const isMatch = currentHost.includes(target);

                sendToServer("match_result", {
                    success: isMatch,
                    host: currentHost,
                    model: target
                });
                break;

            case "new_chat":
                doNewChat();
                break;

            case "send_prompt":
                doSendPrompt(msg.params?.prompt || "");
                break;

            case "remove_msg": // 新增指令：删除首条消息
                doRemoveFirstElement();
                break;
        }
    }

    // ==========================================
    // 3. XHR 劫持 (核心：提取流发给 Golang)
    // ==========================================
    const originalOpen = window.XMLHttpRequest.prototype.open;
    const originalSend = window.XMLHttpRequest.prototype.send;

    window.XMLHttpRequest.prototype.open = function (method, url) {
        this._interceptUrl = typeof url === 'string' ? url : (url?.toString() || '');
        return originalOpen.apply(this, arguments);
    };

    window.XMLHttpRequest.prototype.send = function () {
        let lastLength = 0;
        let buffer = "";

        this.addEventListener('readystatechange', function () {
            if (this._interceptUrl && this._interceptUrl.includes('/chat/completion')) {
                if (this.readyState === 3 || this.readyState === 4) {
                    const currentText = this.responseText;
                    const newData = currentText.substring(lastLength);
                    lastLength = currentText.length;

                    if (!newData) return;

                    buffer += newData;
                    const parts = buffer.split('\n');
                    buffer = parts.pop();

                    for (const part of parts) {
                        const line = part.trim();
                        if (!line || !line.startsWith('data:')) continue;

                        const jsonStr = line.substring(5).trim();
                        if (jsonStr === '[DONE]') {
                            sendToServer("token", { done: true });
                            continue;
                        }

                        try {
                            const parsedObj = JSON.parse(jsonStr);
                            sendToServer("token", parsedObj);
                        } catch (e) { }
                    }
                }
            }
        });
        return originalSend.apply(this, arguments);
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
        const btnText = isConnected ? '🟢 WS已连接(8765)' : '🔴 WS未连接(点击重连)';

        window.FloatingBallAPI.registerAction(
            SCRIPT_ID,
            'btn_ws_toggle',
            btnText,
            () => { if (!isConnected) connectWS(); }
        );

        window.FloatingBallAPI.registerAction(
            SCRIPT_ID,
            'btn_clear_first',
            '🗑️ 删除首条消息',
            () => { doRemoveFirstElement(); }
        );

        window.FloatingBallAPI.registerAction(
            SCRIPT_ID,
            'btn_hello_world',
            '✍️ 输入 HelloWorld',
            () => { doSendPrompt('helloworld'); }
        );
    }

    // 初始化运行
    connectWS();
    registerUI();

})();