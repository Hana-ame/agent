/**
 * DeepSeek 增强脚本 (整合 WebSocket + XHR 劫持版)
 * 1. 无损劫持 XHR 流式数据
 * 2. 自动连接本地 Golang WebSocket Server
 * 3. 接收并执行 Python 发来的网页自动化控制指令
 * 4. 注册悬浮球 UI 按钮展示状态
 */
(function() {
    console.log('🚀 DeepSeek 增强脚本已注入 (XHR + WS 桥接模式)！');

    const SCRIPT_ID = 'deepseek-tools'; // 需与你的 JSON 里的 ID 一致
    const WS_URL = "ws://localhost:8080/ws/browser";
    
    let ws = null;
    let uiRegistered = false;

    // ==========================================
    // 1. WebSocket 客户端与 Golang 通信
    // ==========================================
    function connectWS() {
        if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) {
            return;
        }

        ws = new WebSocket(WS_URL);

        ws.onopen = () => {
            console.log("🟢 成功连接到 Golang Server");
            updateUI(); // 更新悬浮球按钮状态
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
        } else {
            console.warn("⚠️ WS 未连接，无法发送数据到 Golang");
        }
    }

    // ==========================================
    // 2. 指令处理中枢 (响应 Python 请求)
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
                // 模拟点击"新对话" (匹配 DeepSeek 左侧栏的新建按钮)
                const newChatBtn = document.querySelector('div[role="button"]:has(svg)') || document.evaluate("//div[contains(text(), '新对话')]", document, null, XPathResult.FIRST_ORDERED_NODE_TYPE, null).singleNodeValue;
                if (newChatBtn) {
                    newChatBtn.click();
                    console.log("✅ 已执行新建对话");
                }
                break;

            case "send_prompt":
                const promptText = msg.params?.prompt || "";
                // const enableThinking = msg.params?.enable_thinking; // 预留
                // const enableSearch = msg.params?.enable_search;     // 预留

                // DeepSeek 的输入框一般是 textarea
                const textarea = document.querySelector('#chat-input') || document.querySelector('textarea');
                
                if (textarea) {
                    // 聚焦
                    textarea.focus();
                    
                    // 绕过 React 给 textarea 赋值
                    const nativeSetter = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, "value").set;
                    nativeSetter.call(textarea, promptText);
                    
                    // 通知 React 状态更新
                    textarea.dispatchEvent(new Event('input', { bubbles: true }));
                    
                    // 给页面一点时间响应 input 事件使发送按钮高亮
                    setTimeout(() => {
                        // 尝试找到并点击发送按钮 (DeepSeek 通常是在输入框旁边的特定 div/button)
                        const sendBtnContainer = textarea.parentElement.parentElement;
                        // 找带有 "发送" aria-label 或者包含特定 SVG 的按钮
                        const sendBtn = sendBtnContainer.querySelector('div[role="button"]') || document.querySelector('div[aria-label="发送"]');
                        
                        if (sendBtn) {
                            sendBtn.click();
                            console.log("✅ 提示词已发送 (按钮点击)");
                        } else {
                            // 备用方案：模拟键盘按下回车键发送
                            textarea.dispatchEvent(new KeyboardEvent('keydown', { 
                                key: 'Enter', 
                                code: 'Enter', 
                                keyCode: 13, 
                                which: 13, 
                                bubbles: true 
                            }));
                            console.log("✅ 提示词已发送 (回车触发)");
                        }
                    }, 200);
                } else {
                    console.error("❌ 未找到输入框");
                }
                break;
        }
    }

    // ==========================================
    // 3. XHR 劫持 (核心：提取流发给 Golang)
    // ==========================================
    const originalOpen = window.XMLHttpRequest.prototype.open;
    const originalSend = window.XMLHttpRequest.prototype.send;

    window.XMLHttpRequest.prototype.open = function(method, url) {
        this._interceptUrl = typeof url === 'string' ? url : (url?.toString() || '');
        return originalOpen.apply(this, arguments);
    };

    window.XMLHttpRequest.prototype.send = function() {
        let lastLength = 0;
        let buffer = ""; 

        this.addEventListener('readystatechange', function() {
            // 目标接口为 chat/completion
            if (this._interceptUrl && this._interceptUrl.includes('/chat/completion')) {
                // 状态 3: 正在接收数据 (流式)；状态 4: 接收完成
                if (this.readyState === 3 || this.readyState === 4) {
                    
                    const currentText = this.responseText;
                    const newData = currentText.substring(lastLength);
                    lastLength = currentText.length;

                    if (!newData) return;

                    buffer += newData;
                    // SSE 通常以 \n\n 结尾，这里保险起见用 \n 切分
                    const parts = buffer.split('\n');
                    
                    // 弹出最后一个（大概率不完整，留在 buffer 里等下一个网络包拼起来）
                    buffer = parts.pop();

                    for (const part of parts) {
                        const line = part.trim();
                        if (!line || !line.startsWith('data:')) continue;

                        const jsonStr = line.substring(5).trim();
                        if (jsonStr === '[DONE]') {
                            // 可选：告诉后端这一轮回答结束了
                            sendToServer("token", { done: true });
                            continue;
                        }

                        try {
                            const parsedObj = JSON.parse(jsonStr);
                            // 将提取到的 Object 原封不动发给 Golang 中继
                            sendToServer("token", parsedObj);
                        } catch (e) {
                            // 偶尔切包太碎可能会解析失败
                            console.warn("⚠️ 解析 SSE JSON 失败 (可忽略):", jsonStr);
                        }
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
        const btnText = isConnected ? '🟢 WS已连接(点击断开)' : '🔴 WS未连接(点击重连)';

        // 注册控制 WS 的按钮
        window.FloatingBallAPI.registerAction(
            SCRIPT_ID, 
            'btn_ws_toggle', 
            btnText, 
            () => {
                if (isConnected) {
                    ws.close();
                } else {
                    connectWS();
                }
            }
        );
        
        // 保留原有的快捷测试按钮
        window.FloatingBallAPI.registerAction(
            SCRIPT_ID, 
            'btn_hello_world', 
            '✍️ 输入 HelloWorld', 
            () => {
                handleCommand({ command: 'send_prompt', params: { prompt: 'helloworld' } });
            }
        );
    }

    // 初始化运行
    connectWS();
    registerUI();

})();