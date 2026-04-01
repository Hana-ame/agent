// ==UserScript==
// @name         DeepSeek 增强脚本 (整合 WebSocket + XHR 劫持版)
// @version      2.0
// @description  匹配 Golang Bridge Server API V2，支持 DOM 操作、流拦截与全新 Channel 路由
// @match        *://*.deepseek.com/*
// @grant        none
// ==/UserScript==

(function () {
    // [START] SCRIPT-INIT
    // version: 002
    // 上下文：脚本注入后立即执行。先决调用：无。后续调用：XHR-HIJACK 覆写、WS-CONNECT 建立通信、UI-REGISTER 挂载组件。
    // 输入参数：无
    // 输出参数：无
    // 预留扩展空间：可在此处初始化全局拦截器开关或读取本地缓存的配置项。
    console.log('🚀 DeepSeek 增强脚本已注入 (全新 Channel 路由版)！');

    const SCRIPT_ID = 'deepseek-tools';
    const WS_URL = "wss://d.810114.xyz/ws/browser"; // 与 Go Server 地址和端口对齐

    let ws = null;
    let uiRegistered = false;
    let isPaired = false; // 记录当前是否处于配对状态
    // [END] SCRIPT-INIT

    // [START] DOM-FIND-CLICK
    // version: 0.0.2
    // 上下文：当接收到远程 DOM 操作指令，需要模拟物理点击时调用。先决调用：目标 SVG Path 渲染完成。后续调用：触发浏览器原生事件流。
    // 输入参数：pathD (string) 
    // 输出参数：是否成功找到并点击 (boolean)
    // 辅助function
    function findAndClickByPath(pathD) {
        const buttons = Array.from(document.querySelectorAll('div[role="button"], button'));
        const targetBtn = buttons.find(btn => btn.innerHTML.includes(pathD));

        if (targetBtn) {
            if (targetBtn.getAttribute('aria-disabled') === 'true') {
                console.warn("⚠️ 按钮当前处于禁用状态，跳过点击");
                return false;
            }
            targetBtn.click();
            return true;
        }
        return false;
    }
    //[END] DOM-FIND-CLICK

    // [START] DOM-NEW-CHAT
    // version: 0.0.2
    // 上下文：收到 new_chat 指令时调用。先决调用：页面中存在包含特定 SVG Path 的新建对话按钮。后续调用：DOM-FIND-CLICK 执行动作。
    // 输入参数：无
    // 输出参数：无
    // 功能function
    function doNewChat() {
        const path = 'M10 1.22943C5.15604';
        if (findAndClickByPath(path)) {
            console.log("✅ 新建对话按钮已点击");
        } else {
            console.error("❌ 未找到新建对话按钮");
        }
    }
    // [END] DOM-NEW-CHAT

    // [START] DOM-SEND-PROMPT
    // version: 0.0.2
    // 上下文：收到 send_prompt 指令时调用。先决调用：DOM 中存在聊天输入框。后续调用：原生 set 方法修改 value，抛出 input 事件，延时调用 DOM-FIND-CLICK。
    // 输入参数：text (string)
    // 输出参数：无
    // 功能function
    function doSendPrompt(text) {
        const textarea = document.querySelector('#chat-input') || document.querySelector('textarea');
        if (!textarea) return console.error("❌ 未找到输入框");

        textarea.focus();
        const nativeSetter = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, "value").set;
        nativeSetter.call(textarea, text);
        textarea.dispatchEvent(new Event('input', { bubbles: true }));
        // 2. 合并：模拟图片文件粘贴事件
        const simulateImagePaste = (target) => {
            // 构造一个简单的透明 1x1 PNG 图片 Blob (或者你可以传入真实的 File 对象)
            const base64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=";
            const byteString = atob(base64);
            const arrayBuffer = new ArrayBuffer(byteString.length);
            const uint8Array = new Uint8Array(arrayBuffer);
            for (let i = 0; i < byteString.length; i++) uint8Array[i] = byteString.charCodeAt(i);
            const blob = new Blob([uint8Array], { type: "image/png" });
            const file = new File([blob], "pasted_image.png", { type: "image/png" });

            // 创建 DataTransfer 并注入文件
            const dataTransfer = new DataTransfer();
            dataTransfer.items.add(file);

            // 触发 paste 事件
            const pasteEvent = new ClipboardEvent('paste', {
                clipboardData: dataTransfer,
                bubbles: true,
                cancelable: true
            });
            target.dispatchEvent(pasteEvent);
        };

        simulateImagePaste(textarea);
        
        setTimeout(() => {
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
    // [END] DOM-SEND-PROMPT

    // [START] DOM-REMOVE-MSG
    // version: 0.0.2
    // 上下文：收到 remove_msg 指令时调用。先决调用：消息列表中已存在对话。后续调用：DOM 节点移除操作。
    // 输入参数：无
    // 输出参数：无
    // 功能function
    function doRemoveElements() {
        const sampleItem = document.querySelector('[style*="--assistant-last-margin-bottom: 32px"]');
        if (!sampleItem) return console.error("❌ 未能定位消息列表");

        const container = sampleItem.parentElement.parentElement;
        const firstChild = container.firstElementChild;

        if (firstChild) {
            if (!firstChild.querySelector('input, textarea')) {
                firstChild.remove();
                console.log("✅ 成功删除所有消息元素");
            } else {
                console.warn("⚠️ 所有元素包含输入区域，已取消操作");
            }
        }
    }
    // [END] DOM-REMOVE-MSG

    // [START] WS-CONNECT
    // version: 0.0.2
    // 上下文：脚本初始化或连接断开重连时调用。先决调用：后端 Server 处于监听状态。后续调用：触发 onopen 进行系统级注册；触发 onmessage 根据 Channel 分流。
    // 输入参数：无
    // 输出参数：无
    function connectWS() {
        if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) {
            return;
        }

        ws = new WebSocket(WS_URL);

        ws.onopen = () => {
            console.log("🟢 成功连接到 Bridge Server");
            
            // 🔥 根据 V2 接口协议，发送 system 频道的 register 动作
            const registerPayload = {
                action: "register",
                type: "deepseek", // 固定浏览器类型
                title: document.title, // 取代 UUID 作为部分标识
                created_at: Date.now()
            };
            sendToServer("system", registerPayload);
            updateUI();
        };

        ws.onclose = () => {
            console.log("🔴 连接断开，5秒后尝试重连...");
            ws = null;
            isPaired = false;
            updateUI();
            setTimeout(connectWS, 5000);
        };

        ws.onmessage = (event) => {
            try {
                const msg = JSON.parse(event.data);
                
                // 核心分流逻辑：按 Channel 拆分
                if (msg.channel === "system") {
                    handleSystemMessage(msg.payload);
                } else if (msg.channel === "client") {
                    // 接收透传自 Client 的指令
                    handleClientCommand(msg.payload);
                }
            } catch (e) {
                console.error("解析 WebSocket 消息失败", e, event.data);
            }
        };
    }
    // [END] WS-CONNECT

    // [START] WS-SEND
    // version: 0.0.2
    // 上下文：需要向 Go Server 发送系统指令或向 Client 透传数据时调用。先决调用：WS-CONNECT 且 readyState 为 OPEN。后续调用：网络数据流出。
    // 输入参数：channel (string: "system" | "browser"), payload (object)
    // 输出参数：无
    function sendToServer(channel, payload) {
        if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({
                channel: channel,
                payload: payload
            }));
        }
    }
    //[END] WS-SEND

    // [START] WS-SYS-HANDLER
    // version: 0.0.2
    // 上下文：接收到 channel 属 system 的消息时调用。先决调用：WS-CONNECT 分发拦截。后续调用：更新本地配对状态与 UI。
    // 输入参数：payload (object)
    // 输出参数：无
    function handleSystemMessage(payload) {
        console.log("⚙️ 收到系统通知:", payload.action);
        switch (payload.action) {
            case "register_success":
                console.log("✅ 浏览器节点注册成功，等待配对...");
                break;
            case "paired_by_client":
                isPaired = true;
                console.log("🔗 已被 Client 成功锁定配对！");
                updateUI();
                break;
            case "unpaired":
                isPaired = false;
                console.log(`🔓 已解除配对，原因: ${payload.content}`);
                updateUI();
                break;
            default:
                console.log("ℹ️ 未知系统动作:", payload.action);
        }
    }
    // [END] WS-SYS-HANDLER

    // [START] WS-CMD-HANDLER
    // version: 0.0.2
    // 上下文：收到 channel 属 client 且被 Server 透传的指令时调用。先决调用：已被配对 (isPaired=true)。后续调用：触发相应的 DOM 操作。
    // 输入参数：payload (object)
    // 输出参数：无
    function handleClientCommand(payload) {
        console.log("📥 收到 Client 业务透传指令:", payload);
        
        const command = payload.command;
        const params = payload.params || {};

        switch (command) {
            case "match":
                const target = params.target || "";
                const currentHost = location.hostname;
                const isMatch = currentHost.includes(target);
                
                // 向对端（Client）透传自身验证结果，通道固定用自身角色 browser
                sendToServer("browser", {
                    type: "match_result",
                    success: isMatch,
                    host: currentHost,
                    model: target
                });
                break;
            case "new_chat":
                doNewChat();
                break;
            case "send_prompt":
                doSendPrompt(params.prompt || "");
                break;
            case "remove_msg":
                doRemoveFirstElement();
                break;
            default:
                console.log("⚠️ 未知 Client 业务指令:", command);
        }
    }
    // [END] WS-CMD-HANDLER

    // [START] XHR-HIJACK
    // version: 0.0.2
    // 上下文：页面初始化劫持原生 XHR。先决调用：必须在页面核心 JS 发起请求前执行。后续调用：劫持大模型流式响应并推给远端 Client。
    // 输入参数：无
    // 输出参数：无
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
                        
                        // 发送业务 Token 到 Client 端，走 browser 通道
                        if (jsonStr === '[DONE]') {
                            sendToServer("browser", { type: "token", data: { done: true } });
                            continue;
                        }

                        try {
                            const parsedObj = JSON.parse(jsonStr);
                            sendToServer("browser", { type: "token", data: parsedObj });
                        } catch (e) { }
                    }
                }
            }
        });
        return originalSend.apply(this, arguments);
    };
    // [END] XHR-HIJACK

    // [START] UI-REGISTER
    // version: 002
    // 上下文：脚本启动轮询检查依赖是否就绪。先决调用：等待外部 FloatingBallAPI 挂载到 window。后续调用：UI-UPDATE 注入菜单。
    // 输入参数：无
    // 输出参数：无
    function registerUI() {
        if (window.FloatingBallAPI) {
            uiRegistered = true;
            updateUI();
        } else {
            setTimeout(registerUI, 500);
        }
    }
    // [END] UI-REGISTER

    //[START] UI-UPDATE
    // version: 0.0.2
    // 上下文：WS 状态发生改变或 UI 首次就绪时调用。先决调用：UI-REGISTER 确认 API 存在。后续调用：外部 API 渲染 DOM。
    // 输入参数：无
    // 输出参数：无
    function updateUI() {
        if (!uiRegistered || !window.FloatingBallAPI) return;

        const isConnected = ws && ws.readyState === WebSocket.OPEN;
        let btnText = '🔴 WS 未连接';
        if (isConnected) {
            btnText = isPaired ? '🟢 已配对 (转发中)' : '🟡 已连接 (等配对)';
        }

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
    // [END] UI-UPDATE

    connectWS();
    registerUI();
})();