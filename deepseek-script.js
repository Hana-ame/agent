// ==UserScript==
// @name         DeepSeek 增强脚本 (整合 WebSocket + XHR 劫持版)
// @version      1.1
// @description  匹配 Golang Bridge Server API，支持 DOM 操作与流拦截
// @match        *://*.deepseek.com/*
// @grant        none
// ==/UserScript==

(function () {
    // [START] SCRIPT-INIT
    // version: 001
    // 上下文：脚本注入后立即执行。先决调用：无。后续调用：XHR-HIJACK 覆写、WS-CONNECT 建立通信、UI-REGISTER 挂载组件。
    // 输入参数：无
    // 输出参数：无
    // 预留扩展空间：可在此处初始化全局拦截器开关或读取本地缓存的配置项。
    console.log('🚀 DeepSeek 增强脚本已注入 (XHR + WS 路径识别版)！');

    const SCRIPT_ID = 'deepseek-tools';
    const WS_URL = "wss://d.810114.xyz/ws/browser"; // 修改端口为 8765

    let ws = null;
    let uiRegistered = false;
    // [END] SCRIPT-INIT

    // [START] DOM-FIND-CLICK
    // version: 001
    // 上下文：当接收到远程 DOM 操作指令，需要模拟物理点击时调用。先决调用：目标 SVG Path 渲染完成。后续调用：触发浏览器原生事件流。
    // 输入参数：pathD (string) 
    // 输出参数：是否成功找到并点击 (boolean)
    // 预留扩展空间：可增加基于 XPath 或 CSS 属性的组合查询备用方案。
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
    // [END] DOM-FIND-CLICK

    // [START] DOM-NEW-CHAT
    // version: 001
    // 上下文：收到 new_chat 指令时调用。先决调用：页面中存在包含特定 SVG Path 的新建对话按钮。后续调用：DOM-FIND-CLICK 执行动作。
    // 输入参数：无
    // 输出参数：无
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
    // version: 001
    // 上下文：收到 send_prompt 指令时调用。先决调用：DOM 中存在聊天输入框。后续调用：原生 set 方法修改 value，抛出 input 事件，延时调用 DOM-FIND-CLICK。
    // 输入参数：text (string)
    // 输出参数：无
    // 预留扩展空间：可增加针对富文本编辑器（如 ContentEditable）的兼容注入。
    function doSendPrompt(text) {
        const textarea = document.querySelector('#chat-input') || document.querySelector('textarea');
        if (!textarea) return console.error("❌ 未找到输入框");

        textarea.focus();
        const nativeSetter = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, "value").set;
        nativeSetter.call(textarea, text);
        textarea.dispatchEvent(new Event('input', { bubbles: true }));

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
    // version: 001
    // 上下文：收到 remove_msg 指令时调用。先决调用：消息列表中已存在对话。后续调用：DOM 节点移除操作。
    // 输入参数：无
    // 输出参数：无
    function doRemoveFirstElement() {
        const sampleItem = document.querySelector('[style*="--assistant-last-margin-bottom: 32px"]');
        if (!sampleItem) return console.error("❌ 未能定位消息列表");

        const container = sampleItem.parentElement.parentElement;
        const firstChild = container.firstElementChild;

        if (firstChild) {
            if (!firstChild.querySelector('input, textarea')) {
                firstChild.remove();
                console.log("✅ 成功删除首个消息元素");
            } else {
                console.warn("⚠️ 首个元素包含输入区域，已取消操作");
            }
        }
    }
    // [END] DOM-REMOVE-MSG

    // [START] WS-CONNECT
    // version: 001
    // 上下文：脚本初始化或连接断开重连时调用。先决调用：后端 Server 处于监听状态。后续调用：触发 onopen 进行 WS-SEND (register)；触发 onmessage 进行 WS-CMD-HANDLER。
    // 输入参数：无
    // 输出参数：无
    // 预留扩展空间：可增加心跳超时主动断开机制，或指数退避重连策略。
    function connectWS() {
        if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) {
            return;
        }

        ws = new WebSocket(WS_URL);

        ws.onopen = () => {
            console.log("🟢 成功连接到 Golang Server");
            // 🔥 适配 Go Server：同时上报 model 和 title，供客户端做精确配对
            sendToServer("register", null, { 
                model: "deepseek",
                title: document.title 
            });
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
    // [END] WS-CONNECT

    // [START] WS-SEND
    // version: 001
    // 上下文：需要向 Go Server 发送数据或响应时调用。先决调用：WS-CONNECT 且 readyState 为 OPEN。后续调用：网络数据流出。
    // 输入参数：type (string), content (any), extra (object)
    // 输出参数：无
    function sendToServer(type, content = null, extra = {}) {
        if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ type, content, ...extra }));
        }
    }
    // [END] WS-SEND

    // [START] WS-CMD-HANDLER
    // version: 001
    // 上下文：WS-CONNECT 监听到远端推送消息时调用。先决调用：解析 event.data 为 JSON。后续调用：分配至各 DOM-* 动作函数或反馈 WS-SEND。
    // 输入参数：msg (object)
    // 输出参数：无
    // 预留扩展空间：可在此实现拦截器，校验 msg 签名或过滤来源不明的 command。
    function handleCommand(msg) {
        if (msg.type !== "command") return;
        console.log("📥 收到远程指令:", msg.command, msg.params);

        switch (msg.command) {
            case "match":
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
            case "remove_msg":
                doRemoveFirstElement();
                break;
        }
    }
    // [END] WS-CMD-HANDLER

    // [START] XHR-HIJACK
    // version: 001
    // 上下文：页面初始化劫持原生 XHR 原型链。先决调用：必须在页面核心 JS 发起请求前执行。后续调用：劫持响应流分片提取数据并通过 WS-SEND 推送至服务端。
    // 输入参数：无 (通过原型链注入 arguments)
    // 输出参数：无 (返回原生方法调用结果)
    // 预留扩展空间：可扩展支持 Fetch API 的劫持，或过滤其他指定路径（如 /api/v1/user）。
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
    // [END] XHR-HIJACK

    // [START] UI-REGISTER
    // version: 001
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

    // [START] UI-UPDATE
    // version: 001
    // 上下文：WS 状态发生改变或 UI 首次就绪时调用。先决调用：UI-REGISTER 确认 API 存在。后续调用：外部 API 渲染 DOM。
    // 输入参数：无
    // 输出参数：无
    function updateUI() {
        if (!uiRegistered || !window.FloatingBallAPI) return;

        const isConnected = ws && ws.readyState === WebSocket.OPEN;
        const btnText = isConnected ? '🟢 WS已连接(8080)' : '🔴 WS未连接(点击重连)';

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