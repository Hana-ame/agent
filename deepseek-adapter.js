// ==UserScript==
// @name         DeepSeek 增强脚本 (OOP 架构 + WS启停 + 严格协议版)
// @version      3.0
// @description  匹配 Golang Bridge Server API V2，严格遵循面向对象结构，支持 DOM 操作、流拦截、WS 启停与 Channel 路由
// @match        *://*.deepseek.com/*
// @grant        none
// ==/UserScript==

(function () {
    console.log('🚀 DeepSeek 增强脚本已注入 (OOP 架构版)！');

    const SCRIPT_ID = 'deepseek-tools';
    const WS_URL = "wss://d.810114.xyz/ws/browser";

    // ==========================================
    // 1. Utils 辅助工具类
    // 负责通用的 DOM 查找与交互底层逻辑
    // ==========================================
    class Utils {
        /**
         * 通过 SVG Path 查找对应的按钮并执行点击
         * @param {string} pathD - 目标按钮内包含的 SVG Path 字符串
         * @returns {boolean} - 是否成功找到并点击
         */
        static findAndClickByPath(pathD) {
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
    }

    // ==========================================
    // 2. Adapter 适配器类
    // 负责具体的业务功能实现与 DOM 操作解耦
    // ==========================================
    class Adapter {
        /**
         * 新建对话
         */
        doNewChat() {
            const path = 'M10 1.22943C5.15604';
            if (Utils.findAndClickByPath(path)) {
                console.log("✅ 新建对话按钮已点击");
            } else {
                console.error("❌ 未找到新建对话按钮");
            }
        }

        /**
         * 设置输入框文本 (触发原生 set 与 input 事件)
         * @param {string} text - 提示词内容
         */
        doSetPrompt(text) {
            if (!text) return;
            const textarea = document.querySelector('#chat-input') || document.querySelector('textarea');
            if (!textarea) return console.error("❌ 未找到输入框");

            textarea.focus();
            const nativeSetter = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, "value").set;
            nativeSetter.call(textarea, text);
            textarea.dispatchEvent(new Event('input', { bubbles: true }));
            console.log("✅ 文本已填入");
        }

        /**
         * 模拟粘贴图片 (支持外部传入的 Base64 字符串)
         * @param {string} base64Str - 图片的 base64 数据
         */
        doAddImage(base64Str) {
            if (!base64Str) return;
            const textarea = document.querySelector('#chat-input') || document.querySelector('textarea');
            if (!textarea) return console.error("❌ 未找到输入框，无法粘贴图片");

            try {
                // 清理可能包含的 Data URL 前缀
                const pureBase64 = base64Str.replace(/^data:image\/(png|jpeg|jpg);base64,/, '');
                const byteString = atob(pureBase64);
                const arrayBuffer = new ArrayBuffer(byteString.length);
                const uint8Array = new Uint8Array(arrayBuffer);
                for (let i = 0; i < byteString.length; i++) uint8Array[i] = byteString.charCodeAt(i);
                
                const blob = new Blob([uint8Array], { type: "image/png" });
                const file = new File([blob], "pasted_image.png", { type: "image/png" });

                const dataTransfer = new DataTransfer();
                dataTransfer.items.add(file);

                const pasteEvent = new ClipboardEvent('paste', {
                    clipboardData: dataTransfer,
                    bubbles: true,
                    cancelable: true
                });
                textarea.dispatchEvent(pasteEvent);
                console.log("✅ 图片已注入剪贴板");
            } catch (e) {
                console.error("❌ 解析并粘贴图片失败", e);
            }
        }

        /**
         * 统一中控：处理发送、深度思考、联网搜索的开关状态
         * @param {Object} config - {send: boolean, thinking: boolean, search: boolean}
         */
        doControl(config) {
            // 1. 深度思考与搜索开关 (如果有特定的 SVG 或按钮标识，可以在这里扩展 findAndClickByPath)
            if (config.thinking) {
                console.log("🧠 尝试开启深度思考...");
                // 示例预留：Utils.findAndClickByPath('深度思考SVG');
            }
            if (config.search) {
                console.log("🌐 尝试开启联网搜索...");
                // 示例预留：Utils.findAndClickByPath('联网搜索SVG');
            }

            // 2. 发送操作 (延时等待前置操作如粘贴图片完成)
            if (config.send) {
                setTimeout(() => {
                    const sendPath = 'M8.3125 0.981587C8.66767 1.0545 8.97902 1.20558 9.2627 1.43374';
                    const textarea = document.querySelector('#chat-input') || document.querySelector('textarea');
                    
                    if (Utils.findAndClickByPath(sendPath)) {
                        console.log("✅ 消息已发送 (通过点击)");
                    } else if (textarea) {
                        console.log("⚠️ 尝试模拟回车发送...");
                        textarea.dispatchEvent(new KeyboardEvent('keydown', {
                            key: 'Enter', code: 'Enter', keyCode: 13, which: 13, bubbles: true
                        }));
                    }
                }, 300);
            }
        }

        /**
         * 删除页面上的首条(或所有)消息历史
         */
        doRemove() {
            const sampleItem = document.querySelector('[style*="--assistant-last-margin-bottom: 32px"]');
            if (!sampleItem) return console.error("❌ 未能定位消息列表");

            const container = sampleItem.parentElement.parentElement;
            const firstChild = container.firstElementChild;

            if (firstChild) {
                if (!firstChild.querySelector('input, textarea')) {
                    firstChild.remove();
                    console.log("✅ 成功删除目标消息元素");
                } else {
                    console.warn("⚠️ 目标元素包含输入区域，已取消删除操作");
                }
            }
        }
    }

    // ==========================================
    // 3. WsHandler WebSocket 与网络处理类
    // 负责通信建立、断线重连、手动启停、流拦截与消息路由
    // ==========================================
    class WsHandler {
        constructor(adapter, uiRef) {
            this.adapter = adapter;
            this.uiRef = uiRef;      // 引用 UI 实例以更新状态
            this.ws = null;
            this.isPaired = false;
            this.isPaused = false;   // 核心状态：是否由用户手动暂停链接，默认 false (自动连接)
        }

        /**
         * 初始化/建立 WebSocket 连接
         */
        connect() {
            if (this.isPaused) return; // 如果被手动暂停，则不允许自动连接
            if (this.ws && (this.ws.readyState === WebSocket.OPEN || this.ws.readyState === WebSocket.CONNECTING)) return;

            this.ws = new WebSocket(WS_URL);

            this.ws.onopen = () => {
                console.log("🟢 成功连接到 Bridge Server");
                // 遵循新格式发送系统注册指令
                this.send("system", { 
                    command: "register", 
                    message: document.title // 用标题作为标识
                });
                if(this.uiRef) this.uiRef.update();
            };

            this.ws.onclose = () => {
                this.ws = null;
                this.isPaired = false;
                if(this.uiRef) this.uiRef.update();
                
                // 仅在非手动暂停状态下自动重连
                if (!this.isPaused) {
                    console.log("🔴 连接断开，5秒后尝试重连...");
                    setTimeout(() => this.connect(), 5000);
                } else {
                    console.log("⏸️ 连接已挂起 (用户手动暂停)。");
                }
            };

            this.ws.onmessage = (event) => {
                try {
                    const msg = JSON.parse(event.data);
                    // 严格按 Channel 分发
                    if (msg.channel === "system") {
                        this.handleSystemMessage(msg.payload);
                    } else if (msg.channel === "client") {
                        this.handleClientMessage(msg.payload);
                    }
                } catch (e) {
                    console.error("❌ 解析 WebSocket 消息失败", e, event.data);
                }
            };
        }

        /**
         * 切换连接状态 (连接 <-> 暂停)
         */
        toggleConnection() {
            if (this.isPaused) {
                // 当前是暂停状态，切换为连接
                this.isPaused = false;
                console.log("▶️ 恢复 WebSocket 连接...");
                this.connect();
            } else {
                // 当前是连接状态，切换为暂停
                this.isPaused = true;
                this.isPaired = false;
                if (this.ws) {
                    console.log("⏸️ 正在断开 WebSocket 连接...");
                    this.ws.close();
                }
            }
            if(this.uiRef) this.uiRef.update();
        }

        /**
         * 遵循协议标准向服务端发送数据
         * 格式: { channel: channel, payload: payload_object }
         */
        send(channel, payload) {
            if (this.ws && this.ws.readyState === WebSocket.OPEN) {
                this.ws.send(JSON.stringify({
                    channel: channel,
                    payload: payload
                }));
            }
        }

        /**
         * 处理 Channel: system 的消息
         */
        handleSystemMessage(payload) {
            const command = payload.command;
            console.log("⚙️ 收到系统通知:", command);
            
            switch (command) {
                case "register_success":
                    console.log("✅ 浏览器节点注册成功，等待配对...");
                    break;
                case "paired_by_client":
                    this.isPaired = true;
                    console.log("🔗 已被 Client 成功锁定配对！");
                    if(this.uiRef) this.uiRef.update();
                    break;
                case "unpaired":
                    this.isPaired = false;
                    console.log(`🔓 已解除配对，原因: ${payload.message}`);
                    if(this.uiRef) this.uiRef.update();
                    break;
            }
        }

        /**
         * 处理 Channel: client 的透传业务指令
         * payload 格式: { command: string, message: string, image?: string }
         */
        handleClientMessage(payload) {
            console.log("📥 收到 Client 业务指令:", payload);
            const { command, message, image } = payload;

            switch (command) {
                case "match":
                    const isMatch = location.hostname.includes(message);
                    this.send("browser", {
                        command: "match_result",
                        message: isMatch.toString()
                    });
                    break;
                case "new_chat":
                    this.adapter.doNewChat();
                    break;
                case "send_prompt":
                    // 组合调用 Adapter 完成输入、贴图与发送
                    if (message) this.adapter.doSetPrompt(message);
                    if (image) this.adapter.doAddImage(image);
                    // 解析控制参数，此处默认触发 send
                    this.adapter.doControl({ send: true, thinking: true, search: true });
                    break;
                case "remove_msg":
                    this.adapter.doRemove();
                    break;
                default:
                    console.log("⚠️ 未知 Client 业务指令:", command);
            }
        }

        /**
         * 劫持原生 XMLHttpRequest，拦截大模型流式输出
         * payload 格式直接传入 SSE object
         */
        handleXHR() {
            const originalOpen = window.XMLHttpRequest.prototype.open;
            const originalSend = window.XMLHttpRequest.prototype.send;
            const self = this; // 保存 WsHandler 实例的引用

            window.XMLHttpRequest.prototype.open = function (method, url) {
                this._interceptUrl = typeof url === 'string' ? url : (url?.toString() || '');
                return originalOpen.apply(this, arguments);
            };

            window.XMLHttpRequest.prototype.send = function () {
                let lastLength = 0;
                let buffer = "";

                this.addEventListener('readystatechange', function () {
                    // 针对 DeepSeek 的流式对话接口进行拦截
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
                                
                                // 根据要求，载荷如果是 SSE 数据，直接包装为 { SSEobject }
                                if (jsonStr === '[DONE]') {
                                    self.send("browser", { done: true });
                                    continue;
                                }

                                try {
                                    const parsedObj = JSON.parse(jsonStr);
                                    self.send("browser", parsedObj);
                                } catch (e) { }
                            }
                        }
                    }
                });
                return originalSend.apply(this, arguments);
            };
        }
    }

    // ==========================================
    // 4. UI 界面类
    // 负责悬浮球 API 接入与界面渲染
    // ==========================================
    class UI {
        constructor() {
            this.wsHandler = null; 
            this.adapter = null;
            this.uiRegistered = false;
        }

        /**
         * 注入核心依赖并启动轮询注册
         */
        init(wsHandler, adapter) {
            this.wsHandler = wsHandler;
            this.adapter = adapter;
            this.registerLoop();
        }

        /**
         * 轮询检查外部悬浮球环境是否就绪
         */
        registerLoop() {
            if (window.FloatingBallAPI) {
                this.uiRegistered = true;
                this.update();
            } else {
                setTimeout(() => this.registerLoop(), 500);
            }
        }

        /**
         * 渲染 / 更新悬浮球菜单
         */
        update() {
            if (!this.uiRegistered || !window.FloatingBallAPI) return;

            // 根据状态机决定按钮文本与表现
            const isConnected = this.wsHandler.ws && this.wsHandler.ws.readyState === WebSocket.OPEN;
            const isPaused = this.wsHandler.isPaused;
            const isPaired = this.wsHandler.isPaired;

            let btnText = '🔴 WS 离线 (点击连接)';
            if (isPaused) {
                btnText = '⏸️ WS 已暂停 (点击恢复)';
            } else if (isConnected) {
                btnText = isPaired ? '🟢 已配对/点击暂停' : '🟡 待配对/点击暂停';
            }

            // 1. WebSocket 启停按钮
            window.FloatingBallAPI.registerAction(
                SCRIPT_ID,
                'btn_ws_toggle',
                btnText,
                () => { this.wsHandler.toggleConnection(); }
            );

            // 2. 清理首条消息按钮
            window.FloatingBallAPI.registerAction(
                SCRIPT_ID,
                'btn_clear_first',
                '🗑️ 删除首条消息',
                () => { this.adapter.doRemove(); }
            );

            // 3. 测试发信按钮
            window.FloatingBallAPI.registerAction(
                SCRIPT_ID,
                'btn_test_send',
                '✍️ 测试发送消息',
                () => { 
                    this.adapter.doSetPrompt('测试消息 - By Script'); 
                    this.adapter.doControl({send: true}); 
                }
            );
        }
    }

    // ==========================================
    // 5. 组装与启动过程 (IoC/DI 初始化)
    // ==========================================
    const adapter = new Adapter();
    const ui = new UI();
    const wsHandler = new WsHandler(adapter, ui);

    // 双向绑定 (UI 渲染依赖 WS 的状态)
    ui.init(wsHandler, adapter);

    // 默认执行网络与拦截初始化
    wsHandler.handleXHR();
    wsHandler.connect(); // 默认进入连接状态

})();