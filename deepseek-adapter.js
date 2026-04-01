// ==UserScript==
// @name         DeepSeek 增强脚本 (OOP 架构 + WS启停 + 严格协议版)
// @version      3.2
// @description  匹配 Golang Bridge Server API V2，严格遵循面向对象结构，支持 DOM 操作、流拦截、WS 启停与 Channel 路由
// @match        *://*.deepseek.com/*
// @grant        none
// ==/UserScript==

(function () {
    console.log('🚀 DeepSeek 增强脚本已注入 (OOP 架构版 V3.2)！');

    const SCRIPT_ID = 'deepseek-tools';
    // 默认 WS 地址，若用户未设置过则使用此地址
    const DEFAULT_WS_URL = "ws://127.26.3.1:8080/ws/browser";

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
         * @param {Object} config - {send?: boolean, thinking?: boolean, search?: boolean}
         */
        doControl(config) {
            // 1. 深度思考与搜索开关控制
            if (config.hasOwnProperty('thinking')) {
                console.log(`🧠 尝试将深度思考设为: ${config.thinking}`);
                // 此处预留执行真实 DOM 切换的逻辑，如：
                // 如果当前状态与 config.thinking 不一致，则 Utils.findAndClickByPath('深度思考SVG')
            }
            if (config.hasOwnProperty('search')) {
                console.log(`🌐 尝试将联网搜索设为: ${config.search}`);
                // 此处预留执行真实 DOM 切换的逻辑
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
    // ==========================================
    class WsHandler {
        constructor(adapter, uiRef) {
            this.adapter = adapter;
            this.uiRef = uiRef;      // 引用 UI 实例以更新状态
            this.ws = null;
            this.wsUrl = localStorage.getItem('deepseek_ws_url') || DEFAULT_WS_URL;
            this.isPaired = false;
            this.isPaused = false;   // 核心状态：是否由用户手动暂停链接，默认 false
        }

        /**
         * 修改并持久化 WebSocket Endpoint
         */
        setEndpoint(newUrl) {
            if (!newUrl || newUrl === this.wsUrl) return;
            this.wsUrl = newUrl;
            localStorage.setItem('deepseek_ws_url', newUrl);
            console.log(`🔧 WS Endpoint 已更新为: ${newUrl}`);

            if (this.ws) {
                this.ws.close();
            } else if (!this.isPaused) {
                this.connect();
            }
            if (this.uiRef) this.uiRef.update();
        }

        connect() {
            if (this.isPaused) return;
            if (this.ws && (this.ws.readyState === WebSocket.OPEN || this.ws.readyState === WebSocket.CONNECTING)) return;

            this.ws = new WebSocket(this.wsUrl);

            this.ws.onopen = () => {
                console.log(`🟢 成功连接到 Bridge Server [${this.wsUrl}]`);
                this.send("system", {
                    command: "register",
                    title: document.title,          // 对应 Golang 的 payload.Title
                    type: "deepseek-webapp"         // 对应 Golang 的 payload.Type (可自定义你的浏览器类型标识)
                });
                if (this.uiRef) this.uiRef.update();
            };

            this.ws.onclose = () => {
                this.ws = null;
                this.isPaired = false;
                if (this.uiRef) this.uiRef.update();

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

        toggleConnection() {
            if (this.isPaused) {
                this.isPaused = false;
                console.log("▶️ 恢复 WebSocket 连接...");
                this.connect();
            } else {
                this.isPaused = true;
                this.isPaired = false;
                if (this.ws) {
                    console.log("⏸️ 正在断开 WebSocket 连接...");
                    this.ws.close();
                }
            }
            if (this.uiRef) this.uiRef.update();
        }

        send(channel, payload) {
            if (this.ws && this.ws.readyState === WebSocket.OPEN) {
                this.ws.send(JSON.stringify({ channel: channel, payload: payload }));
            }
        }

        handleSystemMessage(payload) {
            const command = payload.command;
            switch (command) {
                case "register_success":
                    console.log("✅ 浏览器节点注册成功，等待配对...");
                    break;
                case "paired_by_client":
                    this.isPaired = true;
                    console.log("🔗 已被 Client 成功锁定配对！");
                    if (this.uiRef) this.uiRef.update();
                    break;
                case "unpaired":
                    this.isPaired = false;
                    console.log(`🔓 已解除配对，原因: ${payload.message}`);
                    if (this.uiRef) this.uiRef.update();
                    break;
            }
        }

        handleClientMessage(payload) {
            const { command, message, image } = payload;
            switch (command) {
                case "match":
                    const isMatch = location.hostname.includes(message);
                    this.send("browser", { command: "match_result", message: isMatch.toString() });
                    break;
                case "new_chat":
                    this.adapter.doNewChat();
                    break;
                case "send_prompt":
                    if (message) this.adapter.doSetPrompt(message);
                    if (image) this.adapter.doAddImage(image);
                    // 透传指令调用发送，不需要显式改变思考和搜索状态
                    this.adapter.doControl({ send: true });
                    break;
                case "remove_msg":
                    this.adapter.doRemove();
                    break;
            }
        }

        handleXHR() {
            const originalOpen = window.XMLHttpRequest.prototype.open;
            const originalSend = window.XMLHttpRequest.prototype.send;
            const self = this;

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

        init(wsHandler, adapter) {
            this.wsHandler = wsHandler;
            this.adapter = adapter;
            this.registerLoop();
        }

        registerLoop() {
            if (window.FloatingBallAPI) {
                this.uiRegistered = true;
                this.update();
            } else {
                setTimeout(() => this.registerLoop(), 500);
            }
        }

        update() {
            if (!this.uiRegistered || !window.FloatingBallAPI) return;

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
            window.FloatingBallAPI.registerAction(SCRIPT_ID, 'btn_ws_toggle', btnText, () => {
                this.wsHandler.toggleConnection();
            });

            // 2. 修改 WS Endpoint 按钮
            window.FloatingBallAPI.registerAction(SCRIPT_ID, 'btn_change_ws', '🔧 修改 WS 地址', () => {
                const currentUrl = this.wsHandler.wsUrl;
                const newUrl = prompt("请输入新的 WebSocket 地址:", currentUrl);
                if (newUrl) {
                    this.wsHandler.setEndpoint(newUrl.trim());
                }
            });

            // 3. 直接通过 doControl 操作思考状态
            window.FloatingBallAPI.registerAction(SCRIPT_ID, 'btn_thinking_on', '🧠 开启思考', () => {
                this.adapter.doControl({ thinking: true });
            });
            window.FloatingBallAPI.registerAction(SCRIPT_ID, 'btn_thinking_off', '🧠 关闭思考', () => {
                this.adapter.doControl({ thinking: false });
            });

            // 4. 直接通过 doControl 操作搜索状态
            window.FloatingBallAPI.registerAction(SCRIPT_ID, 'btn_search_on', '🌐 开启搜索', () => {
                this.adapter.doControl({ search: true });
            });
            window.FloatingBallAPI.registerAction(SCRIPT_ID, 'btn_search_off', '🌐 关闭搜索', () => {
                this.adapter.doControl({ search: false });
            });

            // 5. 清理首条消息按钮
            window.FloatingBallAPI.registerAction(SCRIPT_ID, 'btn_clear_first', '🗑️ 删除首条消息', () => {
                this.adapter.doRemove();
            });
        }
    }

    // ==========================================
    // 5. 组装与启动过程 (IoC/DI 初始化)
    // ==========================================
    const adapter = new Adapter();
    const ui = new UI();
    const wsHandler = new WsHandler(adapter, ui);

    ui.init(wsHandler, adapter);

    wsHandler.handleXHR();
    wsHandler.connect();

})();