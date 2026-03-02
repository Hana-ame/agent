// ==UserScript==
// @name         悬浮球脚本加载器 (精简版)
// @namespace    http://tampermonkey.net/
// @version      2.1.0
// @description  悬浮球控制外部脚本加载，支持外部脚本向菜单注册自定义按钮，动态配置JSON列表
// @author       You
// @match        *://*/*
// @grant        GM_getValue
// @grant        GM_setValue
// @grant        GM_addStyle
// @grant        GM_addElement
// @grant        GM_registerMenuCommand
// @grant        GM_xmlhttpRequest
// @grant        unsafeWindow
// @run-at       document-start
// @updateURL    https://github.com/Hana-ame/agent/raw/refs/heads/tampermonkey/ball.user.js
// @downloadURL  https://github.com/Hana-ame/agent/raw/refs/heads/tampermonkey/ball.user.js
// ==/UserScript==

(function() {
    'use strict';

    // ============================================
    // 工具函数
    // ============================================

    function safeGetValue(key, defaultValue) {
        try {
            if (typeof GM_getValue !== 'undefined') {
                const val = GM_getValue(key);
                return val !== undefined ? val : defaultValue;
            }
            const stored = localStorage.getItem('floating_ball_' + key);
            return stored ? JSON.parse(stored) : defaultValue;
        } catch (e) {
            return defaultValue;
        }
    }

    function safeSetValue(key, value) {
        try {
            if (typeof GM_setValue !== 'undefined') {
                GM_setValue(key, value);
            }
            localStorage.setItem('floating_ball_' + key, JSON.stringify(value));
        } catch (e) {
            console.error('Failed to save setting:', e);
        }
    }

    function matchPattern(pattern, url) {
        if (!pattern || pattern === "*://*/*") return true;
        try {
            const currentUrl = url || window.location.href;
            const regexPattern = pattern
                .replace(/\./g, '\\.')
                .replace(/\*/g, '.*')
                .replace(/\?/g, '\\?');
            return new RegExp('^' + regexPattern + '$', 'i').test(currentUrl);
        } catch (e) {
            return false;
        }
    }

    function isScriptApplicable(script) {
        if (!script.patterns || script.patterns.length === 0) return true;
        return script.patterns.some(pattern => matchPattern(pattern));
    }

    function loadConfigFromUrl(url) {
        return new Promise((resolve, reject) => {
            if (!url) return reject(new Error('URL is empty'));
            if (typeof GM_xmlhttpRequest !== 'undefined') {
                GM_xmlhttpRequest({
                    method: 'GET',
                    url: url,
                    headers: { 'Cache-Control': 'no-cache' },
                    onload: (response) => {
                        try {
                            if (response.status === 200) {
                                resolve(JSON.parse(response.responseText));
                            } else {
                                reject(new Error(`HTTP ${response.status}`));
                            }
                        } catch (e) {
                            reject(new Error('JSON parse error'));
                        }
                    },
                    onerror: reject
                });
            } else {
                fetch(url, { cache: 'no-store' }).then(r => r.json()).then(resolve).catch(reject);
            }
        });
    }

    // ============================================
    // 开放给外部脚本的 API
    // ============================================

    const extAPI = {
        actions: [],
        onUpdateUI: null,

        registerAction: function(scriptId, actionId, name, callback) {
            const existing = this.actions.find(a => a.actionId === actionId && a.scriptId === scriptId);
            if (existing) {
                existing.name = name;
                existing.callback = callback;
            } else {
                this.actions.push({ scriptId, actionId, name, callback });
            }
            if (this.onUpdateUI) this.onUpdateUI();
        },

        unregisterAction: function(scriptId, actionId) {
            this.actions = this.actions.filter(a => !(a.scriptId === scriptId && a.actionId === actionId));
            if (this.onUpdateUI) this.onUpdateUI();
        },

        clearScriptActions: function(scriptId) {
            this.actions = this.actions.filter(a => a.scriptId !== scriptId);
            if (this.onUpdateUI) this.onUpdateUI();
        }
    };

    unsafeWindow.FloatingBallAPI = extAPI;

    // ============================================
    // 配置管理
    // ============================================

    let SCRIPTS_CONFIG_URL = safeGetValue('config_url', '');

    function promptConfigUrl() {
        const newUrl = prompt('请输入脚本配置列表的 JSON URL:', SCRIPTS_CONFIG_URL);
        if (newUrl !== null) {
            safeSetValue('config_url', newUrl.trim());
            SCRIPTS_CONFIG_URL = newUrl.trim();
            location.reload();
        }
    }

    if (typeof GM_registerMenuCommand !== 'undefined') {
        GM_registerMenuCommand('⚙️ 设置/修改 JSON 列表源', promptConfigUrl);
    }

    // ============================================
    // 脚本加载器
    // ============================================

    class ScriptLoader {
        constructor() {
            this.loadedScripts = new Map();
            this.loadingScripts = new Set();
            this.scriptStates = safeGetValue('scriptStates', {});
        }

        isScriptEnabled(script) {
            const storedState = this.scriptStates[script.id];
            if (storedState !== undefined && storedState !== null) {
                return storedState === true;
            }
            return script.enabled === true;
        }

        async loadScript(script) {
            if (this.loadedScripts.has(script.id)) return;
            if (this.loadingScripts.has(script.id)) return;
            if (!script.url) return;

            this.loadingScripts.add(script.id);
            try {
                await this.injectScript(script);
                this.loadedScripts.set(script.id, true);
            } catch (e) {
                console.error(`❌ Failed: ${script.name}`, e);
            } finally {
                this.loadingScripts.delete(script.id);
            }
        }

        injectScript(script) {
            return new Promise((resolve, reject) => {
                try {
                    if (typeof GM_addElement !== 'undefined') {
                        const el = GM_addElement('script', {
                            src: script.url,
                            type: 'text/javascript',
                            'data-script-id': script.id
                        });
                        el.onload = resolve;
                        el.onerror = () => reject(new Error('Load error'));
                    } else {
                        const el = document.createElement('script');
                        el.src = script.url;
                        el.type = 'text/javascript';
                        el.setAttribute('data-script-id', script.id);
                        el.onload = resolve;
                        el.onerror = () => reject(new Error('Load error'));
                        (document.body || document.head).appendChild(el);
                    }
                } catch (e) {
                    reject(e);
                }
            });
        }

        unloadScript(scriptId) {
            this.scriptStates[scriptId] = false;
            safeSetValue('scriptStates', this.scriptStates);
            const el = document.querySelector(`script[data-script-id="${scriptId}"]`);
            if (el) el.remove();
            this.loadedScripts.delete(scriptId);
            unsafeWindow.FloatingBallAPI.clearScriptActions(scriptId);
        }

        setScriptState(scriptId, enabled) {
            this.scriptStates[scriptId] = enabled;
            safeSetValue('scriptStates', this.scriptStates);
        }
    }

    // ============================================
    // 悬浮球 UI
    // ============================================

    class FloatingBall {
        constructor(scriptLoader, scripts) {
            this.scriptLoader = scriptLoader;
            this.scripts = scripts || [];
            this.isDragging = false;
            this.isMenuOpen = false;
            this.dragStartPos = { x: 0, y: 0 };
            this.position = this.loadPosition();

            unsafeWindow.FloatingBallAPI.onUpdateUI = () => {
                if (this.isMenuOpen) this.updateMenu();
            };

            this.init();
        }

        loadPosition() {
            const saved = safeGetValue('ballPosition', null);
            if (saved && saved.x >= 0 && saved.x <= window.innerWidth - 40 &&
                saved.y >= 0 && saved.y <= window.innerHeight - 40) {
                return saved;
            }
            return { x: window.innerWidth - 60, y: 100 };
        }

        clampPosition(x, y) {
            return {
                x: Math.max(5, Math.min(x, window.innerWidth - 45)),
                y: Math.max(5, Math.min(y, window.innerHeight - 45))
            };
        }

        savePosition() {
            safeSetValue('ballPosition', this.position);
        }

        init() {
            this.addStyles();
            this.createBall();
            this.createMenu();
            this.bindEvents();
            this.autoLoadScripts();
        }

        autoLoadScripts() {
            const checkBody = () => {
                if (document.body) {
                    this.scripts
                        .filter(s => isScriptApplicable(s))
                        .sort((a, b) => a.order - b.order)
                        .forEach(script => {
                            if (this.scriptLoader.isScriptEnabled(script)) {
                                this.scriptLoader.loadScript(script);
                            }
                        });
                } else {
                    requestAnimationFrame(checkBody);
                }
            };
            checkBody();
        }

        addStyles() {
            const css = `
                .fb-container { position: fixed; z-index: 2147483647; font-family: system-ui, sans-serif; user-select: none; }
                .fb-ball { 
                    width: 40px; height: 40px; border-radius: 50%; 
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    box-shadow: 0 4px 15px rgba(0,0,0,0.3); cursor: pointer; 
                    display: flex; align-items: center; justify-content: center; 
                    opacity: 0.7; transition: opacity 0.2s, transform 0.2s;
                    touch-action: none; /* 关键：防止移动端拖动时页面滚动 */
                }
                .fb-ball:hover { opacity: 0.95; transform: scale(1.05); }
                .fb-ball.dragging { opacity: 0.5; transform: scale(1.1); cursor: grabbing; }
                .fb-ball-icon { width: 20px; height: 20px; fill: white; pointer-events: none; }

                .fb-menu { position: fixed; z-index: 2147483647; background: rgba(30,30,40,0.98); border-radius: 12px; box-shadow: 0 10px 40px rgba(0,0,0,0.5); padding: 0; min-width: 280px; max-width: 320px; max-height: 70vh; overflow-y: auto; opacity: 0; visibility: hidden; transform: scale(0.9); transition: all 0.2s; border: 1px solid rgba(255,255,255,0.1); display: flex; flex-direction: column; }
                .fb-menu.open { opacity: 1; visibility: visible; transform: scale(1); }

                .fb-section-title { font-size: 11px; color: #888; text-transform: uppercase; margin: 12px 16px 4px; font-weight: bold; letter-spacing: 1px; }

                .fb-actions-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; padding: 8px 16px; }
                .fb-action-btn { background: rgba(102, 126, 234, 0.2); border: 1px solid rgba(102, 126, 234, 0.4); color: #e0e0e0; padding: 8px; border-radius: 6px; font-size: 12px; cursor: pointer; transition: all 0.15s; text-align: center; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
                .fb-action-btn:hover { background: rgba(102, 126, 234, 0.4); color: #fff; transform: translateY(-1px); }
                .fb-empty-hint { font-size: 12px; color: #666; padding: 8px 16px; font-style: italic; }

                .fb-menu-item { padding: 10px 16px; display: flex; align-items: center; cursor: pointer; color: #e0e0e0; transition: background 0.15s; }
                .fb-menu-item:hover { background: rgba(255,255,255,0.05); }
                .fb-script-info { flex: 1; min-width: 0; }
                .fb-script-name { font-size: 13px; font-weight: 500; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }

                .fb-toggle { position: relative; width: 40px; height: 22px; background: rgba(255,255,255,0.2); border-radius: 12px; transition: background 0.2s; flex-shrink: 0; margin-left: 10px; cursor: pointer; }
                .fb-toggle.on { background: #667eea; }
                .fb-toggle-knob { position: absolute; top: 2px; left: 2px; width: 18px; height: 18px; background: white; border-radius: 50%; transition: transform 0.2s; box-shadow: 0 2px 4px rgba(0,0,0,0.2); }
                .fb-toggle.on .fb-toggle-knob { transform: translateX(18px); }

                .fb-menu-footer { padding: 12px 16px; border-top: 1px solid rgba(255,255,255,0.1); display: flex; gap: 8px; flex-shrink: 0; }
                .fb-btn { flex: 1; padding: 8px 12px; border: none; border-radius: 6px; font-size: 12px; cursor: pointer; transition: all 0.15s; font-weight: 500; }
                .fb-btn-primary { background: #667eea; color: white; }
                .fb-btn-primary:hover { background: #5a6fd6; }
                .fb-btn-secondary { background: rgba(255,255,255,0.1); color: #ccc; }
                .fb-btn-secondary:hover { background: rgba(255,255,255,0.15); }

                .fb-status { width: 8px; height: 8px; border-radius: 50%; margin-right: 10px; flex-shrink: 0; }
                .fb-status.loaded { background: #4ade80; box-shadow: 0 0 6px #4ade80; }
                .fb-status.unloaded { background: #6b7280; }
                .fb-status.loading { background: #fbbf24; animation: fb-pulse 1s infinite; }
                @keyframes fb-pulse { 0%,100%{opacity:1} 50%{opacity:0.5} }
                .fb-menu::-webkit-scrollbar { width: 6px; }
                .fb-menu::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.2); border-radius: 3px; }
            `;

            if (typeof GM_addStyle !== 'undefined') {
                GM_addStyle(css);
            } else {
                const style = document.createElement('style');
                style.textContent = css;
                (document.head || document.documentElement).appendChild(style);
            }
        }

        createBall() {
            const checkBody = () => {
                if (document.body) {
                    this.container = document.createElement('div');
                    this.container.className = 'fb-container';
                    this.container.style.left = this.position.x + 'px';
                    this.container.style.top = this.position.y + 'px';

                    this.ball = document.createElement('div');
                    this.ball.className = 'fb-ball';
                    this.ball.innerHTML = '<svg class="fb-ball-icon" viewBox="0 0 24 24"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z"/></svg>';

                    this.container.appendChild(this.ball);
                    document.body.appendChild(this.container);
                } else {
                    requestAnimationFrame(checkBody);
                }
            };
            checkBody();
        }

        createMenu() {
            const checkBody = () => {
                if (document.body) {
                    this.menu = document.createElement('div');
                    this.menu.className = 'fb-menu';
                    document.body.appendChild(this.menu);
                } else {
                    requestAnimationFrame(checkBody);
                }
            };
            checkBody();
        }

        updateMenu() {
            const applicable = this.scripts.filter(isScriptApplicable).sort((a, b) => a.order - b.order);
            const customActions = unsafeWindow.FloatingBallAPI.actions;

            let html = ``;

            // 1. 扩展功能区域
            html += `<div class="fb-section-title">扩展功能</div>`;
            if (customActions.length > 0) {
                html += `<div class="fb-actions-grid">`;
                customActions.forEach((action, index) => {
                    html += `<button class="fb-action-btn" data-index="${index}" title="${action.name}">${action.name}</button>`;
                });
                html += `</div>`;
            } else {
                html += `<div class="fb-empty-hint">暂无可用扩展</div>`;
            }

            // 2. 脚本管理区域
            html += `<div class="fb-section-title">模块开关</div>`;
            if (SCRIPTS_CONFIG_URL) {
                applicable.forEach(script => {
                    const isOn = this.scriptLoader.isScriptEnabled(script);
                    const isLoaded = this.scriptLoader.loadedScripts.has(script.id);
                    const isLoading = this.scriptLoader.loadingScripts.has(script.id);
                    const statusClass = isLoaded ? 'loaded' : (isLoading ? 'loading' : 'unloaded');

                    html += `
                        <div class="fb-menu-item" data-id="${script.id}">
                            <div class="fb-status ${statusClass}"></div>
                            <div class="fb-script-info">
                                <div class="fb-script-name">${script.name}</div>
                            </div>
                            <div class="fb-toggle ${isOn ? 'on' : ''}" data-id="${script.id}">
                                <div class="fb-toggle-knob"></div>
                            </div>
                        </div>
                    `;
                });
            } else {
                html += `<div class="fb-empty-hint" style="color:#fbbf24;cursor:pointer" id="fb-set-url-hint">⚠️ 点击设置 JSON 列表</div>`;
            }

            // 3. 底栏
            html += `
                <div class="fb-menu-footer">
                    <button class="fb-btn fb-btn-secondary" id="fb-set-url">设置源</button>
                    <button class="fb-btn fb-btn-primary" id="fb-refresh">刷新</button>
                </div>
            `;

            if (this.menu) {
                this.menu.innerHTML = html;
                this.bindMenuEvents();
            }
        }

        bindEvents() {
            const waitForElements = () => {
                if (!this.ball || !this.menu) {
                    requestAnimationFrame(waitForElements);
                    return;
                }

                this.ball.addEventListener('mousedown', e => {
                    this.dragStartPos = { x: e.clientX, y: e.clientY };
                    this.isDragging = false;
                });

                this.ball.addEventListener('touchstart', e => {
                    const t = e.touches[0];
                    this.dragStartPos = { x: t.clientX, y: t.clientY };
                    this.isDragging = false;
                }, { passive: true });

                const onMove = (cx, cy) => {
                    if (this.dragStartPos.x === 0) return;
                    const dx = cx - this.dragStartPos.x;
                    const dy = cy - this.dragStartPos.y;
                    if (Math.sqrt(dx*dx + dy*dy) > 5) {
                        if (!this.isDragging) {
                            this.isDragging = true;
                            this.ball.classList.add('dragging');
                        }
                        const newPos = this.clampPosition(this.position.x + dx, this.position.y + dy);
                        this.container.style.left = newPos.x + 'px';
                        this.container.style.top = newPos.y + 'px';
                        this.dragStartPos = { x: cx, y: cy };
                        this.position = newPos;
                    }
                };

                document.addEventListener('mousemove', e => onMove(e.clientX, e.clientY));
                document.addEventListener('touchmove', e => {
                    if (this.isDragging) e.preventDefault(); // 拖动球时禁止页面滚动
                    onMove(e.touches[0].clientX, e.touches[0].clientY);
                }, { passive: false });

                const onEnd = () => {
                    if (this.isDragging) {
                        this.ball.classList.remove('dragging');
                        this.savePosition();
                    }
                    this.dragStartPos = { x: 0, y: 0 };
                };
                document.addEventListener('mouseup', onEnd);
                document.addEventListener('touchend', onEnd);

                this.ball.addEventListener('click', () => {
                    if (!this.isDragging) this.toggleMenu();
                    this.isDragging = false;
                });

                document.addEventListener('click', e => {
                    if (this.isMenuOpen && !this.container.contains(e.target) && !this.menu.contains(e.target)) {
                        this.closeMenu();
                    }
                });

                window.addEventListener('resize', () => {
                    const c = this.clampPosition(this.position.x, this.position.y);
                    this.position = c;
                    this.container.style.left = c.x + 'px';
                    this.container.style.top = c.y + 'px';
                    if (this.isMenuOpen) this.positionMenu();
                });
            };

            waitForElements();
        }

        bindMenuEvents() {
            this.menu.querySelectorAll('.fb-action-btn').forEach(btn => {
                btn.addEventListener('click', (e) => {
                    e.stopPropagation();
                    const index = parseInt(btn.dataset.index);
                    const action = unsafeWindow.FloatingBallAPI.actions[index];
                    if (action && typeof action.callback === 'function') {
                        action.callback();
                    }
                    this.closeMenu();
                });
            });

            this.menu.querySelectorAll('.fb-toggle').forEach(toggle => {
                toggle.addEventListener('click', async e => {
                    e.stopPropagation();
                    const id = toggle.dataset.id;
                    const script = this.scripts.find(s => s.id === id);
                    if (!script) return;

                    const isOn = toggle.classList.contains('on');
                    toggle.classList.toggle('on');
                    this.scriptLoader.setScriptState(id, !isOn);

                    if (!isOn) {
                        await this.scriptLoader.loadScript(script);
                    } else {
                        this.scriptLoader.unloadScript(id);
                    }
                    this.updateMenu();
                });
            });

            this.menu.querySelector('#fb-set-url')?.addEventListener('click', () => { promptConfigUrl(); });
            this.menu.querySelector('#fb-set-url-hint')?.addEventListener('click', () => { promptConfigUrl(); });
            this.menu.querySelector('#fb-refresh')?.addEventListener('click', () => location.reload());
        }

        toggleMenu() {
            this.isMenuOpen = !this.isMenuOpen;
            this.menu.classList.toggle('open', this.isMenuOpen);
            if (this.isMenuOpen) {
                this.positionMenu();
                this.updateMenu();
            }
        }

        closeMenu() {
            this.isMenuOpen = false;
            this.menu.classList.remove('open');
        }

        positionMenu() {
            const mw = 280, mh = Math.min(this.menu.offsetHeight || 300, window.innerHeight * 0.7);
            let x = this.position.x + 50, y = this.position.y;
            if (x + mw > window.innerWidth) x = this.position.x - mw - 10;
            if (y + mh > window.innerHeight) y = window.innerHeight - mh - 10;
            y = Math.max(10, y);
            this.menu.style.left = x + 'px';
            this.menu.style.top = y + 'px';
        }
    }

    async function init() {
        const scriptLoader = new ScriptLoader();
        let scripts = [];

        if (SCRIPTS_CONFIG_URL) {
            try {
                const config = await loadConfigFromUrl(SCRIPTS_CONFIG_URL);
                scripts = Array.isArray(config) ? config : (config.scripts || []);
            } catch (e) {
                console.warn('⚠️ JSON 配置加载失败');
            }
        }

        new FloatingBall(scriptLoader, scripts);

        if (typeof GM_registerMenuCommand !== 'undefined') {
            GM_registerMenuCommand('✨ 打开菜单', () => document.querySelector('.fb-ball')?.click());
        }
    }

    init();
})();