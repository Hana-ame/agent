// ==UserScript==
// @name         悬浮球脚本加载器 (移动端滑动优化版)
// @namespace    http://tampermonkey.net/
// @version      2.3.0
// @description  彻底修复移动端拖拽后页面无法滑动的 Bug
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
    // Trusted Types 兼容处理
    // ============================================
    let ttPolicy;
    if (window.trustedTypes && window.trustedTypes.createPolicy) {
        ttPolicy = window.trustedTypes.createPolicy('fb-policy', { createHTML: (s) => s });
    }
    function setSafeInnerHTML(el, html) {
        if (ttPolicy) el.innerHTML = ttPolicy.createHTML(html);
        else el.innerHTML = html;
    }

    // ============================================
    // 工具函数
    // ============================================
    const safeGetValue = (k, d) => {
        try { return (typeof GM_getValue !== 'undefined') ? GM_getValue(k, d) : JSON.parse(localStorage.getItem('fb_'+k)) || d; }
        catch(e) { return d; }
    };
    const safeSetValue = (k, v) => {
        try { if (typeof GM_setValue !== 'undefined') GM_setValue(k, v); localStorage.setItem('fb_'+k, JSON.stringify(v)); }
        catch(e) {}
    };

    const matchPattern = (p) => {
        if (!p || p === "*://*/*") return true;
        return new RegExp('^' + p.replace(/\./g, '\\.').replace(/\*/g, '.*') + '$', 'i').test(window.location.href);
    };

    // ============================================
    // API 接口
    // ============================================
    const extAPI = {
        actions: [],
        onUpdateUI: null,
        registerAction(scriptId, actionId, name, callback) {
            const act = { scriptId, actionId, name, callback };
            const idx = this.actions.findIndex(a => a.actionId === actionId && a.scriptId === scriptId);
            if (idx > -1) this.actions[idx] = act; else this.actions.push(act);
            if (this.onUpdateUI) this.onUpdateUI();
        },
        clearScriptActions(scriptId) {
            this.actions = this.actions.filter(a => a.scriptId !== scriptId);
            if (this.onUpdateUI) this.onUpdateUI();
        }
    };
    unsafeWindow.FloatingBallAPI = extAPI;

    // ============================================
    // 加载管理
    // ============================================
    class ScriptLoader {
        constructor() {
            this.scriptStates = safeGetValue('states', {});
            this.loaded = new Set();
        }
        isEn(s) { return this.scriptStates[s.id] !== undefined ? this.scriptStates[s.id] === true : s.enabled === true; }
        async load(s) {
            if (this.loaded.has(s.id)) return;
            const el = document.createElement('script');
            el.src = s.url; el.type = 'text/javascript'; el.setAttribute('data-sid', s.id);
            (document.head || document.documentElement).appendChild(el);
            this.loaded.add(s.id);
        }
    }

    // ============================================
    // 悬浮球 UI
    // ============================================
    class FloatingBall {
        constructor(loader, scripts) {
            this.loader = loader;
            this.scripts = scripts || [];
            this.pos = safeGetValue('pos', { x: window.innerWidth - 60, y: 100 });
            this.isDragging = false;
            this.isMenuOpen = false;
            unsafeWindow.FloatingBallAPI.onUpdateUI = () => this.isMenuOpen && this.updateMenu();
            this.init();
        }

        init() {
            this.addStyles();
            this.createBall();
            this.createMenu();
            this.bindEvents();
            this.autoLoad();
        }

        autoLoad() {
            const check = () => {
                if (document.body) {
                    this.scripts.forEach(s => { if(matchPattern(s.patterns?.[0]) && this.loader.isEn(s)) this.loader.load(s); });
                } else requestAnimationFrame(check);
            };
            check();
        }

        addStyles() {
            const css = `
                .fb-container { position: fixed; z-index: 2147483647; touch-action: none; user-select: none; }
                .fb-ball {
                    width: 40px !important; height: 40px !important; border-radius: 50%;
                    background: linear-gradient(135deg, #667eea, #764ba2);
                    box-shadow: 0 4px 12px rgba(0,0,0,0.3); cursor: pointer;
                    display: flex; align-items: center; justify-content: center;
                    opacity: 0.8; touch-action: none;
                }
                .fb-ball-icon { width: 20px; height: 20px; fill: white; pointer-events: none; }
                .fb-menu {
                    position: fixed; z-index: 2147483647; background: #1e1e28;
                    border-radius: 12px; width: 260px; max-height: 70vh; overflow-y: auto;
                    display: none; border: 1px solid #333; box-shadow: 0 8px 32px rgba(0,0,0,0.5); color: white;
                }
                .fb-menu.open { display: block; }
                .fb-section-title { font-size: 11px; color: #777; padding: 12px 16px 4px; text-transform: uppercase; }
                .fb-actions-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; padding: 8px 16px; }
                .fb-action-btn { background: #333; color: #eee; border: 1px solid #444; padding: 8px; border-radius: 6px; font-size: 12px; cursor: pointer; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;}
                .fb-menu-item { padding: 10px 16px; display: flex; align-items: center; font-size: 13px; }
                .fb-toggle { width: 34px; height: 18px; background: #444; border-radius: 9px; position: relative; margin-left: auto; cursor: pointer; transition: 0.2s; }
                .fb-toggle.on { background: #667eea; }
                .fb-toggle-knob { width: 14px; height: 14px; background: white; border-radius: 50%; position: absolute; top: 2px; left: 2px; transition: 0.2s; }
                .fb-toggle.on .fb-toggle-knob { transform: translateX(16px); }
                .fb-footer { padding: 12px; display: flex; gap: 8px; border-top: 1px solid #333; }
                .fb-btn { flex: 1; padding: 8px; border: none; border-radius: 4px; cursor: pointer; font-size: 12px; background: #444; color: white; }
            `;
            if (typeof GM_addStyle !== 'undefined') GM_addStyle(css);
            else { const s = document.createElement('style'); s.textContent = css; document.head.appendChild(s); }
        }

        createBall() {
            this.container = document.createElement('div');
            this.container.className = 'fb-container';
            this.container.style.left = this.pos.x + 'px';
            this.container.style.top = this.pos.y + 'px';
            this.ball = document.createElement('div');
            this.ball.className = 'fb-ball';
            setSafeInnerHTML(this.ball, '<svg class="fb-ball-icon" viewBox="0 0 24 24"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z"/></svg>');
            this.container.appendChild(this.ball);
            document.body.appendChild(this.container);
        }

        createMenu() {
            this.menu = document.createElement('div');
            this.menu.className = 'fb-menu';
            document.body.appendChild(this.menu);
        }

        updateMenu() {
            const acts = unsafeWindow.FloatingBallAPI.actions;
            let html = `<div class="fb-section-title">扩展功能</div><div class="fb-actions-grid">`;
            acts.forEach((a, i) => { html += `<button class="fb-action-btn" data-idx="${i}">${a.name}</button>`; });
            html += `</div><div class="fb-section-title">脚本开关</div>`;
            this.scripts.forEach(s => {
                const on = this.loader.isEn(s);
                html += `<div class="fb-menu-item"><span>${s.name}</span><div class="fb-toggle ${on?'on':''}" data-id="${s.id}"><div class="fb-toggle-knob"></div></div></div>`;
            });
            html += `<div class="fb-footer"><button class="fb-btn" id="fb-set-url">配置源</button><button class="fb-btn" id="fb-refresh">刷新</button></div>`;
            setSafeInnerHTML(this.menu, html);
            this.bindMenuEvents();
        }

        bindEvents() {
            let startX, startY, ballX, ballY, raf;

            const onMove = (e) => {
                if (e.cancelable) e.preventDefault(); // 关键：只有正在拖拽球时才阻止滚动
                const t = e.touches ? e.touches[0] : e;
                const dx = t.clientX - startX;
                const dy = t.clientY - startY;

                if (!this.isDragging && Math.sqrt(dx*dx + dy*dy) > 5) {
                    this.isDragging = true;
                }

                if (this.isDragging) {
                    if (raf) cancelAnimationFrame(raf);
                    raf = requestAnimationFrame(() => {
                        let nx = ballX + dx;
                        let ny = ballY + dy;
                        nx = Math.max(0, Math.min(nx, window.innerWidth - 40));
                        ny = Math.max(0, Math.min(ny, window.innerHeight - 40));
                        this.container.style.left = nx + 'px';
                        this.container.style.top = ny + 'px';
                        this.pos = { x: nx, y: ny };
                    });
                }
            };

            const onEnd = () => {
                if (this.isDragging) safeSetValue('pos', this.pos);
                // 彻底解除监听，释放页面滚动
                document.removeEventListener('mousemove', onMove);
                document.removeEventListener('touchmove', onMove);
                document.removeEventListener('mouseup', onEnd);
                document.removeEventListener('touchend', onEnd);
                document.removeEventListener('touchcancel', onEnd);
            };

            const onStart = (e) => {
                const t = e.touches ? e.touches[0] : e;
                startX = t.clientX; startY = t.clientY;
                ballX = this.pos.x; ballY = this.pos.y;
                this.isDragging = false;

                // 只有按下时才动态绑定 move 和 end，减少对页面的干预
                document.addEventListener('mousemove', onMove, { passive: false });
                document.addEventListener('touchmove', onMove, { passive: false });
                document.addEventListener('mouseup', onEnd);
                document.addEventListener('touchend', onEnd);
                document.addEventListener('touchcancel', onEnd);
            };

            this.ball.addEventListener('mousedown', onStart);
            this.ball.addEventListener('touchstart', onStart, { passive: true });

            this.ball.addEventListener('click', () => {
                if (!this.isDragging) {
                    this.isMenuOpen = !this.isMenuOpen;
                    this.menu.classList.toggle('open', this.isMenuOpen);
                    if (this.isMenuOpen) { this.updateMenu(); this.positionMenu(); }
                }
            });

            document.addEventListener('mousedown', (e) => {
                if (this.isMenuOpen && !this.menu.contains(e.target) && !this.ball.contains(e.target)) {
                    this.isMenuOpen = false;
                    this.menu.classList.remove('open');
                }
            });
        }

        positionMenu() {
            let x = this.pos.x + 45;
            let y = this.pos.y;
            if (x + 260 > window.innerWidth) x = this.pos.x - 265;
            if (y + this.menu.offsetHeight > window.innerHeight) y = window.innerHeight - this.menu.offsetHeight - 10;
            this.menu.style.left = Math.max(5, x) + 'px';
            this.menu.style.top = Math.max(5, y) + 'px';
        }

        bindMenuEvents() {
            this.menu.querySelectorAll('.fb-action-btn').forEach(btn => {
                btn.onclick = () => {
                    const a = unsafeWindow.FloatingBallAPI.actions[btn.dataset.idx];
                    a?.callback();
                    this.isMenuOpen = false;
                    this.menu.classList.remove('open');
                };
            });
            this.menu.querySelectorAll('.fb-toggle').forEach(t => {
                t.onclick = () => {
                    const id = t.dataset.id;
                    this.loader.scriptStates[id] = !t.classList.contains('on');
                    safeSetValue('states', this.loader.scriptStates);
                    location.reload();
                };
            });
            this.menu.querySelector('#fb-refresh').onclick = () => location.reload();
            this.menu.querySelector('#fb-set-url').onclick = () => {
                const url = prompt('JSON URL:', safeGetValue('config_url', ''));
                if (url) { safeSetValue('config_url', url); location.reload(); }
            };
        }
    }

    async function init() {
        const loader = new ScriptLoader();
        const url = safeGetValue('config_url', '');
        let scripts = [];
        if (url) {
            try { const res = await new Promise((res, rej) => {
                GM_xmlhttpRequest({ method: 'GET', url: url, onload: r => res(JSON.parse(r.responseText)), onerror: rej });
            }); scripts = res.scripts || res; } catch(e) {}
        }
        new FloatingBall(loader, scripts);
    }
    init();
})();