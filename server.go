package main

import (
	"fmt"
	"log"
	"net/http"
	"sync"

	"github.com/gorilla/websocket"
)

var upgrader = websocket.Upgrader{
	CheckOrigin: func(r *http.Request) bool { return true },
}

type Message struct {
	Type    string                 `json:"type"`
	Model   string                 `json:"model,omitempty"`
	Source  string                 `json:"source,omitempty"`
	Command string                 `json:"command,omitempty"`
	Params  map[string]interface{} `json:"params,omitempty"`
	Content interface{}            `json:"content,omitempty"`
}

type Client struct {
	conn    *websocket.Conn
	role    string // "browser" 或 "client"
	model   string // "deepseek", "qwen" 等
	partner *Client
	mu      sync.Mutex
}

func (c *Client) WriteJSON(msg Message) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.conn.WriteJSON(msg)
}

type Hub struct {
	sync.RWMutex
	clients map[*Client]bool
}

var hub = &Hub{clients: make(map[*Client]bool)}

func (h *Hub) PairClient(pythonClient *Client, targetModel string) (bool, string) {
	h.Lock()
	defer h.Unlock()

	// 查找是否有匹配且未被占用的浏览器
	for b := range h.clients {
		if b.role == "browser" {
			if b.model == targetModel {
				if b.partner == nil {
					pythonClient.partner = b
					b.partner = pythonClient
					return true, "Successfully paired"
				} else {
					return false, fmt.Sprintf("Target browser (%s) is already busy with another client", targetModel)
				}
			}
		}
	}

	// 如果没找到，打印当前所有可用的浏览器，方便调试
	var available []string
	for b := range h.clients {
		if b.role == "browser" && b.partner == nil {
			available = append(available, b.model)
		}
	}
	return false, fmt.Sprintf("No free browser found for model '%s'. Available idle browsers: %v", targetModel, available)
}

func (h *Hub) Unpair(c *Client) {
	if c.partner != nil {
		log.Printf("💔 [%s-%s] 由于一侧断开，解除配对", c.role, c.model)
		// 通知另一侧
		c.partner.WriteJSON(Message{Type: "system", Content: "partner_disconnected"})
		c.partner.partner = nil
		c.partner = nil
	}
}

func wsHandler(role string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		conn, err := upgrader.Upgrade(w, r, nil)
		if err != nil {
			log.Printf("❌ [%s] 升级协议失败: %v", role, err)
			return
		}

		c := &Client{conn: conn, role: role}
		hub.Lock()
		hub.clients[c] = true
		hub.Unlock()

		log.Printf("➕ [%s] 新连接接入", role)

		defer func() {
			hub.Lock()
			hub.Unpair(c)
			delete(hub.clients, c)
			hub.Unlock()
			conn.Close()
			log.Printf("➖ [%s] 连接已断开并从 Hub 移除", role)
		}()

		for {
			var msg Message
			if err := conn.ReadJSON(&msg); err != nil {
				log.Printf("ℹ️ [%s] 读取消息失败 (可能已断开): %v", role, err)
				break
			}

			// 1. 浏览器注册身份
			if role == "browser" && msg.Type == "register" {
				c.model = msg.Model
				log.Printf("🌐 [Browser] 注册模型身份为: %s", c.model)
				continue
			}

			// 2. Python 请求配对
			if role == "client" && msg.Type == "pair_request" {
				c.model = msg.Model
				log.Printf("🔍 [Client] 正在请求配对模型: %s", c.model)
				success, detail := hub.PairClient(c, c.model)
				if success {
					log.Printf("🔗 [Client] 配对成功: %s", c.model)
				} else {
					log.Printf("⚠️ [Client] 配对失败: %s", detail)
				}
				c.WriteJSON(Message{Type: "pair_result", Content: success, Params: map[string]interface{}{"detail": detail}})
				continue
			}

			// 3. 消息转发
			if c.partner != nil {
				msg.Source = role
				if err := c.partner.WriteJSON(msg); err != nil {
					log.Printf("❌ [%s] 转发消息到对方失败: %v", role, err)
				}
			} else {
				log.Printf("🚫 [%s] 发送了 %s，但当前未配对任何目标", role, msg.Type)
			}
		}
	}
}

func main() {
	http.HandleFunc("/ws/browser", wsHandler("browser"))
	http.HandleFunc("/ws/client", wsHandler("client"))
	fmt.Println("🚀 WebSocket Bridge Server started on ws://localhost:8765")
	log.Fatal(http.ListenAndServe(":8765", nil))
}

/*
在多个链接并发的时候并不好用。
browser池，应该保存一个ID，ID对每个ws/browser唯一，
配对请求消耗池子中的一个链接，当配对断开时，链接返回池子。
当browser断开连接时，这个消失，链接也断开。

需要表明自己是deepseek（或者今后可能的其他模型）
配对请求也是需要带有模型摸着不带的。配对成功后，需要告诉python client配对了个什么模型。


*/
