package main

import (
	"fmt"
	"log"
	"net/http"
	"sync"
	"sync/atomic"
	"time" // 新增导入

	"github.com/gorilla/websocket"
)

var upgrader = websocket.Upgrader{
	CheckOrigin: func(r *http.Request) bool { return true },
}

// 消息结构
type Message struct {
	Type    string                 `json:"type"`
	Model   string                 `json:"model,omitempty"`
	Source  string                 `json:"source,omitempty"`
	Command string                 `json:"command,omitempty"`
	Params  map[string]interface{} `json:"params,omitempty"`
	Content interface{}            `json:"content,omitempty"`
	BID     string                 `json:"bid,omitempty"` // Browser ID
}

type Client struct {
	conn    *websocket.Conn
	role    string  // "browser" 或 "client"
	model   string  // 模型名称
	id      string  // 唯一会话ID
	partner *Client // 配对对象
	mu      sync.Mutex
	done    chan struct{} // 用于停止心跳的通道
}

func (c *Client) WriteJSON(msg Message) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.conn.WriteJSON(msg)
}

// Hub 管理中心
type Hub struct {
	sync.RWMutex
	browsers  map[string]*Client // 所有连接的浏览器 BID -> Client
	idCounter int64
}

var hub = &Hub{
	browsers: make(map[string]*Client),
}

// 生成唯一 ID
func (h *Hub) nextID(prefix string) string {
	return fmt.Sprintf("%s_%d", prefix, atomic.AddInt64(&h.idCounter, 1))
}

// 配对逻辑
func (h *Hub) TryPair(pythonClient *Client, targetModel string) (bool, *Client) {
	h.Lock()
	defer h.Unlock()

	for _, b := range h.browsers {
		// 规则：角色是浏览器 && 模型匹配（如果指定了的话） && 当前没有配对对象
		modelMatch := targetModel == "" || b.model == targetModel
		if b.partner == nil && modelMatch {
			pythonClient.partner = b
			b.partner = pythonClient
			return true, b
		}
	}
	return false, nil
}

// 释放配对（Client 断开，Browser 回归池子）
func (h *Hub) ReleasePair(c *Client) {
	if c.partner != nil {
		p := c.partner
		log.Printf("🔓 解除配对: [Client %s] <-> [Browser %s]", c.id, p.id)

		// 通知对方
		p.WriteJSON(Message{Type: "system", Content: "partner_disconnected"})

		// 双向解绑
		p.partner = nil
		c.partner = nil
	}
}

func wsHandler(role string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		conn, err := upgrader.Upgrade(w, r, nil)
		if err != nil {
			return
		}

		c := &Client{
			conn: conn,
			role: role,
			id:   hub.nextID(role),
			done: make(chan struct{}),
		}

		log.Printf("➕ [%s] 建立连接, ID: %s", role, c.id)

		if role == "browser" {
			hub.Lock()
			hub.browsers[c.id] = c
			hub.Unlock()
		}

		// 启动心跳 goroutine (每秒发送 ping)
		go func() {
			ticker := time.NewTicker(1 * time.Second)
			defer ticker.Stop()
			for {
				select {
				case <-ticker.C:
					c.mu.Lock()
					err := c.conn.WriteMessage(websocket.PingMessage, []byte{})
					c.mu.Unlock()
					if err != nil {
						// 连接已关闭，退出 goroutine
						return
					}
				case <-c.done:
					return
				}
			}
		}()

		defer func() {
			close(c.done) // 通知心跳 goroutine 退出
			hub.Lock()
			hub.ReleasePair(c)
			if role == "browser" {
				delete(hub.browsers, c.id)
			}
			hub.Unlock()
			conn.Close()
			log.Printf("➖ [%s] 断开连接, ID: %s", role, c.id)
		}()

		for {
			var msg Message
			if err := conn.ReadJSON(&msg); err != nil {
				break
			}

			// 处理业务逻辑
			switch msg.Type {
			case "register": // Browser 注册模型
				if role == "browser" {
					c.model = msg.Model
					log.Printf("🌐 [Browser %s] 注册模型: %s", c.id, c.model)
				}

			case "pair_request": // Client 请求配对
				if role == "client" {
					log.Printf("🔍 [Client %s] 尝试配对模型: %s", c.id, msg.Model)
					success, targetBrowser := hub.TryPair(c, msg.Model)
					if success {
						log.Printf("🔗 [Client %s] 成功锁定 [Browser %s](%s)", c.id, targetBrowser.id, targetBrowser.model)
						c.WriteJSON(Message{
							Type:    "pair_result",
							Content: true,
							Model:   targetBrowser.model,
							BID:     targetBrowser.id,
						})
					} else {
						c.WriteJSON(Message{Type: "pair_result", Content: false})
					}
				}

			default: // 路由转发
				if c.partner != nil {
					msg.Source = c.id
					c.partner.WriteJSON(msg)
				}
			}
		}
	}
}

func main() {
	http.HandleFunc("/ws/browser", wsHandler("browser"))
	http.HandleFunc("/ws/client", wsHandler("client"))
	fmt.Println("🚀 Bridge Server 运行在 127.26.3.1:8080")
	log.Fatal(http.ListenAndServe("127.26.3.1:8080", nil))
}