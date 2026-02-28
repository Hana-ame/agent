package main

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"sync"

	"github.com/gorilla/websocket"
)

var upgrader = websocket.Upgrader{
	CheckOrigin: func(r *http.Request) bool { return true }, // 允许跨域，方便油猴脚本连接
}

// 简单的并发安全管理器，存储当前连接的 browser 和 client
type Hub struct {
	sync.Mutex
	browser *websocket.Conn
	client  *websocket.Conn
}

var hub = &Hub{}

// 统一定义消息格式
type Message struct {
	Type    string                 `json:"type"`              // 消息类型: "token", "command", "match_result"
	Source  string                 `json:"source,omitempty"`  // 来源: "browser" 或 "client"
	Command string                 `json:"command,omitempty"` // 指令名称: "new_chat", "send_prompt", "match"
	Params  map[string]interface{} `json:"params,omitempty"`  // 指令参数
	Content interface{}            `json:"content,omitempty"` // 承载的数据 (如 token 字符串或 object)
}

func wsHandler(role string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		conn, err := upgrader.Upgrade(w, r, nil)
		if err != nil {
			log.Println("升级 WS 失败:", err)
			return
		}
		defer conn.Close()

		hub.Lock()
		switch role {
		case "browser":
			hub.browser = conn
			log.Println("🌐 浏览器端 (Tampermonkey) 已连接!")
		case "client":
			hub.client = conn
			log.Println("🐍 客户端 (Python) 已连接!")
		}
		hub.Unlock()

		// 循环读取消息并转发
		for {
			var msg Message
			err := conn.ReadJSON(&msg)
			if err != nil {
				log.Printf("[%s] 断开连接: %v\n", role, err)
				break
			}

			// 标记来源并进行路由转发
			msg.Source = role
			forwardMessage(msg, role)
		}
	}
}

func forwardMessage(msg Message, fromRole string) {
	hub.Lock()
	defer hub.Unlock()

	// 序列化消息以便转发
	data, err := json.Marshal(msg)
	if err != nil {
		return
	}

	if fromRole == "browser" && hub.client != nil {
		// 浏览器发来的数据 (如 token, 匹配结果) 转发给 Python
		hub.client.WriteMessage(websocket.TextMessage, data)
	} else if fromRole == "client" && hub.browser != nil {
		// Python 发来的指令 (如 input, match) 转发给浏览器
		hub.browser.WriteMessage(websocket.TextMessage, data)
	} else {
		fmt.Printf("⚠️ 收到来自 %s 的消息，但目标端未连接: %s\n", fromRole, string(data))
	}
}

func main() {
	http.HandleFunc("/ws/browser", wsHandler("browser"))
	http.HandleFunc("/ws/client", wsHandler("client"))

	fmt.Println("🚀 WebSocket Server 启动在 ws://localhost:8765")
	fmt.Println("浏览器连接端点: ws://localhost:8765/ws/browser")
	fmt.Println("Python 连接端点: ws://localhost:8765/ws/client")

	err := http.ListenAndServe(":8765", nil)
	if err != nil {
		log.Fatal("服务器启动失败: ", err)
	}
}
