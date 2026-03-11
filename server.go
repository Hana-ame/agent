// [START] PKG-MAIN
// version: 001
// 上下文：程序编译和运行的起点。先决调用：无。后续调用：IMPORT-DEPS。
// 输入参数：无
// 输出参数：无
package main

// [END] PKG-MAIN

// [START] IMPORT-DEPS
// version: 001
// 上下文：包加载阶段自动解析。先决调用：PKG-MAIN。后续调用：INIT-FUNC。
// 输入参数：无
// 输出参数：无
import (
	"fmt"
	"log"
	"math/rand"
	"net/http"
	"sync"
	"sync/atomic"
	"time"

	"github.com/gorilla/websocket"
)

// [END] IMPORT-DEPS

// [START] INIT-FUNC
// version: 001
// 上下文：程序 main 函数执行前由运行时自动调用。先决调用：IMPORT-DEPS。后续调用：MAIN-ENTRY。
// 输入参数：无
// 输出参数：无
func init() {
	rand.Seed(time.Now().UnixNano())
}

// [END] INIT-FUNC

// [START] WS-UPGRADE
// version: 001
// 上下文：全局变量装载阶段。先决调用：IMPORT-DEPS。后续调用：在 WS-CONN-INIT 中被实际调用。
// 输入参数：无
// 输出参数：无
// 预留扩展空间：CheckOrigin 函数内预留接入统一的跨域白名单校验逻辑。
var upgrader = websocket.Upgrader{
	CheckOrigin: func(r *http.Request) bool { return true },
}

// [END] WS-UPGRADE

// [START] STRUCT-MSG
// version: 001
// 上下文：在 WS-MSG-LOOP 读取网络字节流以及 METH-CLIENT-WRITE 写入字节流时作为反序列化/序列化的载体。先决调用：无。后续调用：无。
// 输入参数：无
// 输出参数：无
// 预留扩展空间：Ext 字段预留用于传递 TraceID、签名 Token 或指令附加元数据。
type Message struct {
	Type    string                 `json:"type"`
	Model   string                 `json:"model,omitempty"`
	Title   string                 `json:"title,omitempty"`
	Source  string                 `json:"source,omitempty"`
	Command string                 `json:"command,omitempty"`
	Params  map[string]interface{} `json:"params,omitempty"`
	Content interface{}            `json:"content,omitempty"`
	BID     string                 `json:"bid,omitempty"`
	Ext     map[string]interface{} `json:"ext,omitempty"`
}

// [END] STRUCT-MSG

// [START] STRUCT-CRITERIA
// version: 001
// 上下文：在 HNDL-PAIR 中根据 Message 参数构造，传递给 METH-HUB-PAIR 使用。先决调用：无。后续调用：无。
// 输入参数：无
// 输出参数：无
// 预留扩展空间：可增加 Tags[]string、VersionRange 等字段满足更复杂的筛选。
type PairCriteria struct {
	Model string
	Title string
	BID   string
}

// [END] STRUCT-CRITERIA

// [START] STRUCT-CLIENT
// version: 001
// 上下文：在 WS-CONN-INIT 阶段实例化。先决调用：无。后续调用：贯穿该连接的整个生命周期上下文。
// 输入参数：无
// 输出参数：无
// 预留扩展空间：Metadata 用于挂载请求级别的独立状态（如客户端系统版本、当前并发数）。
type Client struct {
	conn     *websocket.Conn
	role     string
	model    string
	title    string
	id       string
	partner  *Client
	mu       sync.Mutex
	done     chan struct{}
	Metadata map[string]string
}

// [END] STRUCT-CLIENT

// [START] METH-CLIENT-WRITE
// version: 001
// 上下文：任何需要向下游连接下发指令的场景（如 HNDL-LIST、HNDL-PAIR、METH-HUB-UNREG）。先决调用：WS-CONN-INIT 成功建立连接。后续调用：继续监听或触发对端响应。
// 输入参数：msg (Message)
// 输出参数：error
func (c *Client) WriteJSON(msg Message) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.conn.WriteJSON(msg)
}

// [END] METH-CLIENT-WRITE

// [START] STRUCT-HUB
// version: 001
// 上下文：全局单例，于 init 阶段装载。先决调用：无。后续调用：各 METH-HUB-* 方法。
// 输入参数：无
// 输出参数：无
type Hub struct {
	sync.RWMutex
	browsers  map[string]*Client
	idCounter int64
}

var hub = &Hub{
	browsers: make(map[string]*Client),
}

// [END] STRUCT-HUB

// [START] METH-HUB-ID
// version: 001
// 上下文：在 WS-CONN-INIT 中创建 Client 对象时调用。先决调用：STRUCT-HUB 实例化。后续调用：客户端 ID 赋值。
// 输入参数：prefix (string)
// 输出参数：唯一会话标识 (string)
func (h *Hub) nextID(prefix string) string {
	return fmt.Sprintf("%s_%d", prefix, atomic.AddInt64(&h.idCounter, 1))
}

// [END] METH-HUB-ID

// [START] METH-HUB-REG
// version: 001
// 上下文：当角色为 browser 且在 WS-REG-BROWSER 节点命中时调用。先决调用：WS-CONN-INIT 完成。后续调用：可被 METH-HUB-LIST / METH-HUB-PAIR 检索。
// 输入参数：c (*Client)
// 输出参数：无
func (h *Hub) RegisterBrowser(c *Client) {
	h.Lock()
	defer h.Unlock()
	h.browsers[c.id] = c
}

// [END] METH-HUB-REG

// [START] METH-HUB-LIST
// version: 001
// 上下文：在处理 client 的 list 消息 (HNDL-LIST) 时调用。先决调用：若干 browser 已执行 METH-HUB-REG。后续调用：METH-CLIENT-WRITE 返回数据。
// 输入参数：无
// 输出参数：空闲浏览器属性集合 ([]map[string]interface{})
// 预留扩展空间：可在此接口接入分页逻辑，或根据 Map 中附加的延时、权重值进行排序筛选后再输出。
func (h *Hub) GetAvailableBrowsers() []map[string]interface{} {
	h.RLock()
	defer h.RUnlock()

	var list []map[string]interface{}
	for _, b := range h.browsers {
		if b.partner == nil {
			list = append(list, map[string]interface{}{
				"id":    b.id,
				"model": b.model,
				"title": b.title,
			})
		}
	}
	return list
}

// [END] METH-HUB-LIST

// [START] METH-HUB-PAIR
// version: 001
// 上下文：在 HNDL-PAIR 解析完用户条件后调用。先决调用：METH-HUB-REG。后续调用：成功或失败均通过 METH-CLIENT-WRITE 反馈。
// 输入参数：pythonClient (*Client), criteria (PairCriteria)
// 输出参数：是否成功标识 (bool), 目标对象引用 (*Client)
func (h *Hub) TryPair(pythonClient *Client, criteria PairCriteria) (bool, *Client) {
	h.Lock()
	defer h.Unlock()

	var candidates []*Client

	if criteria.BID != "" {
		if b, ok := h.browsers[criteria.BID]; ok && b.partner == nil {
			candidates = append(candidates, b)
		}
	} else {
		for _, b := range h.browsers {
			if b.partner != nil {
				continue
			}
			modelMatch := criteria.Model == "" || b.model == criteria.Model
			titleMatch := criteria.Title == "" || b.title == criteria.Title

			if modelMatch && titleMatch {
				candidates = append(candidates, b)
			}
		}
	}

	if len(candidates) == 0 {
		return false, nil
	}

	selected := candidates[rand.Intn(len(candidates))]
	pythonClient.partner = selected
	selected.partner = pythonClient

	return true, selected
}

//[END] METH-HUB-PAIR

// [START] METH-HUB-UNREG
// version: 001
// 上下文：在 WS-TEARDOWN 断开连接的清理环节调用。先决调用：WS-CONN-INIT 曾成功执行。后续调用：套接字关闭清理完成。
// 输入参数：c (*Client)
// 输出参数：无
func (h *Hub) UnregisterAndRelease(c *Client) {
	h.Lock()
	defer h.Unlock()

	if c.partner != nil {
		p := c.partner
		log.Printf("🔓 解除配对:[Client %s] <->[Browser %s]", c.id, p.id)

		p.WriteJSON(Message{Type: "system", Content: "partner_disconnected"})
		p.partner = nil
		c.partner = nil
	}

	if c.role == "browser" {
		delete(h.browsers, c.id)
	}
}

// [END] METH-HUB-UNREG

// [START] ROUTER-MAP
// version: 001
// 上下文：被 WS-MSG-LOOP 在死循环中不断检索调用。先决调用：各 HNDL-* 函数已被系统加载。后续调用：命中即触发对应的 HNDL-* 函数。
// 输入参数：无
// 输出参数：无
type MessageHandler func(c *Client, msg Message)

var messageRouter = map[string]MessageHandler{
	"register":      handleRegister,
	"list_browsers": handleListBrowsers,
	"pair_request":  handlePairRequest,
}

// [END] ROUTER-MAP

// [START] HNDL-REG
// version: 001
// 上下文：当 WS-MSG-LOOP 收到 type 为 "register" 时被触发路由。先决调用：WS-CONN-INIT 及消息反序列化成功。后续调用：重新回到 WS-MSG-LOOP 阻塞等待。
// 输入参数：c (*Client), msg (Message)
// 输出参数：无
func handleRegister(c *Client, msg Message) {
	if c.role == "browser" {
		c.model = msg.Model
		c.title = msg.Title
		log.Printf("🌐[Browser %s] 注册模型: %s, 标题: %s", c.id, c.model, c.title)
	}
}

// [END] HNDL-REG

// [START] HNDL-LIST
// version: 001
// 上下文：当 WS-MSG-LOOP 收到 type 为 "list_browsers" 时被触发路由。先决调用：WS-CONN-INIT 完成。后续调用：METH-HUB-LIST 收集数据，而后 METH-CLIENT-WRITE 发送数据。
// 输入参数：c (*Client), msg (Message)
// 输出参数：无
func handleListBrowsers(c *Client, msg Message) {
	if c.role == "client" {
		c.WriteJSON(Message{
			Type:    "browser_list",
			Content: hub.GetAvailableBrowsers(),
		})
	}
}

// [END] HNDL-LIST

// [START] HNDL-PAIR
// version: 001
// 上下文：当 WS-MSG-LOOP 收到 type 为 "pair_request" 时被触发路由。先决调用：对端 browser 已完成 METH-HUB-REG。后续调用：转入 METH-HUB-PAIR 核心撮合机制，随后通过 METH-CLIENT-WRITE 输出交互。
// 输入参数：c (*Client), msg (Message)
// 输出参数：无
func handlePairRequest(c *Client, msg Message) {
	if c.role != "client" {
		return
	}

	criteria := PairCriteria{
		Model: msg.Model,
		Title: msg.Title,
		BID:   msg.BID,
	}

	log.Printf("🔍[Client %s] 尝试配对: model=%s, title=%s, bid=%s", c.id, criteria.Model, criteria.Title, criteria.BID)

	success, target := hub.TryPair(c, criteria)
	if success {
		log.Printf("🔗 [Client %s] 成功锁定[Browser %s](%s - %s)", c.id, target.id, target.model, target.title)
		c.WriteJSON(Message{
			Type:    "pair_result",
			Content: true,
			Model:   target.model,
			Title:   target.title,
			BID:     target.id,
		})
	} else {
		c.WriteJSON(Message{Type: "pair_result", Content: false})
	}
}

// [END] HNDL-PAIR

// [START] HNDL-FWD
// version: 001
// 上下文：当 ROUTER-MAP 未命中任何路由，作为 fallback 兜底触发。先决调用：METH-HUB-PAIR 成功建立 partner 联系。后续调用：对端连接的 METH-CLIENT-WRITE。
// 输入参数：c (*Client), msg (Message)
// 输出参数：无
func handleForward(c *Client, msg Message) {
	if c.partner != nil {
		msg.Source = c.id
		c.partner.WriteJSON(msg)
	}
}

// [END] HNDL-FWD

// [START] WS-HANDLER
// version: 001
// 上下文：作为 HTTP 请求对应的最终端点。先决调用：MAIN-ENTRY 注册 HTTP Mux 路由规则后发生 HTTP 请求。后续调用：进入闭包内部流程流转（WS-CONN-INIT 开始）。
// 输入参数：role (string)
// 输出参数：http.HandlerFunc
func wsHandler(role string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {

		// [START] WS-CONN-INIT
		// version: 001
		// 上下文：闭包函数激活的第一步。先决调用：WS-HANDLER 接收触发。后续调用：如果是 browser 顺延到 WS-REG-BROWSER，随后进入并发分支 WS-HEARTBEAT 和阻塞循环 WS-MSG-LOOP。
		// 输入参数：w (http.ResponseWriter), r (*http.Request)
		// 输出参数：无 (作用域内输出 conn 与 c 实例)
		conn, err := upgrader.Upgrade(w, r, nil)
		if err != nil {
			return
		}

		c := &Client{
			conn:     conn,
			role:     role,
			id:       hub.nextID(role),
			done:     make(chan struct{}),
			Metadata: make(map[string]string),
		}
		log.Printf("➕ [%s] 建立连接, ID: %s", role, c.id)
		// [END] WS-CONN-INIT

		// [START] WS-REG-BROWSER
		// version: 001
		// 上下文：连接刚刚升级完毕。先决调用：WS-CONN-INIT，且 role 必须为 "browser"。后续调用：挂载入池以备后续的配对查询。
		// 输入参数：c (*Client)
		// 输出参数：无
		if role == "browser" {
			hub.RegisterBrowser(c)
		}
		// [END] WS-REG-BROWSER

		// [START] WS-HEARTBEAT
		// version: 001
		// 上下文：与 WS-MSG-LOOP 同步进行。先决调用：WS-CONN-INIT 连接准备完毕。后续调用：周期内向外层网关发送 Ping，直到通道 done 关闭触发 WS-TEARDOWN。
		// 输入参数：协程捕获的外部 c (*Client)
		// 输出参数：无
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
						return
					}
				case <-c.done:
					return
				}
			}
		}()
		// [END] WS-HEARTBEAT

		// [START] WS-TEARDOWN
		// version: 001
		// 上下文：由于网络异常抛错或手动退出引发函数 Return 前执行。先决调用：必须完成 WS-CONN-INIT。后续调用：触发 METH-HUB-UNREG，进而影响到另一侧关联方的状态更改。
		// 输入参数：协程捕获的外部 conn (*websocket.Conn) 与 c (*Client)
		// 输出参数：无
		defer func() {
			close(c.done)
			hub.UnregisterAndRelease(c)
			conn.Close()
			log.Printf("➖ [%s] 断开连接, ID: %s", role, c.id)
		}()
		// [END] WS-TEARDOWN

		// [START] WS-MSG-LOOP
		// version: 001
		// 上下文：长连接维持生命期的业务心脏。先决调用：WS-CONN-INIT 就绪完毕。后续调用：根据不同的 MsgType 到 ROUTER-MAP 映射并调用具体 HNDL 处理器。
		// 输入参数：协程捕获的外部 conn (*websocket.Conn) 与 c (*Client)
		// 输出参数：无 (通过路由内部或跳出循环触发清理)
		// 预留扩展空间：循环入口处预留埋点拦截器位置（Interceptor），可用于高频风控限流拦截。
		for {
			var msg Message
			if err := conn.ReadJSON(&msg); err != nil {
				break
			}

			if handler, exists := messageRouter[msg.Type]; exists {
				handler(c, msg)
			} else {
				handleForward(c, msg)
			}
		}
		// [END] WS-MSG-LOOP
	}
}

// [END] WS-HANDLER

// [START] MAIN-ENTRY
// version: 001
// 上下文：初始化加载完毕后进入主干流。先决调用：INIT-FUNC。后续调用：WS-HANDLER 路由挂载，最后转入底层 Http Listener 阻塞。
// 输入参数：无
// 输出参数：无
func main() {
	http.HandleFunc("/ws/browser", wsHandler("browser"))
	http.HandleFunc("/ws/client", wsHandler("client"))

	fmt.Println("🚀 Bridge Server 运行在 127.26.3.1:8080")
	log.Fatal(http.ListenAndServe("127.26.3.1:8080", nil))
}

// [END] MAIN-ENTRY
