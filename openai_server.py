#!/usr/bin/env python3
"""
HTTP Server providing OpenAI-compatible API for DeepSeek Web UI
Zero external dependencies - uses only Python standard library
"""

import json
import asyncio
import http.server
import socketserver
import threading
import urllib.parse
import time
import traceback
import hashlib
from typing import Dict, List, Optional
from datetime import datetime

# Import our OpenAI adapter
import sys
sys.path.insert(0, '.')
from openai_adapter import OpenAIAdapter, SessionManager


class OpenAIHTTPHandler(http.server.BaseHTTPRequestHandler):
    """HTTP handler for OpenAI-compatible API endpoints."""
    
    def __init__(self, *args, **kwargs):
        self.adapter = OpenAIAdapter()
        super().__init__(*args, **kwargs)
    
    def log_message(self, format, *args):
        """Override to clean up logs."""
        pass
    
    def _send_response(self, status_code: int, data: Optional[Dict] = None, 
                      error: Optional[str] = None):
        """Send JSON response."""
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.end_headers()
        
        if data is None and error is not None:
            data = {
                "error": {
                    "message": error,
                    "type": "internal_error" if status_code >= 500 else "invalid_request_error"
                }
            }
        
        if data:
            self.wfile.write(json.dumps(data, ensure_ascii=False).encode("utf-8"))
    
    def do_OPTIONS(self):
        """Handle CORS preflight requests."""
        self._send_response(200)
    
    def do_GET(self):
        """Handle GET requests."""
        start_time = time.time()
        
        try:
            if self.path == "/" or self.path == "/health":
                # Health check endpoint
                self._send_response(200, {
                    "status": "ok",
                    "service": "DeepSeek OpenAI Adapter",
                    "timestamp": datetime.now().isoformat(),
                    "version": "1.0.0",
                    "endpoints": {
                        "GET /health": "Health check",
                        "GET /v1/models": "List available models",
                        "POST /v1/chat/completions": "Chat completion"
                    }
                })
                
            elif self.path == "/v1/models":
                # List available models
                self._send_response(200, {
                    "object": "list",
                    "data": [
                        {
                            "id": "deepseek-chat",
                            "object": "model",
                            "created": int(time.time()),
                            "owned_by": "deepseek-webui"
                        }
                    ]
                })
                
            else:
                self._send_response(404, error=f"Not found: {self.path}")
                
        except Exception as e:
            self._send_response(500, error=str(e))
            
        finally:
            duration = (time.time() - start_time) * 1000
            print(f"[{datetime.now().strftime('%H:%M:%S')}] GET {self.path} -> {self.send_header} ({duration:.1f}ms)")
    
    def do_POST(self):
        """Handle POST requests (chat completions)."""
        start_time = time.time()
        
        try:
            # Read request body
            content_length = int(self.headers.get("Content-Length", 0))
            if content_length == 0:
                self._send_response(400, error="Empty request body")
                return
            
            body = self.rfile.read(content_length).decode("utf-8")
            request_data = json.loads(body)
            
            # Validate request
            if "messages" not in request_data:
                self._send_response(400, error="Missing required field: messages")
                return
            
            # Handle different endpoints
            if self.path == "/v1/chat/completions":
                # Process chat completion request
                try:
                    # This would need to be async, but we're in a sync HTTP handler
                    # For now, we'll run the adapter in a thread
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    
                    response = loop.run_until_complete(
                        self.adapter.handle_request(request_data)
                    )
                    
                    self._send_response(200, response)
                    
                except ValueError as e:
                    self._send_response(400, error=str(e))
                except Exception as e:
                    self._send_response(500, error=str(e))
                    traceback.print_exc()
                    
            else:
                self._send_response(404, error=f"Not found: {self.path}")
                
        except json.JSONDecodeError:
            self._send_response(400, error="Invalid JSON in request body")
        except Exception as e:
            self._send_response(500, error=str(e))
            traceback.print_exc()
            
        finally:
            duration = (time.time() - start_time) * 1000
            print(f"[{datetime.now().strftime('%H:%M:%S')}] POST {self.path} -> {self.send_header} ({duration:.1f}ms)")


class OpenAIServer:
    """HTTP server manager."""
    
    def __init__(self, host: str = "127.0.0.1", port: int = 8000):
        self.host = host
        self.port = port
        self.server = None
        self.thread = None
    
    def start(self):
        """Start the HTTP server in background thread."""
        print(f"🚀 Starting OpenAI-compatible API server on http://{self.host}:{self.port}")
        print("=" * 60)
        print("📡 Available endpoints:")
        print("  GET  /health                - Health check")
        print("  GET  /v1/models             - List available models")
        print("  POST /v1/chat/completions   - Chat completion")
        print("")
        print("🔌 WebSocket URL: wss://c.810114.xyz/ws/client")
        print("🧠 Similarity threshold: 20% (auto new_chat)")
        print("=" * 60)
        print("")
        
        # Create server
        self.server = socketserver.ThreadingTCPServer(
            (self.host, self.port),
            OpenAIHTTPHandler
        )
        
        # Start in background thread
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        
        print("✅ Server started successfully!")
        print("")
        print("📝 Example requests:")
        print(f'  curl http://{self.host}:{self.port}/health')
        print(f'  curl -X POST http://{self.host}:{self.port}/v1/chat/completions \\')
        print('    -H "Content-Type: application/json" \\')
        print('    -d \'{"model":"deepseek-chat","messages":[{"role":"user","content":"Hello"}]}\'')
        print("")
        print("🔄 To clear context, include system message: {\"role\": \"system\", \"content\": \"[NEW_CHAT]\"}")
        print("📊 Server logs will show similarity decisions and new_chat triggers")
        print("")
        
        return True
    
    def stop(self):
        """Stop the HTTP server."""
        if self.server:
            print("🛑 Stopping server...")
            self.server.shutdown()
            self.server.server_close()
            print("✅ Server stopped")
    
    def wait(self):
        """Wait for server thread to finish."""
        if self.thread:
            self.thread.join()


def main():
    """Main entry point."""
    import sys
    
    # Parse command line arguments
    host = "127.0.0.1"
    port = 8000
    
    for arg in sys.argv[1:]:
        if arg.startswith("--host="):
            host = arg.split("=")[1]
        elif arg.startswith("--port="):
            port = int(arg.split("=")[1])
        elif arg in ["-h", "--help"]:
            print("Usage: python openai_server.py [--host=127.0.0.1] [--port=8000]")
            print("")
            print("Starts an OpenAI-compatible HTTP server for DeepSeek Web UI automation.")
            print("")
            print("Features:")
            print("  • Zero external dependencies")
            print("  • Similarity-based new_chat detection (20% threshold)")
            print("  • Multi-session support with client_id")
            print("  • Full OpenAI API compatibility")
            print("  • WebSocket connection to existing DeepSeek automation server")
            return
    
    # Start server
    server = OpenAIServer(host, port)
    
    try:
        server.start()
        print(f"\n✨ Server is running! Press Ctrl+C to stop.\n")
        
        # Keep main thread alive
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n👋 Shutting down...")
        server.stop()
    except Exception as e:
        print(f"\n❌ Server error: {e}")
        traceback.print_exc()
        server.stop()


if __name__ == "__main__":
    main()