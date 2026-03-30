#!/usr/bin/env python3
"""
OpenAI API Adapter for DeepSeek Web UI Automation
- Receives OpenAI-compatible HTTP requests
- Manages WebSocket connections to DeepSeek browsers
- Uses similarity detection to decide when to send new_chat
"""

import json
import asyncio
import traceback
import math
import hashlib
import time
from typing import Dict, List, Optional, Tuple
import websockets
from adapters.deepseek_webapp import DeepSeekWebApp


class SessionManager:
    """Manages multiple WebSocket sessions with similarity-based routing."""
    
    def __init__(self):
        self.sessions: Dict[str, Session] = {}
        self.history_limit = 10
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate Jaccard similarity between two texts."""
        if not text1 or not text2:
            return 0.0
        
        # Simple word-based similarity
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    def find_best_session(self, new_text: str, client_id: Optional[str] = None) -> Tuple[Optional[str], float]:
        """
        Find the best existing session for the new text.
        Returns (session_id, similarity_score)
        """
        if client_id and client_id in self.sessions:
            # Use specified client_id if provided
            session = self.sessions[client_id]
            if session.history:
                similarity = self.calculate_similarity(new_text, session.history[-1])
                return client_id, similarity
            return client_id, 0.0
        
        # Find best matching session
        best_id = None
        best_score = 0.0
        
        for session_id, session in self.sessions.items():
            if session.history:
                score = self.calculate_similarity(new_text, session.history[-1])
                if score > best_score:
                    best_score = score
                    best_id = session_id
        
        return best_id, best_score
    
    def should_create_new_chat(self, similarity: float) -> bool:
        """Determine if a new_chat should be sent based on similarity."""
        # Lower similarity = less related = needs new chat
        return similarity < 0.2  # 20% similarity threshold
    
    async def get_or_create_session(self, ws_url: str, client_id: Optional[str] = None) -> 'Session':
        """Get existing session or create new one."""
        if client_id is None:
            client_id = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
        
        if client_id not in self.sessions:
            session = Session(ws_url, client_id)
            await session.connect()
            self.sessions[client_id] = session
        
        return self.sessions[client_id]
    
    async def process_message(self, text: str, client_id: Optional[str] = None, 
                              force_new_chat: bool = False, ws_url: str = "wss://moonchan.publicvm.com/ws/client") -> str:
        """
        Process a message through similarity-based routing.
        Returns the AI response.
        """
        # Get or create session
        session = await self.get_or_create_session(ws_url, client_id)
        
        # Find best session and calculate similarity
        best_id, similarity = self.find_best_session(text, client_id)
        
        # Decide if we need new_chat
        needs_new_chat = force_new_chat or self.should_create_new_chat(similarity)
        
        # If we found a better session, use it
        if best_id and best_id != client_id:
            session = self.sessions[best_id]
            needs_new_chat = force_new_chat or self.should_create_new_chat(similarity)
        
        # Process the message
        response = await session.send_message(text, new_chat=needs_new_chat)
        
        # Update history
        session.add_to_history(text)
        
        return response
    
    def cleanup(self):
        """Clean up inactive sessions."""
        for session_id in list(self.sessions.keys()):
            if not self.sessions[session_id].is_active():
                del self.sessions[session_id]


class Session:
    """Manages a single WebSocket connection and conversation history."""
    
    def __init__(self, ws_url: str, session_id: str):
        self.ws_url = ws_url
        self.session_id = session_id
        self.client: Optional[DeepSeekWebApp] = None
        self.history: List[str] = []
        self.connected = False
        self.last_active = time.time()
    
    async def connect(self):
        """Establish WebSocket connection and pair with browser."""
        try:
            ws = await websockets.connect(self.ws_url)
            self.client = DeepSeekWebApp(ws)
            
            # Start listening in background
            asyncio.create_task(self.client.listen())
            
            # Wait for connection and pairing
            await asyncio.sleep(2.0)
            
            # Request browser list
            await self.client.send("system", {"command": "list"})
            await asyncio.sleep(1.0)
            
            if not self.client.available_browsers:
                raise Exception("No available browsers")
            
            # Pair with first browser
            target = self.client.available_browsers[0]
            await self.client.send("system", {
                "command": "pair",
                "title": target.get("title"),
                "type": target.get("type"),
                "timestamp": target.get("timestamp")
            })
            await asyncio.sleep(1.0)
            
            if not self.client.paired:
                raise Exception("Pairing failed")
            
            # Navigate to DeepSeek chat
            await self.client.call_match("chat.deepseek.com")
            await asyncio.sleep(1.0)
            
            self.connected = True
            self.last_active = time.time()
            
        except Exception as e:
            self.connected = False
            raise
    
    async def send_message(self, text: str, new_chat: bool = False) -> str:
        """Send a message and wait for response."""
        if not self.connected or not self.client:
            await self.connect()
        
        if new_chat:
            await self.client.call_new_chat()
            await asyncio.sleep(1.0)
        
        await self.client.call_send_prompt(text=text)
        
        # Wait for response
        think, response = await self.client.pop_response()
        
        self.last_active = time.time()
        return response if response else think
    
    def add_to_history(self, text: str):
        """Add message to conversation history."""
        self.history.append(text)
        if len(self.history) > 20:  # Limit history length
            self.history = self.history[-10:]
    
    def is_active(self) -> bool:
        """Check if session is still active (within 5 minutes)."""
        return time.time() - self.last_active < 300  # 5 minutes
    
    async def close(self):
        """Close the WebSocket connection."""
        if self.client and self.client.ws:
            await self.client.ws.close()
            self.connected = False


class OpenAIAdapter:
    """HTTP server that provides OpenAI-compatible API."""
    
    def __init__(self, host: str = "127.0.0.1", port: int = 8000):
        self.host = host
        self.port = port
        self.session_manager = SessionManager()
    
    async def handle_request(self, request_data: Dict) -> Dict:
        """Handle OpenAI-compatible chat completion request."""
        try:
            # Extract parameters
            messages = request_data.get("messages", [])
            model = request_data.get("model", "deepseek-chat")
            stream = request_data.get("stream", False)
            
            # Extract the LAST user message (OpenAI API format)
            user_message = ""
            force_new_chat = False
            client_id = None
            
            # Find the last user message in the conversation
            for msg in reversed(messages):
                role = msg.get("role", "")
                content = msg.get("content", "")
                
                if role == "user":
                    user_message = content
                    break
            
            if not user_message:
                raise ValueError("No user message found in messages array")
            
            # Check for new_chat command in any system message
            for msg in messages:
                role = msg.get("role", "")
                content = msg.get("content", "")
                
                if role == "system" and content.strip().upper() == "[NEW_CHAT]":
                    force_new_chat = True
                    break
            
            # Extract client_id from system messages
            for msg in messages:
                if msg.get("role") == "system":
                    content = msg.get("content", "")
                    if content.startswith("client_id:"):
                        client_id = content.split(":")[1].strip()
                        break
            
            # Process through session manager
            response_text = await self.session_manager.process_message(
                text=user_message,
                client_id=client_id,
                force_new_chat=force_new_chat
            )
            
            # Build OpenAI-compatible response
            response = {
                "id": f"chatcmpl-{hashlib.md5(str(time.time()).encode()).hexdigest()[:24]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model,
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response_text
                    },
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": len(user_message) // 4,
                    "completion_tokens": len(response_text) // 4,
                    "total_tokens": (len(user_message) + len(response_text)) // 4
                }
            }
            
            return response
            
        except Exception as e:
            raise
    
    async def start_server(self):
        """Start HTTP server (placeholder - needs implementation)."""
        print(f"OpenAI Adapter server would start on {self.host}:{self.port}")
        print("Note: HTTP server implementation requires additional dependencies")
        print("For now, use the adapter programmatically or with a separate HTTP server")


async def test_openai_adapter():
    """Test the OpenAI adapter functionality."""
    print("Testing OpenAI Adapter...")
    
    adapter = OpenAIAdapter()
    
    # Test request
    test_request = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": "[NEW_CHAT]"},
            {"role": "user", "content": "Hello, how are you?"}
        ],
        "stream": False
    }
    
    try:
        response = await adapter.handle_request(test_request)
        print(f"Response: {json.dumps(response, indent=2, ensure_ascii=False)}")
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    # For testing
    asyncio.run(test_openai_adapter())
