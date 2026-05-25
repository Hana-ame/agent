import requests
import time
import subprocess

def test_messagebox():
    # Start server in background
    proc = subprocess.Popen(["python3", "messagebox_service.py"])
    time.sleep(2)
    
    base_url = "http://127.0.0.1:8000/api"
    
    try:
        # Test Channel A
        print("Testing Channel A...")
        res1 = requests.post(f"{base_url}/chanA/message", json={"content": "Hello ChanA 1"}).json()
        res2 = requests.post(f"{base_url}/chanA/message", json={"content": "Hello ChanA 2"}).json()
        
        # Test Channel B
        print("Testing Channel B...")
        res3 = requests.post(f"{base_url}/chanB/message", json={"content": "Hello ChanB 1"}).json()
        
        # Verify latest
        latest_a = requests.get(f"{base_url}/chanA/latest").json()
        assert latest_a['content'] == "Hello ChanA 2"
        
        latest_b = requests.get(f"{base_url}/chanB/latest").json()
        assert latest_b['content'] == "Hello ChanB 1"
        
        # Verify next
        next_a = requests.get(f"{base_url}/chanA/next", params={"after_id": res1['id']}).json()
        assert next_a['content'] == "Hello ChanA 2"
        
        # Verify list
        list_a = requests.get(f"{base_url}/chanA/list").json()
        assert len(list_a) == 2
        
        print("✅ All tests passed!")
        
    finally:
        proc.terminate()

if __name__ == "__main__":
    test_messagebox()
