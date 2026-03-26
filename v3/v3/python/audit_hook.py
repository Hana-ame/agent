import sys

if hasattr(sys, 'addaudithook'):
    def audit_hook(event, args):
        if event in ('import', 'open'):
            print(f"Audit: {event} {args}")
    sys.addaudithook(audit_hook)
    
    # 触发事件
    with open('/dev/null', 'w') as f:
        pass
    import sys
else:
    print("addaudithook requires Python 3.8+")
