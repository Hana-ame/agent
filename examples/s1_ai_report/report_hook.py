import json
import os

def on_receive(data, channel, settings):
    # This vertex acts as a sink. We will append the incoming AI summary to a list
    # Because _data_store overwrites by channel, we can read the existing list,
    # append to it, and return the updated list.
    
    # Wait, the framework does `data_store[key] = data`. 
    # But `on_receive` doesn't have access to `data_store`.
    # We can just keep a global list in the module.
    
    if not hasattr(on_receive, "reports"):
        on_receive.reports = []
        
    on_receive.reports.append(data)
    
    # Generate report
    report_lines = ["# S1 AI Discussion Report\n"]
    for i, report in enumerate(on_receive.reports):
        report_lines.append(f"## Thread {i+1}")
        report_lines.append(report)
        report_lines.append("\n---\n")
        
    report_content = "\n".join(report_lines)
    
    # Write to a file
    out_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(out_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write(report_content)
        
    return on_receive.reports
