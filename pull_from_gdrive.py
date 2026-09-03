#!/usr/bin/env python3
"""Sync experiment artifacts (scorecard CSV, report MD, checkpoints) from Google Drive."""
import os
import sys
import json
import urllib.request
import urllib.parse
import subprocess

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(ROOT, "recovered_artifacts")

def get_token():
    try:
        return subprocess.check_output(["gcloud", "auth", "print-access-token"]).decode().strip()
    except Exception as e:
        print(f"❌ 获取 gcloud access token 失败: {e}")
        sys.exit(1)

def drive_request(url, token):
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(req) as resp:
        return resp.read()

def list_files_in_folder(folder_id, token):
    q = f"'{folder_id}' in parents and trashed = false"
    url = "https://www.googleapis.com/drive/v3/files?pageSize=100&fields=files(id,name,mimeType,size,modifiedTime)&q=" + urllib.parse.quote(q)
    data = json.loads(drive_request(url, token).decode("utf-8"))
    return data.get("files", [])

def download_file(file_id, dest_path, token):
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    url = f"https://www.googleapis.com/drive/v3/files/{file_id}?alt=media"
    content = drive_request(url, token)
    with open(dest_path, "wb") as f:
        f.write(content)
    print(f"  ✓ 下载完成: {os.path.relpath(dest_path, ROOT)} ({len(content):,} 字节)")

def sync_all(download_ckpts=False):
    print("=" * 65)
    print("🔄 开始从 Google Drive 同步 SimpleAI 实验产物...")
    print("=" * 65)
    token = get_token()

    # 1. 查找 SimpleAI_Experiments 根文件夹
    q = "name = 'SimpleAI_Experiments' and mimeType = 'application/vnd.google-apps.folder' and trashed = false"
    url = "https://www.googleapis.com/drive/v3/files?pageSize=10&fields=files(id,name)&q=" + urllib.parse.quote(q)
    folders = json.loads(drive_request(url, token).decode("utf-8")).get("files", [])
    if not folders:
        print("❌ 未在 Google Drive 中找到 'SimpleAI_Experiments' 文件夹！")
        return
    folder_id = folders[0]["id"]
    print(f"📁 找到 Google Drive 目录: SimpleAI_Experiments (id: {folder_id})")

    # 2. 列出并下载根目录文件 (CSV, MD)
    root_items = list_files_in_folder(folder_id, token)
    runs_folder_id = None
    for item in root_items:
        if item["name"] == "runs" and item["mimeType"] == "application/vnd.google-apps.folder":
            runs_folder_id = item["id"]
        elif item["mimeType"] != "application/vnd.google-apps.folder":
            dest = os.path.join(ROOT, item["name"])
            print(f"📥 同步大表/报告: {item['name']}")
            download_file(item["id"], dest, token)

    # 3. 同步 runs 目录下的各实验 Checkpoints
    if runs_folder_id and download_ckpts:
        print("\n📦 同步 Checkpoints 权重...")
        exp_folders = list_files_in_folder(runs_folder_id, token)
        for exp_f in exp_folders:
            if exp_f["mimeType"] == "application/vnd.google-apps.folder":
                exp_name = exp_f["name"]
                exp_files = list_files_in_folder(exp_f["id"], token)
                for f in exp_files:
                    if f["name"] in ("checkpoint_final.pt", "resolved_config.json"):
                        dest = os.path.join(OUT_DIR, "runs", exp_name, f["name"])
                        download_file(f["id"], dest, token)

    print("\n🎉 Google Drive 产物同步完毕！")

if __name__ == "__main__":
    download_ckpts = "--with-ckpts" in sys.argv
    sync_all(download_ckpts=download_ckpts)
