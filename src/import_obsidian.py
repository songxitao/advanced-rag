import os
import glob
import requests
import sys
import io

if sys.platform.startswith('win'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

def import_markdown_logs(log_dir: str, api_url: str = "http://127.0.0.1:8000/add_file"):
    """
    遍历指定目录下的所有 .md 文件，并通过 API 录入到 RAG 向量数据库中。
    """
    if not os.path.exists(log_dir):
        print(f"❌ 目录不存在: {log_dir}")
        return

    # 寻找所有的 .md 文件
    search_path = os.path.join(log_dir, "**", "*.md")
    md_files = glob.glob(search_path, recursive=True)

    if not md_files:
        print(f"⚠️ 未在 {log_dir} 中找到任何 .md 文件。")
        return

    print(f"📂 找到 {len(md_files)} 个 Markdown 文件，开始导入...")

    success_count = 0
    fail_count = 0

    for idx, file_path in enumerate(md_files, 1):
        # 转换为正斜杠，防止 Windows 路径转义报错
        normalized_path = file_path.replace("\\", "/")
        print(f"[{idx}/{len(md_files)}] 正在导入: {os.path.basename(normalized_path)}...")
        
        try:
            response = requests.post(
                api_url,
                json={"file_path": normalized_path},
                timeout=30
            )
            if response.status_code == 200:
                print(f"   ✅ 成功: {response.json().get('message')}")
                success_count += 1
            else:
                print(f"   ❌ 失败: HTTP {response.status_code} - {response.text}")
                fail_count += 1
        except Exception as e:
            print(f"   ❌ 网络或请求错误: {str(e)}")
            fail_count += 1

    print("\n📊 导入任务结束：")
    print(f"   - 成功: {success_count} 个")
    print(f"   - 失败: {fail_count} 个")

if __name__ == "__main__":
    # 默认路径
    default_dir = r"D:\Obsidian\log"
    
    # 支持命令行参数输入路径
    target_dir = sys.argv[1] if len(sys.argv) > 1 else default_dir
    
    print(f"🚀 开始从 {target_dir} 批量导入项目日志到 RAG...")
    print("💡 提示：请确保已启动 RAG API 服务（python -m uvicorn src.app:app）\n")
    
    import_markdown_logs(target_dir)
