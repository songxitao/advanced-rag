import os

class LogCleaner:
    def clean_log(self, file_path: str) -> str:
        if not os.path.exists(file_path):
            print("文件找不到了哦！")
            return ""
            
        ext = os.path.splitext(file_path)[1].lower()
        if ext == '.log':
            print(f"开始清洗日志文件: {file_path}")
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
            clean_lines = []
            for line in lines:
                # 1. 先把这一行前后空格和换行符去掉
                line_clean = line.strip()
                
                # 2. 判断开头是否是 [WARNING] 或 [ERROR]
                # startswith 接收一个由括号包围的元组，只要满足其中一个前缀就返回 True
                if line_clean.startswith(('[WARNING]', '[ERROR]')):
                    clean_lines.append(line_clean)
                    
            return "\n".join(clean_lines).strip()
        else:
            print("只支持读取 .log 文件哦！")
            return ""

# 实例化并运行测试
if __name__ == "__main__":
    cleaner = LogCleaner()
    result = cleaner.clean_log('E:/project/advanced-rag/learn/app.log')
    print("\n--- 清洗后的日志结果 ---")
    print(result)