import json
import sys

if sys.platform.startswith('win'):
    sys.stdout.reconfigure(encoding='utf-8')

def main():
    p = r"e:/project/advanced-rag/tests/temp_data/extracted_names.json"
    with open(p, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    targets = ['刘备', '玄德', '关羽', '云长', '张飞', '翼德', '曹操', '孟德', '诸葛亮', '孔明', '督邮', '孙权', '周瑜']
    found = [x for x in data if x['name'] in targets]
    
    print("查找到的目标人物词频：")
    print(json.dumps(found, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
