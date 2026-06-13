import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# 配置 HuggingFace 相关的环境变量（确保直接使用下载好的本地缓存）
os.environ['HF_HOME'] = r'D:\my_huggingface_cache'
os.environ['SENTENCE_TRANSFORMERS_HOME'] = r'D:\my_huggingface_cache'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

app = FastAPI(title="Advanced RAG Service", description="工业级混合检索与分层切片 RAG 服务端 API")

# 定义请求与返回的数据格式
class QueryRequest(BaseModel):
    query: str
    top_k: int = 5

class QueryResponse(BaseModel):
    context: str

class AddFileRequest(BaseModel):
    file_path: str

class AddFileResponse(BaseModel):
    status: str
    message: str

@app.on_event("startup")
def startup_event():
    """在服务启动时延迟加载 RAG 组件，支持测试时注入 Mock 实例"""
    if not hasattr(app.state, "coordinator") or app.state.coordinator is None:
        print("正在初始化全局生产 RAG 编排器（离线模式）...")
        # 读取环境变量 RAG_DEVICE 决定加载设备，默认使用 cpu
        device = os.environ.get("RAG_DEVICE", "cpu").lower()
        print(f"RAG 模型加载运行设备定位为: {device}")
        
        # 局部导入以避免在测试不需要加载模型时拖慢启动速度
        from src.loader import DocumentLoader
        from src.splitter import SemanticParentChildSplitter
        from src.embedding import LocalEmbeddingService
        from src.database import ChromaAdapter
        from src.reranker import RerankerService
        from src.coordinator import RAGCoordinator

        db_dir = "./vector_db"
        loader = DocumentLoader()
        embedding_service = LocalEmbeddingService(device=device)
        splitter = SemanticParentChildSplitter(embedding_service=embedding_service, threshold=None, child_size=150)
        db_adapter = ChromaAdapter(db_dir=db_dir)
        reranker_service = RerankerService(device=device)

        app.state.coordinator = RAGCoordinator(
            loader=loader,
            splitter=splitter,
            embedding_service=embedding_service,
            db_adapter=db_adapter,
            reranker=reranker_service
        )
        print("全局生产 RAG 编排服务初始化成功！")

@app.post("/retrieve", response_model=QueryResponse)
def retrieve(request: QueryRequest):
    """
    进行混合检索与重排精选，返回父块拼接上下文
    """
    if not hasattr(app.state, "coordinator") or app.state.coordinator is None:
        raise HTTPException(status_code=503, detail="RAG 服务未就绪")
    
    try:
        context = app.state.coordinator.query(request.query)
        return QueryResponse(context=context)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"检索处理出错: {str(e)}")

@app.post("/add_file", response_model=AddFileResponse)
def add_file(request: AddFileRequest):
    """
    动态将本地文件导入向量数据库中
    """
    if not hasattr(app.state, "coordinator") or app.state.coordinator is None:
        raise HTTPException(status_code=503, detail="RAG 服务未就绪")
    
    file_path = request.file_path
    if not os.path.exists(file_path):
        raise HTTPException(status_code=400, detail=f"文件不存在: {file_path}")

    try:
        app.state.coordinator.add_file(file_path)
        return AddFileResponse(status="success", message=f"文件 {os.path.basename(file_path)} 导入并索引成功")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"文件导入处理出错: {str(e)}")
