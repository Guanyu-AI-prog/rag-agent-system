"""
嵌入模型封装
通过 SiliconFlow API 调用 BAAI/bge-large-zh-v1.5 (1024维)
兼容 LangChain Embeddings 接口
"""
import os
from typing import List
from langchain_core.embeddings import Embeddings
import requests
import time


class LocalBGEEmbeddings(Embeddings):
    """通过 SiliconFlow API 调用嵌入模型"""

    def __init__(self, model_name: str = "BAAI/bge-large-zh-v1.5"):
        self.model_name = model_name
        self.api_key = os.getenv("SILICONFLOW_API_KEY", "")
        self.api_base = os.getenv("SILICONFLOW_API_BASE", "https://api.siliconflow.cn/v1")
        if not self.api_key:
            raise ValueError("请设置 SILICONFLOW_API_KEY 环境变量")

    def _call_api(self, texts: List[str]) -> List[List[float]]:
        """调用 SiliconFlow embedding API"""
        url = f"{self.api_base}/embeddings"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model_name,
            "input": texts,
            "encoding_format": "float"
        }

        for attempt in range(3):
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=60)
                resp.raise_for_status()
                data = resp.json()
                # 按 index 排序，保证顺序一致
                sorted_data = sorted(data["data"], key=lambda x: x["index"])
                return [item["embedding"] for item in sorted_data]
            except Exception as e:
                if attempt < 2:
                    time.sleep(1)
                else:
                    raise

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """批量嵌入文档"""
        # SiliconFlow API 每次最多处理有限条数，分批处理
        batch_size = 10
        # 截断超长文本，防止API 400错误
        texts = [t[:500] for t in texts]
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            embeddings = self._call_api(batch)
            all_embeddings.extend(embeddings)
        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        """嵌入单条查询"""
        return self._call_api([text])[0]


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    embedder = LocalBGEEmbeddings()

    docs = ["你好世界", "今天天气怎么样"]
    doc_embeddings = embedder.embed_documents(docs)
    print(f"文档嵌入维度: {len(doc_embeddings[0])}")

    query_embedding = embedder.embed_query("测试查询")
    print(f"查询嵌入维度: {len(query_embedding)}")

    print("SiliconFlow 嵌入模型测试通过")
