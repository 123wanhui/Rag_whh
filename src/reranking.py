import os
from dotenv import load_dotenv
from openai import OpenAI
import requests
import src.prompts as prompts
from concurrent.futures import ThreadPoolExecutor


# JinaReranker：基于Jina API的重排器，适用于多语言场景
class JinaReranker:
    def __init__(self):
        # 初始化Jina重排API地址和请求头
        self.url = 'https://api.jina.ai/v1/rerank'
        self.headers = self.get_headers()
        
    def get_headers(self):
        # 加载Jina API密钥，组装请求头
        load_dotenv()
        jina_api_key = os.getenv("JINA_API_KEY")    
        headers = {'Content-Type': 'application/json',
                   'Authorization': f'Bearer {jina_api_key}'}
        return headers
    
    def rerank(self, query, documents, top_n = 10):
        # 调用Jina API进行重排，返回top_n相关文档
        data = {
            "model": "jina-reranker-v2-base-multilingual",
            "query": query,
            "top_n": top_n,
            "documents": documents
        }

        response = requests.post(url=self.url, headers=self.headers, json=data)

        return response.json()

# LLMReranker：基于大模型的重排器，支持单条和批量重排
class LLMReranker:
    def __init__(self, provider: str = "dashscope"):
        # 支持 openai/dashscope，默认 dashscope
        self.provider = provider.lower()
        self.llm = self.set_up_llm()
        self.system_prompt_rerank_single_block = prompts.RerankingPrompt.system_prompt_rerank_single_block
        self.system_prompt_rerank_multiple_blocks = prompts.RerankingPrompt.system_prompt_rerank_multiple_blocks
        self.schema_for_single_block = prompts.RetrievalRankingSingleBlock
        self.schema_for_multiple_blocks = prompts.RetrievalRankingMultipleBlocks
      
    def set_up_llm(self):
        # 根据 provider 初始化 LLM 客户端
        load_dotenv()
        if self.provider == "openai":
            return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        elif self.provider == "dashscope":
            import dashscope
            dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")
            return dashscope
        else:
            raise ValueError(f"不支持的 LLM provider: {self.provider}")
    
    @staticmethod
    def _extract_score(text: str) -> float:
        """从 LLM 文本输出中提取 relevance_score（0~1），提取失败返回 0.5"""
        import re
        if not text:
            return 0.5
        # 尝试多种模式匹配
        patterns = [
            r'"relevance_score"\s*:\s*([\d.]+)',   # JSON 格式
            r'relevance_score[:\s]+([\d.]+)',       # 键值对格式
            r'相关性分数[：:]\s*([\d.]+)',            # 中文格式
            r'评分[：:]\s*([\d.]+)',                  # 中文简写
            r'\b([0-9]\.[0-9])\b',                   # 裸小数 0.x
        ]
        for pat in patterns:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                score = float(m.group(1))
                return max(0.0, min(1.0, score))  # clamp to [0, 1]
        return 0.5

    def get_rank_for_single_block(self, query, retrieved_document):
        # 针对单个文本块，调用LLM进行相关性评分
        user_prompt = f'/nHere is the query:/n"{query}"/n/nHere is the retrieved text block:/n"""/n{retrieved_document}/n"""/n'
        if self.provider == "openai":
            completion = self.llm.beta.chat.completions.parse(
                model="gpt-4o-mini-2024-07-18",
                temperature=0,
                messages=[
                    {"role": "system", "content": self.system_prompt_rerank_single_block},
                    {"role": "user", "content": user_prompt},
                ],
                response_format=self.schema_for_single_block
            )
            response = completion.choices[0].message.parsed
            response_dict = response.model_dump()
            return response_dict
        elif self.provider == "dashscope":
            # dashscope 只返回字符串，暂不做结构化解析
            messages = [
                {"role": "system", "content": self.system_prompt_rerank_single_block},
                {"role": "user", "content": user_prompt},
            ]
            try:
                rsp = self.llm.Generation.call(
                    model="qwen-plus",  # qwen-turbo 额度用完
                    messages=messages,
                    temperature=0,
                    result_format='message'
                )
            except Exception as e:
                raise RuntimeError(f"DashScope Generation.call 调用失败: {e}")

            # 多重兜底：从响应中安全提取文本内容
            content = None
            try:
                # 优先使用属性访问（dashscope SDK 对象）
                if hasattr(rsp, 'output') and rsp.output is not None:
                    out = rsp.output
                    if hasattr(out, 'choices') and out.choices:
                        content = out.choices[0].message.content
                # 兜底：尝试 dict 方式访问
                if content is None and isinstance(rsp, dict):
                    out = rsp.get('output') or {}
                    choices = out.get('choices', []) if isinstance(out, dict) else []
                    if choices:
                        content = choices[0].get('message', {}).get('content', None)
            except Exception:
                content = None

            if content is None:
                raise RuntimeError(f"DashScope重排返回格式异常，无法提取content，完整响应: {rsp}")
            score = self._extract_score(content)
            return {"relevance_score": score, "reasoning": content}
        else:
            raise ValueError(f"不支持的 LLM provider: {self.provider}")

    def get_rank_for_multiple_blocks(self, query, retrieved_documents):
        # 针对多个文本块，批量调用LLM进行相关性评分
        formatted_blocks = "\n\n---\n\n".join([f'Block {i+1}:\n\n"""\n{text}\n"""' for i, text in enumerate(retrieved_documents)])
        user_prompt = (
            f"Here is the query: \"{query}\"\n\n"
            "Here are the retrieved text blocks:\n"
            f"{formatted_blocks}\n\n"
            f"You should provide exactly {len(retrieved_documents)} rankings, in order."
        )
        if self.provider == "openai":
            completion = self.llm.beta.chat.completions.parse(
                model="gpt-4o-mini-2024-07-18",
                temperature=0,
                messages=[
                    {"role": "system", "content": self.system_prompt_rerank_multiple_blocks},
                    {"role": "user", "content": user_prompt},
                ],
                response_format=self.schema_for_multiple_blocks
            )
            response = completion.choices[0].message.parsed
            response_dict = response.model_dump()
            return response_dict
        elif self.provider == "dashscope":
            messages = [
                {"role": "system", "content": self.system_prompt_rerank_multiple_blocks},
                {"role": "user", "content": user_prompt},
            ]
            try:
                rsp = self.llm.Generation.call(
                    model="qwen-plus",  # qwen-turbo 额度用完
                    messages=messages,
                    temperature=0,
                    result_format='message'
                )
            except Exception as e:
                raise RuntimeError(f"DashScope Generation.call 调用失败: {e}")

            # 多重兜底：从响应中安全提取文本内容
            content = None
            try:
                # 优先使用属性访问（dashscope SDK 对象）
                if hasattr(rsp, 'output') and rsp.output is not None:
                    out = rsp.output
                    if hasattr(out, 'choices') and out.choices:
                        content = out.choices[0].message.content
                # 兜底：尝试 dict 方式访问
                if content is None and isinstance(rsp, dict):
                    out = rsp.get('output') or {}
                    choices = out.get('choices', []) if isinstance(out, dict) else []
                    if choices:
                        content = choices[0].get('message', {}).get('content', None)
            except Exception:
                content = None

            if content is None:
                raise RuntimeError(f"DashScope重排返回格式异常，无法提取content，完整响应: {rsp}")
            # 尝试从文本中为每个文档提取独立评分
            import re, json as _json
            block_scores = []
            n = len(retrieved_documents)
            # 尝试 JSON 解析
            try:
                parsed = _json.loads(content)
                if isinstance(parsed, dict) and 'block_rankings' in parsed:
                    for br in parsed['block_rankings']:
                        block_scores.append(float(br.get('relevance_score', 0.5)))
            except Exception:
                pass
            # 如果 JSON 解析失败，用正则匹配每个 Block 的分数
            if len(block_scores) < n:
                # 匹配 "Block N: relevance_score: X.X" 或类似模式
                for i in range(n):
                    block_pat = rf'Block\s*{i+1}.*?relevance[_ ]?score[:\s]*([\d.]+)'
                    m = re.search(block_pat, content, re.IGNORECASE)
                    if m:
                        block_scores.append(max(0.0, min(1.0, float(m.group(1)))))
            # 兜底：用全局提取的分数
            if len(block_scores) < n:
                fallback = self._extract_score(content)
                while len(block_scores) < n:
                    block_scores.append(fallback)

            return {"block_rankings": [
                {"relevance_score": block_scores[i], "reasoning": content}
                for i in range(n)
            ]}
        else:
            raise ValueError(f"不支持的 LLM provider: {self.provider}")

    def rerank_documents(self, query: str, documents: list, documents_batch_size: int = 4, llm_weight: float = 0.7):
        """
        使用多线程并行方式对多个文档进行重排。
        结合向量相似度和LLM相关性分数，采用加权平均融合。
        参数：
            query: 查询语句
            documents: 待重排的文档列表，每个元素需包含'text'和'distance'
            documents_batch_size: 每批送入LLM的文档数
            llm_weight: LLM分数权重（0-1），其余为向量分数权重
        返回：
            按融合分数降序排序的文档列表
        """
        # 按batch分组
        doc_batches = [documents[i:i + documents_batch_size] for i in range(0, len(documents), documents_batch_size)]
        vector_weight = 1 - llm_weight
        
        if documents_batch_size == 1:
            def process_single_doc(doc):
                # 单文档重排
                ranking = self.get_rank_for_single_block(query, doc['text'])
                
                doc_with_score = doc.copy()
                doc_with_score["relevance_score"] = ranking["relevance_score"]
                # 计算融合分数，distance越小越相关
                doc_with_score["combined_score"] = round(
                    llm_weight * ranking["relevance_score"] + 
                    vector_weight * doc['distance'],
                    4
                )
                return doc_with_score

            # 多线程并行处理，max_workers=1 保证 dashscope LLM 串行调用，避免 QPS 超限
            with ThreadPoolExecutor(max_workers=1) as executor:
                all_results = list(executor.map(process_single_doc, documents))
                
        else:
            def process_batch(batch):
                # 批量重排
                import traceback as _tb
                try:
                    texts = [doc['text'] for doc in batch]
                    rankings = self.get_rank_for_multiple_blocks(query, texts)
                    results = []
                    block_rankings = rankings.get('block_rankings', [])

                    if len(block_rankings) < len(batch):
                        print(f"\nWarning: Expected {len(batch)} rankings but got {len(block_rankings)}")
                        for i in range(len(block_rankings), len(batch)):
                            doc = batch[i]
                            print(f"Missing ranking for document on page {doc.get('page', 'unknown')}:")
                            print(f"Text preview: {doc['text'][:100]}...\n")

                        for _ in range(len(batch) - len(block_rankings)):
                            block_rankings.append({
                                "relevance_score": 0.0,
                                "reasoning": "Default ranking due to missing LLM response"
                            })

                    for doc, rank in zip(batch, block_rankings):
                        doc_with_score = doc.copy()
                        doc_with_score["relevance_score"] = rank["relevance_score"]
                        doc_with_score["combined_score"] = round(
                            llm_weight * rank["relevance_score"] +
                            vector_weight * doc['distance'],
                            4
                        )
                        results.append(doc_with_score)
                    return results
                except Exception as e:
                    print(f"[process_batch ERROR] {type(e).__name__}: {e}")
                    print(_tb.format_exc())
                    raise

            # 多线程并行处理，max_workers=1 保证 dashscope LLM 串行调用，避免 QPS 超限
            with ThreadPoolExecutor(max_workers=1) as executor:
                batch_results = list(executor.map(process_batch, doc_batches))
            
            # 扁平化结果
            all_results = []
            for batch in batch_results:
                all_results.extend(batch)
        
        # 按融合分数降序排序
        all_results.sort(key=lambda x: x["combined_score"], reverse=True)
        return all_results
