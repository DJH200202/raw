from openai import AsyncOpenAI
import json
import os
from tqdm.asyncio import tqdm as async_tqdm
import httpx
import asyncio

# 请在运行前设置环境变量: 
# Windows: set OPENAI_API_KEY=your_api_key_here
# Linux/Mac: export OPENAI_API_KEY=your_api_key_here
# 或者取消下面一行的注释并填入你的API密钥
# os.environ['OPENAI_API_KEY'] = "your_api_key_here"

# 并发控制：同时最多进行的请求数量
MAX_CONCURRENT_REQUESTS = 20
# 最大重试次数
MAX_RETRIES = 3
# 重试延迟（秒）
RETRY_DELAY = 1

async def extract_entities_with_llm(text, semaphore, retry_count=0):
    """Extract named entities from text using LLM (async version with retry)"""
    async with semaphore:  # 控制并发数量
        client = AsyncOpenAI(
            http_client=httpx.AsyncClient(
                proxy="http://127.0.0.1:7890",
                timeout=60.0  # 设置60秒超时
            ),
            api_key=os.environ["OPENAI_API_KEY"]
        )
        # client = AsyncOpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com")

        user_prompt = f"""You are a professional Named Entity Recognition system. Please analyze the question, perform Named Entity Recognition, extract key entities, and minimize irrelevant ones. Follow these rules strictly:

**Entity Types**:
- Person (e.g., "Elon Musk")
- Location (e.g., "Eiffel Tower")
- Work of Art (e.g., "The Starry Night")
- Product (e.g., "PlayStation 5")
- Organization (e.g., "United Nations")

**Extraction Rules**:
1. Extract only proper nouns and named entities, avoiding general terms.
2. Retain original text casing.
3. Combine compound names when necessary (e.g., "Theodred II (Bishop of Elmham)" as a single entity), but remove unnecessary qualifiers when they do not contribute to entity identity (e.g., "Lothair I of the Franks" → "Lothair I").
4. Return output in the following strict JSON format:

**Output Format**:
```json
{{
    "entities": [
    {{
        "text": "exact_text",
        "type": "entity_type"
    }}
    ]
}}
```

Question: {text}
"""

        try:
            response = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an assistant specialized in named entity recognition. Please return results in JSON format only.",
                    },
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0,
            )
            response_content = response.choices[0].message.content
            # 清理并返回 JSON 内容
            cleaned_content = response_content.strip()
            if cleaned_content.startswith('```json'):
                cleaned_content = cleaned_content[7:]
            if cleaned_content.endswith('```'):
                cleaned_content = cleaned_content[:-3]
            cleaned_content = cleaned_content.strip()
            
            return json.loads(cleaned_content)
        except Exception as e:
            if retry_count < MAX_RETRIES:
                print(f"错误 (重试 {retry_count + 1}/{MAX_RETRIES}): {str(e)[:100]}")
                await asyncio.sleep(RETRY_DELAY * (retry_count + 1))  # 指数退避
                return await extract_entities_with_llm(text, semaphore, retry_count + 1)
            else:
                print(f"处理失败（已达到最大重试次数）: {text[:100]}...")
                print(f"错误信息: {str(e)}")
                return {"entities": []}

async def process_single_item(item, index, semaphore):
    """Process a single item and return result with index to maintain order"""
    question = item["question"]
    entities = await extract_entities_with_llm(question, semaphore)
    result = {
        "question": question,
        "entities": entities["entities"],
    }
    return index, result

async def process_hotpotqa_questions():
    """Process questions from 2wiki dataset and perform entity recognition (async version)"""
    # Read dataset
    with open(os.path.join("data", "hippo", "nq_rear.json"), "r") as f:
        dataset = json.load(f)

    output_dir = "results/question_entities"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "nq_question_entities.json")
    temp_file = os.path.join(output_dir, "nq_question_entities_temp.json")
    
    # 检查是否存在临时文件（断点续传）
    processed_indices = set()
    existing_results = {}
    if os.path.exists(temp_file):
        try:
            with open(temp_file, "r", encoding="utf-8") as f:
                temp_data = json.load(f)
                for item in temp_data:
                    idx = item.get("_index")
                    if idx is not None:
                        processed_indices.add(idx)
                        existing_results[idx] = item
            print(f"检测到临时文件，已处理 {len(processed_indices)} 个问题，将继续处理剩余问题...")
        except Exception as e:
            print(f"读取临时文件失败，将重新开始: {e}")
            processed_indices = set()
            existing_results = {}

    # 创建信号量控制并发数量
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    
    # 只处理未完成的任务
    tasks = []
    for idx, item in enumerate(dataset):
        if idx not in processed_indices:
            tasks.append(process_single_item(item, idx, semaphore))
    
    # 并行处理所有任务，同时显示进度条
    total_questions = len(dataset)
    remaining_questions = len(tasks)
    print(f"总问题数: {total_questions}")
    print(f"已处理: {len(processed_indices)}")
    print(f"待处理: {remaining_questions}")
    print(f"最大并发数: {MAX_CONCURRENT_REQUESTS}")
    
    if remaining_questions > 0:
        indexed_results = list(existing_results.items())  # 已有的结果
        save_interval = max(10, remaining_questions // 20)  # 每处理5%或至少10个保存一次
        processed_count = 0
        
        for coro in async_tqdm.as_completed(tasks, total=remaining_questions):
            idx, result = await coro
            result["_index"] = idx  # 添加索引用于断点续传
            indexed_results.append((idx, result))
            processed_count += 1
            
            # 定期保存临时结果
            if processed_count % save_interval == 0:
                temp_results = [result for _, result in indexed_results]
                with open(temp_file, "w", encoding="utf-8") as f:
                    json.dump(temp_results, f, ensure_ascii=False, indent=2)
    else:
        indexed_results = list(existing_results.items())
        print("所有问题已处理完成！")
    
    # 按照原始顺序排序结果
    indexed_results.sort(key=lambda x: x[0])
    results = [result for _, result in indexed_results]
    
    # 移除临时索引字段
    for result in results:
        result.pop("_index", None)
    
    # Save final results
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # 删除临时文件
    if os.path.exists(temp_file):
        os.remove(temp_file)
    
    print(f"\n✓ 处理完成！")
    print(f"✓ 总共处理了 {len(results)} 个问题")
    print(f"✓ 结果已保存到 {output_file}")

if __name__ == "__main__":
    asyncio.run(process_hotpotqa_questions())

