import asyncio
from sciverse import AgentToolsClient  # 把 AsyncSciverseClient 换成这个

async def verify_doi_availability():
    async with AgentToolsClient() as c:
        # 1. 基础检索：检查返回结果中的 doi 字段
        print(">>> 步骤 1：基础检索")
        # result = await c.search_papers(
        #     query="Attention is All You Need",
        #     page_size=5
        # )
        result = await c.search_papers(query="Probing the Kitaev honeycomb model on a neutral-atom quantum computer", page_size=5)
        print(f"total hits: {result.get('total', 0)}")
        print(f"raw keys: {list(result.keys())}")
        print(f"raw hits: {result.get('hits', [])[:2]}")  # 只打印前两条
        for paper in result.get("hits", []):
            title = paper.get("title", "N/A")
            doi = paper.get("doi", "❌ 未返回")
            print(f"  - {title[:60]}... | DOI: {doi}")

        # # 2. 反向验证：用 DOI 精确过滤
        # print("\n>>> 步骤 2：用 DOI 过滤验证")
        # target_doi = "10.1038/s41586-026-10898-6"
        # doi_result = await c.search_papers(
        #     filters_advanced=[
        #         {
        #             "field": "doi",
        #             "operator": "FILTER_OP_EQ",
        #             "value": target_doi
        #         }
        #     ],
        #     page_size=5
        # )
        # hits = doi_result.get("hits", [])
        # if hits:
        #     print(f"  ✅ 通过 DOI 成功找到：{hits[0].get('title', 'N/A')[:60]}...")
        # else:
        #     print(f"  ⚠️ 未找到 DOI 为 {target_doi} 的论文")

asyncio.run(verify_doi_availability())