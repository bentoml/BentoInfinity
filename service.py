import bentoml
import typing as t

@bentoml.service(
    traffic={"timeout": 120},
    resources={
        "gpu": 1,
        "memory": "8Gi",
    },
)
class InfinityBento:
    """
    This class is inspired by the implementation shown in the infinity project.
    Source: https://github.com/michaelfeil/infinity
    """

    def __init__(self):
        from infinity_emb import AsyncEngineArray, EngineArgs

        self.array = AsyncEngineArray.from_args([
          EngineArgs(model_name_or_path = "BAAI/bge-small-en-v1.5", engine="torch", embedding_dtype="float32", dtype="auto"),
          EngineArgs(model_name_or_path = "mixedbread-ai/mxbai-rerank-xsmall-v1", engine="torch"),
          EngineArgs(model_name_or_path = "wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M", engine="torch")
        ])

    @bentoml.api
    async def embed(self, sentences: list[str], model_name: str) -> t.Dict:
        await self.array.astart()
        embeddings, usage = await self.array[model_name].embed(sentences=sentences)
        return {"embeddings_image": embeddings, "usage": usage}

    @bentoml.api
    async def rerank(self, query: str, docs: list[str], model_name: str) -> t.Dict:
        await self.array.astart()
        rankings, usage = await self.array[model_name].rerank(query=query, docs=docs)
        return {"rankings": rankings, "usage": usage}

    @bentoml.api
    async def embed(self, images: list[str], model_name: str) -> t.Dict:
        await self.array.astart()
        embeddings_image, usage = await self.array[model_name].image_embed(images=images)
        return {"embeddings": embeddings_image, "usage": usage}
    
