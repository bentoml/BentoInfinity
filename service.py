import bentoml
import typing as t
import numpy as np

@bentoml.service(
    traffic={"timeout": 120},
    resources={
        "gpu": 1,
        "memory": "8Gi",
    },
)
class INFINITY:
    """
    This class is inspired by the implementation shown in the infinity project.
    Source: https://github.com/michaelfeil/infinity
    """

    def __init__(self):
        from infinity_emb import AsyncEngineArray, EngineArgs

        self.ea = AsyncEngineArray.from_args(
            [
                EngineArgs(
                    model_name_or_path="BAAI/bge-small-en-v1.5",
                    engine="torch",
                    embedding_dtype="float32",
                    batch_size=8,
                ),
                EngineArgs(
                    model_name_or_path="mixedbread-ai/mxbai-rerank-xsmall-v1",
                    engine="torch",
                    batch_size=8,
                ),
                EngineArgs(
                    model_name_or_path="wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M",
                    engine="torch",
                    batch_size=8,
                ),
            ]
        )

    @bentoml.api
    async def embeddings(self, input: list[str], model: str) -> t.Dict:
        await self.ea.astart()
        embeddings, usage = await self.ea[model].embed(sentences=input)
        embeddings = np.array(embeddings).tolist()
        return {"embeddings": embeddings, "usage": usage}

    @bentoml.api
    async def rerank(self, query: str, docs: list[str], model: str) -> t.Dict:
        await self.ea.astart()
        rankings, usage = await self.ea[model].rerank(query=query, docs=docs)
        return {"rankings": rankings, "usage": usage}

    @bentoml.api
    async def imageembed(self, image_urls: list[str], model: str) -> t.Dict:
        await self.ea.astart()
        embeddings_image, usage = await self.ea[model].image_embed(images=image_urls)
        embeddings_image = np.array(embeddings_image).tolist()
        return {"embeddings": embeddings_image, "usage": usage}
