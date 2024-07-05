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

        self.array = EngineArgs.from_args

    @bentoml.api
    def embed(self, sentences: list[str], model_name: str) -> t.Dict:
        raise ValueError("not implemented")

    @bentoml.api
    def rerank(self, query: str, docs: list[str], model_name: str) -> t.Dict:
        raise ValueError("not implemented")
