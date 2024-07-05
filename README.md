<div align="center">
    <h1 align="center">Self-host LLMs with Infinity and BentoML</h1>
</div>

This is a BentoML example project, showing you how to serve and deploy open-source embedding and reranking Models using [michaelfeil/Infinity](https://github.com/michaelfeil/infinity), which enables high-throughput deployments for clip, sentence-transformer, reranking and classification models.

See [here](https://github.com/bentoml/BentoML/tree/main/examples) for a full list of BentoML example projects.

## Prerequisites

- You have installed Python 3.9+ and `pip`. See the [Python downloads page](https://www.python.org/downloads/) to learn more.
- You have a basic understanding of key concepts in BentoML, such as Services. We recommend you read [Quickstart](https://docs.bentoml.com/en/1.2/get-started/quickstart.html) first.
- You have installed Docker as this example depends on a base Docker image `michaelf34/infinity` to set up Infinity.
- (Optional) We recommend you create a virtual environment for dependency isolation for this project. See the [Conda documentation](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) or the [Python documentation](https://docs.python.org/3/library/venv.html) for details.

## Set up the environment

Clone the repo.

```bash
git clone https://github.com/bentoml/BentoInfinity.git
cd BentoInfinity
```

Make sure you are in the `BentoInfinity` directory and mount it from your host machine (`${PWD}`) into a Docker container at `/BentoInfinity`. This means that the files and folders in the current directory are available inside the container at the `/BentoInfinity`.

```bash
docker run --runtime=nvidia --gpus all -v ${PWD}:/BentoInfinity -v ~/bentoml:/root/bentoml -p 7997:7997 --entrypoint /bin/bash -it --workdir /BentoInfinity michaelf34/infinity v2
```

Install dependencies.

```bash
cd multi-model-deployment
pip install -r requirements.txt
```

## Download the model

Run the script to download Llama 3 to the BentoML [Model Store](https://docs.bentoml.com/en/latest/guides/model-store.html).

```bash
python import_model.py
```

## Run the BentoML Service

We have defined a BentoML Service in `service.py`. Run `bentoml serve` in your project directory to start the Service.

```bash
$ bentoml serve .
2024-06-06T10:31:45+0000 [INFO] [cli] Starting production HTTP BentoServer from "service:TGI" listening on http://localhost:3000 (Press CTRL+C to quit)
```

The server is now active at [http://localhost:7997](http://localhost:7997/). You can interact with it using the Swagger UI or in other different ways.

<details>

<summary>CURL</summary>

```bash
curl -X 'POST' \
  'http://localhost:7997/embeddings' \
  -H 'Content-Type: application/json' \
  -d '{
  "input": ["Explain superconductors like I am five years old"],
  "model": "BAAI/bge-small-en-v1.5"
}'
```

</details>

<details>

<summary>Python client</summary>

```python
import bentoml

with bentoml.SyncHTTPClient("http://localhost:7997") as client:
    response_generator = client.embed( # TODO: Verify this works
        input="Explain superconductors like I am five years old",
        model= "BAAI/bge-small-en-v1.5"
    )
    for response in response_generator:
        print(response, end='')
```

</details>

## Deploy to BentoCloud

After the Service is ready, you can deploy the application to BentoCloud. Make sure you have [logged in to BentoCloud](https://docs.bentoml.com/en/latest/bentocloud/how-tos/manage-access-token.html), then run the following command to deploy it.

```bash
bentoml deploy .
```

Once the application is up and running on BentoCloud, you can access it via the exposed URL.

**Note**: For custom deployment in your own infrastructure, use [BentoML to generate an OCI-compliant image](https://docs.bentoml.com/en/latest/guides/containerization.html).
