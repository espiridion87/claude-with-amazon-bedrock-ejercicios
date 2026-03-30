# Chat Multi-turn

Ejercicio que demuestra cómo implementar una conversación **multi-turn** con Claude,
manteniendo el historial de mensajes para que el modelo recuerde el contexto de la sesión.

## Descripción

La clase `ChatBot` acumula los mensajes en una lista (`self.messages`) y los envía
completos en cada llamada, lo que permite al modelo razonar sobre el contexto previo.

## Implementación actual (Ollama)

> **Nota:** Este ejercicio no cuenta con acceso a Amazon Bedrock, por lo que la función
> `_process` fue implementada utilizando [Ollama](https://ollama.com/) como sustituto
> local del servicio.

```python
# src/claude_with_amazon_bedrock_ejercicio_chat_multi_input/main.py

import ollama

def _process(self):
    response = ollama.chat(
        model='qwen2.5',
        messages=self.messages   # historial completo en cada llamada
    )
    answer: str = response['message']['content']
    self._chat_as(ASSISTANT_ROLE, answer)
    return answer
```

Ollama expone una API local compatible con la estructura de mensajes de Anthropic
(`role` + `content`), por lo que el cambio a Bedrock es directo.

## Implementación equivalente con Amazon Bedrock (boto3)

El siguiente ejemplo muestra cómo reemplazar `_process` para usar el cliente
`boto3` con el servicio `bedrock-runtime`:

```python
import boto3
import json

MODEL_ID = "anthropic.claude-3-5-sonnet-20241022-v2:0"

bedrock = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-east-1",   # ajusta la región según tu configuración
)

def _process(self):
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1024,
        "messages": self.messages,   # mismo historial acumulado
    }

    response = bedrock.invoke_model(
        modelId=MODEL_ID,
        body=json.dumps(body),
    )

    result = json.loads(response["body"].read())
    answer: str = result["content"][0]["text"]
    self._chat_as(ASSISTANT_ROLE, answer)
    return answer
```

### Diferencias clave respecto a la versión Ollama

| Aspecto | Ollama | Amazon Bedrock (boto3) |
|---------|--------|------------------------|
| Cliente | `ollama.chat()` | `bedrock.invoke_model()` |
| Modelo | `qwen2.5` (local) | `anthropic.claude-3-5-sonnet-...` |
| Autenticación | ninguna | credenciales AWS (`~/.aws`) |
| Formato de respuesta | `response['message']['content']` | `result['content'][0]['text']` |
| Campo extra requerido | — | `anthropic_version` en el body |

## Instalación y ejecución

```bash
poetry install
poetry run chat
```
