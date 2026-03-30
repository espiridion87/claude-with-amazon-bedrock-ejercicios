# Claude with Amazon Bedrock — Ejercicios

Este repositorio contiene ejercicios prácticos del curso
[Claude in Amazon Bedrock](https://anthropic-partners.skilljar.com/claude-in-amazon-bedrock).

## Estructura

Cada subdirectorio corresponde a un ejercicio independiente con su propio entorno
de dependencias gestionado con [Poetry](https://python-poetry.org/).

| Ejercicio | Descripción |
|-----------|-------------|
| [chat-multi-input](./chat-multi-input/) | Conversación multi-turn con manejo de historial |

## Requisitos generales

- Python >= 3.14
- Poetry >= 2.0

## Cómo ejecutar un ejercicio

```bash
cd <nombre-ejercicio>
poetry install
poetry run <script>
```
