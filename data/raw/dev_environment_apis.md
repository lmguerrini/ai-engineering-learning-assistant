# Topic: Development Environment and APIs
# Sprint: 1
# Part: 2
# Tags: python, javascript, vscode, agentic-coding, api, openrouter, hugging-face, local-llm, ai-code-assistants, jupyter

## Overview
This section covers the practical development environment for building AI applications.

It includes programming setup, code editors, AI coding assistants, Jupyter notebooks, API-based interaction with large language models (LLMs), and alternative approaches such as no-code tools and local LLM development.

The focus is on how to efficiently build, test, and deploy AI systems using modern tools and APIs.

## Key Concepts

- **Development Environment**  
  The full set of tools required to build AI applications, including programming languages, editors, package management, and API access.

- **Python and JavaScript Tracks**  
  Two primary development paths for building LLM applications.

- **VS Code**  
  A lightweight and extensible code editor with strong support for Python and JavaScript via extensions.

- **Agentic Coding**  
  A development approach where AI tools assist in writing, debugging, and improving code.

- **AI-Powered IDEs**  
  Development environments enhanced with AI features such as autocomplete, refactoring, and error detection.

- **AI Code Assistants**  
  Tools like GitHub Copilot or Claude Code that help generate and improve code.

- **Jupyter Notebooks**  
  Interactive environments for experimentation and step-by-step execution of code.

- **API Access to LLMs**  
  The process of interacting with hosted models using API keys and SDKs.

- **OpenRouter**  
  A unified OpenAI-compatible API gateway that provides access to multiple model providers through one interface.

- **Hugging Face Ecosystem**  
  A platform for exploring and using pre-trained models across multiple AI domains.

- **Local LLM Development**  
  Running open-source models locally instead of using cloud APIs.

- **No-Code AI Platforms**  
  Tools that allow users to build AI workflows without writing code.

- **Environment Variables and .env Files**  
  A standard practice for storing configuration and secrets outside of source code. Libraries like python-dotenv or pydantic-settings load variables from .env files at runtime.

- **Virtual Environments**  
  Isolated Python environments (venv, conda) that prevent dependency conflicts between projects. Essential for reproducible AI application development.

- **Package Management**  
  Tools like pip, poetry, or npm that manage project dependencies. Requirements files (requirements.txt, pyproject.toml) ensure consistent installations across environments.

- **SDK (Software Development Kit)**  
  Client libraries provided by model providers (e.g., openai Python package) that simplify API interaction with type hints, error handling, and retry logic.

## How It Works

### 1. Set Up the Development Environment
- Choose a programming track (Python or JavaScript).
- Install a code editor (VS Code recommended).
- Configure extensions and dependencies.
- Create a virtual environment to isolate project dependencies.
- Use a requirements file to pin dependency versions for reproducibility.

### 2. Use AI-Assisted Coding Tools
- Integrate AI tools such as Copilot or Claude Code.
- Use them for:
  - autocomplete
  - debugging
  - explanations
  - refactoring
  - boilerplate generation
- Always review AI-generated code for correctness and security before committing.

### 3. Experiment and Develop
- Use Jupyter notebooks for exploration and testing.
- Use IDEs for structured application development.
- Transition from notebooks to modules when code matures.

### 4. Call LLMs via API
- Obtain an API key (e.g., OpenRouter).
- Store it securely in environment variables or .env files.
- Use an OpenAI-compatible SDK.
- Send requests with prompts and parameters.
- Select models by changing model identifiers.
- Handle API errors gracefully with retry logic and timeouts.
- Monitor rate limits and implement backoff strategies.

### 5. Choose the Right Model
- Text models → generation and chat
- Embedding models → retrieval and RAG
- Multimodal models → speech, image, video
- Consider cost per token, latency, and quality trade-offs when selecting a model.

### 6. Use Model Ecosystems
- Explore models via Hugging Face.
- Test different architectures and capabilities.
- Use model cards to understand training data, limitations, and intended use.

### 7. Run Models Locally
- Select an open-source model based on hardware constraints.
- Use tools like LM Studio or Ollama.
- Run models locally for:
  - privacy
  - offline usage
  - cost control
  - low latency
- Quantized models (4-bit, 8-bit) reduce memory requirements significantly.

### 8. Secure API Key Management
- Never commit API keys to version control.
- Use .env files with .gitignore to keep secrets out of repositories.
- Use pydantic-settings or python-dotenv to load environment variables.
- Rotate keys periodically and use separate keys for development and production.
- Consider using secret management services (AWS Secrets Manager, HashiCorp Vault) for production deployments.

### 9. Project Structure Best Practices
- Separate configuration (config.py) from application logic.
- Use a src/ layout for modular code organization.
- Keep tests in a dedicated tests/ directory.
- Use logging (e.g., Loguru) instead of print statements for debugging.
- Document setup instructions in README.md for reproducibility.

## Example

### OpenRouter API Example
- Store API key in `OPENROUTER_API_KEY`.
- Use an OpenAI-compatible client.
- Change only the model ID to switch providers.
- Example models:
  - `openai/...`
  - `anthropic/...`
  - `google/...`

### Environment Setup Example
- Create virtual environment: `python -m venv .venv`
- Activate: `source .venv/bin/activate`
- Install dependencies: `pip install -r requirements.txt`
- Copy env template: `cp .env.example .env`
- Add API keys to .env file

### No-Code Example
- Generate HTML with an LLM from a screenshot.
- Copy into a file and open locally.
- Useful for rapid prototyping.

### Local LLM Example
- Run a 7B–13B model locally using LM Studio or Ollama.
- Suitable for privacy-sensitive tasks.
- Use quantized versions for lower memory usage.

## When to Use

- **VS Code and AI Assistants**
  - When building production-ready AI applications.
  - When productivity and iteration speed matter.

- **Jupyter Notebooks**
  - For experimentation and learning.
  - For testing prompts and APIs.

- **API-Based LLM Access**
  - When using powerful hosted models.
  - When building scalable applications.
  - When you need the latest model capabilities.

- **Hugging Face**
  - For experimentation with different models.
  - For fine-tuning and custom model development.

- **No-Code Tools**
  - For simple workflows and rapid prototyping.

- **Local LLMs**
  - When privacy, latency, or cost are critical.
  - When working offline or with sensitive data.

## Common Mistakes

- **Over-relying on AI tools**
  - Not understanding the underlying code or APIs.

- **Expecting full applications from LLMs**
  - LLMs assist development but do not replace engineering.

- **Poor API key management**
  - Exposing keys or committing them to version control. Use .env files and .gitignore.

- **Wrong model selection**
  - Using expensive models unnecessarily. Start with smaller, cheaper models.

- **Ignoring hardware constraints**
  - Running models locally without sufficient resources.

- **Assuming no-code is trivial**
  - No-code tools still require understanding and practice.

- **Overcomplicating local setups**
  - Avoid unnecessary manual configuration when tools exist.

- **Not using virtual environments**
  - Dependency conflicts between projects cause hard-to-debug issues.

- **Skipping error handling for API calls**
  - API calls can fail due to rate limits, network issues, or invalid inputs. Always implement retry logic.

## Best Practices

- Use virtual environments for every project.
- Pin dependency versions in requirements files.
- Store all secrets in .env files, never in source code.
- Use pydantic-settings for type-safe configuration loading.
- Implement structured logging from the start.
- Write tests that mock API calls to avoid costs and flakiness.
- Document setup steps clearly in README.md.
- Use .env.example as a template for required environment variables.

## Related Concepts

- Prompt Engineering  
- Retrieval-Augmented Generation (RAG)  
- Embeddings  
- Agentic Coding  
- OpenAI-Compatible APIs  
- Local Open-Source Models  
- No-Code AI Platforms  
- Privacy and Compliance  
- Configuration Management  
- Secret Management and Security  
- Testing and Mocking API Calls  
- Logging and Observability
