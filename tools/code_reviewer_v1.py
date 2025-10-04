"""
AI-Powered Code Review Tool
Supports GitHub URLs (PR, repos, files) and provides comprehensive code analysis
Uses OpenAI, Anthropic, Google Gemini, and local Ollama models

Author: Stephen Nacion
GitHub: https://github.com/stephennacion06
License: CC BY 4.0 (Creative Commons Attribution 4.0 International)
https://creativecommons.org/licenses/by/4.0/

You are free to:
- Share: copy and redistribute the material in any medium or format
- Adapt: remix, transform, and build upon the material for any purpose

Under the following terms:
- Attribution: You must give appropriate credit, provide a link to the license,
  and indicate if changes were made.
"""

import os
import re
import requests
from typing import Optional, Dict, List
from dotenv import load_dotenv
from openai import OpenAI
import anthropic
from google import genai
from google.genai import types
import gradio as gr
import time

# Load environment variables
load_dotenv(override=True)

# API Keys
openai_api_key = os.getenv('OPENAI_API_KEY')
anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
google_api_key = os.getenv('GOOGLE_API_KEY')

# Initialize clients
available_models = []

# OpenAI
openai_client = None
if openai_api_key:
    openai_client = OpenAI()
    available_models.extend(["GPT-4o-mini", "GPT-4o"])
    print(f"✓ OpenAI API Key loaded: {openai_api_key[:8]}...")

# Anthropic
claude_client = None
if anthropic_api_key:
    claude_client = anthropic.Anthropic()
    available_models.extend(["Claude-3-Haiku", "Claude-3.5-Sonnet"])
    print(f"✓ Anthropic API Key loaded: {anthropic_api_key[:7]}...")

# Google Gemini
gemini_client = None
if google_api_key:
    gemini_client = genai.Client(api_key=google_api_key)
    available_models.extend(["Gemini-2.0-Flash", "Gemini-1.5-Flash"])
    print(f"✓ Google API Key loaded: {google_api_key[:8]}...")

# Ollama - check for local models
try:
    ollama_response = requests.get("http://localhost:11434/api/tags", timeout=2)
    if ollama_response.status_code == 200:
        ollama_models = ollama_response.json().get('models', [])
        for model in ollama_models:
            model_name = model.get('name', '')
            if model_name:
                available_models.append(f"Ollama-{model_name}")
        if ollama_models:
            print(f"✓ Ollama models found: {len(ollama_models)}")
except:
    print("✗ Ollama not available (is it running?)")

if not available_models:
    print("⚠ WARNING: No models available! Please check your .env file or start Ollama")
else:
    print(f"\n📋 Available models: {', '.join(available_models)}")

# System message for code review
REVIEW_SYSTEM_MESSAGE = """You are an expert code reviewer with deep knowledge of software engineering best practices, security, and performance optimization.

When reviewing code, provide a comprehensive analysis covering:

1. **Code Quality**: Readability, maintainability, and adherence to best practices
2. **Security**: Potential vulnerabilities, input validation, and security risks
3. **Performance**: Efficiency issues, optimization opportunities
4. **Bugs & Issues**: Logical errors, edge cases, potential runtime errors
5. **Documentation**: Code comments, docstrings, and clarity
6. **Suggestions**: Specific, actionable improvements with code examples

Format your response in clear Markdown with sections and severity levels (Critical/Major/Minor).
Be constructive, specific, and provide code examples for suggestions."""


class GitHubCodeFetcher:
    """Fetch code from various GitHub URL formats"""
    
    def __init__(self, url: str):
        self.url = url
        self.github_token = os.getenv('GITHUB_TOKEN')  # Optional, for rate limits
    
    def fetch(self) -> Dict[str, str]:
        """Main method to fetch code based on URL type"""
        if '/pull/' in self.url:
            return self._fetch_pull_request()
        elif '/blob/' in self.url:
            return self._fetch_file()
        elif 'github.com/' in self.url:
            return self._fetch_repository()
        else:
            raise ValueError("Invalid GitHub URL format")
    
    def _get_headers(self) -> Dict[str, str]:
        """Get headers for GitHub API requests"""
        headers = {'Accept': 'application/vnd.github.v3+json'}
        if self.github_token:
            headers['Authorization'] = f'token {self.github_token}'
        return headers
    
    def _fetch_pull_request(self) -> Dict[str, str]:
        """Fetch PR diff and metadata"""
        # Extract owner, repo, and PR number from URL
        # Format: https://github.com/owner/repo/pull/123
        match = re.search(r'github\.com/([^/]+)/([^/]+)/pull/(\d+)', self.url)
        if not match:
            raise ValueError("Invalid Pull Request URL")
        
        owner, repo, pr_number = match.groups()
        api_url = f'https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}'
        
        response = requests.get(api_url, headers=self._get_headers())
        response.raise_for_status()
        pr_data = response.json()
        
        # Fetch the diff
        diff_url = pr_data['diff_url']
        diff_response = requests.get(diff_url)
        diff_response.raise_for_status()
        
        return {
            'title': pr_data['title'],
            'description': pr_data.get('body', 'No description'),
            'code': diff_response.text,
            'type': 'Pull Request'
        }
    
    def _fetch_file(self) -> Dict[str, str]:
        """Fetch a single file from GitHub"""
        # Format: https://github.com/owner/repo/blob/branch/path/to/file
        match = re.search(r'github\.com/([^/]+)/([^/]+)/blob/([^/]+)/(.+)', self.url)
        if not match:
            raise ValueError("Invalid file URL")
        
        owner, repo, branch, file_path = match.groups()
        api_url = f'https://api.github.com/repos/{owner}/{repo}/contents/{file_path}?ref={branch}'
        
        response = requests.get(api_url, headers=self._get_headers())
        response.raise_for_status()
        file_data = response.json()
        
        # Decode content from base64
        import base64
        content = base64.b64decode(file_data['content']).decode('utf-8')
        
        return {
            'title': file_path,
            'description': f'File from {owner}/{repo}',
            'code': content,
            'type': 'Single File'
        }
    
    def _fetch_repository(self) -> Dict[str, str]:
        """Fetch key files from a repository"""
        # Format: https://github.com/owner/repo
        match = re.search(r'github\.com/([^/]+)/([^/]+?)(?:\.git)?/?$', self.url)
        if not match:
            raise ValueError("Invalid repository URL")
        
        owner, repo = match.groups()
        
        # Fetch common important files
        files_to_check = [
            'README.md', 'main.py', 'app.py', 'index.js', 'package.json',
            'requirements.txt', 'setup.py', 'Dockerfile'
        ]
        
        code_parts = []
        for filename in files_to_check:
            try:
                api_url = f'https://api.github.com/repos/{owner}/{repo}/contents/{filename}'
                response = requests.get(api_url, headers=self._get_headers())
                if response.status_code == 200:
                    import base64
                    content = base64.b64decode(response.json()['content']).decode('utf-8')
                    code_parts.append(f"=== {filename} ===\n{content}\n")
            except:
                continue
        
        if not code_parts:
            raise ValueError("Could not fetch any files from repository")
        
        return {
            'title': f'{owner}/{repo}',
            'description': f'Repository analysis',
            'code': '\n\n'.join(code_parts),
            'type': 'Repository'
        }


def stream_openai(prompt: str, model: str):
    """Stream response from OpenAI models"""
    messages = [
        {"role": "system", "content": REVIEW_SYSTEM_MESSAGE},
        {"role": "user", "content": prompt}
    ]
    
    model_map = {
        "GPT-4o-mini": "gpt-4o-mini",
        "GPT-4o": "gpt-4o"
    }
    
    stream = openai_client.chat.completions.create(
        model=model_map.get(model, "gpt-4o-mini"),
        messages=messages,
        stream=True,
        temperature=0.3
    )
    
    result = ""
    for chunk in stream:
        result += chunk.choices[0].delta.content or ""
        yield result


def stream_claude(prompt: str, model: str):
    """Stream response from Claude models"""
    model_map = {
        "Claude-3-Haiku": "claude-3-haiku-20240307",
        "Claude-3.5-Sonnet": "claude-3-5-sonnet-20241022"
    }
    
    result = claude_client.messages.stream(
        model=model_map.get(model, "claude-3-haiku-20240307"),
        max_tokens=4000,
        temperature=0.3,
        system=REVIEW_SYSTEM_MESSAGE,
        messages=[{"role": "user", "content": prompt}],
    )
    
    response = ""
    with result as stream:
        for text in stream.text_stream:
            response += text or ""
            yield response


def stream_gemini(prompt: str, model: str):
    """Stream response from Gemini models"""
    time.sleep(0.5)  # Rate limiting
    
    model_map = {
        "Gemini-2.0-Flash": "gemini-2.0-flash-exp",
        "Gemini-1.5-Flash": "gemini-1.5-flash"
    }
    
    response = gemini_client.models.generate_content_stream(
        model=model_map.get(model, "gemini-2.0-flash-exp"),
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=REVIEW_SYSTEM_MESSAGE,
            temperature=0.3
        )
    )
    
    result = ""
    for chunk in response:
        result += chunk.text or ""
        yield result


def stream_ollama(prompt: str, model: str):
    """Stream response from local Ollama models"""
    # Extract model name (remove "Ollama-" prefix)
    ollama_model = model.replace("Ollama-", "")
    
    url = "http://localhost:11434/api/generate"
    full_prompt = f"{REVIEW_SYSTEM_MESSAGE}\n\nUser request: {prompt}"
    
    data = {
        "model": ollama_model,
        "prompt": full_prompt,
        "stream": True
    }
    
    response = requests.post(url, json=data, stream=True)
    result = ""
    
    for line in response.iter_lines():
        if line:
            import json
            chunk = json.loads(line)
            if 'response' in chunk:
                result += chunk['response']
                yield result


def perform_code_review(github_url: str, model: str, custom_focus: str = ""):
    """Main function to perform code review"""
    
    # Validate inputs
    if not github_url or not github_url.strip():
        yield "❌ Please enter a GitHub URL"
        return
    
    if not model:
        yield "❌ Please select a model"
        return
    
    # Initial status
    yield "🔍 Fetching code from GitHub...\n"
    
    try:
        # Fetch code from GitHub
        fetcher = GitHubCodeFetcher(github_url)
        code_data = fetcher.fetch()
        
        yield f"✓ Fetched {code_data['type']}: **{code_data['title']}**\n\n🤖 Analyzing code with {model}...\n\n"
        
        # Build prompt
        prompt = f"""Please review the following code:

**Type**: {code_data['type']}
**Title**: {code_data['title']}
**Description**: {code_data['description']}

"""
        if custom_focus:
            prompt += f"**Special Focus**: {custom_focus}\n\n"
        
        prompt += f"""**Code**:
```
{code_data['code'][:15000]}  # Limit to avoid token issues
```

Provide a comprehensive code review with specific, actionable feedback."""

        # Route to appropriate model
        if model.startswith("GPT"):
            if not openai_client:
                yield "❌ OpenAI not configured. Please check your .env file."
                return
            yield from stream_openai(prompt, model)
        
        elif model.startswith("Claude"):
            if not claude_client:
                yield "❌ Anthropic not configured. Please check your .env file."
                return
            yield from stream_claude(prompt, model)
        
        elif model.startswith("Gemini"):
            if not gemini_client:
                yield "❌ Google Gemini not configured. Please check your .env file."
                return
            yield from stream_gemini(prompt, model)
        
        elif model.startswith("Ollama"):
            yield from stream_ollama(prompt, model)
        
        else:
            yield f"❌ Unknown model: {model}"
    
    except Exception as e:
        yield f"\n\n❌ **Error**: {str(e)}\n\nPlease check:\n- URL is valid\n- GitHub is accessible\n- For private repos, set GITHUB_TOKEN in .env"


# Create Gradio Interface
def create_interface():
    """Create the Gradio UI"""
    
    with gr.Blocks(title="AI Code Reviewer") as demo:
        gr.Markdown("""
        # 🔍 AI-Powered Code Review Tool

        Review code from GitHub using multiple AI models (OpenAI, Anthropic, Google, Ollama)
        
        **Supported URLs:**
        - Pull Requests: `https://github.com/owner/repo/pull/123`
        - Files: `https://github.com/owner/repo/blob/main/file.py`
        - Repositories: `https://github.com/owner/repo`

        **👨‍💻 Created by:** [Stephen Nacion](https://github.com/stephennacion06)  
        **📜 License:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) - Free to use with attribution
        """)
        
        with gr.Row():
            with gr.Column(scale=2):
                url_input = gr.Textbox(
                    label="GitHub URL",
                    placeholder="https://github.com/owner/repo/pull/123",
                    lines=1
                )
            with gr.Column(scale=1):
                model_dropdown = gr.Dropdown(
                    choices=available_models,
                    label="Select AI Model",
                    value=available_models[0] if available_models else None
                )
        
        custom_focus = gr.Textbox(
            label="Custom Review Focus (Optional)",
            placeholder="e.g., 'Focus on security vulnerabilities' or 'Check for performance issues'",
            lines=2
        )
        
        submit_btn = gr.Button("🚀 Review Code", variant="primary", size="lg")
        
        output = gr.Markdown(label="Code Review Results")
        
        # Examples
        gr.Examples(
            examples=[
                ["https://github.com/gradio-app/gradio/pull/1", available_models[0] if available_models else "", ""],
                ["https://github.com/openai/openai-python/blob/main/README.md", available_models[0] if available_models else "", "Focus on documentation quality"],
            ],
            inputs=[url_input, model_dropdown, custom_focus]
        )
        
        submit_btn.click(
            fn=perform_code_review,
            inputs=[url_input, model_dropdown, custom_focus],
            outputs=output
        )
    
    return demo


if __name__ == "__main__":
    if not available_models:
        print("\n⚠️  WARNING: No AI models available!")
        print("Please either:")
        print("1. Add API keys to your .env file (OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY)")
        print("2. Start Ollama with: ollama serve")
        print("3. Pull an Ollama model with: ollama pull llama2")
    
    demo = create_interface()
    demo.launch(
        inbrowser=True,
        share=False
    )