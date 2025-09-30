# LLM Engineering - WSL (Windows Subsystem for Linux) Setup Guide

## Complete setup instructions for Ubuntu 22.04 WSL

This guide will help you set up the LLM Engineering course environment on Windows Subsystem for Linux (WSL) with Ubuntu 22.04. The instructions have been tested and refined based on real-world setup experiences.

### Prerequisites

Before starting, ensure you have:
- Windows 10/11 with WSL2 enabled
- Ubuntu 22.04 LTS installed via WSL
- Basic familiarity with terminal/command line

---

## Part 1: System Dependencies

### 1.1 Update System Packages

First, update your Ubuntu system:

```bash
sudo apt update
sudo apt upgrade -y
```

### 1.2 Install Essential Dependencies

Install Python 3, pip, and other essential tools:

```bash
sudo apt install python3 python3-pip python3-venv git curl build-essential -y
```

### 1.3 Verify Python Installation

Check your Python version (should be 3.10+ for Ubuntu 22.04):

```bash
python3 --version
pip3 --version
```

---

## Part 2: Clone the Repository

### 2.1 Navigate to Your Projects Directory

```bash
# Create a projects directory if it doesn't exist
mkdir -p ~/Projects
cd ~/Projects
```

### 2.2 Clone the Repository

```bash
git clone https://github.com/ed-donner/llm_engineering.git
cd llm_engineering
```

---

## Part 3: Python Environment Setup

### 3.1 Create Virtual Environment

Create and activate a Python virtual environment:

```bash
python3 -m venv llms
source llms/bin/activate
```

**Note**: You should see `(llms)` in your terminal prompt, indicating the environment is active.

### 3.2 Upgrade pip

```bash
pip install --upgrade pip
```

### 3.3 Handle Corporate Network Issues (If Applicable)

If you're on a corporate network and encounter artifactory/proxy issues, you may need to:

1. **Use only public PyPI** (recommended):
```bash
pip install --index-url https://pypi.org/simple/ --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org -r requirements.txt
```

2. **Or reset pip configuration**:
```bash
mkdir -p ~/.config/pip
cat > ~/.config/pip/pip.conf << EOF
[global]
index-url = https://pypi.org/simple/
EOF
```

### 3.4 Install Dependencies

Install the required packages:

```bash
pip install -r requirements.txt
```

If you encounter dependency conflicts, install core packages individually:

```bash
# Core dependencies
pip install jupyter notebook ipykernel
pip install openai anthropic google-generativeai
pip install python-dotenv requests pandas numpy matplotlib
```

### 3.5 Install Additional Missing Dependencies

Install commonly missing dependencies:

```bash
pip install filelock huggingface-hub regex safetensors tokenizers networkx distro jiter pydantic pytz tzdata aiohttp click rich typer contourpy cycler fonttools kiwisolver pyparsing langsmith SQLAlchemy tiktoken jsonpatch tenacity
```

---

## Part 4: API Keys Setup (Optional but Recommended)

### 4.1 Create API Accounts

Set up accounts with AI providers:

1. **OpenAI**: https://platform.openai.com/
   - Minimum credit required (usually $5)
   - Create API key at: https://platform.openai.com/api-keys

2. **Anthropic** (for Claude): https://console.anthropic.com/
   - Create API key in console settings

3. **Google** (for Gemini): https://ai.google.dev/gemini-api
   - Create API key in Google AI Studio

4. **HuggingFace**: https://huggingface.co
   - Free account, create token in Settings > Access Tokens

### 4.2 Create Environment File

Create a `.env` file in your project root directory:

```bash
# Make sure you're in the project root
cd ~/Projects/llm_engineering

# Create the .env file
nano .env
```

Add your API keys to the file:

```env
OPENAI_API_KEY=sk-proj-your-key-here
GOOGLE_API_KEY=AIza-your-key-here  
ANTHROPIC_API_KEY=sk-ant-your-key-here
DEEPSEEK_API_KEY=your-deepseek-key-here
HF_TOKEN=your-huggingface-token-here
```

Save and exit nano:
- `Ctrl + O` (save)
- `Enter` (confirm)
- `Ctrl + X` (exit)

### 4.3 Verify Environment File

```bash
ls -la | grep .env
```

You should see the `.env` file listed.

---

## Part 5: Install Ollama (Local AI Models)

### 5.1 Install Ollama

```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

### 5.2 Start Ollama Service

```bash
# Start Ollama in background
ollama serve &
```

### 5.3 Pull a Model

```bash
# Pull Llama 3.2 (recommended for most machines)
ollama pull llama3.2

# For smaller machines, use the 1B parameter version
ollama pull llama3.2:1b
```

**Important**: Avoid `llama3.3` as it's too large (70B parameters) for most home computers.

---

## Part 6: WSL-Specific Configurations

### 6.1 Enable GUI Applications (Optional)

If you want to use GUI applications (like matplotlib plots), ensure WSLg is enabled:

```bash
# Check if WSLg is working
echo $DISPLAY
```

If empty, you may need to update WSL or enable WSLg in Windows.

### 6.2 Memory Configuration

For large model processing, you might want to increase WSL memory limits. Create or edit `%UserProfile%\.wslconfig` on Windows:

```ini
[wsl2]
memory=8GB
processors=4
swap=2GB
```

Then restart WSL:
```bash
# In Windows PowerShell (as Administrator)
wsl --shutdown
wsl
```

---

## Part 7: Test Your Setup

### 7.1 Start Jupyter

```bash
# Make sure you're in the project directory with virtual environment active
cd ~/Projects/llm_engineering
source llms/bin/activate
jupyter notebook
```

This should open your browser to `http://localhost:8888`.

### 7.2 Run Diagnostics

Test your setup with the provided diagnostics:

```bash
cd week1
python diagnostics.py
```

### 7.3 Test Basic Functionality

Create a new Jupyter notebook and run:

```python
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Test environment
print("Python environment test:")
print("✅ Python import successful")

# Test API keys (if configured)
if "OPENAI_API_KEY" in os.environ:
    print("✅ OpenAI API key loaded")
else:
    print("ℹ️ OpenAI API key not found (this is optional)")

if "GOOGLE_API_KEY" in os.environ:
    print("✅ Google API key loaded")
else:
    print("ℹ️ Google API key not found (this is optional)")

# Test libraries
try:
    import openai
    print("✅ OpenAI library imported")
except ImportError:
    print("❌ OpenAI library failed to import")

try:
    import google.generativeai as genai
    print("✅ Google GenAI library imported")
except ImportError:
    print("❌ Google GenAI library failed to import")

print("\nSetup verification complete!")
```

### 7.4 Test Ollama Integration

```python
from openai import OpenAI

# Test Ollama integration
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

try:
    response = client.chat.completions.create(
        model="llama3.2",
        messages=[{"role": "user", "content": "Hello! Can you confirm you're working?"}],
        max_tokens=50
    )
    print("✅ Ollama integration working!")
    print(f"Response: {response.choices[0].message.content}")
except Exception as e:
    print(f"❌ Ollama integration failed: {e}")
    print("Make sure Ollama is running: ollama serve")
```

---

## Part 8: Course Structure and Next Steps

### 8.1 Repository Organization

- **week1/**: Introduction and basic LLM interactions
- **week2/**: Advanced prompting and API usage
- **week3-8/**: Progressive projects building AI applications
- **community-contributions/**: Student projects and contributions

### 8.2 Alternative to Paid APIs

If you prefer not to use paid APIs, you can substitute OpenAI calls with Ollama:

```python
# Instead of this:
# client = OpenAI()

# Use this:
client = OpenAI(
    base_url="http://localhost:11434/v1", 
    api_key="ollama"
)

# And replace model names like "gpt-4o-mini" with "llama3.2"
```

### 8.3 Starting Your Journey

1. Navigate to `week1/day1.ipynb` to begin
2. Follow along with the course materials
3. Run each cell and experiment with the code
4. Complete the challenges and exercises

---

## Troubleshooting Common Issues

### Issue: Corporate Network/Proxy Problems
**Solution**: Use the pip configuration from Part 3.3, or contact your IT department for proxy settings.

### Issue: "Module not found" errors
**Solution**: Ensure your virtual environment is activated and reinstall the missing package:
```bash
source llms/bin/activate
pip install [missing-package-name]
```

### Issue: Jupyter notebook kernel not found
**Solution**: Install and configure the kernel:
```bash
pip install ipykernel
python -m ipykernel install --user --name=llms --display-name="LLM Engineering"
```

### Issue: Ollama connection refused
**Solution**: Make sure Ollama is running:
```bash
# Check if Ollama is running
ps aux | grep ollama

# If not running, start it:
ollama serve &
```

### Issue: Permission denied errors
**Solution**: Ensure proper file permissions and you're in the right directory:
```bash
# Fix permissions if needed
chmod +x ollama
# Make sure you're in the project directory
pwd  # Should show /home/[username]/Projects/llm_engineering
```

---

## Cost Management

- **OpenAI**: Monitor usage at https://platform.openai.com/usage
- **Anthropic**: Check costs at https://console.anthropic.com/settings/cost  
- **Google**: View billing at https://console.cloud.google.com/apis/api/generativelanguage.googleapis.com/cost

**Tip**: Use the cheapest models for learning:
- OpenAI: `gpt-4o-mini` instead of `gpt-4o`
- Anthropic: `claude-3-haiku-20240307`
- Or use Ollama for completely free local inference

---

## Getting Help

If you encounter issues:

1. **Check the troubleshooting section** above
2. **Review the error messages** carefully
3. **Search the course materials** in the `docs/` folder
4. **Contact the instructor**:
   - Email: ed@edwarddonner.com
   - LinkedIn: https://www.linkedin.com/in/eddonner/

## Next Steps

Once your setup is complete:

1. **Explore Week 1**: Start with `week1/day1.ipynb`
2. **Join the community**: Consider contributing to `community-contributions/`
3. **Experiment freely**: The best way to learn is by doing!

Happy learning! 🚀

---

*Last updated: December 2024*
*Based on Ubuntu 22.04 WSL setup experience*