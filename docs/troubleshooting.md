# Troubleshooting Guide

## Table of Contents
- [Environment Setup Issues](#environment-setup-issues)
- [Package Installation Problems](#package-installation-problems)
- [Python Version Conflicts](#python-version-conflicts)
- [Import Errors](#import-errors)
- [Performance Issues](#performance-issues)
- [Common Error Messages](#common-error-messages)

## Environment Setup Issues

### Creating Conda Environment

**Problem**: Need to create a new conda environment with Python 3.12

**Solution**: Use the following commands:

```bash
# Create a new conda environment with Python 3.12
conda create -n llm_engineering python=3.12 -y

# Activate the environment
conda activate llm_engineering

# Install common packages for LLM engineering
conda install numpy pandas matplotlib scikit-learn jupyter -y
conda install pytorch torchvision torchaudio -c pytorch -y
pip install transformers datasets openai langchain

# If you have a requirements.txt file, install from it
pip install -r requirements.txt

# Verify installation
python --version
conda list
```

**Problem**: Environment activation fails

**Solution**:
```bash
# Initialize conda for your shell
conda init

# Restart terminal and try again
conda activate llm_engineering
```

## Package Installation Problems

### Issue: Package not found
**Error**: `PackageNotFoundError: Packages missing in current channels`

**Solution**:
```bash
# Try different channels
conda install -c conda-forge package_name
# Or use pip instead
pip install package_name
```

### Issue: Conflicting dependencies
**Error**: `UnsatisfiableError: The following specifications were found to be incompatible`

**Solution**:
```bash
# Create a fresh environment
conda create -n new_env python=3.12
conda activate new_env
# Install packages one by one to identify conflicts
```

## Python Version Conflicts

### Issue: Wrong Python version
**Problem**: Project requires specific Python version

**Solution**:
```bash
# Check current version
python --version

# Create environment with specific version
conda create -n project_env python=3.12.0
conda activate project_env
```

## Import Errors

### Issue: Module not found
**Error**: `ModuleNotFoundError: No module named 'module_name'`

**Solution**:
```bash
# Verify environment is activated
conda activate llm_engineering

# Install missing module
pip install module_name

# Check if module is in the right environment
pip list | grep module_name
```

### Issue: CUDA/GPU related errors
**Error**: CUDA version mismatch or GPU not available

**Solution**:
```bash
# Check CUDA version
nvidia-smi

# Install compatible PyTorch version
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

## Performance Issues

### Issue: Slow model loading
**Problem**: Models take too long to load

**Solutions**:
- Use model caching: `cache_dir="./model_cache"`
- Load models in half precision: `torch_dtype=torch.float16`
- Use CPU if GPU memory is insufficient

### Issue: Out of memory errors
**Error**: `CUDA out of memory` or `RuntimeError: out of memory`

**Solutions**:
```python
# Reduce batch size
batch_size = 1

# Use gradient checkpointing
model.gradient_checkpointing_enable()

# Clear GPU cache
torch.cuda.empty_cache()
```

## Common Error Messages

### SSL Certificate Errors
**Error**: `SSL: CERTIFICATE_VERIFY_FAILED`

**Solution**:
```bash
# Trust certificates
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org package_name
```

### Permission Errors
**Error**: `PermissionError: [WinError 5] Access is denied`

**Solution**:
```bash
# Run as administrator or use user flag
pip install --user package_name
```

### Conda Command Not Found
**Error**: `'conda' is not recognized as an internal or external command`

**Solution**:
```bash
# Add conda to PATH or use full path
C:\Users\username\anaconda3\Scripts\conda.exe activate llm_engineering
```

## Getting Help

### Check Environment Details
```bash
# List all environments
conda env list

# Show environment info
conda info

# List installed packages
conda list
pip list
```

### Debugging Commands
```bash
# Verbose installation
pip install -v package_name

# Check pip configuration
pip config list

# Show package information
pip show package_name
```

### Useful Resources
- [Conda Documentation](https://docs.conda.io/)
- [PyTorch Installation Guide](https://pytorch.org/get-started/locally/)
- [Transformers Troubleshooting](https://huggingface.co/docs/transformers/troubleshooting)

---

**Note**: Always ensure your conda environment is activated before installing packages or running scripts.