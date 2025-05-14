# LLM Code Doc Gen

**LLM Code Doc Gen** is a Python application that automatically adds documentation comments to code files using various LLM providers. It supports multiple programming languages and can add both module-level and function-level documentation while preserving the original files.

## Features

* **Multiple LLM Providers**: Supports Azure OpenAI, OpenAI, and Hugging Face models
* **Multiple Languages**: Supports Python and C++ documentation styles
* **Module Documentation**: Adds comprehensive module/file-level documentation
* **Function Documentation**: Generates detailed docstrings for functions and methods
* **Non-Destructive**: Original files remain untouched; enhanced files receive a `_c` suffix
* **Simple CLI**: Accepts target directory as a command-line argument
* **Documentation Summary**: Auto-generates `log.txt` listing processed files

## Prerequisites

* Python 3.8 or newer
* Access to one of the supported LLM providers:
  - Azure OpenAI
  - OpenAI
  - Hugging Face

## Installation

1. Clone this repository or copy the script into your project directory.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file in the root directory with your preferred LLM provider configuration:

   For Azure OpenAI:
   ```env
   LLM_PROVIDER=azure
   AZURE_OPENAI_ENDPOINT=https://<your-resource-name>.openai.azure.com/
   AZURE_OPENAI_API_KEY=<your-azure-openai-api-key>
   AZURE_OPENAI_DEPLOYMENT=<your-deployment-id>
   AZURE_OPENAI_API_VERSION=2024-12-01-preview
   ```

   For OpenAI:
   ```env
   LLM_PROVIDER=openai
   OPENAI_API_KEY=<your-openai-api-key>
   OPENAI_MODEL=gpt-3.5-turbo
   ```

   For Hugging Face:
   ```env
   LLM_PROVIDER=huggingface
   HF_API_TOKEN=<your-huggingface-api-token>
   HF_MODEL=<model-name>
   ```

## Usage

You can use the app via the command line or the graphical user interface (GUI).

### GUI Usage (Recommended)

1. Run the script without arguments:

   ```bash
   python auto_docstring.py
   ```

2. In the GUI:
   - Select the programming language from the dropdown menu (Python, C++, Java)
   - Click "Browse Directory" to choose the folder containing your code files
   - Click "Run" to start processing
   - The log window will display a summary of processed files
   - Click "Close" to exit

### Command-Line Usage

You can still run the script with a target directory as an argument (the language will be taken from the .env file, but GUI selection is recommended):

```bash
python auto_docstring.py <target_directory>
```

* `<target_directory>`: Path to the folder containing your code files.

After execution (in either mode):
- Each original file (e.g., `example.py`) will have an enhanced counterpart (e.g., `example_c.py`)
- A `log.txt` summary will appear in the same directory
- Both module-level and function-level documentation will be added

## Configuration

* **LLM Provider**: Set via `LLM_PROVIDER` in the `.env` file (`azure`, `openai`, or `huggingface`)
* **API Keys and Endpoints**: Set the relevant keys and endpoints for your provider in `.env`
* **Language Selection**: Now handled in the GUI (no need to set `TARGET_LANGUAGE` in `.env`)
* **Temperature**: Set to 0 for consistent, accurate documentation (hardcoded in script)
* **Prompt Templates**: Located at the top of the script (`MODULE_DOCSTRING_TEMPLATE` and `FUNCTION_DOCSTRING_TEMPLATE`)

## Contributing

Contributions are welcome! Feel free to:

* Add support for additional programming languages
* Implement new LLM providers
* Improve documentation templates
* Add unit tests
* Enhance error handling

## License

This project is released under the MIT License. See `LICENSE` for details.
