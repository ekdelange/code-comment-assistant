# LLM Code Doc Gen

**LLM Code Doc Gen** is a simple Python application that automatically adds docstrings to Python functions in a specified directory using the Azure OpenAI API. It processes all `.py` files (excluding those ending with `_c.py`), generates or updates docstrings for functions lacking them, and writes the enhanced code to new files with a `_c.py` suffix. A summary `documentation.txt` is also generated.

---

## Features

* **Docstring Generation**: Uses Azure OpenAI to generate clear, concise Python docstrings.
* **Non-Destructive**: Original files remain untouched; enhanced files receive a `_c.py` suffix.
* **Simple CLI**: Accepts target directory as a command-line argument.
* **Documentation Summary**: Auto-generates `documentation.txt` listing processed files.

---

## Prerequisites

* Python 3.8 or newer
* Azure OpenAI resource with deployment
* An environment supporting Azure Key Credential authentication

---

## Installation

1. Clone this repository or copy the script into your project directory.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file in the root directory:

   ```env
   AZURE_OPENAI_ENDPOINT=https://<your-resource-name>.openai.azure.com/
   AZURE_OPENAI_API_KEY=<your-azure-openai-api-key>
   AZURE_DEPLOYMENT=<your-deployment-id>
   ```

---

## Usage

Run the script with the directory you want to process:

```bash
python auto_docstring.py <target_directory>
```

* `<target_directory>`: Path to the folder containing your Python files.

**Example**:

```bash
python auto_docstring.py ./my_project
```

After execution, each original file `foo.py` will have an enhanced counterpart `foo_c.py`, and a `documentation.txt` summary will appear in the same directory.

---

## Configuration

* **API Version**: The script uses Azure OpenAI API version `2024-12-01-preview`. Update in code if necessary.
* **Prompt Template**: Located at the top of the script (`PROMPT_TEMPLATE`). Modify to adjust docstring style or tone.
* **Temperature**: Set to `0.3` for consistent, accurate docstrings.

---

## Contributing

Contributions are welcome! Feel free to:

* Improve the prompt for specialized docstring styles.
* Extend the script to handle nested functions or classes.
* Add unit tests for validation.

---

## License

This project is released under the MIT License. See `LICENSE` for details.
