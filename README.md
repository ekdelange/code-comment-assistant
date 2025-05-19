# Comment Assistant

A versatile documentation generator that automatically adds docstrings and comments to code files using an LLM (Large Language Model). Supports Python, C++, and C. Choose from Azure OpenAI, OpenAI, or Hugging Face as the LLM provider.

## Features
- **Automatic docstring/comment generation** for Python, C++, and C source files
- **Supports multiple LLM providers**: Azure OpenAI, OpenAI, Hugging Face
- **Easy-to-use GUI**: Select language, directory, and start processing with a click
- **Log window**: View processed files and errors in real time
- **Original files are preserved**: Documented versions are saved with a `_c` suffix

## Installation
1. Clone this repository.
2. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```
3. Copy `.env.example` to `.env` and fill in your LLM provider credentials.

## Usage
1. Run the app:
   ```powershell
   python comment-assistant.py
   ```
2. In the GUI:
   - Select the programming language (Python, C++, or C)
   - Choose the directory containing your code files
   - Click **Start** to process files
   - View the log for results and errors
   - Click **Close** to exit

## Supported File Types
- **Python**: `.py`
- **C++**: `.cpp`, `.hpp`, `.h`, `.cc`, `.cxx`
- **C**: `.c`, `.h`

## Configuration
- Set your LLM provider and credentials in `.env` (see `.env.example`).
- The programming language is now selected in the GUI, not in `.env`.

## Notes
- The tool creates new files with a `_c` suffix (e.g., `main_c.py`).
- A `log.txt` file is generated in the target directory with a summary of processed files.

## License
MIT
