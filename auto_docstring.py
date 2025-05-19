"""
A versatile documentation generator that automatically adds docstrings to code files.
Supports multiple programming languages and various LLM providers including Azure OpenAI,
OpenAI, and Hugging Face. The tool processes files in a given directory and creates
documented versions with '_c' suffix while maintaining the original files.
"""

import os
import sys
from abc import ABC, abstractmethod
from dotenv import load_dotenv


class LLMProvider(ABC):
    @abstractmethod
    def generate_completion(self, prompt: str, max_tokens: int = 1024, temperature: float = 0) -> str:
        pass

class AzureOpenAIProvider(LLMProvider):
    def __init__(self):
        from openai import AzureOpenAI
        self.client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
        )
        self.deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")

    def generate_completion(self, prompt: str, max_tokens: int = 1024, temperature: float = 0) -> str:
        response = self.client.chat.completions.create(
            model=self.deployment,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content.strip()

class OpenAIProvider(LLMProvider):
    def __init__(self):
        from openai import OpenAI
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")

    def generate_completion(self, prompt: str, max_tokens: int = 1024, temperature: float = 0) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content.strip()

class HuggingFaceProvider(LLMProvider):
    def __init__(self):
        try:
            from huggingface_hub import InferenceClient
            self.client = InferenceClient(token=os.getenv("HF_API_TOKEN"))
            self.model = os.getenv("HF_MODEL")
        except ImportError as exc:
            raise ImportError("huggingface-hub package is required for HuggingFace support. Install it with 'pip install huggingface-hub'") from exc

    def generate_completion(self, prompt: str, max_tokens: int = 1024, temperature: float = 0) -> str:
        response = self.client.text_generation(
            prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
        )
        return response

def get_llm_provider() -> LLMProvider:
    provider_type = os.getenv("LLM_PROVIDER", "azure").lower()
    if provider_type == "azure":
        return AzureOpenAIProvider()
    elif provider_type == "openai":
        return OpenAIProvider()
    elif provider_type == "huggingface":
        return HuggingFaceProvider()
    else:
        raise ValueError(f"Unsupported LLM provider: {provider_type}")

# Load environment variables
load_dotenv()

# Initialize the LLM provider
llm_client = get_llm_provider()

# Get the programming language from environment or default to Python
language = os.getenv("TARGET_LANGUAGE", "Python")

# Language-specific docstring formats
DOCSTRING_FORMATS = {
    "Python": {
        "start": '"""',
        "end": '"""',
        "indent": "    "
    },
    "C++": {
        "start": "/**",
        "end": " */",
        "indent": " * "
    },
    "C": {
        "start": "/*",
        "end": "*/",
        "indent": " * "
    },
    "Java": {
        "start": "/**",
        "end": " */",
        "indent": " * "
    }
}

MODULE_DOCSTRING_TEMPLATE = """
You are an expert {language} programmer. Given the following file contents:

{code}

Add a clear, concise, and helpful module/file-level documentation comment at the top of the file in the correct format for {language}.
The documentation should explain the purpose and main functionality of this module/file.

Reply strictly only with the documentation comment as plain text, no other text and no markdown code fences.
"""

FUNCTION_DOCSTRING_TEMPLATE = """
You are an expert {language} programmer. Given the following {language} function or method:

{code}

If a documentation comment is missing, add a clear, concise, and helpful documentation in the correct format for {language}.
For Python, use docstrings. For C++ and Java, use Javadoc (/** ... */) style comments.

Reply strictly only with the function with the documentation as plain text, no other text and no markdown code fences.
"""

def add_module_docstring(code: str, lang_override=None) -> str:
    """
    Adds a module-level documentation comment at the top of the file.

    Args:
        code (str): The full source code of the file
        lang_override (str, optional): Override the language for this file.

    Returns:
        str: The code with a module-level documentation comment added at the top
    """
    lang = lang_override or language
    response = llm_client.generate_completion(
        prompt=MODULE_DOCSTRING_TEMPLATE.format(code=code, language=lang),
        max_tokens=1024,
        temperature=0,
    )
    formats = DOCSTRING_FORMATS[lang]
    module_doc = response.strip()
    # If the response already includes the comment delimiters, use it as is
    if not (module_doc.startswith(formats["start"]) or module_doc.startswith("/*")):
        if lang == "Python":
            module_doc = f'{formats["start"]}\n{module_doc}\n{formats["end"]}\n'
        elif lang == "Java":
            lines = module_doc.split("\n")
            formatted_lines = [f"{formats['indent']}{line}" for line in lines]
            module_doc = f"{formats['start']}\n" + "\n".join(formatted_lines) + f"\n{formats['end']}\n"
        else:  # C++ or C
            lines = module_doc.split("\n")
            formatted_lines = [f"{formats['indent']}{line}" for line in lines]
            module_doc = f"{formats['start']}\n" + "\n".join(formatted_lines) + f"\n{formats['end']}\n"
    return module_doc + "\n" + code

def add_docstrings_to_code(code, lang_override=None):
    """
    Adds docstrings or comments to the provided code using a language model.

    Args:
        code (str): The code to which docstrings/comments need to be added.
        lang_override (str, optional): Override the language for this code.

    Returns:
        str: The code with added docstrings/comments.
    """
    lang = lang_override or language
    response = llm_client.generate_completion(
        prompt=FUNCTION_DOCSTRING_TEMPLATE.format(code=code, language=lang),
        max_tokens=1024,
        temperature=0,
    )
    raw = response
    # Remove markdown code fences if present
    lines = raw.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].startswith("```"):
        lines = lines[:-1]
    # For C, ensure comments are /* ... */ or //, not triple quotes
    if lang == "C":
        # Replace any accidental triple quotes with /* ... */
        joined = "\n".join(lines)
        if '"""' in joined or "'''" in joined:
            joined = joined.replace('"""', '/*').replace("'''", '/*')
            # Try to close with */ if not present
            if not joined.rstrip().endswith('*/'):
                joined = joined.rstrip() + '\n*/'
            return joined
    elif lang == "Java":
        # Ensure Javadoc style for Java
        joined = "\n".join(lines)
        if '"""' in joined or "'''" in joined:
            joined = joined.replace('"""', '/**').replace("'''", '/**')
            if not joined.rstrip().endswith('*/'):
                joined = joined.rstrip() + '\n*/'
            return joined
    return "\n".join(lines)

def process_file(filepath: str, language_override=None) -> str:
    """
    Process a source code file to add documentation comments.

    Args:
        filepath (str): The path to the file to be processed.
        language_override (str, optional): Override the language for this file.

    Returns:
        str: The modified code with added documentation comments.
    """
    with open(filepath, 'r', encoding='utf-8') as file:
        content = file.read()
    lang = language_override or language
    # First add the module docstring if one doesn't exist
    has_module_doc = False
    # Check for existing module documentation
    if lang == "Python":
        has_module_doc = content.lstrip().startswith('"""') or content.lstrip().startswith("'''")
    elif lang == "Java":
        has_module_doc = content.lstrip().startswith("/**")
    else:  # C++ or C
        has_module_doc = content.lstrip().startswith("/*") or content.lstrip().startswith("/**")
    if not has_module_doc:
        content = add_module_docstring(content, lang_override=lang)
    # Now process functions
    if lang == "Python":
        return process_python_functions(content)
    elif lang == "Java":
        return process_java_functions(content)
    else:  # C++ or C
        return process_cpp_functions(content, lang_override=lang)

def process_python_functions(content: str) -> str:
    """
    Process Python source code to add docstrings to functions.

    Args:
        content (str): The Python source code to process.

    Returns:
        str: The processed Python code with added docstrings.
    """
    lines = content.splitlines(True)  # Keep line endings
    modified_code = []
    current_function = []
    
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.strip().startswith("def "):
            current_function = [line]
            i += 1
            # Capture the function body
            while i < len(lines) and (lines[i].startswith("    ") or lines[i].strip() == ""):
                current_function.append(lines[i])
                i += 1
            # Check if the function already has a docstring
            if len(current_function) > 1 and '"""' in current_function[1]:
                modified_code.extend(current_function)
            else:
                new_block = add_docstrings_to_code("".join(current_function))
                modified_code.extend(new_block.splitlines(True))
            continue
        modified_code.append(line)
        i += 1
    
    return "".join(modified_code)

def process_cpp_functions(content: str, lang_override=None) -> str:
    """
    Process C or C++ source code to add documentation comments to functions.

    Args:
        content (str): The C/C++ source code to process.
        lang_override (str, optional): Override the language for this code.

    Returns:
        str: The processed code with added documentation comments.
    """
    lang = lang_override or language
    lines = content.splitlines(True)  # Keep line endings
    modified_code = []
    i = 0
    while i < len(lines):
        line = lines[i]
        # Look for function declarations
        if (any(type in line for type in ["void", "int", "bool", "char", "float", "double", "string"]) and 
            "(" in line and 
            not line.strip().startswith("//")):
            # Check if there's already a documentation comment
            has_doc = False
            if i > 0:
                prev_lines = "".join(lines[max(0, i-3):i])
                has_doc = "/**" in prev_lines or "/*" in prev_lines
            if not has_doc:
                # Get the full function signature
                func_block = [line]
                peek = i + 1
                # Capture multi-line function signatures
                while peek < len(lines) and "{" not in lines[peek-1]:
                    func_block.append(lines[peek])
                    peek += 1
                new_block = add_docstrings_to_code("".join(func_block), lang_override=lang)
                modified_code.extend(new_block.splitlines(True))
                i = peek
                continue
        modified_code.append(line)
        i += 1
    return "".join(modified_code)

def process_java_functions(content: str) -> str:
    """
    Process Java source code to add Javadoc comments to classes and methods.

    Args:
        content (str): The Java source code to process.

    Returns:
        str: The processed Java code with added Javadoc comments.
    """
    lines = content.splitlines(True)
    modified_code = []
    i = 0
    while i < len(lines):
        line = lines[i]
        is_class = line.strip().startswith("class ") or line.strip().startswith("public class ")
        is_method = (any(ret in line for ret in ["void", "int", "boolean", "char", "float", "double", "String"]) and "(" in line and ")" in line and "{" in line)
        if is_class or is_method:
            has_doc = False
            if i > 0:
                prev_lines = "".join(lines[max(0, i-3):i])
                has_doc = "/**" in prev_lines or "/*" in prev_lines
            if not has_doc:
                block = [line]
                peek = i + 1
                while peek < len(lines) and "{" not in lines[peek-1]:
                    block.append(lines[peek])
                    peek += 1
                new_block = add_docstrings_to_code("".join(block), lang_override="Java")
                modified_code.extend(new_block.splitlines(True))
                i = peek
                continue
        modified_code.append(line)
        i += 1
    return "".join(modified_code)

def process_directory(directory: str, selected_language=None, log_callback=None, progress_callback=None) -> None:
    """
    Processes all source code files in the given directory, excluding those ending with '_c.<ext>'.
    For each file, it generates a new file with '_c.<ext>' suffix and writes the processed code to it.
    Additionally, it creates a logfile listing all processed files.

    Args:
        directory (str): The path to the directory containing the source code files to be processed.
        selected_language (str, optional): The language to process (overrides env).
        log_callback (callable, optional): Function to call with log messages.
        progress_callback (callable, optional): Function to call with progress updates.
    """
    documentation = "Documentation:\n\n"
    extensions = {
        "Python": [".py"],
        "C++": [".cpp", ".hpp", ".h", ".cc", ".cxx"],
        "C": [".c", ".h"],
        "Java": [".java"]
    }
    lang = selected_language or language
    file_extensions = extensions[lang]
    files = [f for f in os.listdir(directory) if any(f.endswith(ext) for ext in file_extensions) and not f.endswith("_c" + file_extensions[0])]
    total = len(files)
    if progress_callback:
        progress_callback(0, maximum=total if total > 0 else 1)
    processed = 0
    for file in files:
        filepath = os.path.join(directory, file)
        new_filepath = filepath[:-len(file_extensions[0])] + "_c" + file_extensions[0]
        msg = f"Processing {filepath}"
        print(msg)
        if log_callback:
            log_callback(msg)
        try:
            new_code = process_file(filepath, language_override=lang)
            with open(new_filepath, 'w', encoding='utf-8') as f:
                f.write(new_code)
            documentation += f"Processed file: {file}\n"
            if log_callback:
                log_callback(f"Processed file: {file}")
        except (IOError, OSError) as e:
            documentation += f"Error processing {file}: {str(e)}\n"
            if log_callback:
                log_callback(f"Error processing {file}: {str(e)}")
        processed += 1
        if progress_callback:
            progress_callback(processed)
    with open(os.path.join(directory, "log.txt"), "w", encoding='utf-8') as doc_file:
        doc_file.write(documentation)

if __name__ == "__main__":
    import tkinter as tk
    from tkinter import filedialog, messagebox, scrolledtext, ttk
    import threading

    class ProgressPopup:
        def __init__(self, master, total):
            self.top = tk.Toplevel(master)
            self.top.title("Processing...")
            self.top.geometry("420x80")
            self.top.transient(master)
            self.top.grab_set()
            tk.Label(self.top, text="Processing files...").pack(pady=(10, 0))
            self.progress = ttk.Progressbar(self.top, orient="horizontal", length=380, mode="determinate")
            self.progress.pack(pady=10)
            self.progress["maximum"] = total if total > 0 else 1
            self.progress["value"] = 0
            self.top.protocol("WM_DELETE_WINDOW", lambda: None)  # Disable close
        def set_progress(self, value):
            self.progress["value"] = value
            self.top.update_idletasks()
        def close(self):
            self.top.grab_release()
            self.top.destroy()

    class CommentAssistantGUI:
        def __init__(self, master):
            self.master = master
            master.title("Comment Assistant")
            self.selected_language = tk.StringVar(value="Python")
            self.selected_dir = tk.StringVar()
            # Language dropdown
            tk.Label(master, text="Language:").grid(row=0, column=0, sticky="w")
            lang_menu = tk.OptionMenu(master, self.selected_language, "Python", "C++", "C", "Java")
            lang_menu.grid(row=0, column=1, sticky="ew")
            # Directory chooser
            tk.Label(master, text="Directory:").grid(row=1, column=0, sticky="w")
            tk.Entry(master, textvariable=self.selected_dir, width=40).grid(row=1, column=1, sticky="ew")
            tk.Button(master, text="Choose...", command=self.choose_dir).grid(row=1, column=2)
            # Start button
            tk.Button(master, text="Start", command=self.start_processing).grid(row=2, column=1, sticky="ew")
            # Log window
            self.log = scrolledtext.ScrolledText(master, width=60, height=15, state="disabled")
            self.log.grid(row=3, column=0, columnspan=3, pady=10)
            # Close button at the bottom
            tk.Button(master, text="Close", command=master.quit).grid(row=4, column=0, columnspan=3, pady=(0,10))
            self.progress_popup = None
        def choose_dir(self):
            dirname = filedialog.askdirectory()
            if dirname:
                self.selected_dir.set(dirname)
        def log_message(self, msg):
            self.log.config(state="normal")
            self.log.insert(tk.END, msg + "\n")
            self.log.see(tk.END)
            self.log.config(state="disabled")
        def start_processing(self):
            directory = self.selected_dir.get()
            lang = self.selected_language.get()
            if not directory:
                messagebox.showerror("Error", "Please select a directory.")
                return
            # Count files to process
            extensions = {
                "Python": [".py"],
                "C++": [".cpp", ".hpp", ".h", ".cc", ".cxx"],
                "C": [".c", ".h"],
                "Java": [".java"]
            }
            file_extensions = extensions[lang]
            files = [f for f in os.listdir(directory) if any(f.endswith(ext) for ext in file_extensions) and not f.endswith("_c" + file_extensions[0])]
            total = len(files)
            if total == 0:
                messagebox.showinfo("No files", f"No {lang} files found in the selected directory.")
                return
            self.log_message(f"Starting processing for {lang} files in {directory}...")
            self.progress_popup = ProgressPopup(self.master, total)
            # Run processing in a separate thread to keep GUI responsive
            threading.Thread(target=self._process_files_thread, args=(directory, lang, total), daemon=True).start()
        def _process_files_thread(self, directory, lang, total):
            try:
                process_directory(
                    directory,
                    selected_language=lang,
                    log_callback=self.log_message,
                    progress_callback=self._progress_callback_threadsafe(total)
                )
                self.log_message("Done.")
            except Exception as e:
                self.log_message(f"Error: {str(e)}")
                messagebox.showerror("Error", str(e))
            if self.progress_popup:
                self.master.after(100, self.progress_popup.close)
                self.progress_popup = None
        def _progress_callback_threadsafe(self, total):
            # Returns a callback that safely updates the popup progress bar from any thread
            def cb(value, maximum=None):
                def update():
                    if self.progress_popup:
                        self.progress_popup.set_progress(value)
                self.master.after(0, update)
            return cb

    root = tk.Tk()
    gui = CommentAssistantGUI(root)
    root.mainloop()
