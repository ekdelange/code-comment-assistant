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
For Python, use docstrings. For C++, use /** */ style comments.

Reply strictly only with the function with the documentation as plain text, no other text and no markdown code fences.
"""

def add_module_docstring(code: str) -> str:
    """
    Adds a module-level documentation comment at the top of the file.

    Args:
        code (str): The full source code of the file

    Returns:
        str: The code with a module-level documentation comment added at the top
    """
    response = llm_client.generate_completion(
        prompt=MODULE_DOCSTRING_TEMPLATE.format(code=code, language=language),
        max_tokens=1024,
        temperature=0,
    )
    
    formats = DOCSTRING_FORMATS[language]
    module_doc = response.strip()
    
    # If the response already includes the comment delimiters, use it as is
    if not (module_doc.startswith(formats["start"]) or module_doc.startswith("/*")):
        if language == "Python":
            module_doc = f'{formats["start"]}\n{module_doc}\n{formats["end"]}\n'
        else:  # C++
            lines = module_doc.split("\n")
            formatted_lines = [f"{formats['indent']}{line}" for line in lines]
            module_doc = f"{formats['start']}\n" + "\n".join(formatted_lines) + f"\n{formats['end']}\n"
    
    # Add a newline after the module docstring
    return module_doc + "\n" + code

def add_docstrings_to_code(code):
    """
    Adds docstrings to the provided Python code using a language model.

    Args:
        code (str): The Python code to which docstrings need to be added.

    Returns:
        str: The Python code with added docstrings.
    """
    response = llm_client.generate_completion(
        prompt=FUNCTION_DOCSTRING_TEMPLATE.format(code=code, language=language),
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
    return "\n".join(lines)

def process_file(filepath: str) -> str:
    """
    Process a source code file to add documentation comments.

    Args:
        filepath (str): The path to the file to be processed.

    Returns:
        str: The modified code with added documentation comments.
    """
    with open(filepath, 'r', encoding='utf-8') as file:
        content = file.read()
        
    # First add the module docstring if one doesn't exist
    has_module_doc = False
    
    # Check for existing module documentation
    if language == "Python":
        has_module_doc = content.lstrip().startswith('"""') or content.lstrip().startswith("'''")
    else:  # C++
        has_module_doc = content.lstrip().startswith("/*") or content.lstrip().startswith("/**")
    
    if not has_module_doc:
        content = add_module_docstring(content)
    
    # Now process functions
    if language == "Python":
        return process_python_functions(content)
    else:  # C++
        return process_cpp_functions(content)

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

def process_cpp_functions(content: str) -> str:
    """
    Process C++ source code to add documentation comments to functions.

    Args:
        content (str): The C++ source code to process.

    Returns:
        str: The processed C++ code with added documentation comments.
    """
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
                
                new_block = add_docstrings_to_code("".join(func_block))
                modified_code.extend(new_block.splitlines(True))
                i = peek
                continue
        
        modified_code.append(line)
        i += 1
    
    return "".join(modified_code)

def process_directory(directory: str) -> None:
    """
    Processes all source code files in the given directory, excluding those ending with '_c.<ext>'.
    For each file, it generates a new file with '_c.<ext>' suffix and writes the processed code to it.
    Additionally, it creates a logfile listing all processed files.

    Args:
        directory (str): The path to the directory containing the source code files to be processed.
    """
    documentation = "Documentation:\n\n"
    extensions = {
        "Python": [".py"],
        "C++": [".cpp", ".hpp", ".h", ".cc", ".cxx"]
    }
    
    file_extensions = extensions[language]
    
    for file in os.listdir(directory):
        if any(file.endswith(ext) for ext in file_extensions) and not file.endswith("_c" + file_extensions[0]):
            filepath = os.path.join(directory, file)
            new_filepath = filepath[:-len(file_extensions[0])] + "_c" + file_extensions[0]
            print(f"Processing {filepath}")
            try:
                new_code = process_file(filepath)
                with open(new_filepath, 'w', encoding='utf-8') as f:
                    f.write(new_code)
                documentation += f"Processed file: {file}\n"
            except (IOError, OSError) as e:
                documentation += f"Error processing {file}: {str(e)}\n"

    with open(os.path.join(directory, "log.txt"), "w", encoding='utf-8') as doc_file:
        doc_file.write(documentation)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <target_directory>")
        sys.exit(1)
    target_directory = sys.argv[1]
    process_directory(target_directory)
