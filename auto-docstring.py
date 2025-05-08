import os
import sys
from openai import AzureOpenAI
from dotenv import load_dotenv


load_dotenv()

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
language = "Python"

PROMPT_TEMPLATE = """
You are an expert {language} programmer. Given the following file with {language} function:

{code}

If a docstring is missing, add a clear, concise, and helpful docstring in the correct format for {language}.

Reply strictly only with the function with the docstring as plain text, no other text and no markdown code fences.
"""
def add_docstrings_to_code(code):
    """
    Adds docstrings to the provided Python code using a language model.

    Args:
        code (str): The Python code to which docstrings need to be added.

    Returns:
        str: The Python code with added docstrings.
    """
    response = client.chat.completions.create(
        model=DEPLOYMENT,
        messages=[{"role": "user", "content": PROMPT_TEMPLATE.format(code=code, language=language)}],
        max_tokens=1024,
        temperature=0,
    )
    raw = response.choices[0].message.content.strip()
    # Remove markdown code fences if present
    lines = raw.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines)
def process_file(filepath):
    """
    Process a Python file to ensure all functions have docstrings.

    Args:
        filepath (str): The path to the Python file to be processed.

    Returns:
        str: The modified code with added docstrings where necessary.
    """
    with open(filepath, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    modified_code = ""
    inside_function = False
    function_block = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.strip().startswith("def "):
            inside_function = True
            function_block = [line]
            i += 1
            # Capture the function body
            while i < len(lines) and (lines[i].startswith("    ") or lines[i].strip() == ""):
                function_block.append(lines[i])
                i += 1
            # Check if the function already has a docstring
            if len(function_block) > 1 and '"""' in function_block[1]:
                modified_code += "".join(function_block)
            else:
                new_block = add_docstrings_to_code("".join(function_block))
                modified_code += new_block + '\n'
            inside_function = False
        else:
            modified_code += line
            i += 1

    return modified_code
def process_directory(directory):
    """
    Processes all Python files in the given directory, excluding those ending with '_c.py'.
    For each file, it generates a new file with '_c.py' suffix and writes the processed code to it.
    Additionally, it creates a logfile listing all processed files.

    Args:
        directory (str): The path to the directory containing the Python files to be processed.
    """
    documentation = "Documentation:\n\n"

    for file in os.listdir(directory):
        if file.endswith(".py") and not file.endswith("_c.py"):
            filepath = os.path.join(directory, file)
            new_filepath = filepath.replace('.py', '_c.py')
            print(f"Processing {filepath}")
            new_code = process_file(filepath)
            with open(new_filepath, 'w', encoding='utf-8') as f:
                f.write(new_code)
            documentation += f"Processed file: {file}\n"

    with open(os.path.join(directory, "log.txt"), "w", encoding='utf-8') as doc_file:
        doc_file.write(documentation)
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <target_directory>")
        sys.exit(1)
    target_directory = sys.argv[1]
    process_directory(target_directory)
