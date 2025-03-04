import pandas as pd
import requests
import time
import argparse
import os
import re


def call_ollama(prompt, model_name, temperature=0.7, max_tokens=2048, thinking_mode=False):
    """
    Calls the Ollama API with a given prompt and model name.

    Args:
        prompt (str): The prompt to send to the model.
        model_name (str): The name of the model to use.
        temperature (float): Sampling temperature, higher is more creative.
        max_tokens (int): Maximum tokens to generate.
        thinking_mode (bool): If True, removes "<think>...</think>" sections from the response.

    Returns:
        str: The cleaned response from the model.
    """
    url = "http://localhost:11434/api/generate"

    payload = {
        "model": model_name,
        "prompt": prompt,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False
    }

    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        full_response = response.json().get("response", "Error: No response found")

        # Remove "<think>...</think>" section for thinking models
        if thinking_mode:
            full_response = re.sub(r"<think>.*?</think>", "", full_response, flags=re.DOTALL).strip()

        return full_response

    except requests.exceptions.RequestException as e:
        return f"Error calling Ollama API: {str(e)}"


def process_excel_with_ollama(
        input_file,
        prompt_column,
        model_names,
        output_file=None,
        temperature=0.7,
        max_tokens=2048,
        batch_size=10,
        delay=1,
        thinking_models=None
):
    """
    Processes an Excel file with prompts, sends them to Ollama models, and saves responses.

    Args:
        input_file (str): Path to the input Excel file.
        prompt_column (str): Name of the column containing prompts.
        model_names (list): List of model names to use.
        output_file (str, optional): Path to save the output Excel file.
        temperature (float): Sampling temperature for the models.
        max_tokens (int): Maximum number of tokens to generate.
        batch_size (int): Number of prompts to process before saving.
        delay (int): Delay in seconds between API calls to avoid rate limiting.
        thinking_models (list, optional): List of models that have a "thinking" phase.
    """
    if thinking_models is None:
        thinking_models = []

    # Set output file name
    if output_file is None:
        base, ext = os.path.splitext(input_file)
        output_file = f"{base}_responses{ext}"

    print(f"Reading Excel file: {input_file}")
    df = pd.read_excel(input_file)

    # Validate prompt column exists
    if prompt_column not in df.columns:
        raise ValueError(f"Prompt column '{prompt_column}' not found in the Excel file")

    # Create response columns if they don't exist
    for model_name in model_names:
        response_column = f"{model_name}_response"
        if response_column not in df.columns:
            df[response_column] = ""

    total_prompts = len(df)
    print(f"Found {total_prompts} prompts to process")

    # Process each prompt row
    for i, (idx, row) in enumerate(df.iterrows()):
        prompt = row[prompt_column]

        # Skip empty prompts
        if pd.isna(prompt) or prompt.strip() == "":
            print(f"Skipping empty prompt at row {idx + 2}")
            continue

        prompt_preview = prompt[:50] + "..." if len(prompt) > 50 else prompt
        print(f"Processing prompt {i + 1}/{total_prompts}: '{prompt_preview}'")

        # Get responses from each model
        for model_name in model_names:
            response_column = f"{model_name}_response"

            # Skip if response already exists
            if pd.notna(row[response_column]) and row[response_column].strip() != "":
                print(f"  Model {model_name}: Response already exists, skipping")
                continue

            print(f"  Calling model {model_name}...")
            response = call_ollama(
                prompt,
                model_name,
                temperature,
                max_tokens,
                thinking_mode=model_name in thinking_models
            )
            df.at[idx, response_column] = response

            # Add delay to prevent rate limiting
            time.sleep(delay)

        # Save periodically
        if (i + 1) % batch_size == 0 or i == total_prompts - 1:
            print(f"Saving progress to {output_file} ({i + 1}/{total_prompts})")
            df.to_excel(output_file, index=False)

    print(f"Processing complete. Results saved to {output_file}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process Excel prompts using Ollama models")
    parser.add_argument("--input", "-i", required=True, help="Input Excel file")
    parser.add_argument("--output", "-o", help="Output Excel file (optional)")
    parser.add_argument("--prompt-column", "-p", required=True, help="Name of the column containing prompts")
    parser.add_argument("--models", "-m", required=True, nargs="+", help="Space-separated list of model names")
    parser.add_argument("--temperature", "-t", type=float, default=0.7, help="Temperature for generation (default: 0.7)")
    parser.add_argument("--max-tokens", type=int, default=2048, help="Maximum tokens to generate (default: 2048)")
    parser.add_argument("--batch-size", "-b", type=int, default=10, help="How often to save progress (default: every 10 prompts)")
    parser.add_argument("--delay", "-d", type=int, default=1, help="Delay between API calls in seconds (default: 1)")
    parser.add_argument("--thinking-models", "-th", nargs="*", default=[], help="List of models with a thinking phase")

    args = parser.parse_args()

    process_excel_with_ollama(
        args.input,
        args.prompt_column,
        args.models,
        args.output,
        args.temperature,
        args.max_tokens,
        args.batch_size,
        args.delay,
        args.thinking_models
    )
