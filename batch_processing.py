import asyncio
import os
import json
import google.generativeai as genai

# Configure the API key
try:
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
except AttributeError:
    print("API Key not found. Please set the GOOGLE_API_KEY environment variable.")
    exit()

# --- Configuration ---
MODEL_NAME = "gemini-1.5-flash"
INPUT_FILE = "prompts.jsonl"
OUTPUT_FILE = "outputs.jsonl"
GENERATION_CONFIG = {
    "temperature": 0.7,
    "top_p": 1,
    "top_k": 1,
    "max_output_tokens": 256,
}

async def generate_content(model, request_id, prompt):
    """Asynchronously calls the Gemini API for a single prompt."""
    try:
        print(f"Sending prompt for ID: {request_id}")
        response = await model.generate_content_async(
            prompt,
            generation_config=GENERATION_CONFIG
        )
        return {"id": request_id, "prompt": prompt, "response": response.text}
    except Exception as e:
        # Handle potential API errors for a single request
        print(f"Error processing ID '{request_id}': {e}")
        return {"id": request_id, "prompt": prompt, "error": str(e)}

async def main():
    """Main function to run the batch processing from a JSONL file."""
    model = genai.GenerativeModel(model_name=MODEL_NAME)
    tasks = []

    # Read the input file and create concurrent tasks
    try:
        with open(INPUT_FILE, 'r') as f:
            for line in f:
                data = json.loads(line)
                tasks.append(generate_content(model, data["id"], data["prompt"]))
    except FileNotFoundError:
        print(f"Error: Input file '{INPUT_FILE}' not found.")
        return

    # Run all tasks concurrently
    results = await asyncio.gather(*tasks)

    # Write the results to the output file
    with open(OUTPUT_FILE, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')

    print(f"\n--- Batch Processing Complete ---")
    print(f"Results have been saved to '{OUTPUT_FILE}'.")

# Run the asynchronous main function
if __name__ == "__main__":
    asyncio.run(main())
    
    
r"""

I want you to implement gemini question generation and utilize batch processing in the processes involved. 
In the code for the gemini_question_gen.py i want you to utilize batch processing when obtaining the questions. 
A batch for a given course, topic, level, department should yeild 50 questions. Each individual request would output 5 questions for calculations type and 10 questions for theoretical.
All question generation batches should use the topics which is first obtained from the course outline (Topics) from the code. 
All of these processes should utilize RAG.
Use a jsonl file for batch processing and each of those requests for the  course, topic, level, department combo would have the same RAG prompt


"""