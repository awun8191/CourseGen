import base64
import os
from google import genai
from google.genai import types
from chromadb_query import ChromaQuery, MetaData
from courses import DataFormatting


TEMPERATURE = 0.15
THINKING_BUDGET = 12700
TOP_P = 0.9
MAX_OUTPUT_TOKENS = 15500

class GeminiQuestionGen:
    
    
    def __init__(self, is_thinking = True):
        self.is_thinking = is_thinking
    
    
    def _build_prompt(self, course: str, topic: str, difficulty: str, is_calculation: True):
        if is_calculation:
            return f"""Generate {difficulty} difficulty questions of which 5 calculation multiple choice questions on the topic {topic} for an {course} course. Do this in a json format with the following format:
    
    {{
        "question": "question string",
        "explanation": "explanation string",
        "steps": {{
            "1": "step 1",
            "2": "step 2"
        }},
        "options": ["a", "b", "c", "d"],
        "answer": "answer string"
    }}
    
    Generate questions, answers, explanations, options and the optional calculation steps.
    Ensure the calculation steps have at least 3 steps and no more than 8 steps keeping them short and concise.
    Always solve before providing the answer.
    Ensure proper latex formatting
    """
    
        else:
             return f"""Generate {difficulty} difficulty questions of which 10 theoretical multiple choice questions on the topic {topic} for an {course} course aimed at building an understanding for students. Do this in a json format with the following format:
    
    {{
        "question": "question string",
        "explanation": "explanation string",
        "options": ["a", "b", "c", "d"],
        "answer": "answer string"
    }}
    
    Generate questions, answers, explanations and options.
    Ensure proper latex formatting
    """
    
    
    def _build_RAG_prompt(self, course: str, topic: str, difficulty: str, is_calculation: True, RAG_data):
        if is_calculation:
            return f"""
        f{RAG_data}
        
      YOU  are a profrssional question generator who generates questions for engineering students based on thier study materials. Utilize the study materials above to perform your tasks.
        
        
        Generate {difficulty} difficulty questions of which 5 calculation multiple choice questions on the topic {topic} for an {course} course. Do this in a json format with the following format:
    
    {{
        "question": "question string",
        "explanation": "explanation string",
        "steps": {{
            "1": "step 1",
            "2": "step 2"
        }},
        "options": ["a", "b", "c", "d"],
        "answer": "answer string"
    }}
    
    Generate questions, answers, explanations, options and the optional calculation steps.
    Ensure the calculation steps have at least 3 steps and no more than 8 steps keeping them short and concise.
    Always solve before providing the answer.
    Ensure proper latex formatting
    """
    
        else:
             return f"""
               f{RAG_data}
        
      YOU  are a profrssional question generator who generates questions for engineering students based on thier study materials. Utilize the study materials above to perform your tasks.
        
        
         
         
         Generate {difficulty} difficulty questions of which 10 theoretical multiple choice questions on the topic {topic} for an {course} course aimed at building an understanding for students. Do this in a json format with the following format:
    
    {{
        "question": "question string",
        "explanation": "explanation string",
        "options": ["a", "b", "c", "d"],
        "answer": "answer string"
    }}
    
    Generate questions, answers, explanations and options.
    Ensure proper latex formatting
    """
    
    
    
    def _sanitize(self, text: str):
        """Extract JSON from `text` and return the parsed Python object.

        Strategy:
        - First, try to find a fenced code block that starts with ```json and ends with ```.
        - If not found, try to parse the entire text as JSON.
        - If that fails, try to extract JSON-like content using regex for { ... }.
        - Then, try to parse the extracted content with json.loads.
        - On JSON decode error, attempt fallbacks (escape backslashes, or use ast.literal_eval).
        """
        import re
        import json

        code = None
        # Try to find the first ```json ... ``` block
        m = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL | re.IGNORECASE)
        if m:
            code = m.group(1).strip()
        else:
            # Try to parse the whole text as JSON
            try:
                return json.loads(text.strip())
            except json.JSONDecodeError:
                pass
            # Try to extract JSON using regex for outermost { ... }
            m2 = re.search(r'\{.*\}', text, re.DOTALL)
            if m2:
                code = m2.group(0).strip()
            else:
                raise ValueError("No JSON content found in text")

        # Now parse the code
        # Try direct JSON parse first
        try:
            return json.loads(code)
        except json.JSONDecodeError:
            # Try escaping single backslashes (common when LaTeX backslashes are present)
            try:
                fixed = code.replace('\\', '\\\\')
                return json.loads(fixed)
            except json.JSONDecodeError:
                # Last resort: try ast.literal_eval after converting JSON literals to Python
                import ast
                pyish = code.replace('true', 'True').replace('false', 'False').replace('null', 'None')
                try:
                    return ast.literal_eval(pyish)
                except Exception as e:
                    raise ValueError('Failed to parse JSON content') from e



    def generate_with_RAG(self, course="ELECTRONICS", topic="TRANSISTORS", department ="ELECTRICAL ELECTONICS", difficulty="medium", is_calculation = False, metadata: MetaData = MetaData(COURSE_FOLDER="EEE 313"), variation = False):
        # Chroma expects a plain dict for the `where` filter. If a MetaData
        # dataclass is provided, convert it to a dict using `to_where()`.
        where_arg = metadata.to_where() if isinstance(metadata, MetaData) else (metadata or {})
        if variation:
            rag = ChromaQuery().search_with_temperature(
                f"Obtain texbooks and questions for the course f{course} on the {topic} for an f{department} engineering student",
                where=where_arg,
                topk=50,
                tau=0.35,
                min_sim=0.65,
                seed=None,
                show_snippet=True
            )
        else:
            rag = ChromaQuery().search(
                f"Obtain texbooks and questions for the course f{course} on the {topic} for an f{department} engineering student",
                where=where_arg,
                show_snippet=True
            )
        
        if self.is_thinking:
            client = genai.Client(
                api_key=os.environ.get("GOOGLE_API_KEY"),
            )

            model = "gemini-2.5-flash-lite"
            prompt = self._build_RAG_prompt(course, topic, difficulty, is_calculation=is_calculation, RAG_data=rag)
            contents = [
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_text(text=prompt),
                    ],
                ),
            ]
            
            generate_content_config = types.GenerateContentConfig(
                temperature=TEMPERATURE,
                top_p=TOP_P,
                max_output_tokens=MAX_OUTPUT_TOKENS,
                thinking_config = types.ThinkingConfig(
                    thinking_budget=THINKING_BUDGET,
                ),
            )

            # Accumulate streamed pieces safely (some chunk.text or part.text may be None)
            response = self._accumulate_chunks(client.models.generate_content_stream(model=model, contents=contents, config=generate_content_config))
            response = self._sanitize(response)
            print(response)
        else: 
            client = genai.Client(
                api_key=os.environ.get("GOOGLE_API_KEY"),
            )

            model = "gemma-3-27b-it"
            prompt = self._build_prompt(course, topic, difficulty, is_calculation=is_calculation)
            contents = [
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_text(text=prompt),
                    ],
                ),
            ]
            
            generate_content_config = types.GenerateContentConfig(
                temperature=TEMPERATURE,
                top_p=TOP_P,
                max_output_tokens=MAX_OUTPUT_TOKENS,
                
            )

            # Accumulate streamed pieces safely (some chunk.text or part.text may be None)
            response = self._accumulate_chunks(client.models.generate_content_stream(model=model, contents=contents, config=generate_content_config))
            response = self._sanitize(response)
            print(response)



    def generate(self, course="Contol Systems", topic="Laplace Transform", difficulty="medium", is_calculation = False, use_RAG = True):
        if self.is_thinking:
            client = genai.Client(
                api_key=os.environ.get("GOOGLE_API_KEY"),
            )

            model = "gemini-2.5-flash-lite"
            prompt = self._build_prompt(course, topic, difficulty, is_calculation=is_calculation) if not use_RAG else self._build_RAG_prompt(course, topic, difficulty, is_calculation=is_calculation)
            contents = [
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_text(text=prompt),
                    ],
                ),
            ]
            
            generate_content_config = types.GenerateContentConfig(
                temperature=TEMPERATURE,
                top_p=TOP_P,
                max_output_tokens=MAX_OUTPUT_TOKENS,
                thinking_config = types.ThinkingConfig(
                    thinking_budget=THINKING_BUDGET,
                ),
            )

            # Accumulate streamed pieces safely (some chunk.text or part.text may be None)
            response = self._accumulate_chunks(client.models.generate_content_stream(model=model, contents=contents, config=generate_content_config))
            response = self._sanitize(response)
            print(response)
        else: 
            client = genai.Client(
                api_key=os.environ.get("GOOGLE_API_KEY"),
            )

            model = "gemma-3-27b-it"
            prompt = self._build_prompt(course, topic, difficulty, is_calculation=is_calculation)
            contents = [
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_text(text=prompt),
                    ],
                ),
            ]
            
            generate_content_config = types.GenerateContentConfig(
                temperature=TEMPERATURE,
                top_p=TOP_P,
                max_output_tokens=MAX_OUTPUT_TOKENS,
                
            )

            # Accumulate streamed pieces safely (some chunk.text or part.text may be None)
            response = self._accumulate_chunks(client.models.generate_content_stream(model=model, contents=contents, config=generate_content_config))
            response = self._sanitize(response)
            print(response)


    
    def _build_Course_Outline_Prompt(self,  RAG_data, course="Contol Systems", department="Electrical Electronics"):
        return f"""
            {RAG_data}
            
            
            
            I want you to generate only the description and a very relevant course outline for a {course} course for a {department} student.
            
            Do it in a json format:
            
            
            {{
                "description": "Text",
                "topics": []
            }}
        
    
    """


    def course_outline_RAG(self, course="Electronics", department ="ELECTRICAL ELECTONICS", metadata: MetaData = MetaData(COURSE_FOLDER="EEE 313"), variation = True):
        # Chroma expects a plain dict for the `where` filter. If a MetaData
        # dataclass is provided, convert it to a dict using `to_where()`.
        where_arg = metadata.to_where() if isinstance(metadata, MetaData) else (metadata or {})
        
        if variation:
            rag = ChromaQuery().search_with_temperature(
                f"COURSE OUTLINE, SUMMARY, OBJECTIVES, OVERVIEW for {course} {department}",
                where=where_arg,
                topk=50,
                tau=0.35,
                min_sim=0.65,
                seed=None,
                show_snippet=True
            )
        else:
            rag = ChromaQuery().search(
                f"COURSE OUTLINE, SUMMARY, OBJECTIVES, OVERVIEW for {course} {department}",
                where=where_arg,
                show_snippet=True
            )
        
        if self.is_thinking:
            client = genai.Client(
                api_key=os.environ.get("GOOGLE_API_KEY"),
            )

            model = "gemini-2.5-flash-lite"
            prompt = self._build_Course_Outline_Prompt(course=course, RAG_data=rag)
            contents = [
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_text(text=prompt),
                    ],
                ),
            ]
            
            generate_content_config = types.GenerateContentConfig(
                temperature=TEMPERATURE,
                top_p=TOP_P,
                max_output_tokens=MAX_OUTPUT_TOKENS,
                thinking_config = types.ThinkingConfig(
                    thinking_budget=THINKING_BUDGET,
                ),
            )

            # Accumulate streamed pieces safely (some chunk.text or part.text may be None)
            response = self._accumulate_chunks(client.models.generate_content_stream(model=model, contents=contents, config=generate_content_config))
            response = self._sanitize(response)
            print(response)
        else: 
            client = genai.Client(
                api_key=os.environ.get("GOOGLE_API_KEY"),
            )

            model = "gemma-3-27b-it"
            prompt = self._build_Course_Outline_Prompt(course=course, RAG_data=rag)
            contents = [
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_text(text=prompt),
                    ],
                ),
            ]
            
            generate_content_config = types.GenerateContentConfig(
                temperature=TEMPERATURE,
                top_p=TOP_P,
                max_output_tokens=MAX_OUTPUT_TOKENS,
                
            )

            # Accumulate streamed pieces safely (some chunk.text or part.text may be None)
            response = self._accumulate_chunks(client.models.generate_content_stream(model=model, contents=contents, config=generate_content_config))
            response = self._sanitize(response)
            print(response)


    def _accumulate_chunks(self, stream, print_chunks: bool = False):
        """Safely accumulate text from a generate_content_stream iterator.

        Some chunks or parts may have a `text` attribute set to None. This helper
        only concatenates actual string content.
        """
        response = ""
        for chunk in stream:
            text = getattr(chunk, "text", None)
            if isinstance(text, str) and text:
                if print_chunks:
                    print(f"Chunk: {text}")
                response += text
                continue

            parts = getattr(chunk, "parts", None)
            if parts:
                for part in parts:
                    part_text = getattr(part, "text", None)
                    if isinstance(part_text, str) and part_text:
                        response += part_text

        return response



if __name__ == "__main__":
    # gemini = GeminiQuestionGen(is_thinking=False).generate(is_calculation=True, difficulty="medium")
    # gemini = GeminiQuestionGen(is_thinking=True).generate_with_RAG(is_calculation=True, difficulty="medium", variation=False)
    course_info = DataFormatting().search_course("EEE 313")
    
    
    
    gemini = GeminiQuestionGen(is_thinking=True).course_outline_RAG(variation=False, course=course_info[0].title, department=", 0".join(course_info[1]), metadata=MetaData(COURSE_FOLDER=course_info[0].code))