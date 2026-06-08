from typing import Dict, List, Any, Optional
from app.models.schemas import Question
from app.services.prompts.base_prompt_service import BasePromptService
import os

try:
    import google.genai as genai
    GENAI_AVAILABLE = True
except Exception:
    try:
        import google.generativeai as genai
        GENAI_AVAILABLE = True
    except Exception:
        genai = None
        GENAI_AVAILABLE = False


class QuestionGenerator(BasePromptService):
    """
    Generate interview questions tailored to candidate resume and target roles.
    Uses JSON mode for fast, direct JSON responses.
    Inherits optimized formatting and parsing utilities from BasePromptService.
    """
    
    def __init__(self):
        """Initialize question generator service."""
        super().__init__()
        self.api_key = os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is required")
        if not GENAI_AVAILABLE or not genai:
            raise ImportError("google-generativeai package is not available")
        from app.services.genai_compat import configure_genai
        configure_genai(self.api_key)

    @property
    def model(self):
        """Get the generative AI model instance with JSON mode enabled."""
        if self._model is None:
            if not GENAI_AVAILABLE or not genai:
                raise ImportError("google-generativeai package is not available")
            try:
                try:
                    json_config = genai.types.GenerationConfig(
                        response_mime_type="application/json"
                    )
                except Exception:
                    json_config = None
                from app.services.genai_compat import get_model
                self._model = get_model(self.DEFAULT_MODEL, generation_config=json_config)
        return self._model

    async def generate(self, resume_data: Dict[str, Any], **kwargs) -> List[Question]:
        """
        Generate general interview questions based on resume data.
        Args:
            resume_data: Parsed resume data dictionary
            **kwargs: Additional arguments (not used)
        Returns:
            List of Question objects
        """
        model = self.model
        prompt = self._create_prompt(resume_data)
        response = await model.generate_content_async(prompt)

        try:
            raw_questions = self._parse_questions(response.text)
            questions = [
                Question(
                    type=q["type"],
                    question=q["question"],
                    context=q.get("context", "Generated based on resume analysis")
                )
                for q in raw_questions
            ]
            return questions
        except Exception as e:
            raise ValueError(f"Failed to generate questions: {str(e)}")

    async def generate_for_role(
        self,
        resume_data: Dict[str, Any],
        target_role: str,
        job_description: Optional[str] = None
    ) -> List[Question]:
        """
        Generate role-specific interview questions.
        Args:
            resume_data: Parsed resume data dictionary
            target_role: Target job role
            job_description: Optional job description
        Returns:
            List of Question objects
        """
        model = self.model
        prompt = self._create_role_specific_prompt(resume_data, target_role, job_description)
        response = await model.generate_content_async(prompt)

        try:
            raw_questions = self._parse_questions(response.text)
            questions = [
                Question(
                    type=q["type"],
                    question=q["question"],
                    context=q.get("context", f"Generated for {target_role} position")
                )
                for q in raw_questions
            ]
            return questions
        except Exception as e:
            raise ValueError(f"Failed to generate role-specific questions: {str(e)}")

    # ========== PROMPT CREATION METHODS ==========
    def _create_prompt(self, resume_data: Dict[str, Any]) -> str:
        """Create optimized structured prompt for general interview questions."""
        skills = self.format_skills(resume_data.get("skills", []))
        experience_summary = self.format_work_experience(resume_data.get("workExperience", []))
        education_summary = self.format_education(resume_data.get("education", []))
        highlights = self.format_highlights(resume_data.get("highlights", []))

        prompt = f"""ROLE: Expert Technical Interviewer.

TASK: Generate 8-10 interview questions for this candidate.

CANDIDATE_PROFILE:
Skills: {skills}
Work_Experience: {experience_summary}
Education: {education_summary}
Highlights: {highlights}

INSTRUCTIONS:
1. Generate 8-10 insightful questions targeting skills, experience, and knowledge gaps.
2. Include mix of "technical", "behavioral", and "experience" types.
3. context: 1-2 sentence explanation of WHY this question is asked.
4. Questions must be specific to resume content and test real-world understanding.

RESPONSE SCHEMA (MUST FOLLOW):
You MUST output a valid JSON array with exactly this structure:
[
  {{
    "type": "string (required: 'technical'|'behavioral'|'experience')",
    "question": "string (required)",
    "context": "string (required: 1-2 sentence reason)"
  }}
]

OUTPUT: Return ONLY the JSON array. No additional text before or after."""
        return prompt

    def _create_role_specific_prompt(
        self,
        resume_data: Dict[str, Any],
        target_role: str,
        job_description: Optional[str] = None
    ) -> str:
        """Create optimized structured prompt for role-specific interview questions."""
        skills = self.format_skills(resume_data.get("skills", []))
        experience_summary = self.format_work_experience(resume_data.get("workExperience", []))
        education_summary = self.format_education(resume_data.get("education", []))
        highlights = self.format_highlights(resume_data.get("highlights", []))

        job_section = f"Job_Description: {job_description[:300]}\n" if job_description else ""

        prompt = f"""ROLE: Expert Technical Interviewer for {target_role}.

TASK: Generate 8-10 role-specific interview questions.

CANDIDATE_PROFILE:
Skills: {skills}
Work_Experience: {experience_summary}
Education: {education_summary}
Highlights: {highlights}

{job_section}
INSTRUCTIONS:
1. Generate 8-10 questions specific to {target_role} position.
2. Include role requirements, scenario-based, and fit assessment questions.
3. Focus on how candidate skills translate to {target_role}.
4. Include mix of "role-specific", "technical", "behavioral", "scenario-based".
5. context: 1-2 sentence explanation of WHY this question matters for {target_role}.

RESPONSE SCHEMA (MUST FOLLOW):
You MUST output a valid JSON array with exactly this structure:
[
  {{
    "type": "string (required: 'role-specific'|'technical'|'behavioral'|'scenario-based')",
    "question": "string (required)",
    "context": "string (required: 1-2 sentence reason)"
  }}
]

OUTPUT: Return ONLY the JSON array. No additional text before or after."""
        return prompt

    # ========== RESPONSE PARSING METHODS ==========
    def _parse_questions(self, response_text: str) -> List[Dict[str, str]]:
        """Parse interview questions from AI response with validation."""
        questions = self.parse_json_array_response(response_text)
        
        # Validate structure
        if not isinstance(questions, list):
            raise ValueError(f"Response must be a JSON array, got {type(questions).__name__}")
        
        if len(questions) == 0:
            raise ValueError("Response array is empty - no questions generated")
        
        validated_qs = []
        for i, q in enumerate(questions):
            if not isinstance(q, dict):
                raise ValueError(f"Question {i} is not a JSON object: {type(q).__name__}")
            
            # Check required fields
            if "type" not in q:
                raise ValueError(f"Question {i} missing required field: type")
            if "question" not in q:
                raise ValueError(f"Question {i} missing required field: question")
            if "context" not in q:
                raise ValueError(f"Question {i} missing required field: context")
            
            validated_qs.append(q)
        
        return validated_qs
