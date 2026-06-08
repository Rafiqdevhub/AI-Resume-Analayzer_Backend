import os
from typing import Dict, List, Any, Optional
from app.models.schemas import RoleRecommendation
from app.services.prompts.base_prompt_service import BasePromptService

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


class HiredeskService(BasePromptService):
    """
    Prompt service for /hiredesk-analyze endpoint.
    Handles:
    - Comprehensive resume analysis
    - Role recommendations with best fit detection
    - Role fit analysis with target role and job description
    - Question generation for interviews
    - Resume scoring and personality analysis
    - Career path prediction
    Maintains same logic as existing services combined (RoleRecommender, QuestionGenerator, AdvancedAnalyzer)
    """
    def __init__(self):
        """Initialize the hiredesk analysis service."""
        super().__init__()
        self.api_key = os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is required")
        if not GENAI_AVAILABLE or not genai:
            raise ImportError("google-generativeai package is not available")
        from app.services.genai_compat import configure_genai
        configure_genai(self.api_key)
        self._model = None

    @property
    def model(self):
        """Get the generative AI model instance with JSON mode enabled."""
        if self._model is None:
            if not GENAI_AVAILABLE or not genai:
                raise ImportError("google-generativeai package is not available")
            # Force JSON output from the model
            try:
                try:
                    json_config = genai.types.GenerationConfig(  # type: ignore[attr-defined]
                        response_mime_type="application/json"
                    )
                except Exception:
                    json_config = None
                from app.services.genai_compat import get_model
                self._model = get_model(self.DEFAULT_MODEL, generation_config=json_config)
            except Exception:
                # If get_model raised, surface the error to caller
                raise
        return self._model

    async def generate(self, resume_data: Dict[str, Any], **kwargs) -> List[RoleRecommendation]:
        """
        Generate general role recommendations for comprehensive analysis.
        Args:
            resume_data: Parsed resume data dictionary
            **kwargs: Additional arguments (not used)
        Returns:
            List of RoleRecommendation objects
        """
        model = self.model
        prompt = self._create_role_prompt(resume_data)
        response = await model.generate_content_async(prompt)
        
        try:
            role_recommendations = self._parse_recommendations(response.text)
            return [
                RoleRecommendation(
                    roleName=rec["roleName"],
                    matchPercentage=rec["matchPercentage"],
                    reasoning=rec["reasoning"],
                    requiredSkills=rec.get("requiredSkills", []),
                    missingSkills=rec.get("missingSkills", [])
                ) for rec in role_recommendations
            ]
        except Exception as e:
            raise ValueError(f"Failed to generate role recommendations: {str(e)}")

    async def analyze_role_fit(
        self,
        resume_data: Dict[str, Any],
        target_role: str,
        job_description: Optional[str] = None
    ) -> List[RoleRecommendation]:
        """
        Analyze if candidate fits the target role.
        Args:
            resume_data: Parsed resume data dictionary
            target_role: Target job role to analyze fit for
            job_description: Optional job description for better analysis
        Returns:
            List of RoleRecommendation objects with target role as primary
        """
        model = self.model
        prompt = self._create_role_fit_prompt(resume_data, target_role, job_description)
        response = await model.generate_content_async(prompt)
        
        try:
            role_recommendations = self._parse_recommendations(response.text)
            return [
                RoleRecommendation(
                    roleName=rec["roleName"],
                    matchPercentage=rec["matchPercentage"],
                    reasoning=rec["reasoning"],
                    requiredSkills=rec.get("requiredSkills", []),
                    missingSkills=rec.get("missingSkills", [])
                ) for rec in role_recommendations
            ]
        except Exception as e:
            raise ValueError(f"Failed to analyze role fit: {str(e)}")

    async def generate_interview_questions(
        self,
        resume_data: Dict[str, Any],
        target_role: Optional[str] = None,
        job_description: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """
        Generate interview questions tailored to the candidate and role.
        Args:
            resume_data: Parsed resume data dictionary
            target_role: Optional target role for role-specific questions
            job_description: Optional job description for better questions
        Returns:
            List of question dictionaries with type, question, and context
        """
        model = self.model
        
        if target_role:
            prompt = self._create_role_specific_questions_prompt(
                resume_data, target_role, job_description
            )
        else:
            prompt = self._create_general_questions_prompt(resume_data)
        
        response = await model.generate_content_async(prompt)
        
        try:
            questions = self._parse_questions(response.text)
            return questions
        except Exception as e:
            raise ValueError(f"Failed to generate interview questions: {str(e)}")

    # ========== PROMPT CREATION METHODS ==========
    def _create_role_prompt(self, resume_data: Dict[str, Any]) -> str:
        """Create optimized structured prompt for role recommendations."""
        profile_block = self.render_candidate_profile(resume_data)
        prompt = (
            "ROLE: Expert Career Advisor & Technical Recruiter.\n"
            "TASK: Produce JSON array of exactly 5 best-fit roles sorted by matchPercentage (integer 0-100).\n"
            f"{self.MATCH_RUBRIC}\n\n"
            "OUTPUT FIELDS:\n"
            "- roleName\n"
            "- matchPercentage (int 0-100)\n"
            "- reasoning (<=2 sentences grounded in resume data)\n"
            "- requiredSkills (top 2-3 skills the candidate already has)\n"
            "- missingSkills (top 2-3 critical gaps)\n\n"
            f"CANDIDATE PROFILE:\n{profile_block}\n\n"
            "RETURN: JSON array only, no narration."
        )
        return prompt

    def _create_role_fit_prompt(
        self,
        resume_data: Dict[str, Any],
        target_role: str,
        job_description: Optional[str] = None
    ) -> str:
        """Create optimized structured prompt for role fit analysis."""
        profile_block = self.render_candidate_profile(resume_data)
        job_section = ""
        if job_description:
            job_section = f"\nJOB DESCRIPTION (truncated to 300 chars):\n{job_description[:300]}"

        prompt = (
            "ROLE: Expert Recruiter & Career Analyst.\n"
            f"TASK: Score candidate fit for {target_role} and list the strongest alternatives as a JSON array sorted by matchPercentage. First entry must be {target_role}.\n"
            f"{self.MATCH_RUBRIC}\n\n"
            "OUTPUT FIELDS:\n"
            "- roleName\n"
            "- matchPercentage (int 0-100)\n"
            "- reasoning (<=2 sentences tied to evidence)\n"
            "- requiredSkills (top 2-3 proven skills)\n"
            "- missingSkills (top 2-3 gaps)\n\n"
            f"CANDIDATE PROFILE:\n{profile_block}{job_section}\n\n"
            "RETURN: JSON array only, no commentary."
        )
        return prompt

    def _create_general_questions_prompt(self, resume_data: Dict[str, Any]) -> str:
        """Create optimized structured prompt for general interview questions."""
        profile_block = self.render_candidate_profile(
            resume_data,
            include_personal_info=False,
            include_highlights=False
        )

        prompt = (
            "ROLE: Senior AI Hiring Manager.\n"
            "TASK: Output JSON array of 8-10 interview questions covering technical, behavioral, and problem-solving angles.\n\n"
            "EACH ITEM:\n"
            "- type: one of \"technical\" | \"behavioral\" | \"problem-solving\"\n"
            "- question: targeted to resume evidence\n"
            "- context: 1 sentence explaining the intent\n\n"
            f"CANDIDATE PROFILE:\n{profile_block}\n\n"
            "RETURN: JSON array only."
        )
        return prompt

    def _create_role_specific_questions_prompt(
        self,
        resume_data: Dict[str, Any],
        target_role: str,
        job_description: Optional[str] = None
    ) -> str:
        """Create optimized structured prompt for role-specific questions."""
        profile_block = self.render_candidate_profile(
            resume_data,
            include_personal_info=False,
            include_highlights=False
        )
        job_section = ""
        if job_description:
            job_section = f"\nJOB DESCRIPTION (truncated to 300 chars):\n{job_description[:300]}"

        prompt = (
            f"ROLE: Expert Technical Interviewer for {target_role}.\n"
            "TASK: Output JSON array of 8-10 questions focused on this role.\n\n"
            "EACH ITEM:\n"
            "- type: \"role-specific\" | \"technical\" | \"behavioral\" | \"scenario-based\"\n"
            f"- question: tailored to {target_role}\n"
            f"- context: 1 sentence explaining why the question matters for {target_role}\n\n"
            f"CANDIDATE PROFILE:\n{profile_block}{job_section}\n\n"
            "RETURN: JSON array only."
        )
        return prompt

    # ========== RESPONSE PARSING METHODS ==========
    def _parse_recommendations(self, response_text: str) -> List[Dict[str, Any]]:
        """Parse role recommendations from AI response with validation."""
        recommendations = self.parse_json_array_response(response_text)
        
        # Validate structure
        if not isinstance(recommendations, list):
            raise ValueError(f"Response must be a JSON array, got {type(recommendations).__name__}")
        
        if len(recommendations) == 0:
            raise ValueError("Response array is empty - no recommendations provided")
        
        validated_recs = []
        for i, rec in enumerate(recommendations):
            if not isinstance(rec, dict):
                raise ValueError(f"Recommendation {i} is not a JSON object: {type(rec).__name__}")
            
            # Check required fields
            if "roleName" not in rec:
                raise ValueError(f"Recommendation {i} missing required field: roleName")
            if "matchPercentage" not in rec:
                raise ValueError(f"Recommendation {i} missing required field: matchPercentage")
            if "reasoning" not in rec:
                raise ValueError(f"Recommendation {i} missing required field: reasoning")
            
            # Validate matchPercentage is a number
            try:
                match_pct = float(rec["matchPercentage"])
                if not (0 <= match_pct <= 100):
                    raise ValueError(f"matchPercentage {match_pct} not in range 0-100")
                rec["matchPercentage"] = int(match_pct)
            except (ValueError, TypeError) as e:
                raise ValueError(f"Recommendation {i}: matchPercentage must be a number 0-100: {str(e)}")
            
            validated_recs.append(rec)
        
        return validated_recs

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
