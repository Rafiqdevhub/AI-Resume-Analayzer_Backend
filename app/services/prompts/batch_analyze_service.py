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


class BatchAnalyzeService(BasePromptService):
    """
    Prompt service for /batch-analyze endpoint.
    Handles:
    - Optimized prompts for batch processing of multiple resumes
    - Faster analysis without unnecessary details
    - Role recommendations for batch context
    - Maintains consistency with single resume analysis
    - Designed for efficiency in batch operations
    Maintains same logic as existing batch analysis in router
    """

    def __init__(self):
        """Initialize the batch analysis service."""
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
                raise
        return self._model

    async def generate(self, resume_data: Dict[str, Any], **kwargs) -> List[RoleRecommendation]:
        """
        Generate general role recommendations for batch processing.
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
        Analyze if candidate fits the target role for batch processing.
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

    # ========== PROMPT CREATION METHODS ==========
    def _create_role_prompt(self, resume_data: Dict[str, Any]) -> str:
        """
        Create an optimized prompt for batch role recommendation.
        Ultra-concise format for fast batch processing.
        Args:
            resume_data: Parsed resume data dictionary
        Returns:
            Formatted prompt string for AI model
        """
        profile_block = self.render_candidate_profile(
            resume_data,
            include_personal_info=False,
            include_highlights=False
        )
        prompt = (
            "ROLE: Expert Career Advisor & Technical Recruiter.\n"
            "TASK: Produce JSON array of exactly 5 best-fit roles sorted by matchPercentage (integer 0-100).\n"
            f"{self.MATCH_RUBRIC}\n\n"
            "OUTPUT FIELDS:\n"
            "- roleName\n"
            "- matchPercentage (int 0-100)\n"
            "- reasoning (<=2 direct sentences)\n"
            "- requiredSkills (top 2-3 relevant skills candidate already has)\n"
            "- missingSkills (top 2-3 critical gaps)\n\n"
            "CONCISENESS: facts only, no fluff.\n\n"
            f"CANDIDATE SNAPSHOT:\n{profile_block}\n\n"
            "RETURN: JSON array only."
        )
        return prompt

    def _create_role_fit_prompt(
        self,
        resume_data: Dict[str, Any],
        target_role: str,
        job_description: Optional[str] = None
    ) -> str:
        """
        Create an optimized prompt for batch role fit analysis.
        Ultra-concise format for fast batch processing.
        Args:
            resume_data: Parsed resume data dictionary
            target_role: Target role to analyze
            job_description: Optional job description 
        Returns:
            Formatted prompt string for AI model
        """
        profile_block = self.render_candidate_profile(
            resume_data,
            include_personal_info=False,
            include_highlights=False
        )
        job_section = ""
        if job_description:
            job_section = f"\nJOB DESCRIPTION (truncated to 300 chars):\n{job_description[:300]}"

        prompt = (
            "ROLE: Expert Recruiter & Career Analyst.\n"
            f"TASK: Score candidate fit for {target_role} and return JSON array of the target role plus strongest alternatives sorted by matchPercentage (integer 0-100). First entry must be {target_role}.\n"
            f"{self.MATCH_RUBRIC}\n\n"
            "OUTPUT FIELDS:\n"
            "- roleName\n"
            "- matchPercentage (int 0-100)\n"
            "- reasoning (<=2 direct sentences)\n"
            "- requiredSkills (top 2-3 proven skills)\n"
            "- missingSkills (top 2-3 gaps)\n\n"
            "CONCISENESS: direct facts only.\n\n"
            f"CANDIDATE SNAPSHOT:\n{profile_block}{job_section}\n\n"
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
