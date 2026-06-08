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


class CompareResumesService(BasePromptService):
    """
    Prompt service for /compare-resumes endpoint.
    Handles:
    - Comparative analysis prompts for multiple resumes
    - Ranking and scoring logic for resume comparison
    - Identification of strengths and weaknesses for each candidate
    - Comparative insights across all candidates
    - Designed for fast and efficient resume comparison
    Maintains same logic as existing compare resumes in router
    """

    def __init__(self):
        """Initialize the compare resumes service."""
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
        Generate role recommendations for comparison context.
        Not typically used in compare-resumes flow.   
        Args:
            resume_data: Parsed resume data dictionary
            **kwargs: Additional arguments 
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

    async def generate_comparison_analysis(
        self,
        resumes_data: List[Dict[str, Any]],
        target_role: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate comparative analysis across multiple resumes.
        Args:
            resumes_data: List of parsed resume data dictionaries
            target_role: Optional target role for comparison context
        Returns:
            Dictionary containing comparative analysis and insights
        """
        model = self.model
        prompt = self._create_comparison_prompt(resumes_data, target_role)
        response = await model.generate_content_async(prompt)
        
        try:
            analysis = self._parse_comparison_analysis(response.text)
            return analysis
        except Exception as e:
            raise ValueError(f"Failed to generate comparison analysis: {str(e)}")

    async def generate_detailed_comparison(
        self,
        resumes_data: List[Dict[str, Any]],
        target_role: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate detailed pairwise comparison of resumes with strengths/weaknesses.
        Args:
            resumes_data: List of parsed resume data dictionaries
            target_role: Optional target role for comparison context   
        Returns:
            Dictionary containing detailed comparison insights
        """
        model = self.model
        prompt = self._create_detailed_comparison_prompt(resumes_data, target_role)
        response = await model.generate_content_async(prompt)
        
        try:
            analysis = self._parse_detailed_comparison(response.text)
            return analysis
        except Exception as e:
            raise ValueError(f"Failed to generate detailed comparison: {str(e)}")

    # ========== PROMPT CREATION METHODS ==========
    def _create_role_prompt(self, resume_data: Dict[str, Any]) -> str:
        """Create a prompt for single resume role recommendation."""
        profile_block = self.render_candidate_profile(
            resume_data,
            include_personal_info=True,
            include_highlights=False
        )

        prompt = (
            "ROLE: Expert Career Advisor & Technical Recruiter.\n"
            "TASK: Produce JSON array of exactly 3 best-fit roles sorted by matchPercentage (integer 0-100).\n"
            f"{self.MATCH_RUBRIC}\n\n"
            "OUTPUT FIELDS:\n"
            "- roleName\n"
            "- matchPercentage (int 0-100)\n"
            "- reasoning (<=2 direct sentences)\n"
            "- requiredSkills (top 2-3 relevant skills)\n"
            "- missingSkills (top 2-3 critical gaps)\n\n"
            f"CANDIDATE PROFILE:\n{profile_block}\n\n"
            "RETURN: JSON array only."
        )
        return prompt

    def _create_comparison_prompt(
        self,
        resumes_data: List[Dict[str, Any]],
        target_role: Optional[str] = None
    ) -> str:
        """
        Create a prompt for high-level comparison of resumes.
        Concise format for fast comparison with JSON mode.
        Args:
            resumes_data: List of resume data dictionaries
            target_role: Optional target role for comparison context
        Returns:
            Formatted prompt string
        """
        candidates_section = self._format_candidates_section(resumes_data)
        target_section = ""
        if target_role:
            target_section = f"\nTARGET ROLE: {target_role}"

        prompt = (
            "ROLE: Expert Recruiter & Comparative Analyst.\n"
            "TASK: Compare candidates and return JSON object with high-level insights.\n\n"
            f"{candidates_section}\n{target_section}\n\n"
            "OUTPUT FIELDS:\n"
            "- overall_assessment: Brief 1-2 sentence summary\n"
            "- top_performer: Name of best candidate\n"
            "- top_performer_reason: Why they stand out (1-2 sentences)\n"
            "- key_differentiators: Object with candidate names as keys, differentiators as values\n"
            "- relative_strengths: Object with candidate names as keys, strength arrays as values (top 2-3)\n"
            "- recommendations: Interview recommendations (1-2 sentences)\n\n"
            "RETURN: JSON object only."
        )
        return prompt

    def _create_detailed_comparison_prompt(
        self,
        resumes_data: List[Dict[str, Any]],
        target_role: Optional[str] = None
    ) -> str:
        """
        Create a prompt for detailed comparison with strengths/weaknesses analysis.
        Concise format for fast detailed comparison with JSON mode.
        Args:
            resumes_data: List of resume data dictionaries
            target_role: Optional target role for comparison context
        Returns:
            Formatted prompt string
        """
        candidates_section = self._format_candidates_section(resumes_data)
        target_section = ""
        if target_role:
            target_section = f"\nTARGET ROLE: {target_role}"

        prompt = (
            "ROLE: Expert Recruiter & Detailed Analyst.\n"
            "TASK: Provide detailed candidate comparison with scores and analysis as JSON object.\n\n"
            f"{candidates_section}\n{target_section}\n\n"
            "FOR EACH CANDIDATE:\n"
            "- overall_fit_score: 0-100 integer\n"
            "- strengths: Array of top 3 strengths\n"
            "- weaknesses: Array of top 2 weaknesses\n"
            "- technical_proficiency: 'High' | 'Medium' | 'Low'\n"
            "- experience_level: 'Entry-level' | 'Mid-level' | 'Senior'\n"
            "- growth_potential: 'High' | 'Medium' | 'Low'\n\n"
            "OUTPUT FORMAT:\n"
            "- candidates: Array of candidate objects (listed above)\n"
            "- comparative_insights: 2-3 sentence overall comparison\n\n"
            "RETURN: JSON object only."
        )
        return prompt

    # ========== HELPER METHODS ==========
    def _format_candidates_section(self, resumes_data: List[Dict[str, Any]]) -> str:
        """Format candidates section for prompt."""
        candidates_section = "CANDIDATES:\n"
        for idx, resume in enumerate(resumes_data, 1):
            profile_block = self.render_candidate_profile(
                resume,
                include_personal_info=True,
                include_highlights=False
            )
            candidates_section += f"\nCandidate {idx}:\n{profile_block}\n"
        return candidates_section

    # ========== RESPONSE PARSING METHODS ==========
    def _parse_recommendations(self, response_text: str) -> List[Dict[str, Any]]:
        """Parse role recommendations from AI response."""
        recommendations = self.parse_json_array_response(response_text)
        
        # Validate structure
        if not isinstance(recommendations, list):
            raise ValueError("Response should be a JSON array of recommendations")
        
        for rec in recommendations:
            if not isinstance(rec, dict):
                raise ValueError("Each recommendation should be a JSON object")
            if "roleName" not in rec or "matchPercentage" not in rec:
                raise ValueError("Each recommendation must have roleName and matchPercentage")
        
        return recommendations

    def _parse_comparison_analysis(self, response_text: str) -> Dict[str, Any]:
        """Parse high-level comparison analysis from AI response."""
        analysis = self.parse_json_response(response_text)
        
        # Validate structure
        if not isinstance(analysis, dict):
            raise ValueError("Comparison analysis should be a JSON object")
        
        required_fields = ["overall_assessment", "top_performer"]
        for field in required_fields:
            if field not in analysis:
                raise ValueError(f"Analysis must include {field} field")
        
        return analysis

    def _parse_detailed_comparison(self, response_text: str) -> Dict[str, Any]:
        """Parse detailed comparison analysis from AI response."""
        analysis = self.parse_json_response(response_text)
        
        # Validate structure
        if not isinstance(analysis, dict):
            raise ValueError("Detailed comparison should be a JSON object")
        
        if "candidates" not in analysis or not isinstance(analysis["candidates"], list):
            raise ValueError("Detailed comparison must include candidates array")
        
        return analysis
