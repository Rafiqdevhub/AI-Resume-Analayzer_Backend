import json
from typing import Dict, List, Any
from abc import ABC, abstractmethod
try:
    import google.genai as genai
except Exception:
    try:
        import google.generativeai as genai
    except Exception:
        genai = None

class BasePromptService(ABC):
    """
    Base prompt service providing shared utilities and constants for all route-specific prompt services.
    Handles common tasks like formatting, parsing, and AI model interaction.
    Uses JSON mode for guaranteed valid JSON output from AI model.
    """

    DEFAULT_MODEL = "gemini-3.5-flash"

    MATCH_RUBRIC = (
        "MATCH RUBRIC:\n"
        "- 90-100: Perfect fit; ready immediately.\n"
        "- 75-89: Strong fit; minor gaps only.\n"
        "- 60-74: Good fit; notable gaps to close.\n"
        "- <60: Potential fit; major upskilling required."
    )
    
    SUPPORTED_FORMATS = ('.pdf', '.doc', '.docx')
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

    WARNING_THRESHOLD_BATCHES = 10
    WARNING_THRESHOLD_COMPARISONS = 10

    def __init__(self):
        """Initialize base prompt service."""
        self._model = None

    @property
    def model(self):
        """Get or initialize the generative AI model instance with JSON mode enabled.
        Uses JSON mode to force valid JSON output, eliminating need for text parsing.
        This is faster and 100% reliable.
        """
        if self._model is None:
            try:
                if not genai:
                    raise ImportError("google-generativeai package is not available")
                # Enable JSON mode for guaranteed valid JSON output when supported
                try:
                    config_cls = getattr(genai.types, "GenerateContentConfig", None) or getattr(genai.types, "GenerationConfig", None)
                    if config_cls is not None:
                        json_config = config_cls(response_mime_type="application/json")
                    else:
                        json_config = None
                except Exception:
                    json_config = None

                from app.services.genai_compat import get_model
                try:
                    self._model = get_model(self.DEFAULT_MODEL, generation_config=json_config)
                except Exception as e:
                    raise RuntimeError(f"Failed to construct compatible AI model: {str(e)}")
            except ImportError as e:
                raise ImportError(f"Failed to initialize AI model: {str(e)}")
        return self._model

    # ========== FORMATTING UTILITIES ==========
    @staticmethod
    def format_work_experience(experience_list: List[Dict[str, Any]]) -> str:
        """Format work experience section - concise."""
        if not experience_list:
            return "None"
        formatted = [
            f"- {exp['title']} at {exp['company']} ({exp.get('duration', '')})"
            for exp in experience_list
            if exp.get('title') and exp.get('company')
        ]
        return "\n".join(formatted) or "None"

    @staticmethod
    def format_education(education_list: List[Dict[str, Any]]) -> str:
        """Format education section - degree & year only."""
        if not education_list:
            return "None"
        formatted = [
            f"- {edu['degree']} ({edu.get('year')})" if edu.get('year') else f"- {edu['degree']}"
            for edu in education_list if edu.get('degree')
        ]
        return "\n".join(formatted) or "None"

    @staticmethod
    def format_skills(skills_list: List[str]) -> str:
        """Format skills section from resume data."""
        return ", ".join(skills_list) or "No skills listed"

    @staticmethod
    def format_highlights(highlights_list: List[str]) -> str:
        """Format highlights section - concise."""
        formatted = [f"- {h[:100]}" for h in highlights_list[:2]]
        return "\n".join(formatted) or "None"

    def build_candidate_profile(self, resume_data: Dict[str, Any]) -> Dict[str, Any]:
        """Build formatted candidate profile slices for prompt context."""
        return {
            "skills": self.format_skills(resume_data.get("skills", [])),
            "experience": self.format_work_experience(resume_data.get("workExperience", [])),
            "education": self.format_education(resume_data.get("education", [])),
            "highlights": self.format_highlights(resume_data.get("highlights", [])),
            "personal_info": self.extract_personal_info(resume_data)
        }

    def render_candidate_profile(
        self,
        resume_data: Dict[str, Any],
        *,
        include_personal_info: bool = False,
        include_highlights: bool = True
    ) -> str:
        """Render a compact multi-line candidate profile for prompting."""
        profile = self.build_candidate_profile(resume_data)
        lines: List[str] = []

        def _append(label: str, value: str) -> None:
            if value and value not in {"None", "No skills listed"}:
                lines.append(f"{label}: {value}")

        if include_personal_info:
            personal = profile.get("personal_info", {})
            name = personal.get("name") or "Candidate"
            lines.append(f"Name: {name}")

        _append("Skills", profile.get("skills", ""))
        _append("Experience", profile.get("experience", ""))
        _append("Education", profile.get("education", ""))

        if include_highlights:
            _append("Highlights", profile.get("highlights", ""))

        return "\n".join(lines) if lines else "No resume details provided"

    @staticmethod
    def extract_personal_info(resume_data: Dict[str, Any]) -> Dict[str, str]:
        """Extract personal information from resume data."""
        personal_info = resume_data.get("personalInfo", {})
        return {
            "name": personal_info.get("name", "Candidate"),
            "email": personal_info.get("email", "Not provided"),
            "phone": personal_info.get("phone", "Not provided"),
            "location": personal_info.get("location", "Not provided")
        }

    # ========== PARSING UTILITIES ==========
    @staticmethod
    def parse_json_response(response_text: str, start_marker: str = "{", end_marker: str = "}") -> Dict[str, Any]:
        """
        Parse JSON response from AI model.
        With JSON mode enabled, the model ONLY outputs valid JSON, so direct parsing works.
        This eliminates complex text extraction logic.
        """
        try:
            return json.loads(response_text)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON response: {str(e)}\nResponse: {response_text[:200]}")

    @staticmethod
    def parse_json_array_response(response_text: str) -> List[Dict[str, Any]]:
        """
        Parse JSON array response from AI model.
        With JSON mode enabled, the model ONLY outputs valid JSON arrays, so direct parsing works.
        No need to search for '[' and ']' in the text.
        """
        try:
            return json.loads(response_text)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON array response: {str(e)}\nResponse: {response_text[:200]}")

    # ========== VALIDATION UTILITIES ==========
    @staticmethod
    def validate_file_format(filename: str) -> bool:
        """Validate if file has supported format."""
        if not filename:
            return False
        return filename.lower().endswith(BasePromptService.SUPPORTED_FORMATS)

    @staticmethod
    def validate_file_size(file_size: int) -> bool:
        """Validate if file is within size limit."""
        return file_size <= BasePromptService.MAX_FILE_SIZE

    # ========== ERROR HANDLING ==========
    @staticmethod
    def handle_file_error(error_message: str) -> str:
        """Convert error messages to user-friendly messages."""
        if "PDF" in error_message:
            return "Error reading PDF file. Please ensure it's not corrupted or password protected."
        elif "DOCX" in error_message or "DOC" in error_message:
            return "Error reading document file. Please ensure it's a valid Word document."
        else:
            return "Analysis failed. Please try again."

    # ========== ABSTRACT METHODS ==========
    @abstractmethod
    async def generate(self, resume_data: Dict[str, Any], **kwargs) -> Any:
        """Generate output based on prompt service logic. Must be implemented by subclasses."""
        pass
