from typing import Dict, Any
import os
from app.models.schemas import ResumeScore, PersonalityInsights, CareerPathPrediction
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

class AdvancedAnalyzer(BasePromptService):
    def __init__(self):
        super().__init__()
        self.api_key = os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is required")

        if not GENAI_AVAILABLE or not genai:
            raise ImportError("google-generativeai package is not available")

        from app.services.genai_compat import configure_genai
        configure_genai(self.api_key)

    async def generate(self, resume_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Generate comprehensive analysis including score, personality, and career path."""
        return {
            "score": await self.calculate_resume_score(resume_data),
            "personality": await self.analyze_personality(resume_data),
            "career_path": await self.predict_career_path(resume_data)
        }

    async def calculate_resume_score(self, resume_data: Dict[str, Any]) -> ResumeScore:
        """Calculate comprehensive resume score with detailed breakdown"""
        try:
            model = self.model
            prompt = self._create_scoring_prompt(resume_data)
            response = await model.generate_content_async(prompt)
            
            if not response or not response.text:
                raise ValueError("Empty response from AI model")

            score_data = self._parse_score_response(response.text)
            return ResumeScore(**score_data)
        except Exception as e:
            raise ValueError(f"Failed to calculate resume score: {str(e)}")

    async def analyze_personality(self, resume_data: Dict[str, Any]) -> PersonalityInsights:
        """Analyze personality traits from resume content"""
        try:
            model = self.model
            prompt = self._create_personality_prompt(resume_data)
            response = await model.generate_content_async(prompt)
            
            if not response or not response.text:
                raise ValueError("Empty response from AI model")

            personality_data = self._parse_personality_response(response.text)
            return PersonalityInsights(**personality_data)
        except Exception as e:
            raise ValueError(f"Failed to analyze personality: {str(e)}")

    async def predict_career_path(self, resume_data: Dict[str, Any]) -> CareerPathPrediction:
        """Predict career progression and next steps"""
        try:
            model = self.model
            prompt = self._create_career_prompt(resume_data)
            response = await model.generate_content_async(prompt)
            
            if not response or not response.text:
                raise ValueError("Empty response from AI model")

            career_data = self._parse_career_response(response.text)
            return CareerPathPrediction(**career_data)
        except Exception as e:
            raise ValueError(f"Failed to predict career path: {str(e)}")

    def _create_scoring_prompt(self, resume_data: Dict[str, Any]) -> str:
        formatted_experience = self.format_work_experience(resume_data.get("workExperience", []))
        formatted_education = self.format_education(resume_data.get("education", []))
        formatted_skills = self.format_skills(resume_data.get("skills", []))

        return f"""
ROLE: Expert Resume Evaluator
TASK: Score resume across 5 dimensions with concise assessment
INSTRUCTIONS: Reasoning MUST be 1-2 sentences max. Strengths/weaknesses top 3 only. Be direct and concise.

CANDIDATE RESUME:
Skills: {formatted_skills}
Experience: {formatted_experience}
Education: {formatted_education}

SCORING RUBRIC:
- Technical Skills (0-100): Relevance, depth, and currency
- Experience (0-100): Quality, relevance, career progression
- Education (0-100): Relevance to goals, academic achievements
- Communication (0-100): Clarity of writing, presentation
- Overall Score: Weighted (Tech 30%, Exp 35%, Edu 20%, Comm 15%)

RESPONSE_SCHEMA:
{{
  "overall_score": <float>,
  "technical_score": <float>,
  "experience_score": <float>,
  "education_score": <float>,
  "communication_score": <float>,
  "reasoning": "<1-2 sentences max: key assessment>",
  "strengths": ["<top strength>", "<second>", "<third>"],
  "weaknesses": ["<top weakness>", "<second>", "<third>"],
  "improvement_suggestions": ["<actionable step 1>", "<step 2>", "<step 3>"]
}}

OUTPUT: Return ONLY valid JSON. Concise only.
"""

    def _create_personality_prompt(self, resume_data: Dict[str, Any]) -> str:
        # Extract text content for personality analysis
        formatted_experience = self.format_work_experience(resume_data.get("workExperience", []))
        formatted_education = self.format_education(resume_data.get("education", []))
        formatted_skills = self.format_skills(resume_data.get("skills", []))

        return f"""
ROLE: Personality and Work Style Analyst
TASK: Infer personality traits and work preferences from resume
INSTRUCTIONS: Use resume indicators (achievements, roles, skills) to score traits. Analysis MUST be 2-3 sentences max.

RESUME SUMMARY:
Experience: {formatted_experience}
Education: {formatted_education}
Skills: {formatted_skills}

PERSONALITY DIMENSIONS (0-100 scale):
- Extraversion: Social orientation, outgoing vs reserved
- Conscientiousness: Organization, discipline, reliability
- Openness: Curiosity, innovation, comfort with change
- Agreeableness: Cooperation, empathy, teamwork
- Emotional Stability: Resilience, calm under pressure

WORK PREFERENCES:
- Work Style: "Independent", "Collaborative", "Leadership", "Analytical", "Creative"
- Leadership Potential: 0-100 based on track record
- Team Player Score: 0-100 based on collaboration indicators

RESPONSE_SCHEMA:
{{
  "traits": {{
    "extraversion": <0-100>,
    "conscientiousness": <0-100>,
    "openness": <0-100>,
    "agreeableness": <0-100>,
    "emotional_stability": <0-100>
  }},
  "work_style": "<one of the 5 options>",
  "leadership_potential": <0-100>,
  "team_player_score": <0-100>,
  "analysis": "<2-3 sentences: key personality insights derived from resume>"
}}

OUTPUT: Return ONLY valid JSON. Be concise and direct.
"""

    def _create_career_prompt(self, resume_data: Dict[str, Any]) -> str:
        formatted_skills = self.format_skills(resume_data.get("skills", []))
        experience = resume_data.get("workExperience", [])
        education = resume_data.get("education", [])

        current_role = experience[0].get("title", "Entry Level") if experience else "Entry Level"
        years_exp = len(experience) * 2  # Rough estimate

        return f"""
ROLE: Career Development Expert
TASK: Predict career progression and next opportunities
INSTRUCTIONS: Identify current level, top 3 next roles, timeline, and key skill gaps. Be direct and actionable.

CANDIDATE PROFILE:
Skills: {formatted_skills}
Current Role: {current_role}
Years of Experience: Approximately {years_exp}
Education: {self.format_education(education)}

ANALYSIS REQUIREMENTS:
1. Current career level: "Entry Level", "Mid Level", "Senior Level", or "Executive"
2. Next 3 potential roles (top opportunity first)
3. Advancement timeline based on skill/experience gaps
4. Key skill developments needed for progression

RESPONSE_SCHEMA:
{{
  "current_level": "<one of the 4 levels>",
  "next_roles": ["<top opportunity>", "<second option>", "<third option>"],
  "timeline": "<e.g., '2-3 years' for next advancement>",
  "required_development": ["<skill gap 1>", "<skill gap 2>", "<skill gap 3>"]
}}

OUTPUT: Return ONLY valid JSON. Be specific and actionable.
"""

    def _parse_score_response(self, text: str) -> Dict[str, Any]:
        """Parse resume score response using base class JSON parsing."""
        return self.parse_json_response(text)

    def _parse_personality_response(self, text: str) -> Dict[str, Any]:
        """Parse personality response using base class JSON parsing."""
        return self.parse_json_response(text)

    def _parse_career_response(self, text: str) -> Dict[str, Any]:
        """Parse career response using base class JSON parsing."""
        return self.parse_json_response(text)