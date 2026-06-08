import base64
from io import BytesIO

from fastapi import APIRouter, UploadFile, HTTPException, Request, Form, File, Depends, status
from starlette.datastructures import UploadFile as StarletteUploadFile
from pydantic import ValidationError
from typing import Optional, List, Union
import logging
from app.services.resume_parser import ResumeParser
from app.services.advanced_analyzer import AdvancedAnalyzer
from app.services.candidate_selector import CandidateSelector
from app.services.prompts import (
    HiredeskService,
    BatchAnalyzeService,
    CompareResumesService
)
from app.models.schemas import ResumeAnalysisResponse, ResumeData, Question, CandidateSelectionResponse, CandidateSelectionResult
from app.dependencies.auth import get_current_user, TokenData
from app.services.rate_limit_service import rate_limit_service

logger = logging.getLogger(__name__)


router = APIRouter()


class InMemoryUploadFile:
    def __init__(self, filename: str, content: bytes, content_type: str = "application/octet-stream"):
        self.filename = filename
        self.content_type = content_type
        self._file = BytesIO(content)

    async def read(self):
        return self._file.read()

    async def seek(self, offset: int, whence: int = 0):
        return self._file.seek(offset, whence)

    async def close(self):
        return None


def _decode_file_string(file_value: str) -> bytes:
    try:
        return base64.b64decode(file_value, validate=True)
    except Exception:
        return file_value.encode("latin-1")


def _guess_filename(content: bytes) -> str:
    if content.startswith(b"%PDF"):
        return "resume.pdf"
    if content.startswith(b"PK"):
        return "resume.docx"
    if content.startswith(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"):
        return "resume.doc"
    return "resume.bin"


def _infer_filename_from_content(content: bytes, content_type: Optional[str] = None) -> Optional[str]:
    if content.startswith(b"%PDF"):
        return "resume.pdf"
    if content.startswith(b"PK"):
        return "resume.docx"
    if content.startswith(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"):
        return "resume.doc"
    if content_type:
        type_lower = content_type.lower()
        if "pdf" in type_lower:
            return "resume.pdf"
        if "word" in type_lower or "doc" in type_lower:
            return "resume.docx"
    return None


async def _resolve_file_upload(request: Request, file: Optional[UploadFile]) -> Optional[UploadFile]:
    if file is not None:
        return file

    # Fallback for form fields where file content was submitted as a string or alternate field name.
    content_type = request.headers.get("content-type", "").lower()
    if "multipart/form-data" in content_type:
        try:
            form = await request.form()
        except Exception:
            form = None
        if form:
            if "file" in form:
                form_file = form.get("file")
                if isinstance(form_file, (UploadFile, StarletteUploadFile)):
                    return form_file
                if isinstance(form_file, str):
                    bytes_data = _decode_file_string(form_file)
                    filename = _guess_filename(bytes_data)
                    return InMemoryUploadFile(filename=filename, content=bytes_data)

            for key, form_value in form.items():
                if isinstance(form_value, (UploadFile, StarletteUploadFile)):
                    return form_value
                if isinstance(form_value, str) and key.lower() in {"resume", "document", "upload", "file_content", "data"}:
                    bytes_data = _decode_file_string(form_value)
                    filename = _guess_filename(bytes_data)
                    return InMemoryUploadFile(filename=filename, content=bytes_data)

    # Fallback for JSON/other non-multipart payloads.
    body = None
    try:
        body = await request.json()
    except Exception:
        body = None

    if isinstance(body, dict):
        raw_file = None
        for key in ("file", "resume", "resume_file", "document", "data", "file_data"):
            if body.get(key) is not None:
                raw_file = body.get(key)
                break

        if raw_file is not None:
            if isinstance(raw_file, str):
                bytes_data = _decode_file_string(raw_file)
            elif isinstance(raw_file, (bytes, bytearray)):
                bytes_data = bytes(raw_file)
            else:
                bytes_data = None

            if bytes_data is not None:
                filename = body.get("filename") or _guess_filename(bytes_data)
                return InMemoryUploadFile(filename=filename, content=bytes_data)

    # Fallback for raw binary/text body directly sent without JSON.
    try:
        raw_body = await request.body()
    except Exception:
        raw_body = b""

    if raw_body:
        return InMemoryUploadFile(filename=_guess_filename(raw_body), content=raw_body)

    return None


def format_validation_error(error: ValidationError) -> str:
    error_messages = []
    for err in error.errors():
        field = err["loc"][-1] if err["loc"] else "Unknown field"
        if isinstance(field, int):
            parent = err["loc"][-2] if len(err["loc"]) > 1 else "item"
            field = f"{parent} #{field + 1}"
        error_messages.append(f"{field}: {err['msg']}")
    return "Validation Error: " + "; ".join(error_messages)


@router.post("/hiredesk-analyze", response_model=ResumeAnalysisResponse, status_code=status.HTTP_200_OK)
async def hiredesk_analyze(
    request: Request,
    target_role: Optional[str] = Form(None),
    job_description: Optional[str] = Form(None),
    current_user: TokenData = Depends(get_current_user)
):
    try:
        # Parse incoming file from raw request rather than relying on FastAPI body validation.
        file = await _resolve_file_upload(request, None)
        logger.debug("hiredesk_analyze: resolved file type=%s filename=%s content_type=%s",
                     type(file), getattr(file, "filename", None), getattr(file, "content_type", None))

        # Also support JSON payloads that include target_role/job_description.
        if (not target_role or not job_description) and request.headers.get("content-type", "").startswith("application/json"):
            try:
                body = await request.json()
            except Exception:
                body = None
            if isinstance(body, dict):
                target_role = target_role or body.get("target_role")
                job_description = job_description or body.get("job_description")

        if file is None:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={
                    "success": False,
                    "message": "Missing resume file. Ensure you send the file as multipart/form-data using the 'file' or 'resume' field, or provide a JSON body with a base64/raw file value.",
                    "error": "VALIDATION_ERROR"
                }
            )

        if not target_role:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={
                    "success": False,
                    "message": "Missing target_role. Provide a target_role string in multipart form or JSON body.",
                    "error": "VALIDATION_ERROR"
                }
            )

        if not job_description:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={
                    "success": False,
                    "message": "Missing job_description. Provide a job_description string in multipart form or JSON body.",
                    "error": "VALIDATION_ERROR"
                }
            )

        # Check rate limit before processing
        rate_limit_status = await rate_limit_service.check_user_upload_limit(current_user.email)

        if not rate_limit_status["allowed"]:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail={
                    "success": False,
                    "message": "Upload limit reached. You can upload up to 10 files per account.",
                    "error": "RATE_LIMIT_EXCEEDED",
                    "current_count": rate_limit_status["current_count"],
                    "limit": rate_limit_status["limit"],
                    "remaining": rate_limit_status["remaining"]
                }
            )

        file_content = await file.read()
        if len(file_content) > 10 * 1024 * 1024:  # 10MB
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={
                    "success": False,
                    "message": "File too large. Maximum size is 10MB.",
                    "error": "VALIDATION_ERROR"
                }
            )

        resolved_filename = getattr(file, "filename", "") or ""
        if not resolved_filename.lower().endswith(('.pdf', '.doc', '.docx')):
            inferred_name = _infer_filename_from_content(file_content, getattr(file, "content_type", ""))
            if inferred_name:
                resolved_filename = inferred_name
                try:
                    setattr(file, "filename", inferred_name)
                except Exception:
                    pass
            else:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail={
                        "success": False,
                        "message": "Invalid file format. Please upload PDF, DOC, or DOCX files only.",
                        "error": "VALIDATION_ERROR"
                    }
                )

        # Reset file pointer for processing
        await file.seek(0)
        parser = ResumeParser()
        resume_data = await parser.parse(file)

        # Initialize hiredesk service for comprehensive analysis
        hiredesk_service = HiredeskService()
        
        # Get role recommendations
        role_recommendations = await hiredesk_service.generate(resume_data)
        
        # Pick the top recommended role as the best fit
        best_fit_role = role_recommendations[0] if role_recommendations else target_role
        
        # Always use string for role name
        if isinstance(best_fit_role, str):
            best_fit_role_name = best_fit_role
        elif hasattr(best_fit_role, 'roleName'):
            best_fit_role_name = best_fit_role.roleName
        else:
            best_fit_role_name = str(best_fit_role)

        # Analyze fit for the best-fit role
        fit_result = await hiredesk_service.analyze_role_fit(resume_data, best_fit_role_name, job_description)
        if isinstance(fit_result, dict):
            fit_status = "fit" if fit_result.get("fit", False) else "not fit"
            reasoning = fit_result.get("reasoning", "No reasoning provided.")
        else:
            fit_status = "fit" if fit_result else "not fit"
            reasoning = "Analyzed based on resume data."

        # Generate questions for the best-fit role and general resume
        questions = []
        try:
            # General resume-based questions
            general_questions_data = await hiredesk_service.generate_interview_questions(resume_data)
            
            # Role-specific questions if candidate is fit
            role_questions_data = []
            if fit_status == "fit":
                role_questions_data = await hiredesk_service.generate_interview_questions(
                    resume_data, best_fit_role_name, job_description
                )
            
            # Combine and deduplicate questions
            all_questions = general_questions_data + role_questions_data
            seen = set()
            questions = []
            for q in all_questions:
                q_text = q.get("question") if isinstance(q, dict) else getattr(q, "question", None)
                if q_text and q_text not in seen:
                    questions.append(Question(**q) if isinstance(q, dict) else q)
                    seen.add(q_text)
        except Exception:
            questions = []

        # Generate advanced analysis
        advanced_analyzer = AdvancedAnalyzer()
        resume_score = await advanced_analyzer.calculate_resume_score(resume_data)
        personality_insights = await advanced_analyzer.analyze_personality(resume_data)
        career_path = await advanced_analyzer.predict_career_path(resume_data)

        response = ResumeAnalysisResponse(
            resumeData=ResumeData(**resume_data),
            questions=questions,
            roleRecommendations=role_recommendations,
            resumeScore=resume_score,
            personalityInsights=personality_insights,
            careerPath=career_path
        )
        # Increment filesUploaded counter for single file upload
        await rate_limit_service.increment_files_uploaded(current_user.email, 1)

        return {
            "success": True,
            "fit_status": fit_status,
            "reasoning": reasoning,
            "resumeData": response.resumeData,
            "roleRecommendations": response.roleRecommendations,
            "questions": response.questions,
            "best_fit_role": best_fit_role_name,
            "resumeScore": response.resumeScore,
            "personalityInsights": response.personalityInsights,
            "careerPath": response.careerPath
        }
    except HTTPException:
        raise
    except ValidationError as e:
        raise HTTPException(
            status_code=422,
            detail={
                "success": False,
                "message": format_validation_error(e),
                "error": "VALIDATION_ERROR"
            }
        )
    except Exception as e:
        # Preserve the original exception message for clearer diagnostics while
        # mapping obvious file-read issues to user-friendly messages.
        error_message = str(e)
        if "PDF" in error_message:
            user_message = "Error reading PDF file. Please ensure it's not corrupted or password protected."
        elif "DOCX" in error_message:
            user_message = "Error reading DOCX file. Please ensure it's a valid Word document."
        else:
            # Propagate original message to aid debugging instead of masking it.
            user_message = error_message or "Analysis failed. Please try again."

        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "message": user_message,
                "error": "SERVER_ERROR"
            }
        )


@router.post("/batch-analyze", response_model=dict, status_code=status.HTTP_200_OK)
async def batch_analyze_resumes(
    files: List[UploadFile],
    request: Request,
    target_role: Optional[str] = Form(None),
    job_description: Optional[str] = Form(None),
    current_user: TokenData = Depends(get_current_user)
):
    """
    Analyze multiple resumes in batch with rate limiting
    Requirements:
    - Authentication required (JWT token)
    - Maximum 5 files per batch
    - Maximum 10 files per account (free tier)
    - Tracks: batch_analysis counter and filesUploaded counter
    Response includes batch summary and updated usage statistics
    """
    try:
        user_email = current_user.email
        partial_upload = False
        files_rejected = 0

        # ========== STEP 1: VALIDATE BATCH SIZE ==========
        if not files or len(files) == 0:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={
                    "success": False,
                    "message": "No files provided",
                    "error": "VALIDATION_ERROR"
                }
            )

        if len(files) > rate_limit_service.batch_size_limit:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={
                    "success": False,
                    "message": f"Maximum {rate_limit_service.batch_size_limit} files per batch. You submitted {len(files)}.",
                    "error": "BATCH_SIZE_EXCEEDED",
                    "batch_limit": rate_limit_service.batch_size_limit,
                    "submitted": len(files)
                }
            )

        # ========== STEP 2: CHECK RATE LIMIT ==========
        rate_limit_check = await rate_limit_service.check_batch_analysis_limit(
            user_email,
            len(files)
        )

        if not rate_limit_check["allowed"]:

            if rate_limit_check["reason"] == "batch_size_exceeded":
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail={
                        "success": False,
                        "message": rate_limit_check["message"],
                        "error": "BATCH_SIZE_EXCEEDED",
                        "batch_limit": rate_limit_service.batch_size_limit,
                        "submitted": rate_limit_check["submitted"]
                    }
                )
            else:  # file_limit_exceeded
                files_allowed = rate_limit_check.get("files_allowed", 0)
                
                if files_allowed <= 0:
                    # User has hit the limit completely
                    raise HTTPException(
                        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                        detail={
                            "success": False,
                            "message": rate_limit_check["message"],
                            "error": "RATE_LIMIT_EXCEEDED",
                            "current_count": rate_limit_check.get("current_files_uploaded", 0),
                            "batch_size": rate_limit_check.get("batch_size", 0),
                            "limit": rate_limit_check.get("files_limit", 10),
                            "files_allowed": 0,
                            "upgrade_required": True,
                            "upgrade_message": f"You've reached your free limit of {rate_limit_check.get('files_limit', 10)} files. Upgrade to analyze more resumes."
                        }
                    )
                else:
                    # User can upload some files - we'll process only what's allowed
                    # This is communicated in the response instead of throwing an error
                    # Continue with partial file processing
                    files = files[:files_allowed]  # Trim to allowed count
                    partial_upload = True
                    files_rejected = rate_limit_check.get("would_exceed_by", 0)
        else:
            partial_upload = False
            files_rejected = 0

        # ========== STEP 3: PROCESS FILES ==========
        results = []
        successful_files = []
        failed_files = []

        for file in files:
            batch_service = BatchAnalyzeService()
            logger.debug("batch_analyze: processing file type=%s filename=%s content_type=%s",
                         type(file), getattr(file, "filename", None), getattr(file, "content_type", None))
            try:
                if not file.filename:
                    raise ValueError("No filename provided")

                if not file.filename.lower().endswith(('.pdf', '.doc', '.docx')):
                    raise ValueError("Invalid file format. Please upload PDF, DOC, or DOCX files only.")

                file_content = await file.read()
                if len(file_content) > 10 * 1024 * 1024:
                    raise ValueError("File too large. Maximum size is 10MB.")

                # Reset file pointer for processing
                await file.seek(0)

                # Parse resume
                parser = ResumeParser()
                resume_data = await parser.parse(file)

                # Generate role recommendations using batch service
                if target_role:
                    role_recommendations = await batch_service.analyze_role_fit(
                        resume_data, target_role, job_description
                    )
                else:
                    role_recommendations = await batch_service.generate(resume_data)

                advanced_analyzer = AdvancedAnalyzer()
                resume_score = await advanced_analyzer.calculate_resume_score(resume_data)
                personality_insights = await advanced_analyzer.analyze_personality(resume_data)
                career_path = await advanced_analyzer.predict_career_path(resume_data)

                response = ResumeAnalysisResponse(
                    resumeData=ResumeData(**resume_data),
                    questions=[], 
                    roleRecommendations=role_recommendations,
                    resumeScore=resume_score,
                    personalityInsights=personality_insights,
                    careerPath=career_path
                )

                results.append({
                    "file_name": file.filename,
                    "status": "success",
                    "data": response,
                    "error": None
                })
                successful_files.append(file.filename)

            except ValueError as e:
                results.append({
                    "file_name": file.filename,
                    "status": "validation_error",
                    "data": None,
                    "error": str(e)
                })
                failed_files.append((file.filename, str(e)))

            except Exception as e:
                error_message = str(e)
                print(f"DEBUG: Exception in batch_analyze for {file.filename}: {type(e).__name__}: {error_message}")
                import traceback
                traceback.print_exc()
                
                detailed_error = error_message
                
                if "PDF" in error_message:
                    display_error = "Error reading PDF file. Please ensure it's not corrupted."
                elif "DOCX" in error_message:
                    display_error = "Error reading DOCX file. Please ensure it's a valid document."
                else:
                    # Prefer the detailed error from the exception for debugging
                    display_error = detailed_error or "Analysis failed. Please try again."

                results.append({
                    "file_name": file.filename,
                    "status": "error",
                    "data": None,
                    "error": display_error
                })
                failed_files.append((file.filename, display_error))

        # ========== STEP 4: TRACK UPLOADS ==========
        successful_count = len(successful_files)

        if successful_count > 0:
            await rate_limit_service.increment_batch_counter(user_email, successful_count)

        # ========== STEP 5: GET UPDATED STATS ==========
        updated_usage = await rate_limit_service.get_feature_usage(user_email)
        if updated_usage is None:
            updated_usage = {
                "files_uploaded": 0,
                "batch_analysis": 0,
                "compare_resumes": 0
            }

        warning_at_batches = 10  # Warning threshold for batch operations
        approaching_limit = updated_usage["batch_analysis"] >= warning_at_batches

        # ========== STEP 6: BUILD RESPONSE ==========
        batch_response = {
            "success": True,
            "message": f"Batch analysis completed. {successful_count} of {len(files)} files processed successfully.",
            "batch_summary": {
                "total_submitted": len(files),
                "successful": successful_count,
                "failed": len(failed_files),
                "success_rate": f"{(successful_count / len(files) * 100):.1f}%" if len(files) > 0 else "0%"
            },
            "usage_stats": {
                "batches_processed": updated_usage["batch_analysis"],
                "approaching_limit": approaching_limit,
                "approaching_limit_threshold": warning_at_batches
            },
            "results": results
        }

        if partial_upload and files_rejected > 0:
            batch_response["upload_limit_info"] = {
                "reached_limit": True,
                "files_rejected": files_rejected,
                "message": f"{files_rejected} file(s) were not processed because they would exceed your free limit of 10 files.",
                "upgrade_prompt": {
                    "show": True,
                    "message": f"You've reached your free file limit. {files_rejected} file(s) could not be processed.",
                    "cta": "Upgrade now to analyze all your resumes",
                    "upgrade_required": True
                }
            }

        if approaching_limit:
            batch_response["upgrade_prompt"] = {
                "show": True,
                "message": f"You've processed {updated_usage['batch_analysis']} batch operations.",
                "cta": "Upgrade now to process unlimited batches",
                "batches_processed": updated_usage['batch_analysis']
            }

        if failed_files:
            batch_response["failed_files_details"] = [
                {"filename": fname, "error": error}
                for fname, error in failed_files
            ]

        return batch_response

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "success": False,
                "message": "Batch analysis failed. Please try again.",
                "error": "SERVER_ERROR"
            }
        )


@router.post("/compare-resumes")
async def compare_resumes(
    files: List[UploadFile] = File(...),
    current_user: TokenData = Depends(get_current_user)
):
    """
    Compare multiple resumes and rank them.
    Requires: 2-5 resumes
    Rate Limit: 1 comparison per batch (same as batch analysis)
    Authentication: Required (JWT token)
    """
    try:
        user_email = current_user.email

        # ========== STEP 1: VALIDATE FILE COUNT ==========
        if not files or len(files) == 0:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={
                    "success": False,
                    "message": "No files provided",
                    "error": "VALIDATION_ERROR"
                }
            )

        if len(files) < 2:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={
                    "success": False,
                    "message": "Minimum 2 resumes required for comparison",
                    "error": "VALIDATION_ERROR"
                }
            )

        if len(files) > 5:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={
                    "success": False,
                    "message": f"Maximum 5 resumes allowed for comparison. You submitted {len(files)}.",
                    "error": "VALIDATION_ERROR",
                    "limit": 5,
                    "submitted": len(files)
                }
            )

        # ========== STEP 2: CHECK RATE LIMIT ==========
        rate_limit_check = await rate_limit_service.check_batch_analysis_limit(
            user_email,
            len(files)
        )

        if not rate_limit_check["allowed"]:
            if rate_limit_check["reason"] == "batch_size_exceeded":
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail={
                        "success": False,
                        "message": rate_limit_check["message"],
                        "error": "BATCH_SIZE_EXCEEDED",
                        "batch_limit": rate_limit_service.batch_size_limit,
                        "submitted": rate_limit_check["submitted"]
                    }
                )
            else: 
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail={
                        "success": False,
                        "message": rate_limit_check["message"],
                        "error": "RATE_LIMIT_EXCEEDED",
                        "current_count": rate_limit_check["current_files_uploaded"],
                        "files_submitted": len(files),
                        "limit": rate_limit_check["files_limit"],
                        "would_exceed_by": rate_limit_check["would_exceed_by"],
                        "upgrade_required": True,
                        "upgrade_message": f"You've reached your free limit of {rate_limit_check['files_limit']} files. Upgrade to compare more resumes."
                    }
                )

        # ========== STEP 3: PROCESS & ANALYZE RESUMES ==========
        candidates = []
        failed_files = []
        compare_service = CompareResumesService()

        for file in files:
            logger.debug("compare_resumes: processing file type=%s filename=%s content_type=%s",
                         type(file), getattr(file, "filename", None), getattr(file, "content_type", None))
            try:
                # Validate file
                if not file.filename:
                    raise ValueError("No filename provided")

                if not file.filename.lower().endswith(('.pdf', '.doc', '.docx')):
                    raise ValueError("Invalid file format. Please upload PDF, DOC, or DOCX files only.")

                # Check file size (max 10MB)
                file_content = await file.read()
                if len(file_content) > 10 * 1024 * 1024:
                    raise ValueError("File too large. Maximum size is 10MB.")

                # Reset file pointer for processing
                await file.seek(0)

                # Parse resume
                parser = ResumeParser()
                resume_data = await parser.parse(file)

                # Calculate score using compare service
                advanced_analyzer = AdvancedAnalyzer()
                score = await advanced_analyzer.calculate_resume_score(resume_data)

                candidates.append({
                    "filename": file.filename,
                    "resumeData": ResumeData(**resume_data),
                    "score": score.overall_score,
                    "strengths": score.strengths,
                    "weaknesses": score.weaknesses,
                    "status": "success"
                })

            except ValueError as e:
                failed_files.append((file.filename, str(e)))
                candidates.append({
                    "filename": file.filename,
                    "error": str(e),
                    "score": 0,
                    "status": "validation_error"
                })

            except Exception as e:
                error_message = str(e)
                if "PDF" in error_message:
                    user_message = "Error reading PDF file. Please ensure it's not corrupted."
                elif "DOCX" in error_message:
                    user_message = "Error reading DOCX file. Please ensure it's a valid document."
                else:
                    user_message = error_message or "Analysis failed. Please try again."

                failed_files.append((file.filename, user_message))
                candidates.append({
                    "filename": file.filename,
                    "error": user_message,
                    "score": 0,
                    "status": "error"
                })

        successful_candidates = [c for c in candidates if c["status"] == "success"]
        if len(successful_candidates) < 2:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={
                    "success": False,
                    "message": "Failed to analyze enough resumes. Need at least 2 valid resumes for comparison.",
                    "error": "INSUFFICIENT_VALID_FILES",
                    "failed_files": [{"filename": fname, "error": error} for fname, error in failed_files]
                }
            )

        # ========== STEP 4: TRACK COMPARISON ==========
        successful_count = len(successful_candidates)
        
        if successful_count > 0:
            await rate_limit_service.increment_compare_resumes_counter(user_email, successful_count)

        # ========== STEP 5: GET UPDATED STATS ==========
        updated_usage = await rate_limit_service.get_feature_usage(user_email)
        if updated_usage is None:
            updated_usage = {
                "files_uploaded": 0,
                "batch_analysis": 0,
                "compare_resumes": 0
            }

        warning_at_comparisons = 10
        approaching_limit = updated_usage["compare_resumes"] >= warning_at_comparisons

        # ========== STEP 6: RANK & RETURN RESULTS ==========
        ranked_candidates = sorted(
            [c for c in candidates if c["status"] == "success"],
            key=lambda x: x.get("score", 0),
            reverse=True
        )

        comparison_response = {
            "success": True,
            "message": f"Comparison completed. {len(successful_candidates)} resumes analyzed successfully.",
            "comparison_summary": {
                "total_submitted": len(files),
                "successful": len(successful_candidates),
                "failed": len(failed_files),
                "highest_score": max([c.get("score", 0) for c in successful_candidates]) if successful_candidates else 0,
                "average_score": sum([c.get("score", 0) for c in successful_candidates]) / len(successful_candidates) if successful_candidates else 0
            },
            "usage_stats": {
                "comparisons_completed": updated_usage["compare_resumes"],
                "approaching_limit": approaching_limit,
                "approaching_limit_threshold": warning_at_comparisons
            },
            "ranked_candidates": ranked_candidates,
            "recommendations": [
                "Consider top 3 candidates for interviews",
                "Review candidates with scores > 80 for immediate consideration",
                "Candidates with scores < 60 may need additional training"
            ]
        }

        if approaching_limit:
            comparison_response["upgrade_prompt"] = {
                "show": True,
                "message": f"You've completed {updated_usage['compare_resumes']} comparisons.",
                "cta": "Upgrade now to compare unlimited resumes",
                "comparisons_completed": updated_usage['compare_resumes']
            }

        if failed_files:
            comparison_response["failed_files_details"] = [
                {"filename": fname, "error": error}
                for fname, error in failed_files
            ]

        return comparison_response

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "success": False,
                "message": "Unexpected error during comparison",
                "error": str(e)
            }
        )


@router.post("/selection-candidate", response_model=CandidateSelectionResponse, status_code=status.HTTP_200_OK)
async def selection_candidate(
    files: List[UploadFile] = File(...),
    job_title: str = Form(...),
    keywords: str = Form(...),
    current_user: TokenData = Depends(get_current_user)
):
    """
    Evaluate multiple candidates against a job title and keywords.
    Returns FIT or REJECT status for each candidate.
    
    Requirements:
    - Authentication required (JWT token)
    - Maximum 5 files per batch
    - Maximum 10 files per account (free tier)
    - Tracks filesUploaded counter
    
    Args:
        files: Multiple resume files (PDF, DOC, DOCX)
        job_title: Target job position (required)
        keywords: Comma-separated keywords/skills (required)
        current_user: Authenticated user (from JWT token)
    
    Returns:
        CandidateSelectionResponse with evaluation results
    """
    try:
        user_email = current_user.email
        partial_upload = False
        files_rejected = 0

        # ========== STEP 1: VALIDATE BATCH SIZE ==========
        if not files:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={
                    "success": False,
                    "message": "No files provided.",
                    "error": "VALIDATION_ERROR"
                }
            )
        
        if len(files) > rate_limit_service.batch_size_limit:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={
                    "success": False,
                    "message": f"Maximum {rate_limit_service.batch_size_limit} files per batch. You submitted {len(files)}.",
                    "error": "BATCH_SIZE_EXCEEDED",
                    "batch_limit": rate_limit_service.batch_size_limit,
                    "submitted": len(files)
                }
            )
        
        # ========== STEP 2: CHECK RATE LIMIT (INDEPENDENT FROM BATCH-ANALYZE) ==========
        rate_limit_check = await rate_limit_service.check_selected_candidate_limit(
            user_email,
            len(files)
        )

        if not rate_limit_check["allowed"]:
            if rate_limit_check["reason"] == "batch_size_exceeded":
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail={
                        "success": False,
                        "message": rate_limit_check["message"],
                        "error": "BATCH_SIZE_EXCEEDED",
                        "batch_limit": rate_limit_service.batch_size_limit,
                        "submitted": rate_limit_check["submitted"]
                    }
                )
            else:  # file_limit_exceeded
                # ALWAYS REJECT - No partial uploads allowed for selection-candidate
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail={
                        "success": False,
                        "message": rate_limit_check["message"],
                        "error": "RATE_LIMIT_EXCEEDED",
                        "current_count": rate_limit_check.get("current_count", 0),
                        "batch_size": rate_limit_check.get("batch_size", 0),
                        "limit": rate_limit_check.get("files_limit", 10),
                        "files_allowed": 0,
                        "upgrade_required": True,
                        "upgrade_message": f"You've reached your limit of {rate_limit_check.get('files_limit', 10)} candidate selections. Upgrade to select more candidates."
                    }
                )
        
        # ========== STEP 3: VALIDATE JOB TITLE AND KEYWORDS ==========
        if not job_title or not job_title.strip():
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={
                    "success": False,
                    "message": "Job title is required.",
                    "error": "VALIDATION_ERROR"
                }
            )
        
        if not keywords or not keywords.strip():
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={
                    "success": False,
                    "message": "Keywords are required.",
                    "error": "VALIDATION_ERROR"
                }
            )
        
        # Parse keywords
        keywords_list = [k.strip() for k in keywords.split(",") if k.strip()]
        
        if not keywords_list:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={
                    "success": False,
                    "message": "No valid keywords provided.",
                    "error": "VALIDATION_ERROR"
                }
            )
        
        # ========== STEP 4: VALIDATE FILE FORMATS AND SIZES ==========
        validated_files = []
        validation_errors = []
        
        for idx, file in enumerate(files):
            try:
                if not file.filename:
                    validation_errors.append(f"File {idx+1}: No filename")
                    continue
                
                if not file.filename.lower().endswith(('.pdf', '.doc', '.docx')):
                    raise HTTPException(
                        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                        detail={
                            "success": False,
                            "message": f"Invalid file format: {file.filename}. Please upload PDF, DOC, or DOCX files only.",
                            "error": "VALIDATION_ERROR"
                        }
                    )
                
                # Check file size before reading
                file_content = await file.read()
                if len(file_content) > 10 * 1024 * 1024:  # 10MB
                    raise HTTPException(
                        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                        detail={
                            "success": False,
                            "message": f"File {file.filename} is too large. Maximum size is 10MB.",
                            "error": "VALIDATION_ERROR"
                        }
                    )
                
                # CRITICAL: Reset file pointer to beginning for processing
                await file.seek(0)
                validated_files.append(file)
                
            except HTTPException:
                # Re-raise HTTP exceptions immediately
                raise
            except Exception as e:
                # Log validation error but continue processing other files
                error_msg = f"File {idx+1} ({file.filename if hasattr(file, 'filename') else 'unknown'}): {str(e)[:100]}"
                validation_errors.append(error_msg)
                continue
        
        if len(validated_files) == 0:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={
                    "success": False,
                    "message": "No valid files to process. " + "; ".join(validation_errors) if validation_errors else "All files failed validation",
                    "error": "VALIDATION_ERROR"
                }
            )
        
        # ========== STEP 5: EVALUATE CANDIDATES ==========
        selector = CandidateSelector()
        results = await selector.evaluate_candidates(validated_files, job_title, keywords_list)
        
        # Track actual processed files
        successful_files_count = len([r for r in results if r.get("status") in ["FIT", "REJECT"]])
        
        # Count fit and reject
        fit_count = sum(1 for r in results if r["status"] == "FIT")
        reject_count = len(results) - fit_count
        
        # Build response
        selection_results = [
            CandidateSelectionResult(
                candidate=r["candidate"],
                status=r["status"],
                message=r["message"]
            )
            for r in results
        ]
        
        response = CandidateSelectionResponse(
            job_title=job_title,
            keywords=keywords_list,
            total_candidates=len(results),
            fit_count=fit_count,
            reject_count=reject_count,
            results=selection_results
        )
        
        # ========== STEP 6: UPDATE RATE LIMIT COUNTERS ==========
        # Use successful files count for accurate tracking (INDEPENDENT counter)
        if successful_files_count > 0:
            await rate_limit_service.increment_selected_candidate_counter(user_email, successful_files_count)
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        error_message = str(e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "success": False,
                "message": "Error processing candidate selection",
                "error": error_message
            }
        )


