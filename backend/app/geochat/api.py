"""OpenAI-compatible GeoChat endpoint with skill-based orchestration."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
from typing import Literal
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, HTTPException, Response
from openai import OpenAI, OpenAIError
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.database import get_db
from app.runtime_settings import get_effective_settings
from app.storage import ensure_storage_root
from app.geochat.agent import run_agent

router = APIRouter()


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=20000)


class GeoChatRequest(BaseModel):
    conversation_id: str | None = None
    project_id: str | None = Field(default=None)
    messages: list[ChatMessage] = Field(min_length=1, max_length=100)


class GeoChatConversationSummary(BaseModel):
    conversation_id: str
    project_id: str | None
    title: str
    created_at: str
    updated_at: str
    request_count: int = 0
    model: str | None = None


class GeoChatConversationDetail(GeoChatConversationSummary):
    messages: list[ChatMessage]


class GeoChatResponse(BaseModel):
    conversation_id: str
    skill: str | None = None
    ui_action: str | None = None
    ui_action_data: dict | None = None
    message: ChatMessage
    model: str
    created_at: str
    usage: dict[str, int] | None = None
    suggested_questions: list[str] = Field(default_factory=list)


def _skill_questions(skill: str, asked: list[str]) -> list[str]:
    """Follow-up questions a process skill publishes for itself, minus the ones already asked."""
    from importlib import import_module

    try:
        questions = getattr(import_module(f"app.geochat.skills.{skill}.handler"), "SUGGESTED_QUESTIONS", None)
    except ImportError:
        return []
    if not isinstance(questions, (list, tuple)):
        return []
    seen = {item.strip().casefold() for item in asked}
    return [str(item) for item in questions if str(item).strip().casefold() not in seen][:4]


def _suggested_questions(skill: str | None, ui_action: str | None, content: str, asked: list[str] | None = None) -> list[str]:
    """Return concise, implementation-grounded next steps for each assistant turn."""
    if ui_action == "watch_processing_job":
        return ["Show the processing job status.", "List my completed processing jobs.", "Analyze the outputs from this processing job."]
    if ui_action == "switch_project":
        return ["Show the datasets in this project.", "List the wells and surveys in this project.", "Switch to another project."]
    by_skill = {
        "project_management": ["List all my projects.", "Create a new project.", "Switch to another project."],
        "data_inventory": ["List all seismic datasets.", "Show the available well logs.", "Which datasets can I open in a viewer?"],
        "well_survey_inventory": ["List all wells in this project.", "Which surveys are registered?", "Show the wells and surveys together."],
        "processing_catalog": ["List the available processing workflows.", "Which scripts appear runnable?", "Inspect the inputs for a processing script."],
        "processing_job_history": ["Show all processing jobs, including failed jobs.", "Summarize the outputs from the latest completed job.", "Show only successful processing jobs."],
        "geoscience_knowledge": ["Explain this concept using the project's seismic data.", "How could this concept affect interpretation?", "What related geoscience concepts should I know?"],
        "ask_geo_capabilities": ["List my projects and switch between them.", "What datasets, wells, and surveys are available?", "Show completed processing jobs and their outputs."],
    }
    if skill in by_skill:
        return by_skill[skill]
    if skill and skill.startswith("process_"):
        # A process skill knows best what is worth asking next about its own results.
        own = _skill_questions(skill, asked or [])
        if own:
            return own
        if not skill.endswith("_output_analysis"):
            return ["Explain the results of the latest run of this process.", "List my completed processing jobs.", "What other processing workflows are available?"]
        return ["Show the status of the processing job.", "Analyze the outputs from this processing workflow.", "What other processing workflows are available?"]
    if ui_action in {"view_dataset", "open_seismic_viewer", "open_well_viewer"}:
        return ["Explain what I am viewing.", "Show another available dataset.", "What analysis can I do with this data?"]
    if ui_action == "open_upload":
        return ["What file formats are supported?", "Show me the datasets already in this project.", "What can you do for me?"]
    lowered = content.lower()
    if any(word in lowered for word in ("error", "failed", "could not", "cannot")):
        return ["Help me troubleshoot this issue.", "Show the available project data.", "What can you do for me?"]
    return ["What datasets, wells, and surveys are available?", "Show completed processing jobs.", "What processing workflows can I run?"]


def _conversation_id(value: str | None) -> str:
    if value is None:
        return str(uuid4())
    try:
        return str(UUID(value))
    except (ValueError, TypeError, AttributeError) as exc:
        raise HTTPException(status_code=422, detail="conversation_id must be a UUID") from exc


def _write_transcript(path: Path, metadata: dict, messages: list[dict[str, str]]) -> None:
    lines = ["GEOCHAT TRANSCRIPT", "METADATA_JSON:", json.dumps(metadata, ensure_ascii=False, indent=2), "", "MESSAGES:"]
    for message in messages:
        lines.extend([f"{message['role'].upper()}:", message["content"], ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".geochat-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write("\n".join(lines).rstrip() + "\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _read_transcript(path: Path) -> tuple[dict, list[dict[str, str]]]:
    text = path.read_text(encoding="utf-8")
    metadata_marker = "METADATA_JSON:\n"
    messages_marker = "\n\nMESSAGES:\n"
    start = text.index(metadata_marker) + len(metadata_marker)
    end = text.index(messages_marker, start)
    metadata = json.loads(text[start:end])
    body = text[end + len(messages_marker):]
    messages: list[dict[str, str]] = []
    current_role: str | None = None
    current_lines: list[str] = []
    for line in body.splitlines():
        if line in {"USER:", "ASSISTANT:"}:
            if current_role is not None:
                messages.append({"role": current_role, "content": "\n".join(current_lines).rstrip()})
            current_role = line[:-1].lower()
            current_lines = []
        elif current_role is not None:
            current_lines.append(line)
    if current_role is not None:
        messages.append({"role": current_role, "content": "\n".join(current_lines).rstrip()})
    return metadata, messages


def _conversation_summary(path: Path, metadata: dict) -> GeoChatConversationSummary:
    conversation_id = str(metadata.get("conversation_id") or path.stem)
    messages = []
    try:
        _, messages = _read_transcript(path)
    except (ValueError, OSError, json.JSONDecodeError):
        pass
    first_user = next((item["content"].strip() for item in messages if item["role"] == "user"), "New conversation")
    title = first_user.splitlines()[0][:90] or "New conversation"
    return GeoChatConversationSummary(
        conversation_id=conversation_id,
        project_id=metadata.get("project_id"),
        title=title,
        created_at=str(metadata.get("created_at") or ""),
        updated_at=str(metadata.get("updated_at") or metadata.get("created_at") or ""),
        request_count=int(metadata.get("request_count", 0)),
        model=metadata.get("model"),
    )


@router.get("/conversations", response_model=list[GeoChatConversationSummary])
def list_conversations(project_id: str, db: Session = Depends(get_db)) -> list[GeoChatConversationSummary]:
    """List saved conversations belonging to one project only."""
    root = ensure_storage_root(db) / "geochat"
    if not root.is_dir():
        return []
    results: list[GeoChatConversationSummary] = []
    for path in root.glob("*.txt"):
        try:
            metadata, _ = _read_transcript(path)
        except (ValueError, OSError, json.JSONDecodeError):
            continue
        if str(metadata.get("project_id") or "") != project_id:
            continue
        results.append(_conversation_summary(path, metadata))
    results.sort(key=lambda item: item.updated_at, reverse=True)
    return results


@router.get("/conversations/{conversation_id}", response_model=GeoChatConversationDetail)
def get_conversation(conversation_id: str, project_id: str, db: Session = Depends(get_db)) -> GeoChatConversationDetail:
    """Load a conversation only when it belongs to the requested project."""
    chat_id = _conversation_id(conversation_id)
    path = ensure_storage_root(db) / "geochat" / f"{chat_id}.txt"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Conversation not found")
    try:
        metadata, messages = _read_transcript(path)
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=500, detail="Conversation transcript is invalid") from exc
    if str(metadata.get("project_id") or "") != project_id:
        raise HTTPException(status_code=404, detail="Conversation not found in this project")
    summary = _conversation_summary(path, metadata)
    return GeoChatConversationDetail(**summary.model_dump(), messages=messages)


@router.delete("/conversations/{conversation_id}", status_code=204)
def delete_conversation(conversation_id: str, project_id: str, db: Session = Depends(get_db)) -> Response:
    """Delete a saved conversation only when it belongs to the requested project."""
    chat_id = _conversation_id(conversation_id)
    path = ensure_storage_root(db) / "geochat" / f"{chat_id}.txt"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Conversation not found")
    try:
        metadata, _ = _read_transcript(path)
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=500, detail="Conversation transcript is invalid") from exc
    if str(metadata.get("project_id") or "") != project_id:
        raise HTTPException(status_code=404, detail="Conversation not found in this project")
    try:
        path.unlink()
    except OSError as exc:
        raise HTTPException(status_code=500, detail="Could not delete conversation") from exc
    return Response(status_code=204)


@router.post("", response_model=GeoChatResponse)
def chat(payload: GeoChatRequest, db: Session = Depends(get_db)) -> GeoChatResponse:
    if payload.messages[-1].role != "user":
        raise HTTPException(status_code=422, detail="The last message must be from the user")

    chat_id = _conversation_id(payload.conversation_id)
    config = get_effective_settings(db).geochat
    now = datetime.now(timezone.utc).isoformat()
    transcript_path = ensure_storage_root(db) / "geochat" / f"{chat_id}.txt"
    client = OpenAI(base_url=config.base_url.rstrip("/"), api_key="token-not-needed", timeout=config.timeout_seconds)
    generation = {
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
        "top_p": config.top_p,
        "frequency_penalty": config.frequency_penalty,
        "presence_penalty": config.presence_penalty,
    }
    try:
        content, selected_skill, ui_action, ui_action_data, completion = run_agent(
            client,
            model=config.model,
            system_prompt=config.system_prompt,
            messages=[message.model_dump() for message in payload.messages],
            project_id=payload.project_id,
            db=db,
            generation=generation,
        )
    except OpenAIError as exc:
        raise HTTPException(status_code=502, detail=f"LLM request failed: {exc.__class__.__name__}: {exc}") from exc
    except (OSError, RuntimeError) as exc:
        raise HTTPException(status_code=503, detail=f"Could not access GeoChat resources: {exc}") from exc

    # Viewer navigation may intentionally skip LLM prose generation. Keep the
    # API response valid while returning the UI action to the frontend.
    if not content or not content.strip():
        if ui_action == "view_dataset":
            dataset_name = (ui_action_data or {}).get("dataset_name")
            content = f"Opening {dataset_name} in the Well Viewer." if dataset_name else "Opening the dataset in the viewer."
        elif ui_action == "open_upload":
            content = "Opening the data upload dialog."
        else:
            content = "I couldn't generate a response. Please try again."

    transcript_messages = [message.model_dump() for message in payload.messages]
    if content:
        transcript_messages.append({"role": "assistant", "content": content})
    created_at, request_count = now, 1
    if transcript_path.is_file():
        try:
            old_text = transcript_path.read_text(encoding="utf-8")
            marker = "METADATA_JSON:\n"
            start = old_text.index(marker) + len(marker)
            end = old_text.index("\n\nMESSAGES:", start)
            old_metadata = json.loads(old_text[start:end])
            if str(old_metadata.get("project_id") or "") != str(payload.project_id or ""):
                raise HTTPException(status_code=409, detail="Conversation belongs to a different project")
            created_at = old_metadata.get("created_at", now)
            request_count = int(old_metadata.get("request_count", 0)) + 1
        except (ValueError, OSError, json.JSONDecodeError):
            pass

    usage = None
    if completion and completion.usage:
        usage = {"prompt_tokens": completion.usage.prompt_tokens, "completion_tokens": completion.usage.completion_tokens, "total_tokens": completion.usage.total_tokens}
    metadata = {
        "conversation_id": chat_id,
        "project_id": payload.project_id,
        "created_at": created_at,
        "updated_at": now,
        "request_count": request_count,
        "model": config.model,
        "base_url": config.base_url,
        "selected_skill": selected_skill,
        "ui_action": ui_action,
        "ui_action_data": ui_action_data,
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
        "top_p": config.top_p,
        "frequency_penalty": config.frequency_penalty,
        "presence_penalty": config.presence_penalty,
        "timeout_seconds": config.timeout_seconds,
        "system_prompt": config.system_prompt,
        "usage": usage,
    }
    try:
        _write_transcript(transcript_path, metadata, transcript_messages)
    except OSError as exc:
        raise HTTPException(status_code=503, detail=f"Response generated but transcript could not be saved: {exc}") from exc

    return GeoChatResponse(
        conversation_id=chat_id,
        skill=selected_skill,
        ui_action=ui_action,
        ui_action_data=ui_action_data,
        message=ChatMessage(role="assistant", content=content),
        model=(completion.model if completion else config.model) or config.model,
        created_at=now,
        usage=usage,
        suggested_questions=_suggested_questions(
            selected_skill, ui_action, content,
            asked=[message.content for message in payload.messages if message.role == "user"],
        ),
    )
