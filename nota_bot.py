from __future__ import annotations

import logging
import argparse
import re
import sqlite3
import time
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path

import discord
from discord import app_commands

from config import settings
from github_client import GithubWorkspaceManager, WorkspaceProvisionError
from job_queue import JobQueueService
from knowledge_base import KnowledgeBaseService, build_change_report
from security_utils import sanitize_repo_relative_path


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("nota.security")

intents = discord.Intents.default()
bot = discord.Client(intents=intents)
tree = app_commands.CommandTree(bot)

github_manager = GithubWorkspaceManager()
kb = KnowledgeBaseService()


class CommandRateLimiter:
    def __init__(self) -> None:
        self._recent_by_user: dict[int, deque[float]] = defaultdict(deque)

    def check(self, user_id: int) -> tuple[bool, str | None]:
        now = time.time()
        user_hits = self._recent_by_user[user_id]

        while user_hits and now - user_hits[0] > settings.command_window_seconds:
            user_hits.popleft()

        if user_hits and now - user_hits[-1] < settings.command_cooldown_seconds:
            return False, f"紐낅졊?대? ?덈Т 鍮좊Ⅴ寃??몄텧?섍퀬 ?덉뒿?덈떎. {settings.command_cooldown_seconds}珥????ㅼ떆 ?쒕룄??二쇱꽭??"

        if len(user_hits) >= settings.command_max_per_window:
            return False, "?붿껌 ?쒕룄瑜?珥덇낵?덉뒿?덈떎. ?좎떆 ???ㅼ떆 ?쒕룄??二쇱꽭??"

        user_hits.append(now)
        return True, None


class ThreadAccessManager:
    def __init__(self) -> None:
        self._init_db()

    def _init_db(self) -> None:
        conn = sqlite3.connect(settings.sqlite_path)
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS user_thread_mapping (
                    discord_user_id INTEGER PRIMARY KEY,
                    guild_id INTEGER NOT NULL,
                    thread_id INTEGER NOT NULL UNIQUE
                )
                """
            )
            conn.commit()
        finally:
            conn.close()

    def get_thread_id(self, discord_user_id: int) -> int | None:
        conn = sqlite3.connect(settings.sqlite_path)
        try:
            row = conn.execute(
                "SELECT thread_id FROM user_thread_mapping WHERE discord_user_id=?",
                (discord_user_id,),
            ).fetchone()
            return int(row[0]) if row else None
        finally:
            conn.close()

    def is_managed_thread(self, thread_id: int) -> bool:
        conn = sqlite3.connect(settings.sqlite_path)
        try:
            row = conn.execute(
                "SELECT 1 FROM user_thread_mapping WHERE thread_id=?",
                (thread_id,),
            ).fetchone()
            return bool(row)
        finally:
            conn.close()

    def upsert(self, discord_user_id: int, guild_id: int, thread_id: int) -> None:
        conn = sqlite3.connect(settings.sqlite_path)
        try:
            conn.execute(
                """
                INSERT INTO user_thread_mapping(discord_user_id, guild_id, thread_id)
                VALUES(?, ?, ?)
                ON CONFLICT(discord_user_id)
                DO UPDATE SET guild_id=excluded.guild_id, thread_id=excluded.thread_id
                """,
                (discord_user_id, guild_id, thread_id),
            )
            conn.commit()
        finally:
            conn.close()




class ThreadRoleManager:
    def __init__(self) -> None:
        self._init_db()

    def _init_db(self) -> None:
        conn = sqlite3.connect(settings.sqlite_path)
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS thread_roles (
                    thread_id INTEGER NOT NULL,
                    discord_user_id INTEGER NOT NULL,
                    role TEXT NOT NULL,
                    PRIMARY KEY(thread_id, discord_user_id)
                )
                """
            )
            conn.commit()
        finally:
            conn.close()

    def set_roles(self, thread_id: int, owner_id: int, collaborator_ids: set[int]) -> None:
        conn = sqlite3.connect(settings.sqlite_path)
        try:
            conn.execute("DELETE FROM thread_roles WHERE thread_id=?", (thread_id,))
            conn.execute(
                "INSERT INTO thread_roles(thread_id, discord_user_id, role) VALUES (?, ?, 'owner')",
                (thread_id, owner_id),
            )
            for cid in sorted(collaborator_ids):
                if cid == owner_id:
                    continue
                conn.execute(
                    "INSERT INTO thread_roles(thread_id, discord_user_id, role) VALUES (?, ?, 'collaborator')",
                    (thread_id, cid),
                )
            conn.commit()
        finally:
            conn.close()

    def get_role(self, thread_id: int, user_id: int) -> str | None:
        conn = sqlite3.connect(settings.sqlite_path)
        try:
            row = conn.execute(
                "SELECT role FROM thread_roles WHERE thread_id=? AND discord_user_id=?",
                (thread_id, user_id),
            ).fetchone()
            return str(row[0]) if row else None
        finally:
            conn.close()

thread_access_manager = ThreadAccessManager()
thread_role_manager = ThreadRoleManager()
rate_limiter = CommandRateLimiter()
job_queue = JobQueueService()


def _extract_user_ids(raw: str) -> list[int]:
    if not raw.strip():
        return []
    return [int(match) for match in re.findall(r"\d+", raw)]


def _is_command_allowed_outside_thread(interaction: discord.Interaction) -> bool:
    if not interaction.command:
        return False
    return interaction.command.name == "open_thread"


def _audit_log(interaction: discord.Interaction, outcome: str, detail: str = "") -> None:
    command_name = interaction.command.name if interaction.command else "unknown"
    logger.info(
        "audit command=%s user_id=%s guild_id=%s channel_id=%s outcome=%s detail=%s",
        command_name,
        interaction.user.id if interaction.user else "unknown",
        interaction.guild_id,
        interaction.channel_id,
        outcome,
        detail,
    )


def _sanitize_repo_relative_path(path: str) -> Path:
    return sanitize_repo_relative_path(path)


async def _is_valid_private_thread_context(interaction: discord.Interaction) -> tuple[bool, str | None]:
    channel = interaction.channel
    if not isinstance(channel, discord.Thread) or not channel.is_private():
        return False, "紐⑤뱺 紐낅졊? 媛쒖씤 ?곕젅?쒖뿉?쒕쭔 ?ㅽ뻾?????덉뒿?덈떎. 癒쇱? `/open_thread`瑜??ъ슜??二쇱꽭??"
    if channel.parent_id != settings.thread_parent_channel_id:
        return False, "?덉슜??紐낅졊 ?곕젅?쒓? ?꾨떃?덈떎. 吏?뺣맂 遺紐?梨꾨꼸?먯꽌 ?앹꽦???곕젅?쒖뿉???ㅽ뻾??二쇱꽭??"

    if not thread_access_manager.is_managed_thread(channel.id):
        return False, "Nota媛 愿由ы븯??媛쒖씤 ?곕젅?쒓? ?꾨떃?덈떎. `/open_thread`濡??앹꽦???곕젅?쒖뿉???ㅽ뻾??二쇱꽭??"

    if interaction.user.id in settings.admin_user_ids:
        return True, None

    try:
        await channel.fetch_member(interaction.user.id)
    except discord.NotFound:
        return False, "???곕젅?쒖쓽 李몄뿬?먮쭔 紐낅졊?대? ?ㅽ뻾?????덉뒿?덈떎."

    return True, None


async def ensure_thread_context(interaction: discord.Interaction) -> bool:
    if _is_command_allowed_outside_thread(interaction):
        return True

    ok, reason = await _is_valid_private_thread_context(interaction)
    if ok:
        return True

    if interaction.response.is_done():
        await interaction.followup.send(reason, ephemeral=True)
    else:
        await interaction.response.send_message(reason, ephemeral=True)
    _audit_log(interaction, "denied", reason or "")
    return False




COMMAND_PERMISSIONS: dict[str, set[str]] = {
    "delete": {"owner", "admin"},
    "reset": {"owner", "admin"},
    "invite": {"owner", "admin"},
    "setup": {"owner", "admin"},
    "edit": {"owner", "admin"},
    "add": {"owner", "collaborator", "admin"},
    "chat": {"owner", "collaborator", "admin"},
    "list": {"owner", "collaborator", "admin"},
    "history": {"owner", "collaborator", "admin"},
    "summarize_all": {"owner", "admin"},
    "analyze_update": {"owner", "admin"},
    "job_status": {"owner", "collaborator", "admin"},
    "open_thread": {"owner", "collaborator", "admin"},
}


def _resolve_access_role(interaction: discord.Interaction) -> str:
    if interaction.user.id in settings.admin_user_ids:
        return "admin"
    if not isinstance(interaction.channel, discord.Thread):
        return "collaborator"
    role = thread_role_manager.get_role(interaction.channel.id, interaction.user.id)
    return role or "collaborator"


async def enforce_rbac(interaction: discord.Interaction) -> bool:
    if not interaction.command:
        return True
    allowed = COMMAND_PERMISSIONS.get(interaction.command.name)
    if not allowed:
        return True
    role = _resolve_access_role(interaction)
    if role in allowed:
        return True
    reason = f"沅뚰븳 遺議? `{interaction.command.name}` 紐낅졊? owner留??ㅽ뻾?????덉뒿?덈떎."
    if interaction.response.is_done():
        await interaction.followup.send(reason, ephemeral=True)
    else:
        await interaction.response.send_message(reason, ephemeral=True)
    _audit_log(interaction, "denied", reason)
    return False

async def enforce_rate_limit(interaction: discord.Interaction) -> bool:
    if interaction.user.id in settings.admin_user_ids:
        return True
    allowed, reason = rate_limiter.check(interaction.user.id)
    if allowed:
        return True
    if interaction.response.is_done():
        await interaction.followup.send(reason, ephemeral=True)
    else:
        await interaction.response.send_message(reason, ephemeral=True)
    _audit_log(interaction, "rate_limited", reason or "")
    return False




def _commit_message(user_id: int, command_name: str, reason: str) -> str:
    return f"Nota automated update | user_id={user_id} | command={command_name} | reason={reason}"


def _parse_ymd_to_ts(raw: str | None, *, end_of_day: bool = False) -> float | None:
    if not raw:
        return None
    parsed = datetime.strptime(raw, "%Y-%m-%d")
    if end_of_day:
        return parsed.timestamp() + 86399
    return parsed.timestamp()


@bot.event
async def on_ready() -> None:
    await tree.sync()
    logger.info("Logged in as %s (ID: %s)", bot.user, bot.user.id if bot.user else "unknown")


@tree.command(name="setup", description="Create or connect your private GitHub knowledge workspace.")
@app_commands.describe(github_username="(Optional) GitHub username to invite as collaborator")
async def setup(interaction: discord.Interaction, github_username: str | None = None) -> None:
    if not await ensure_thread_context(interaction):
        return
    if not await enforce_rbac(interaction):
        return
    if not await enforce_rate_limit(interaction):
        return
    await interaction.response.defer()
    try:
        repo_full_name = github_manager.create_workspace_for_user(interaction.user.id, github_username)
    except WorkspaceProvisionError as exc:
        _audit_log(interaction, "failed", f"setup: {exc}")
        await interaction.followup.send(f"?뚰겕?ㅽ럹?댁뒪 ?ㅼ젙 ?ㅽ뙣: {exc}")
        return

    _audit_log(interaction, "success", "setup")
    await interaction.followup.send(f"?뚰겕?ㅽ럹?댁뒪 ?ㅼ젙 ?꾨즺: `https://github.com/{repo_full_name}`")


@tree.command(name="invite", description="Invite a collaborator into your workspace repository.")
@app_commands.describe(github_username="GitHub username to invite")
async def invite(interaction: discord.Interaction, github_username: str) -> None:
    if not await ensure_thread_context(interaction):
        return
    if not await enforce_rbac(interaction):
        return
    if not await enforce_rate_limit(interaction):
        return
    await interaction.response.defer()
    try:
        repo_full_name = github_manager.invite_collaborator(interaction.user.id, github_username)
    except WorkspaceProvisionError as exc:
        _audit_log(interaction, "failed", f"invite: {exc}")
        await interaction.followup.send(f"?묒뾽??珥덈? ?ㅽ뙣: {exc}")
        return
    _audit_log(interaction, "success", "invite")
    await interaction.followup.send(
        f"`{github_username}` ?섏쓣 `{repo_full_name}` ??μ냼 Collaborator濡?珥덈??덉뒿?덈떎.\n"
        "GitHub 珥덈? ?섎씫 ?꾧퉴吏??沅뚰븳???쒖꽦?붾릺吏 ?딆뒿?덈떎.",
    )


@tree.error
async def on_app_command_error(interaction: discord.Interaction, error: app_commands.AppCommandError) -> None:
    original = getattr(error, "original", error)
    if isinstance(original, ValueError) and "run /setup first" in str(original):
        message = "?뚰겕?ㅽ럹?댁뒪媛 ?꾩쭅 ?ㅼ젙?섏? ?딆븯?듬땲?? `/setup` ??癒쇱? ?ㅽ뻾??二쇱꽭??"
    else:
        message = "?붿껌 泥섎━ 以??ㅻ쪟媛 諛쒖깮?덉뒿?덈떎. 愿由ъ옄?먭쾶 臾몄쓽??二쇱꽭??"

    _audit_log(interaction, "error", f"{type(original).__name__}: {original}")

    if interaction.response.is_done():
        await interaction.followup.send(message)
    else:
        await interaction.response.send_message(message)


@tree.command(name="add", description="Ingest YouTube/Web/File into your GitHub knowledge repo.")
@app_commands.describe(source_type="youtube|web|file|repo", source="URL or repo-relative path")
async def add(interaction: discord.Interaction, source_type: str, source: str) -> None:
    if not await ensure_thread_context(interaction):
        return
    if not await enforce_rbac(interaction):
        return
    if not await enforce_rate_limit(interaction):
        return

    uid = interaction.user.id
    thread_id = interaction.channel_id or 0

    def handler() -> str:
        with github_manager.cloned_repo(
            uid,
            commit_message=_commit_message(uid, "add", f"source_type={source_type}"),
            author_name=f"discord-{uid}",
            author_email=f"discord-{uid}@nota.local",
        ) as repo_dir:
            action = "?앹꽦"
            if source_type == "youtube":
                md = kb.youtube_to_markdown(source)
                target = repo_dir / "sources" / "youtube" / f"{source.split('=')[-1]}.md"
                reason = "Ingested YouTube transcript"
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(md, encoding="utf-8")
                report_path = str(target.relative_to(repo_dir))
            elif source_type == "web":
                md = kb.web_to_markdown(source)
                filename = source.replace("https://", "").replace("http://", "").replace("/", "_")
                target = repo_dir / "sources" / "web" / f"{filename}.md"
                reason = "?뱁럹?댁? 吏???섏쭛"
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(md, encoding="utf-8")
                report_path = str(target.relative_to(repo_dir))
            elif source_type == "file":
                raise ValueError("蹂댁븞 ?뺤콉???쒕쾭 濡쒖뺄 ?뚯씪 寃쎈줈 ?낅젰? ?덉슜?섏? ?딆뒿?덈떎. ??μ냼 ?뚯씪? source_type=repo瑜??ъ슜??二쇱꽭??")
            elif source_type == "repo":
                safe_rel = _sanitize_repo_relative_path(source)
                repo_target = repo_dir / safe_rel
                if not repo_target.exists() or not repo_target.is_file():
                    raise ValueError("??μ냼 ?대? ?뚯씪 寃쎈줈瑜?李얠? 紐삵뻽?듬땲??")
                reason = "Indexed repository file for RAG"
                action = "update"
                report_path = str(repo_target.relative_to(repo_dir))
            else:
                raise ValueError("source_type? youtube/web/file/repo 以??섎굹?ъ빞 ?⑸땲??")

            report = build_change_report(uid, report_path, reason, action)
            stats = kb.index_markdown_files(uid, repo_dir)
            return (
                f"{report}\n"
                f"利앸텇 ?몃뜳??寃곌낵: chunk={stats.total_chunks}, 蹂寃쏀뙆??{stats.indexed_files}, "
                f"?ㅽ궢?뚯씪={stats.skipped_files}, ??젣?뚯씪={stats.deleted_files}"
            )

    await interaction.response.defer()
    job_id = job_queue.enqueue(user_id=uid, thread_id=thread_id, command="add", handler=handler)
    _audit_log(interaction, "queued", f"add job_id={job_id}")
    await interaction.followup.send(
        f"/add 작업을 큐에 등록했습니다. job_id=`{job_id}`\n"
        f"`/job_status {job_id}`로 상태를 확인하세요."
    )


@tree.command(name="chat", description="Chat with the assistant.")
@app_commands.describe(
    question="Your question",
)
async def chat(
    interaction: discord.Interaction,
    question: str,
) -> None:
    if not await ensure_thread_context(interaction):
        return
    if not await enforce_rbac(interaction):
        return
    if not await enforce_rate_limit(interaction):
        return
    await interaction.response.defer()
    answer, contexts = kb.answer_with_lm_studio_clean(
        interaction.user.id,
        question,
    )

    embed = discord.Embed(title="Nota Chat", description=answer[:3900], color=0x4F46E5)
    embed.add_field(name="Question", value=question[:1024], inline=False)
    if contexts:
        top = contexts[0]
        embed.add_field(name="Top Source", value=f"{top['path']}:{top['start_line']}-{top['end_line']}", inline=False)

    _audit_log(interaction, "success", "chat")
    await interaction.followup.send(embed=embed)


@tree.command(name="summarize_all", description="Generate README/whitepaper/learning guide from whole knowledge base.")
async def summarize_all(interaction: discord.Interaction) -> None:
    if not await ensure_thread_context(interaction):
        return
    if not await enforce_rbac(interaction):
        return
    if not await enforce_rate_limit(interaction):
        return

    uid = interaction.user.id
    thread_id = interaction.channel_id or 0

    def handler() -> str:
        mapping = github_manager.get_mapping(uid)
        if not mapping:
            raise ValueError("워크스페이스가 없습니다. 먼저 /setup을 실행해 주세요.")

        with github_manager.cloned_repo(
            uid,
            commit_message=_commit_message(uid, "summarize_all", "repository synthesis"),
            author_name=f"discord-{uid}",
            author_email=f"discord-{uid}@nota.local",
        ) as repo_dir:
            generated = kb.synthesize_repository_documents(uid, mapping.github_repo_full_name)
            for rel, content in generated.items():
                target = repo_dir / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content, encoding="utf-8")
            stats = kb.index_markdown_files(uid, repo_dir)
        return (
            "README/기술백서/학습가이드 자동 생성 완료. "
            f"(증분 인덱싱 변경파일 {stats.indexed_files}개)"
        )

    await interaction.response.defer()
    job_id = job_queue.enqueue(user_id=uid, thread_id=thread_id, command="summarize_all", handler=handler)
    _audit_log(interaction, "queued", f"summarize_all job_id={job_id}")
    await interaction.followup.send(
        f"/summarize_all 작업을 큐에 등록했습니다. job_id=`{job_id}`\n"
        f"`/job_status {job_id}`로 상태를 확인하세요."
    )


@tree.command(name="analyze_update", description="Analyze a repo file and update docs automatically.")
@app_commands.describe(path="Repo file path", instruction="What kind of analysis/document update to do")
async def analyze_update(interaction: discord.Interaction, path: str, instruction: str) -> None:
    if not await ensure_thread_context(interaction):
        return
    if not await enforce_rbac(interaction):
        return
    if not await enforce_rate_limit(interaction):
        return

    uid = interaction.user.id
    thread_id = interaction.channel_id or 0

    def handler() -> str:
        mapping = github_manager.get_mapping(uid)
        if not mapping:
            raise ValueError("?뚰겕?ㅽ럹?댁뒪媛 ?놁뒿?덈떎. 癒쇱? /setup???ㅽ뻾??二쇱꽭??")

        with github_manager.cloned_repo(
            uid,
            commit_message=_commit_message(uid, "analyze_update", f"path={path}"),
            author_name=f"discord-{uid}",
            author_email=f"discord-{uid}@nota.local",
        ) as repo_dir:
            safe_rel = _sanitize_repo_relative_path(path)
            target = repo_dir / safe_rel
            if not target.exists() or not target.is_file():
                raise ValueError("????뚯씪??議댁옱?섏? ?딆뒿?덈떎.")
            source = target.read_text(encoding="utf-8", errors="ignore")
            prompt = f"?뚯씪 `{safe_rel}` 遺꾩꽍 寃곌낵瑜?臾몄꽌濡??뺣━?댁쨾. ?붿껌?ы빆: {instruction}"
            synthesized, _ = kb.answer_with_lm_studio(uid, f"{prompt}\n\n?뚯뒪:\n{source[:4000]}", mapping.github_repo_full_name)
            out = repo_dir / "docs" / "AUTO_ANALYSIS.md"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(f"# Automated Analysis\n\n{synthesized}\n", encoding="utf-8")
            stats = kb.index_markdown_files(uid, repo_dir)
        return f"臾몄꽌 ?낅뜲?댄듃 ?꾨즺: docs/AUTO_ANALYSIS.md (蹂寃쏀뙆??{stats.indexed_files}媛??몃뜳??"

    await interaction.response.defer()
    job_id = job_queue.enqueue(user_id=uid, thread_id=thread_id, command="analyze_update", handler=handler)
    _audit_log(interaction, "queued", f"analyze_update job_id={job_id}")
    await interaction.followup.send(
        f"/analyze_update 작업을 큐에 등록했습니다. job_id=`{job_id}`\n"
        f"`/job_status {job_id}`로 상태를 확인하세요."
    )


@tree.command(name="list", description="List files in your GitHub workspace.")
async def list_files(interaction: discord.Interaction) -> None:
    if not await ensure_thread_context(interaction):
        return
    if not await enforce_rbac(interaction):
        return
    if not await enforce_rate_limit(interaction):
        return
    await interaction.response.defer()
    files = github_manager.list_files(interaction.user.id)
    if not files:
        await interaction.followup.send("저장소에 파일이 없습니다.")
        return
    _audit_log(interaction, "success", "list")
    await interaction.followup.send("\n".join(f"- {f}" for f in files[:200]))


@tree.command(name="history", description="Show recent file history from git commits.")
@app_commands.describe(path="Relative file path", limit="Commit count (1-20)")
async def history(interaction: discord.Interaction, path: str, limit: app_commands.Range[int, 1, 20] = 10) -> None:
    if not await ensure_thread_context(interaction):
        return
    if not await enforce_rbac(interaction):
        return
    if not await enforce_rate_limit(interaction):
        return

    await interaction.response.defer()
    safe_rel = _sanitize_repo_relative_path(path)
    commits = github_manager.get_file_history(interaction.user.id, str(safe_rel), int(limit))
    if not commits:
        await interaction.followup.send("해당 경로의 커밋 이력을 찾지 못했습니다.")
        return

    lines = [f"최근 커밋 이력: `{safe_rel}`"]
    for c in commits:
        lines.append(
            f"- `{c['hexsha'][:8]}` | {c['summary']} | author={c['author']} ({c['author_email']}) | {c['committed_datetime']}"
        )
    _audit_log(interaction, "success", "history")
    await interaction.followup.send("\n".join(lines)[:3900])


@tree.command(name="job_status", description="Check queued job status/result.")
@app_commands.describe(job_id="Queue job id")
async def job_status(interaction: discord.Interaction, job_id: str) -> None:
    if not await ensure_thread_context(interaction):
        return
    if not await enforce_rbac(interaction):
        return
    if not await enforce_rate_limit(interaction):
        return

    await interaction.response.defer(ephemeral=True)
    status = job_queue.get_status(job_id.strip(), interaction.user.id, interaction.channel_id or 0)
    if not status:
        await interaction.followup.send("해당 job_id를 찾을 수 없거나 조회 권한이 없습니다.", ephemeral=True)
        return

    message = (
        f"job_id=`{status.job_id}`\n"
        f"command=`{status.command}`\n"
        f"state=`{status.state}`\n"
        f"detail={status.detail}\n"
        f"updated_at={datetime.fromtimestamp(status.updated_at).isoformat()}"
    )
    if status.state == "completed" and status.result:
        message += f"\n\n결과:\n{status.result[:2500]}"
    elif status.state == "failed" and status.result:
        message += f"\n\n오류:\n{status.result[:1200]}"

    _audit_log(interaction, "success", f"job_status {status.state}")
    await interaction.followup.send(message, ephemeral=True)


@tree.command(name="edit", description="Edit a file in your repository.")
@app_commands.describe(path="Relative file path", new_content="New text content")
async def edit(interaction: discord.Interaction, path: str, new_content: str) -> None:
    if not await ensure_thread_context(interaction):
        return
    if not await enforce_rbac(interaction):
        return
    if not await enforce_rate_limit(interaction):
        return
    await interaction.response.defer()
    uid = interaction.user.id
    with github_manager.cloned_repo(
        uid,
        commit_message=_commit_message(uid, "edit", f"path={path}"),
        author_name=f"discord-{uid}",
        author_email=f"discord-{uid}@nota.local",
    ) as repo_dir:
        safe_rel = _sanitize_repo_relative_path(path)
        target = repo_dir / safe_rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(new_content, encoding="utf-8")
        kb.index_markdown_files(uid, repo_dir)
    report = build_change_report(uid, str(safe_rel), "?ъ슜??吏곸젒 ?몄쭛 ?붿껌", "?섏젙")
    _audit_log(interaction, "success", "edit")
    await interaction.followup.send(report)


@tree.command(name="delete", description="Delete a file in your repository.")
@app_commands.describe(path="Relative file path")
async def delete(interaction: discord.Interaction, path: str) -> None:
    if not await ensure_thread_context(interaction):
        return
    if not await enforce_rbac(interaction):
        return
    if not await enforce_rate_limit(interaction):
        return
    await interaction.response.defer()
    uid = interaction.user.id
    safe_rel = _sanitize_repo_relative_path(path)
    removed = github_manager.delete_files(
        uid,
        [str(safe_rel)],
        commit_message=_commit_message(uid, "delete", f"path={safe_rel}"),
        author_name=f"discord-{uid}",
        author_email=f"discord-{uid}@nota.local",
    )
    if removed:
        kb.reset_user_index(uid)
        with github_manager.cloned_repo(uid) as repo_dir:
            kb.index_markdown_files(uid, repo_dir)
        report = build_change_report(uid, str(safe_rel), "?ъ슜????젣 ?붿껌", "??젣")
        _audit_log(interaction, "success", "delete")
        await interaction.followup.send(report)
    else:
        await interaction.followup.send("????뚯씪??李얠? 紐삵뻽?듬땲??")


@tree.command(name="reset", description="Delete all repo files and reset local vector index.")
async def reset(interaction: discord.Interaction) -> None:
    if not await ensure_thread_context(interaction):
        return
    if not await enforce_rbac(interaction):
        return
    if not await enforce_rate_limit(interaction):
        return
    await interaction.response.defer()
    uid = interaction.user.id
    with github_manager.cloned_repo(
        uid,
        commit_message=_commit_message(uid, "reset", "clear workspace"),
        author_name=f"discord-{uid}",
        author_email=f"discord-{uid}@nota.local",
    ) as repo_dir:
        for f in [p for p in repo_dir.rglob("*") if p.is_file() and ".git" not in str(p)]:
            f.unlink()
    kb.reset_user_index(uid)
    _audit_log(interaction, "success", "reset")
    await interaction.followup.send("?섍꼍 珥덇린???꾨즺: GitHub ?뚯씪怨?濡쒖뺄 ?몃뜳?ㅻ? ?뺣━?덉뒿?덈떎.")


@tree.command(name="open_thread", description="Create your private Nota command thread.")
@app_commands.describe(collaborators="Optional collaborator mentions/IDs (comma separated)")
async def open_thread(interaction: discord.Interaction, collaborators: str | None = None) -> None:
    if not interaction.guild:
        await interaction.response.send_message("??紐낅졊? ?쒕쾭?먯꽌留??ъ슜?????덉뒿?덈떎.", ephemeral=True)
        return

    if interaction.channel_id != settings.thread_parent_channel_id:
        await interaction.response.send_message(
            "??紐낅졊? 吏?뺣맂 梨꾨꼸?먯꽌留??ㅽ뻾?????덉뒿?덈떎.",
            ephemeral=True,
        )
        return

    if not await enforce_rate_limit(interaction):
        return

    await interaction.response.defer(ephemeral=True)
    existing_thread_id = thread_access_manager.get_thread_id(interaction.user.id)
    if existing_thread_id:
        thread = interaction.guild.get_thread(existing_thread_id)
        if thread:
            await interaction.followup.send(f"?대? 媛쒖씤 ?곕젅?쒓? ?덉뒿?덈떎: {thread.mention}", ephemeral=True)
            return

    parent_channel = interaction.guild.get_channel(settings.thread_parent_channel_id)
    if not isinstance(parent_channel, discord.TextChannel):
        await interaction.followup.send("遺紐?梨꾨꼸???띿뒪??梨꾨꼸???꾨떃?덈떎. ?섍꼍 蹂?섎? ?뺤씤??二쇱꽭??", ephemeral=True)
        return

    thread = await parent_channel.create_thread(
        name=f"nota-{interaction.user.name}-{interaction.user.id}",
        type=discord.ChannelType.private_thread,
        invitable=False,
    )
    await thread.add_user(interaction.user)

    member_ids = set(settings.admin_user_ids)
    if collaborators:
        member_ids.update(_extract_user_ids(collaborators))

    for member_id in member_ids:
        member = interaction.guild.get_member(member_id)
        if member:
            await thread.add_user(member)

    thread_access_manager.upsert(interaction.user.id, interaction.guild.id, thread.id)
    thread_role_manager.set_roles(thread.id, interaction.user.id, member_ids)
    _audit_log(interaction, "success", "open_thread")
    await interaction.followup.send(f"媛쒖씤 ?곕젅?쒕? ?앹꽦?덉뒿?덈떎: {thread.mention}", ephemeral=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Nota Discord bot")
    parser.add_argument(
        "--groq",
        action="store_true",
        help="Use Groq OpenAI-compatible API instead of LM Studio",
    )
    args = parser.parse_args()

    if args.groq:
        settings.llm_provider = "groq"
    else:
        settings.llm_provider = "lmstudio"

    settings.validate()
    logger.info("Starting Nota bot with LLM provider: %s", settings.llm_provider)
    bot.run(settings.discord_token)

