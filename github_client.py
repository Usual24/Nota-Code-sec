from __future__ import annotations

import os
import shutil
import sqlite3
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Iterable

from git import Repo
from git import Actor
from github import Github
from github.GithubException import GithubException, UnknownObjectException

from config import settings


@dataclass(slots=True)
class RepoMapping:
    discord_user_id: int
    github_repo_full_name: str


class WorkspaceProvisionError(RuntimeError):
    pass


class GithubWorkspaceManager:
    def __init__(self) -> None:
        self.gh = Github(settings.github_token)
        self._init_db()

    def _init_db(self) -> None:
        conn = sqlite3.connect(settings.sqlite_path)
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS user_repo_mapping (
                    discord_user_id INTEGER PRIMARY KEY,
                    github_repo_full_name TEXT NOT NULL
                )
                """
            )
            conn.commit()
        finally:
            conn.close()

    def get_mapping(self, discord_user_id: int) -> RepoMapping | None:
        conn = sqlite3.connect(settings.sqlite_path)
        try:
            row = conn.execute(
                "SELECT discord_user_id, github_repo_full_name FROM user_repo_mapping WHERE discord_user_id=?",
                (discord_user_id,),
            ).fetchone()
            if not row:
                return None
            return RepoMapping(discord_user_id=row[0], github_repo_full_name=row[1])
        finally:
            conn.close()

    def _upsert_mapping(self, mapping: RepoMapping) -> None:
        conn = sqlite3.connect(settings.sqlite_path)
        try:
            conn.execute(
                """
                INSERT INTO user_repo_mapping(discord_user_id, github_repo_full_name)
                VALUES(?, ?)
                ON CONFLICT(discord_user_id)
                DO UPDATE SET github_repo_full_name=excluded.github_repo_full_name
                """,
                (mapping.discord_user_id, mapping.github_repo_full_name),
            )
            conn.commit()
        finally:
            conn.close()

    def create_workspace_for_user(self, discord_user_id: int, github_username: str | None = None) -> str:
        repo_name = f"{settings.repo_prefix}-{discord_user_id}"
        owner = self.gh.get_organization(settings.github_org) if settings.github_org else self.gh.get_user()

        try:
            repo = owner.get_repo(repo_name)
        except UnknownObjectException:
            try:
                repo = owner.create_repo(
                    name=repo_name,
                    private=True,
                    description=f"Nota knowledge workspace for Discord user {discord_user_id}",
                    auto_init=True,
                )
            except GithubException as exc:
                if exc.status == 403:
                    raise WorkspaceProvisionError(
                        "GitHub 토큰 권한이 부족해서 저장소를 생성할 수 없습니다. "
                        "Classic PAT는 `repo` scope가 필요하고, Fine-grained PAT는 "
                        "repositories 생성/관리 권한(Administration: Read & Write)이 필요합니다."
                    ) from exc
                raise WorkspaceProvisionError(f"GitHub 저장소 생성에 실패했습니다: {exc.data}") from exc

        if github_username and github_username.lower() != repo.owner.login.lower():
            self._invite(repo.full_name, github_username)

        self._upsert_mapping(RepoMapping(discord_user_id=discord_user_id, github_repo_full_name=repo.full_name))
        return repo.full_name

    def invite_collaborator(self, discord_user_id: int, github_username: str) -> str:
        mapping = self.get_mapping(discord_user_id)
        if not mapping:
            raise ValueError("User workspace is not configured. Please run /setup first.")
        self._invite(mapping.github_repo_full_name, github_username)
        return mapping.github_repo_full_name

    def _invite(self, repo_full_name: str, github_username: str) -> None:
        repo = self.gh.get_repo(repo_full_name)
        try:
            repo.add_to_collaborators(github_username, permission="push")
        except GithubException as exc:
            if exc.status == 404:
                raise WorkspaceProvisionError(f"GitHub 사용자 `{github_username}` 를 찾지 못했거나 초대 권한이 없습니다.") from exc
            if exc.status == 403:
                raise WorkspaceProvisionError("저장소는 준비됐지만 협력자 초대 권한이 부족합니다. 토큰 권한을 확인해 주세요.") from exc
            raise WorkspaceProvisionError(f"협력자 초대에 실패했습니다: {exc.data}") from exc

    @contextmanager
    def cloned_repo(
        self,
        discord_user_id: int,
        *,
        commit_message: str = "Nota automated update",
        author_name: str = "nota-bot",
        author_email: str = "nota-bot@local",
    ) -> Generator[Path, None, None]:
        mapping = self.get_mapping(discord_user_id)
        if not mapping:
            raise ValueError("User workspace is not configured. Please run /setup first.")

        tmp_dir = Path(tempfile.mkdtemp(prefix="nota-repo-"))
        askpass_fd, askpass_path = tempfile.mkstemp(prefix="nota-askpass-", suffix=".sh")
        askpass_file = Path(askpass_path)
        try:
            os.close(askpass_fd)
            askpass_file.write_text(
                "#!/usr/bin/env bash\n"
                "case \"$1\" in\n"
                "  *Username*) echo \"x-access-token\" ;;&\n"
                "  *) echo \"$GITHUB_TOKEN\" ;;&\n"
                "esac\n",
                encoding="utf-8",
            )
            os.chmod(askpass_file, 0o700)
            git_env = {
                **os.environ,
                "GIT_ASKPASS": str(askpass_file),
                "GIT_TERMINAL_PROMPT": "0",
                "GITHUB_TOKEN": settings.github_token,
            }

            clone_url = f"https://github.com/{mapping.github_repo_full_name}.git"
            Repo.clone_from(clone_url, tmp_dir, env=git_env)
            yield tmp_dir
            repo = Repo(tmp_dir)
            repo.git.add(A=True)
            if repo.is_dirty(untracked_files=True):
                actor = Actor(author_name, author_email)
                repo.index.commit(commit_message, author=actor, committer=actor)
                repo.remote("origin").push(settings.default_branch, env=git_env)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            try:
                askpass_file.unlink()
            except FileNotFoundError:
                pass

    def list_files(self, discord_user_id: int) -> list[str]:
        with self.cloned_repo(discord_user_id) as repo_dir:
            return [str(p.relative_to(repo_dir)) for p in repo_dir.rglob("*") if p.is_file() and ".git" not in str(p)]

    def delete_files(
        self,
        discord_user_id: int,
        targets: Iterable[str],
        *,
        commit_message: str = "Nota automated update",
        author_name: str = "nota-bot",
        author_email: str = "nota-bot@local",
    ) -> list[str]:
        removed: list[str] = []
        with self.cloned_repo(
            discord_user_id,
            commit_message=commit_message,
            author_name=author_name,
            author_email=author_email,
        ) as repo_dir:
            for rel in targets:
                target = repo_dir / rel
                if target.exists() and target.is_file():
                    target.unlink()
                    removed.append(rel)
        return removed

    def get_file_history(self, discord_user_id: int, rel_path: str, limit: int = 10) -> list[dict]:
        history: list[dict] = []
        with self.cloned_repo(discord_user_id) as repo_dir:
            repo = Repo(repo_dir)
            for commit in repo.iter_commits(paths=rel_path, max_count=limit):
                history.append(
                    {
                        "hexsha": commit.hexsha,
                        "summary": commit.summary,
                        "author": commit.author.name,
                        "author_email": commit.author.email,
                        "committed_datetime": commit.committed_datetime.isoformat(),
                    }
                )
        return history
