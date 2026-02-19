# Nota-Code 기술 문서

## 1. 개요
Nota-Code는 디스코드 슬래시 명령을 통해 개인 지식 저장소를 운영하는 자동화 봇입니다. 사용자별 GitHub 리포지토리를 워크스페이스로 할당하고, 수집한 콘텐츠를 벡터 인덱스로 관리해 질의응답(RAG)과 문서 자동화를 제공합니다.

핵심 목표는 다음 세 가지입니다.
- 디스코드 안에서 지식 수집/정리/검색을 일관된 명령 체계로 제공
- 사용자별 저장소 및 인덱스 분리로 데이터 격리 유지
- 수동 문서 작업(요약, 분석, 정리)의 자동화

> 🔐 **Security First**: Nota-Code는 편의 기능보다 **데이터 격리·최소 권한·입력 검증·감사 추적**을 우선합니다. 모든 명령은 private thread, RBAC, 레이트리밋, 경로/URL 검증 정책 위에서 동작하도록 설계되어 운영 중 사고 반경을 줄입니다.

---

## 2. 시스템 구성

### 2.1 주요 모듈
- `nota_bot.py`  
  디스코드 이벤트/슬래시 명령 엔트리포인트. 쓰레드 접근 제어, RBAC, 레이트리밋, 작업 큐 연계를 담당합니다.
- `github_client.py`  
  사용자별 GitHub 워크스페이스 생성/매핑/클론/커밋/푸시 및 파일 이력 조회를 담당합니다.
- `knowledge_base.py`  
  문서 수집(YouTube/Web/Repo), 청크 분할, Chroma 인덱싱, 컨텍스트 검색, LM Studio 질의응답을 담당합니다.
- `job_queue.py`  
  비동기 작업 실행용 로컬 큐(스레드 풀 + SQLite 상태 저장)입니다.
- `security_utils.py`  
  저장소 상대경로 검증(경로 순회 방지) 유틸리티입니다.
- `config.py`  
  `.env` 및 환경변수를 로딩하고 실행 설정을 검증합니다.

### 2.2 저장소/데이터 구조
기본 데이터 경로(`NOTA_DATA_DIR`, 기본 `./data`)에 아래 리소스를 생성합니다.
- `nota.sqlite3`: 사용자-리포 매핑, 쓰레드 매핑, 역할, 작업 큐 상태 저장
- `chroma/`: 벡터 인덱스 영속 저장소
- `index_hash_state.json`: 증분 인덱싱용 파일 해시 상태

---

## 3. 기능 명세

### 3.1 워크스페이스/협업 관리
- 사용자별 private GitHub 저장소 자동 생성 또는 기존 저장소 연결
- 선택적 GitHub 협업자 초대
- 사용자-리포 매핑 SQLite 영속화

### 3.2 지식 수집 및 인덱싱
- `youtube`: 자막 추출 후 Markdown 저장
- `web`: 웹 페이지 텍스트 추출 후 Markdown 저장(콘텐츠 타입/용량 정책 적용)
- `repo`: 저장소 내부 파일을 대상으로 즉시 인덱싱
- 파일 해시 기반 증분 인덱싱(변경 없는 파일 스킵, 삭제 파일 인덱스 정리)

### 3.3 질의응답(RAG)
- 벡터 검색 기반 컨텍스트 추출
- 경로 prefix, 소스 타입, 기간 필터 제공
- LM Studio `chat/completions` 연동
- 응답 문장별 인용 강제 및 GitHub 소스 링크 자동 부착
- LM Studio 실패 시 요약형 폴백 응답 제공

### 3.4 문서 자동 생성
- `summarize_all`: README/기술백서/학습가이드 자동 생성
- `analyze_update`: 특정 파일 분석 후 `docs/AUTO_ANALYSIS.md` 업데이트

### 3.5 저장소 파일 관리
- 파일 목록 조회
- 파일 수정/삭제
- 파일 단위 Git 이력 조회
- 저장소 전체 초기화(reset)

### 3.6 실행 제어/보안 운영
- 개인 private thread 기반 명령 실행 강제
- 명령별 RBAC(owner/collaborator/admin)
- 사용자별 레이트리밋(쿨다운 + 윈도우 제한)
- 경로 정규화 및 상위 경로 접근 차단
- 웹 URL 검증(스킴/도메인 정책/사설망 IP 차단)
- 감사 로그(audit) 기록

### 3.8 보안 아키텍처 핵심 원칙
- **격리(Isolation)**: 사용자별 GitHub 저장소/인덱스를 분리해 교차 접근 가능성을 최소화
- **최소 권한(Least Privilege)**: 명령 권한을 owner/collaborator/admin으로 명확히 제한
- **입력 검증(Input Validation)**: 파일 경로, URL, 콘텐츠 타입, 용량을 다층 검증
- **남용 방지(Abuse Prevention)**: 명령 쿨다운 + 윈도우 레이트리밋으로 자동화 공격/오용 완화
- **추적 가능성(Auditability)**: 주요 이벤트를 감사 로그에 기록해 사후 분석 가능성 확보

### 3.7 비동기 작업 큐
아래 명령은 큐 기반으로 실행됩니다.
- `/add`
- `/summarize_all`
- `/analyze_update`

작업 등록 후 `job_id`를 발급하며 `/job_status`로 상태와 결과를 조회합니다.

---

## 4. 디스코드 명령어 레퍼런스

| 명령어 | 설명 | 권한 |
|---|---|---|
| `/open_thread [collaborators]` | 개인 명령 쓰레드 생성 | owner/collaborator/admin |
| `/setup [github_username]` | 사용자 워크스페이스 생성/연결 | owner/admin |
| `/invite <github_username>` | 저장소 협업자 초대 | owner/admin |
| `/add <source_type> <source>` | 콘텐츠 수집/인덱싱(`youtube`,`web`,`repo`) | owner/collaborator/admin |
| `/chat <question> [source_type] [path_prefix] [from_date] [to_date] [topk]` | 지식베이스 질의응답 | owner/collaborator/admin |
| `/summarize_all` | 저장소 문서 자동 생성 | owner/admin |
| `/analyze_update <path> <instruction>` | 특정 파일 분석 및 문서 업데이트 | owner/admin |
| `/list` | 저장소 파일 목록 조회 | owner/collaborator/admin |
| `/history <path> [limit]` | 파일 커밋 이력 조회 | owner/collaborator/admin |
| `/edit <path> <new_content>` | 파일 수정 | owner/admin |
| `/delete <path>` | 파일 삭제 | owner/admin |
| `/reset` | 저장소 파일/인덱스 초기화 | owner/admin |
| `/job_status <job_id>` | 비동기 작업 상태 조회 | owner/collaborator/admin |

> 참고: `source_type=file`은 정책상 차단되어 있으며 오류를 반환합니다.

---

## 5. 환경변수

### 5.1 필수
- `DISCORD_TOKEN`: 디스코드 봇 토큰
- `GITHUB_TOKEN`: GitHub PAT 또는 호환 토큰
- `NOTA_THREAD_PARENT_CHANNEL_ID`: `/open_thread`를 허용할 부모 채널 ID

### 5.2 GitHub/리포 설정
- `GITHUB_ORG`: 조직 저장소로 생성 시 조직명(미설정 시 사용자 계정)
- `NOTA_REPO_PREFIX` (기본 `nota-kb`): 생성 리포 접두사
- `DEFAULT_BRANCH` (기본 `main`): push 대상 브랜치

### 5.3 데이터/모델 설정
- `NOTA_DATA_DIR` (기본 `./data`)
- `EMBEDDING_MODEL` (기본 `sentence-transformers/all-MiniLM-L6-v2`)
- `LM_STUDIO_BASE_URL` (기본 `http://127.0.0.1:1234/v1`)
- `LM_STUDIO_MODEL` (기본 `local-model`)
- `LM_STUDIO_MAX_PROMPT_CHARS` (기본 `3500`)
- `LM_STUDIO_CONTEXT_WINDOW` (기본 `2048`)
- `LM_STUDIO_RESERVED_TOKENS` (기본 `512`)
- `LM_STUDIO_CHARS_PER_TOKEN` (기본 `3.0`)

### 5.4 운영 정책/보안 설정
- `NOTA_ADMIN_USER_IDS`: 관리자 Discord ID 목록(쉼표 구분)
- `NOTA_ALLOWED_LOCAL_FILE_EXTENSIONS` (기본 `txt,json,pdf`)
- `NOTA_LOCAL_FILE_MAX_BYTES` (기본 `3145728`)
- `NOTA_WEB_MAX_BYTES` (기본 `2097152`)
- `NOTA_WEB_REQUEST_TIMEOUT_SECONDS` (기본 `15`)
- `NOTA_ALLOWED_WEB_DOMAINS`: 허용 도메인 정책(비우면 전체 허용)
- `NOTA_COMMAND_COOLDOWN_SECONDS` (기본 `5`)
- `NOTA_COMMAND_WINDOW_SECONDS` (기본 `60`)
- `NOTA_COMMAND_MAX_PER_WINDOW` (기본 `12`)

---

## 6. 설치 및 실행 절차

### 6.1 의존성 설치
```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install discord.py PyGithub GitPython chromadb sentence-transformers requests beautifulsoup4 youtube-transcript-api python-dotenv pypdf
```

### 6.2 환경파일 작성
프로젝트 루트에 `.env` 파일을 생성합니다.

```dotenv
DISCORD_TOKEN=...
GITHUB_TOKEN=...
NOTA_THREAD_PARENT_CHANNEL_ID=123456789012345678

# 선택
GITHUB_ORG=
NOTA_REPO_PREFIX=nota-kb
DEFAULT_BRANCH=main
NOTA_DATA_DIR=./data
LM_STUDIO_BASE_URL=http://127.0.0.1:1234/v1
LM_STUDIO_MODEL=local-model
NOTA_ADMIN_USER_IDS=111111111111111111,222222222222222222
```

### 6.3 실행
```bash
python nota_bot.py
```

정상 기동 시 디스코드에 명령 트리가 동기화됩니다.

---

## 7. 운영 흐름

1. 관리 채널에서 사용자가 `/open_thread` 실행
2. 개인 private thread 생성 및 권한 사용자 등록
3. `/setup`으로 GitHub 워크스페이스 준비
4. `/add`로 지식 소스 누적 및 인덱싱
5. `/chat`으로 검색/질의응답
6. 필요 시 `/summarize_all`, `/analyze_update`로 문서 자동화

---

## 8. 테스트

### 8.1 단위 테스트 실행
```bash
python -m unittest -v tests/test_security_and_indexing.py
```

### 8.2 검증 항목
- 경로 순회 차단
- 로컬호스트 URL 차단
- 답변 인용 강제 후처리
- 증분 인덱싱 삭제 반영

---

## 9. 장애 대응 포인트

- `/setup` 실패 시
  - GitHub 토큰 권한(`repo` 또는 동등 권한) 확인
  - `GITHUB_ORG` 접근 권한 확인
- `/chat` 응답 품질 저하 시
  - 인덱싱 데이터 유무 확인(`/add`, `/list`)
  - LM Studio 가동/엔드포인트 점검
- 큐 작업 지연 시
  - `/job_status` 상태 확인(`queued`, `running`, `completed`, `failed`)
  - `data/nota.sqlite3` jobs 테이블 및 서버 리소스 점검

---

## 10. 보안 운영 권장사항

### 10.1 필수 보안 체크리스트(운영 전)
- [ ] `GITHUB_TOKEN`, `DISCORD_TOKEN`이 로그/히스토리/에러메시지에 노출되지 않는지 점검
- [ ] `NOTA_ALLOWED_WEB_DOMAINS`를 반드시 화이트리스트로 고정
- [ ] `NOTA_ADMIN_USER_IDS`를 최소 인원으로 유지
- [ ] `NOTA_WEB_MAX_BYTES`, `NOTA_WEB_REQUEST_TIMEOUT_SECONDS`를 환경에 맞게 제한
- [ ] `NOTA_COMMAND_MAX_PER_WINDOW` 정책이 트래픽 규모 대비 충분히 보수적인지 검토
- [ ] LM Studio 엔드포인트가 외부 공개되지 않도록 내부망 ACL/프록시 정책 적용

### 10.2 권장 하드닝
- 토큰은 최소 월 1회 회전하고, 퇴사/권한 변경 즉시 폐기
- GitHub 토큰은 가능한 최소 스코프만 부여(`repo` 등 필요한 범위 한정)
- 운영 로그/감사 로그 보관 주기 및 파기 정책을 사전에 정의
- 디스코드 thread 권한 변경 이벤트를 정기 점검하여 권한 드리프트 방지

### 10.3 사고 대응 준비
- 유출 의심 시: 토큰 즉시 폐기 → 재발급 → 최근 감사 로그/커밋 로그 교차 점검
- 비정상 대량 요청 시: 레이트리밋 강화 및 문제 사용자/채널 임시 차단
- 외부 URL 악성 징후 시: 도메인 화이트리스트 축소 후 검증 정책 재적용
