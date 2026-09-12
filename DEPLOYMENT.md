# Deploying TRACE

TRACE ships as two containers (FastAPI backend, static React frontend behind
nginx) plus a small backup sidecar, all wired together in `docker-compose.yml`.

## Quick start (single VM, behind your own load balancer / TLS termination)

```bash
cp backend/.env.example backend/.env
# Edit backend/.env:
#   - ENVIRONMENT=production
#   - JWT_SECRET=<output of: python -c "import secrets; print(secrets.token_urlsafe(64))">
#   - ALLOWED_ORIGINS=https://your-domain.com
#   - FRONTEND_URL=https://your-domain.com

docker compose up -d --build
```

The backend will refuse to start in production if `JWT_SECRET` is still the
placeholder value or under 32 characters — see `config.py`. This is
deliberate: it's better to fail loudly at deploy time than to silently sign
auth tokens with a secret anyone can read out of the example file.

## With automatic HTTPS (no external load balancer)

If you're deploying directly on a VM with a public domain and nothing else
terminating TLS for you, use the Caddy overlay:

```bash
# Edit Caddyfile: replace "your-domain.com" with your real domain
docker compose -f docker-compose.yml -f docker-compose.https.yml up -d --build
```

Caddy takes over ports 80/443 and issues/renews Let's Encrypt certificates
automatically. Point your domain's DNS at the host before starting it — cert
issuance needs port 80 reachable from the internet.

## What's in the stack

| Service    | Role                                                             |
|------------|-------------------------------------------------------------------|
| `backend`  | FastAPI + gunicorn/uvicorn, classifier + Stable Diffusion pipeline |
| `frontend` | Vite build served by nginx, proxies `/api/*` to `backend`          |
| `backup`   | Nightly SQLite snapshot into a separate `trace-backups` volume    |
| `caddy`    | (optional, `docker-compose.https.yml`) automatic HTTPS             |

Data persistence:
- `trace-data` volume — the live SQLite DB (`ENVIRONMENT`, `DATABASE_URL` etc. in `backend/.env`)
- `trace-backups` volume — nightly `.backup` snapshots, 14-day retention (see `backend/scripts/backup_sqlite.sh`)

**Offsite backups**: the `trace-backups` volume protects against DB
corruption or an accidental `docker volume rm`, but it's still on the same
host. For real production, sync it elsewhere too — e.g. a cron'd
`aws s3 sync /var/lib/docker/volumes/.../trace-backups s3://your-bucket/` or
equivalent for your cloud provider.

## Scaling notes

- The backend Dockerfile runs gunicorn with **1 worker** on purpose — the
  Stable Diffusion pipeline is loaded once per process, and a second worker
  would double the memory footprint (each worker needs several GB) for no
  throughput gain on CPU. To handle more traffic, run more `backend`
  *containers* behind a load balancer instead of more workers inside one.
- `ENABLE_GENERATION=false` in `backend/.env` skips loading Stable Diffusion
  entirely if you only need the classical-ML identity matching — this drops
  memory usage from several GB to a few hundred MB and image size
  significantly if you also strip `torch`/`diffusers`/`transformers`/
  `accelerate` from `backend/requirements.txt`.
- For a GPU host: switch the backend Dockerfile's base image to
  `nvidia/cuda:*-runtime` and install the matching CUDA build of `torch`.

## CI

`.github/workflows/ci.yml` runs on every push/PR to `main`:
1. Backend: installs CPU-only torch (avoids the multi-GB CUDA download in
   CI), runs the pytest suite with `ENABLE_GENERATION=false`.
2. Frontend: lints and builds the Vite app.
3. Both Docker images are built (not pushed) to catch Dockerfile
   regressions before merge.

## Health checks

`GET /health` returns `200` with `{"status": "ok", ...}` when the classifier
is loaded and the database is reachable, or `503` with `{"status":
"degraded", ...}` otherwise — this is what docker-compose's `healthcheck`
and any external uptime monitor should poll.

## Known limitations (read before treating this as more than a demo)

- The identity classifier is trained on 18 people from a small research
  dataset (RWMFD) — it is a proof of concept, not a general-purpose face
  recognition system, and the frontend says so directly in its "model
  honesty" section. Don't point this at a real access-control or
  surveillance use case without a much larger, properly consented,
  properly audited training set and a different threat model entirely.
- Password-reset emails aren't actually sent yet — the reset link is logged
  server-side (see `/auth/forgot-password` in `main.py`). Wire in a real
  provider (SES, Postmark, Resend, etc.) before relying on this in
  production; until then, anyone who can read backend logs can complete a
  password reset for any account.