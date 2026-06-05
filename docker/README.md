# Docker Environment for Spider RL MJLab

Isolated container for running Claude Code with `--dangerously-skip-permissions` safely.

## Quick Start

```bash
cd ~/projects/scr_phybot_mjab/docker

# Build image (first time, takes a while)
docker compose build

# Start container
docker compose up -d

# Enter container
docker exec -it spider_mjlab_dev bash

# Inside container: run Claude Code with bypass permissions
conda activate env_mjlab
claude --dangerously-skip-permissions
```

## What's Inside

- **Base**: `nvidia/cuda:12.8.0-devel-ubuntu24.04`
- **Python**: 3.11 (via Miniconda)
- **Key packages**: PyTorch 2.10, MuJoCo 3.6, mjlab 1.2
- **GPU**: Full NVIDIA GPU passthrough
- **Claude Code**: Pre-installed via npm

## Volumes

| Host | Container | Mode |
|---|---|---|
| `../` (project root) | `/workspace` | read-write |
| `~/.claude` | `/root/.claude` | read-write |
| `~/.config` | `/root/.config` | read-only |

## API Key

Set your Anthropic API key before starting:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
docker compose up -d
```

Or it will be picked up from `~/.claude` config.

## Safety

- Container cannot access files outside mounted volumes
- `rm -rf /` inside container won't affect host (except `/workspace`)
- To protect project files too, change the volume to read-only:
  ```yaml
  volumes:
    - ..:/workspace:ro
  ```

## Common Commands

```bash
# Rebuild after changing Dockerfile
docker compose build --no-cache

# Stop container
docker compose down

# View logs
docker compose logs -f

# Reset everything (container + image)
docker compose down --rmi all

cd ~/projects/scr_phybot_mjab/docker
docker compose build
docker compose up -d
docker exec -it spider_mjlab_dev bash

```
