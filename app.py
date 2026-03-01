"""
Auto-Shorts Web Interface
Flask app que executa o pipeline real com logs em tempo real (SSE).
Uso: python app.py  ->  abre http://localhost:5000
"""
import os
import sys
import json
import time
import signal
import subprocess
import threading
import webbrowser
from pathlib import Path
from queue import Queue, Empty
from datetime import datetime

from flask import Flask, render_template, request, jsonify, Response

app = Flask(__name__)
BASE_DIR = Path(__file__).parent

# Detecta o executavel Python correto (mesmo que foi usado pra rodar app.py)
PYTHON_EXE = sys.executable

# ========== ESTADO GLOBAL ==========
process_state = {
    "process": None,       # subprocess.Popen
    "running": False,
    "logs": [],            # historico de logs
    "step": 0,             # etapa atual (1-5)
    "step_label": "",
    "started_at": None,
    "video_count": 0,
    "last_video": None,
}
log_queues = []  # Lista de queues SSE (uma por cliente conectado)
state_lock = threading.Lock()


def broadcast_log(msg, log_type="log"):
    """Envia log para todos os clientes SSE conectados."""
    entry = {
        "time": datetime.now().strftime("%H:%M:%S"),
        "msg": msg.rstrip(),
        "type": log_type,
    }
    with state_lock:
        process_state["logs"].append(entry)
        # Manter max 500 logs
        if len(process_state["logs"]) > 500:
            process_state["logs"] = process_state["logs"][-500:]
    for q in log_queues:
        q.put(entry)


def broadcast_status(step=None, label=None, running=None, video=None):
    """Envia atualizacao de status."""
    with state_lock:
        if step is not None:
            process_state["step"] = step
        if label is not None:
            process_state["step_label"] = label
        if running is not None:
            process_state["running"] = running
        if video is not None:
            process_state["last_video"] = video
            process_state["video_count"] += 1
    
    status = {
        "step": process_state["step"],
        "label": process_state["step_label"],
        "running": process_state["running"],
        "video_count": process_state["video_count"],
    }
    for q in log_queues:
        q.put({"type": "status", "status": status})


def detect_pipeline_step(line):
    """Detecta etapa do pipeline pela saida do console."""
    line_lower = line.lower()
    if "[1/5]" in line or "gerando roteiro" in line_lower:
        broadcast_status(step=1, label="Roteiro (Groq)")
    elif "[2/5]" in line or "gerando narracao" in line_lower or "gerando narração" in line_lower:
        broadcast_status(step=2, label="Narração (TTS)")
    elif "[3/5]" in line or "gerando" in line_lower and "imag" in line_lower:
        broadcast_status(step=3, label="Imagens (FLUX)")
    elif "[4/5]" in line or "legenda" in line_lower:
        broadcast_status(step=4, label="Legendas (Whisper)")
    elif "[5/5]" in line or "montagem" in line_lower:
        broadcast_status(step=5, label="Montagem Final")
    elif "video gerado com sucesso" in line_lower:
        broadcast_status(step=6, label="Concluído", video=True)
    elif "erro" in line_lower or "error" in line_lower:
        broadcast_log(line, "error")
        return
    
    if "titulo:" in line_lower or "sucesso" in line_lower:
        broadcast_log(line, "success")
    elif "gpu" in line_lower or "cuda" in line_lower or "qsv" in line_lower or "nvenc" in line_lower:
        broadcast_log(line, "info")
    else:
        broadcast_log(line, "log")


def run_command(cmd, cwd=None):
    """Executa comando em background com streaming de output."""
    if cwd is None:
        cwd = str(BASE_DIR)
    
    broadcast_status(step=0, label="Iniciando...", running=True)
    broadcast_log(f"$ {cmd}", "info")
    
    try:
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        
        # Converte string em lista para evitar problemas com shell no Windows
        if isinstance(cmd, str):
            import shlex
            if sys.platform == "win32":
                # No Windows, split manual: separa o executavel (entre aspas) dos args
                args = []
                remaining = cmd.strip()
                while remaining:
                    remaining = remaining.strip()
                    if remaining.startswith('"'):
                        end = remaining.index('"', 1)
                        args.append(remaining[1:end])
                        remaining = remaining[end+1:]
                    else:
                        parts = remaining.split(None, 1)
                        args.append(parts[0])
                        remaining = parts[1] if len(parts) > 1 else ""
            else:
                args = shlex.split(cmd)
        else:
            args = cmd
        
        # Insere -u apos o python pra forcar unbuffered
        if len(args) > 1 and args[0].lower().endswith(('python', 'python.exe', 'python3', 'python3.exe')):
            args.insert(1, "-u")
        
        proc = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=cwd,
            encoding="utf-8",
            errors="replace",
            env=env,
        )
        with state_lock:
            process_state["process"] = proc
            process_state["started_at"] = time.time()
        
        for line in iter(proc.stdout.readline, ""):
            if not line:
                break
            detect_pipeline_step(line.strip())
        
        proc.wait()
        
        if proc.returncode == 0:
            broadcast_log("Processo finalizado com sucesso!", "success")
        else:
            broadcast_log(f"Processo encerrado (code {proc.returncode})", "error")
    
    except Exception as e:
        broadcast_log(f"Erro: {e}", "error")
    
    finally:
        with state_lock:
            process_state["process"] = None
        broadcast_status(running=False)


# ========== CONFIGURACAO ==========

def load_config():
    """Le config.py e extrai valores."""
    config_path = BASE_DIR / "config.py"
    config = {}
    if config_path.exists():
        text = config_path.read_text(encoding="utf-8")
        # Parse simples de atribuicoes
        for line in text.splitlines():
            line = line.strip()
            if "=" in line and not line.startswith("#") and not line.startswith("from") and not line.startswith("import"):
                parts = line.split("=", 1)
                key = parts[0].strip()
                val = parts[1].split("#")[0].strip()
                # Ignorar chamadas de funcao e listas complexas
                if "(" not in val or val.startswith('"') or val.startswith("'"):
                    config[key] = val
    return config


def load_env():
    """Le .env e retorna dict."""
    env_path = BASE_DIR / ".env"
    env = {}
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                env[key.strip()] = val.strip().strip('"').strip("'")
    return env


def load_presets():
    """Extrai presets do batch.py."""
    batch_path = BASE_DIR / "batch.py"
    if not batch_path.exists():
        return {}
    try:
        text = batch_path.read_text(encoding="utf-8")
        # Encontrar SERIES_PRESETS
        start = text.find("SERIES_PRESETS = {")
        if start == -1:
            return {}
        # Contar chaves para encontrar o fim
        depth = 0
        end = start
        for i, c in enumerate(text[start:], start):
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        preset_code = text[start:end]
        local_ns = {}
        exec(preset_code, {}, local_ns)
        presets = local_ns.get("SERIES_PRESETS", {})
        return {k: {"count": len(v), "temas": v[:3]} for k, v in presets.items()}
    except Exception:
        return {}


def load_history():
    """Le historico de temas usados."""
    hist_path = BASE_DIR / "temas_usados.json"
    if hist_path.exists():
        try:
            return json.loads(hist_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def get_output_videos():
    """Lista videos gerados na pasta output."""
    output_dir = BASE_DIR / "output"
    if not output_dir.exists():
        return []
    videos = []
    for f in sorted(output_dir.glob("*.mp4"), key=lambda x: x.stat().st_mtime, reverse=True):
        stat = f.stat()
        videos.append({
            "name": f.name,
            "size_mb": round(stat.st_size / (1024 * 1024), 1),
            "created": datetime.fromtimestamp(stat.st_mtime).strftime("%d/%m %H:%M"),
        })
    return videos[:20]


# ========== ROTAS ==========

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/state")
def api_state():
    """Estado atual completo."""
    config = load_config()
    presets = load_presets()
    history = load_history()
    videos = get_output_videos()
    env = load_env()
    
    apis = {
        "groq": bool(env.get("GROQ_API_KEY")),
        "hf": bool(env.get("HF_TOKEN")),
        "cloudflare": bool(env.get("CLOUDFLARE_ACCOUNT_ID") and env.get("CLOUDFLARE_API_TOKEN")),
        "pexels": bool(env.get("PEXELS_API_KEY")),
    }
    
    with state_lock:
        running = process_state["running"]
        step = process_state["step"]
        step_label = process_state["step_label"]
        video_count = process_state["video_count"]
        logs = list(process_state["logs"][-100:])
    
    return jsonify({
        "config": config,
        "presets": presets,
        "history": history,
        "videos": videos,
        "apis": apis,
        "running": running,
        "step": step,
        "step_label": step_label,
        "video_count": video_count,
        "logs": logs,
    })


@app.route("/api/generate", methods=["POST"])
def api_generate():
    """Gera um video avulso."""
    with state_lock:
        if process_state["running"]:
            return jsonify({"error": "Ja tem um processo rodando!"}), 409
    
    data = request.json or {}
    tema = data.get("tema", "").strip()
    if not tema:
        return jsonify({"error": "Tema vazio"}), 400
    
    plataformas = data.get("plataformas", [])
    
    cmd = f'"{PYTHON_EXE}" main.py --tema "{tema}"'
    if plataformas:
        cmd += f' --publicar {" ".join(plataformas)}'
    
    thread = threading.Thread(target=run_command, args=(cmd,), daemon=True)
    thread.start()
    
    return jsonify({"ok": True, "cmd": cmd})


@app.route("/api/batch", methods=["POST"])
def api_batch():
    """Gera videos em lote."""
    with state_lock:
        if process_state["running"]:
            return jsonify({"error": "Ja tem um processo rodando!"}), 409
    
    data = request.json or {}
    mode = data.get("mode", "preset")
    quantidade = data.get("quantidade", 5)
    plataformas = data.get("plataformas", [])
    inicio = data.get("inicio")
    
    cmd = f'"{PYTHON_EXE}" batch.py'
    
    if mode == "preset":
        preset = data.get("preset", "historias_ocultas")
        cmd += f" --preset {preset}"
    elif mode == "serie":
        serie = data.get("serie", "").strip()
        cmd += f' --serie "{serie}"'
    elif mode == "arquivo":
        cmd += " --arquivo temas.txt"
    
    cmd += f" -q {quantidade}"
    
    if plataformas:
        cmd += f' --publicar {" ".join(plataformas)}'
    
    if inicio:
        cmd += f" --inicio {inicio}"
    
    thread = threading.Thread(target=run_command, args=(cmd,), daemon=True)
    thread.start()
    
    return jsonify({"ok": True, "cmd": cmd})


@app.route("/api/stop", methods=["POST"])
def api_stop():
    """Para o processo atual."""
    with state_lock:
        proc = process_state["process"]
        if proc and proc.poll() is None:
            if sys.platform == "win32":
                proc.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                proc.terminate()
            broadcast_log("Processo cancelado pelo usuario.", "error")
            return jsonify({"ok": True})
    return jsonify({"error": "Nenhum processo rodando"}), 404


@app.route("/api/run_cmd", methods=["POST"])
def api_run_cmd():
    """Executa um script Python arbitrario."""
    with state_lock:
        if process_state["running"]:
            return jsonify({"error": "Ja tem um processo rodando!"}), 409
    
    data = request.json or {}
    cmd_arg = data.get("cmd", "").strip()
    if not cmd_arg:
        return jsonify({"error": "Comando vazio"}), 400
    
    cmd = f'"{PYTHON_EXE}" {cmd_arg}'
    thread = threading.Thread(target=run_command, args=(cmd,), daemon=True)
    thread.start()
    return jsonify({"ok": True, "cmd": cmd})


@app.route("/api/tts/test", methods=["POST"])
def api_tts_test():
    """Gera amostra de audio TTS pra preview."""
    data = request.json or {}
    voice = data.get("voice", "pt-BR-AntonioNeural")
    pitch = data.get("pitch", "-2Hz")
    rate = data.get("rate", "+0%")
    texto = data.get("texto", "")
    
    if not texto:
        # Textos de teste por idioma
        if voice.startswith("pt"):
            texto = "Ninguém esperava o que aconteceu naquela noite. O mistério continua sem resposta."
        elif voice.startswith("es"):
            texto = "Nadie esperaba lo que sucedió esa noche. El misterio sigue sin respuesta."
        else:
            texto = "Nobody expected what happened that night. The mystery remains unsolved."
    
    output_path = BASE_DIR / "temp" / "tts_test.mp3"
    
    # Monta comando edge-tts
    pitch_str = pitch if pitch.startswith(("+", "-")) else f"+{pitch}"
    rate_str = rate if rate.startswith(("+", "-")) else f"+{rate}"
    
    cmd = (
        f'"{PYTHON_EXE}" -m edge_tts '
        f'--voice "{voice}" '
        f'--pitch="{pitch_str}" '
        f'--rate="{rate_str}" '
        f'--text "{texto}" '
        f'--write-media "{output_path}"'
    )
    
    try:
        import subprocess as sp
        env = os.environ.copy()
        result = sp.run(
            cmd, shell=True, capture_output=True, text=True,
            timeout=15, cwd=str(BASE_DIR), env=env,
        )
        if result.returncode != 0:
            return jsonify({"error": f"Edge TTS falhou: {result.stderr[:200]}"}), 500
        
        if not output_path.exists():
            return jsonify({"error": "Arquivo de audio nao gerado"}), 500
        
        from flask import send_file
        return send_file(str(output_path), mimetype="audio/mpeg")
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/batch/status", methods=["POST"])
def api_batch_status():
    """Executa batch --status."""
    with state_lock:
        if process_state["running"]:
            return jsonify({"error": "Ja tem um processo rodando!"}), 409
    thread = threading.Thread(target=run_command, args=(f'"{PYTHON_EXE}" batch.py --status',), daemon=True)
    thread.start()
    return jsonify({"ok": True})


@app.route("/api/batch/history", methods=["POST"])
def api_batch_history():
    """Executa batch --historico."""
    with state_lock:
        if process_state["running"]:
            return jsonify({"error": "Ja tem um processo rodando!"}), 409
    thread = threading.Thread(target=run_command, args=(f'"{PYTHON_EXE}" batch.py --historico',), daemon=True)
    thread.start()
    return jsonify({"ok": True})


@app.route("/api/batch/resume", methods=["POST"])
def api_batch_resume():
    """Retoma batch interrompido."""
    with state_lock:
        if process_state["running"]:
            return jsonify({"error": "Ja tem um processo rodando!"}), 409
    thread = threading.Thread(target=run_command, args=(f'"{PYTHON_EXE}" batch.py --resumir',), daemon=True)
    thread.start()
    return jsonify({"ok": True})


@app.route("/api/logs/clear", methods=["POST"])
def api_clear_logs():
    """Limpa historico de logs."""
    with state_lock:
        process_state["logs"].clear()
    return jsonify({"ok": True})


@app.route("/api/logs/stream")
def api_log_stream():
    """Server-Sent Events para logs em tempo real."""
    q = Queue()
    log_queues.append(q)
    
    def generate():
        try:
            while True:
                try:
                    entry = q.get(timeout=30)
                    yield f"data: {json.dumps(entry, ensure_ascii=False)}\n\n"
                except Empty:
                    # Heartbeat pra manter conexao viva
                    yield f"data: {json.dumps({'type': 'heartbeat'})}\n\n"
        except GeneratorExit:
            pass
        finally:
            if q in log_queues:
                log_queues.remove(q)
    
    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@app.route("/api/config/save", methods=["POST"])
def api_config_save():
    """Salva configuracoes no config.py."""
    data = request.json or {}
    config_path = BASE_DIR / "config.py"
    
    if not config_path.exists():
        return jsonify({"error": "config.py nao encontrado"}), 404
    
    try:
        text = config_path.read_text(encoding="utf-8")
        
        # Mapeamento: chave JSON -> (variavel no config.py, formatador)
        replacements = {
            "WHISPER_MODEL": (lambda v: f'"{v}"'),
            "VIDEO_PRESET": (lambda v: f'"{v}"'),
            "VIDEO_WIDTH": (lambda v: str(v)),
            "VIDEO_HEIGHT": (lambda v: str(v)),
            "VIDEO_FPS": (lambda v: str(v)),
            "VIDEO_THREADS": (lambda v: str(v)),
            "TARGET_DURATION": (lambda v: str(v)),
            "TARGET_DURATION_MIN": (lambda v: str(v)),
            "TARGET_DURATION_MAX": (lambda v: str(v)),
            "SUBTITLE_FONT": (lambda v: f'"{v}"'),
            "SUBTITLE_FONT_SIZE": (lambda v: str(v)),
            "SUBTITLE_COLOR": (lambda v: f'"{v}"'),
            "SUBTITLE_HIGHLIGHT_COLOR": (lambda v: f'"{v}"'),
            "SUBTITLE_STROKE_COLOR": (lambda v: f'"{v}"'),
            "SUBTITLE_STROKE_WIDTH": (lambda v: str(v)),
            "SUBTITLE_POSITION_Y": (lambda v: str(v)),
            "SUBTITLE_MAX_CHARS": (lambda v: str(v)),
            "GPU_ENABLED": (lambda v: str(v).capitalize()),
            "TTS_RATE_MIN": (lambda v: str(v)),
            "TTS_RATE_MAX": (lambda v: str(v)),
            "TTS_PITCH": (lambda v: f'"{v}"'),
        }
        
        # Handle TTS_VOICE specially - update TTS_VOICES list
        if "TTS_VOICE" in data:
            voice = data["TTS_VOICE"]
            import re
            pattern = re.compile(
                r'TTS_VOICES\s*=\s*\[.*?\]',
                re.DOTALL,
            )
            if pattern.search(text):
                text = pattern.sub(f'TTS_VOICES = [\n    "{voice}",\n]', text)
        
        for key, fmt in replacements.items():
            if key in data:
                import re
                pattern = re.compile(
                    rf'^({key}\s*=\s*)(.+?)(\s*#.*)?$',
                    re.MULTILINE,
                )
                match = pattern.search(text)
                if match:
                    new_val = fmt(data[key])
                    comment = match.group(3) or ""
                    replacement = f"{match.group(1)}{new_val}{comment}"
                    text = text[:match.start()] + replacement + text[match.end():]
        
        # Regras de postagem (estrutura dict, tratamento especial)
        if any(k.startswith("YT_") or k.startswith("TT_") for k in data):
            # Reconstruir POSTING_RULES
            yt_max = data.get("YT_MAX_DIA", 5)
            yt_int = data.get("YT_INTERVALO", 2.0)
            yt_int_max = round(yt_int * 2.4, 1)
            yt_start = data.get("YT_HORARIO_INICIO", "08:00")
            yt_end = data.get("YT_HORARIO_FIM", "23:00")
            tt_max = data.get("TT_MAX_DIA", 7)
            tt_int = data.get("TT_INTERVALO", 1.5)
            tt_int_max = round(tt_int * 2.3, 1)
            tt_start = data.get("TT_HORARIO_INICIO", "07:00")
            tt_end = data.get("TT_HORARIO_FIM", "23:00")
            
            new_rules = f'''POSTING_RULES = {{
    "youtube": {{
        "max_por_dia": {yt_max},
        "intervalo_min_horas": {yt_int},    # Minimo {yt_int}h entre posts
        "intervalo_max_horas": {yt_int_max},    # Maximo {yt_int_max}h entre posts
        "horario_inicio": "{yt_start}",
        "horario_fim": "{yt_end}",
    }},
    "tiktok": {{
        "max_por_dia": {tt_max},
        "intervalo_min_horas": {tt_int},    # Minimo {tt_int}h entre posts
        "intervalo_max_horas": {tt_int_max},    # Maximo {tt_int_max}h entre posts
        "horario_inicio": "{tt_start}",
        "horario_fim": "{tt_end}",
    }},
}}'''
            # Substituir bloco POSTING_RULES inteiro
            import re
            pattern = re.compile(
                r'POSTING_RULES\s*=\s*\{.*?\n\}',
                re.DOTALL,
            )
            if pattern.search(text):
                text = pattern.sub(new_rules, text)
        
        config_path.write_text(text, encoding="utf-8")
        return jsonify({"ok": True})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/output/open", methods=["POST"])
def api_open_output():
    """Abre pasta output no explorador."""
    output_dir = BASE_DIR / "output"
    output_dir.mkdir(exist_ok=True)
    if sys.platform == "win32":
        os.startfile(str(output_dir))
    return jsonify({"ok": True})


@app.route("/api/analytics")
def api_analytics():
    """Metricas de producao e performance."""
    output_dir = BASE_DIR / "output"
    videos = []
    if output_dir.exists():
        for f in sorted(output_dir.glob("*.mp4"), key=lambda x: x.stat().st_mtime):
            stat = f.stat()
            videos.append({
                "name": f.stem,
                "size_mb": round(stat.st_size / (1024 * 1024), 1),
                "mtime": stat.st_mtime,
                "dt": datetime.fromtimestamp(stat.st_mtime),
            })

    total_size = sum(v["size_mb"] for v in videos)
    avg_size = round(total_size / len(videos), 1) if videos else 0

    from collections import Counter
    today = datetime.now().date()
    daily_counts = Counter()
    for v in videos:
        d = v["dt"].date()
        delta = (today - d).days
        if delta < 14:
            daily_counts[d] = daily_counts.get(d, 0) + 1

    daily = []
    for i in range(13, -1, -1):
        from datetime import timedelta
        d = today - timedelta(days=i)
        daily.append({"label": d.strftime("%d/%m"), "value": daily_counts.get(d, 0)})

    hourly_counts = Counter()
    for v in videos:
        hourly_counts[v["dt"].hour] += 1
    hourly = [{"label": f"{h:02d}h", "value": hourly_counts.get(h, 0)} for h in range(24)]

    recent_videos = sorted(videos, key=lambda x: x["mtime"], reverse=True)[:20]
    sizes = [{"label": v["name"][:8], "value": v["size_mb"]} for v in reversed(recent_videos)]

    history = load_history()
    preset_usage = []
    if history:
        preset_counts = Counter()
        for preset_name, temas in history.items():
            if isinstance(temas, list):
                preset_counts[preset_name] = len(temas)
            elif isinstance(temas, dict):
                for k, v in temas.items():
                    if isinstance(v, list):
                        preset_counts[k] = len(v)
        for name, count in preset_counts.most_common(10):
            preset_usage.append({"name": name.replace("_", " ").title(), "count": count})

    total_temas = sum(p["count"] for p in preset_usage) if preset_usage else 0

    recent_temas = []
    if history:
        for preset_name, temas in history.items():
            if isinstance(temas, list):
                for t in temas[-10:]:
                    if isinstance(t, str):
                        recent_temas.append({"tema": t, "preset": preset_name, "data": ""})
                    elif isinstance(t, dict):
                        recent_temas.append({"tema": t.get("tema", str(t)), "preset": preset_name, "data": t.get("data", "")})
    recent_temas = recent_temas[-20:]
    recent_temas.reverse()

    return jsonify({
        "total_videos": len(videos),
        "total_size_mb": round(total_size, 1),
        "avg_size_mb": avg_size,
        "total_temas": total_temas,
        "presets_used": len(preset_usage),
        "total_duration": f"~{len(videos)}min" if videos else "",
        "daily": daily,
        "hourly": hourly,
        "sizes": sizes,
        "preset_usage": preset_usage,
        "recent_temas": recent_temas,
    })


@app.route("/api/analytics/platforms")
def api_analytics_platforms():
    """Metricas reais do YouTube e TikTok."""
    try:
        from pipeline.analytics import get_full_analytics
        data = get_full_analytics()
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e), "youtube": {"connected": False}, "tiktok": {"connected": False}})


# ========== MAIN ==========

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print()
    print("=" * 50)
    print("  AUTO-SHORTS Web Interface")
    print("=" * 50)
    print(f"  Python: {PYTHON_EXE}")
    print(f"  Abrindo em: http://localhost:{port}")
    print(f"  Pasta: {BASE_DIR}")
    print("  Ctrl+C para encerrar")
    print("=" * 50)
    print()
    
    # Abrir navegador automaticamente
    threading.Timer(1.5, lambda: webbrowser.open(f"http://localhost:{port}")).start()
    
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
