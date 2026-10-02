import os
import json
import logging
import logging.handlers
from pathlib import Path


class StructuredExtraFormatter(logging.Formatter):
    """
    Formatlayıcı: Ekstra alanları (extra fields) konsol ve dosya çıktılarına
    okunabilir JSON formatında ekler.
    """

    def format(self, record):
        base_msg = super().format(record)
        
        # Standart logging özniteliklerini filtrele
        builtin_attrs = {
            "name", "msg", "args", "levelname", "levelno", "pathname",
            "filename", "module", "lineno", "funcName", "created",
            "msecs", "relativeCreated", "thread", "threadName",
            "processName", "process", "getMessage", "exc_info",
            "exc_text", "stack_info", "asctime", "message", "taskName"
        }
        
        extra_fields = {}
        for key, value in record.__dict__.items():
            if key not in builtin_attrs:
                extra_fields[key] = value

        if extra_fields:
            try:
                extra_str = json.dumps(extra_fields, ensure_ascii=False, default=str)
                return f"{base_msg} | EXTRA: {extra_str}"
            except Exception:
                formatted_pairs = [f"{k}={v}" for k, v in extra_fields.items()]
                return f"{base_msg} | {' | '.join(formatted_pairs)}"
                
        return base_msg


def setup_logging():
    """
    Uygulama loglama altyapısını başlatır.
    Konsol, RotatingFileHandler ve opsiyonel Graylog (GELF) desteği sunar.
    """
    # 1. Ortam Değişkenleri & Streamlit Secrets Desteği
    secrets_dict = {}
    try:
        import streamlit as st
        if hasattr(st, "secrets"):
            secrets_dict = dict(st.secrets)
    except Exception:
        secrets_dict = {}

    def get_config(key: str, default: str = "") -> str:
        return os.getenv(key, str(secrets_dict.get(key, default)))

    environment_name = get_config("ENVIRONMENT", "development")
    log_level_name = get_config("LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_name, logging.INFO)
    
    graylog_enabled = get_config("GRAYLOG_ENABLED", "false").lower() in ("true", "1", "yes")
    graylog_host = get_config("GRAYLOG_HOST", "localhost")
    graylog_port = int(get_config("GRAYLOG_PORT", "12201"))
    graylog_protocol = get_config("GRAYLOG_PROTOCOL", "udp").lower()

    # Log dizini (Modülün bulunduğu klasör altında logs/)
    log_dir = Path(__file__).resolve().parent / "logs"
    log_dir.mkdir(exist_ok=True)

    formatter = StructuredExtraFormatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # 1. Dönen Dosya Loglayıcısı (10 MB x 5 yedek)
    file_handler = logging.handlers.RotatingFileHandler(
        filename=log_dir / "app.log",
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(log_level)

    # 2. Konsol Loglayıcısı
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(log_level)

    logger_name = f"hypothesis-engine-{environment_name.lower()}"
    app_logger = logging.getLogger(logger_name)
    app_logger.setLevel(logging.DEBUG)

    if not app_logger.handlers:
        app_logger.addHandler(file_handler)
        app_logger.addHandler(console_handler)

        # 3. Graylog (GELF UDP / TCP / HTTP) Desteği
        if graylog_enabled:
            try:
                import graypy
                
                if graylog_protocol == "tcp":
                    gelf_handler = graypy.GELFTCPHandler(graylog_host, graylog_port)
                elif graylog_protocol in ("http", "https"):
                    gelf_handler = graypy.GELFHTTPHandler(graylog_host, graylog_port)
                else:
                    gelf_handler = graypy.GELFUDPHandler(graylog_host, graylog_port)

                gelf_handler.facility = logger_name
                gelf_handler.setLevel(log_level)
                app_logger.addHandler(gelf_handler)
                app_logger.info(
                    "Graylog handler successfully registered.",
                    extra={"graylog_host": graylog_host, "graylog_port": graylog_port, "protocol": graylog_protocol}
                )
            except Exception as e:
                app_logger.warning(f"Graylog handler could not be initialized: {e}")

    return app_logger


# Modül düzeyinde global logger nesnesi
logger = setup_logging()


def log_query(question: str, json_query: dict, sql: str, result: any = None, duration_ms: float = None, error: str = None):
    """
    Sorgu üretim ve çalıştırma adımlarını yapılandırılmış (structured) biçimde loglar.
    Hem yerel dosyalara yazar hem de Graylog için aranabilir öznitelikler üretir.
    """
    extra_payload = {
        "event_type": "query_execution",
        "question": question,
        "json_query": json_query,
        "sql": sql,
        "result_sample": str(result)[:300] if result is not None else None,
        "duration_ms": duration_ms
    }
    
    if error:
        extra_payload["error"] = error
        logger.error(f"Query Failed: {question}", extra=extra_payload)
    else:
        logger.info(f"Query Executed: {question}", extra=extra_payload)


def log_db_fallback(target_db: str, fallback_db: str = "sqlite:///insight_generation_bot.db", reason: str = None):
    """
    Hedef veritabanına (ClickHouse vb.) bağlanılamadığında SQLite yedeğine geçişi yapılandırılmış biçimde loglar.
    """
    extra_payload = {
        "event_type": "database_fallback",
        "target_db": target_db,
        "fallback_db": fallback_db,
        "reason": str(reason) if reason else None
    }
    logger.warning(
        f"Database fallback triggered: {target_db} is unavailable. Using {fallback_db} instead.",
        extra=extra_payload
    )
