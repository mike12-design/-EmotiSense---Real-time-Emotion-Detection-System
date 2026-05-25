import os
import sys
import types
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = PROJECT_ROOT / "backend"
FRONTEND_DIR = PROJECT_ROOT / "frontend"

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from core.models import Base, User  # noqa: E402


@pytest.fixture(scope="session")
def browser_type_launch_args():
    return {"headless": True}


@pytest.fixture(scope="session")
def browser_context_args():
    return {
        "viewport": {"width": 1440, "height": 900},
        "ignore_https_errors": True,
    }


@pytest.fixture(scope="session")
def frontend_base_url():
    return os.getenv("E2E_BASE_URL", "http://127.0.0.1:5173")


@pytest.fixture(scope="session")
def api_base_url():
    return os.getenv("API_BASE_URL", "http://127.0.0.1:8000")


@pytest.fixture(scope="session")
def frontend_dir():
    return FRONTEND_DIR


@pytest.fixture(scope="session")
def test_db_url(tmp_path_factory):
    db_dir = tmp_path_factory.mktemp("databases")
    return f"sqlite:///{db_dir / 'test_emotisense.db'}"


@pytest.fixture(scope="session")
def test_engine(test_db_url):
    engine = create_engine(test_db_url, connect_args={"check_same_thread": False})
    yield engine
    engine.dispose()


@pytest.fixture(scope="session")
def testing_session_local(test_engine):
    return sessionmaker(autocommit=False, autoflush=False, bind=test_engine)


@pytest.fixture()
def db_session(test_engine, testing_session_local):
    Base.metadata.drop_all(bind=test_engine)
    Base.metadata.create_all(bind=test_engine)

    session = testing_session_local()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture()
def seeded_admin(db_session):
    admin = User(username="admin", password_hash="123456", role="admin")
    db_session.add(admin)
    db_session.commit()
    db_session.refresh(admin)
    return admin


@pytest.fixture(scope="session")
def backend_import_stubs():
    cv2_module = types.ModuleType("cv2")
    cv2_module.data = types.SimpleNamespace(haarcascades="")
    cv2_module.CascadeClassifier = lambda *args, **kwargs: object()
    cv2_module.VideoCapture = lambda *args, **kwargs: types.SimpleNamespace(
        isOpened=lambda: False,
        read=lambda: (False, None),
        release=lambda: None,
    )
    cv2_module.flip = lambda frame, mode: frame
    cv2_module.imencode = lambda ext, frame: (True, types.SimpleNamespace(tobytes=lambda: b""))
    cv2_module.cvtColor = lambda frame, code: frame
    cv2_module.COLOR_BGR2GRAY = 0
    cv2_module.IMREAD_COLOR = 1
    cv2_module.imdecode = lambda *args, **kwargs: None

    deepface_module = types.ModuleType("deepface")
    deepface_module.DeepFace = types.SimpleNamespace(
        analyze=lambda *args, **kwargs: {"emotion": {"neutral": 100.0}},
        represent=lambda *args, **kwargs: [{"embedding": [0.0]}],
    )

    ultralytics_module = types.ModuleType("ultralytics")
    ultralytics_module.YOLO = lambda *args, **kwargs: types.SimpleNamespace(
        __call__=lambda *a, **k: []
    )

    pygame_module = types.ModuleType("pygame")
    pygame_module.mixer = types.SimpleNamespace(
        init=lambda: None,
        music=types.SimpleNamespace(
            set_volume=lambda *args, **kwargs: None,
            get_busy=lambda: False,
            load=lambda *args, **kwargs: None,
            play=lambda *args, **kwargs: None,
            stop=lambda *args, **kwargs: None,
            pause=lambda *args, **kwargs: None,
            unpause=lambda *args, **kwargs: None,
        ),
        Sound=lambda *args, **kwargs: types.SimpleNamespace(
            play=lambda: None,
            get_length=lambda: 0,
        ),
    )

    edge_tts_module = types.ModuleType("edge_tts")
    edge_tts_module.Communicate = lambda *args, **kwargs: types.SimpleNamespace(
        save=lambda *a, **k: None
    )

    sys.modules.setdefault("cv2", cv2_module)
    sys.modules.setdefault("deepface", deepface_module)
    sys.modules.setdefault("ultralytics", ultralytics_module)
    sys.modules.setdefault("pygame", pygame_module)
    sys.modules.setdefault("edge_tts", edge_tts_module)


@pytest.fixture()
def api_client(db_session, backend_import_stubs):
    from app.api import get_db as api_get_db  # noqa: E402
    from app.api import router as api_router  # noqa: E402

    app = FastAPI(title="EmotiSense Test API")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    assets_dir = BACKEND_DIR / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

    app.include_router(api_router)

    def override_get_db():
        yield db_session

    app.dependency_overrides[api_get_db] = override_get_db

    with TestClient(app) as client:
        yield client

    app.dependency_overrides.clear()
