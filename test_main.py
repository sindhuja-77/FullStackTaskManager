
#test_main.py
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from main import app, Base, get_db

#  Create a temporary SQLite test database
SQLALCHEMY_DATABASE_URL = "sqlite:///./test_tasks.db"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
TestingSessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

#  Create the tables in the test DB
Base.metadata.create_all(bind=engine)

#  Dependency override for testing
def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()

app.dependency_overrides[get_db] = override_get_db

client = TestClient(app)

def test_create_task():
    response = client.post(
        "/tasks",
        json={"title": "Test Task", "description": "Testing DB", "completed": False},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["title"] == "Test Task"
    assert data["description"] == "Testing DB"
    assert data["completed"] is False


def test_get_tasks():
    response = client.get("/tasks")
    assert response.status_code == 200
    data = response.json()
    assert len(data) > 0
    assert data[0]["title"] == "Test Task"

def test_update_task():
    # First, create a task to update
    response = client.post(
        "/tasks",
        json={"title": "Old Task", "description": "Old Desc", "completed": False},
    )
    assert response.status_code == 200
    task_id = response.json()["id"]

    # Now, update the task
    response = client.put(
        f"/tasks/{task_id}",
        json={"title": "Updated Task", "description": "Updated Desc", "completed": True},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["title"] == "Updated Task"
    assert data["description"] == "Updated Desc"
    assert data["completed"] is True


def test_delete_task():
    # First, create a task to delete
    response = client.post(
        "/tasks",
        json={"title": "Task to Delete", "description": "Will be deleted", "completed": False},
    )
    assert response.status_code == 200
    task_id = response.json()["id"]

    # then delete the task
    response = client.delete(f"/tasks/{task_id}")
    assert response.status_code == 200
    assert response.json() == {"message": "Task deleted"}

    # Confirm it's gone
    response = client.get("/tasks")
    data = response.json()
    assert all(task["id"] != task_id for task in data)


    