from app import app as fastapi_app
import uvicorn


app = fastapi_app


def main() -> None:
    """Console entrypoint required by OpenEnv validator."""
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
