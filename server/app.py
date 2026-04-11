import uvicorn
from fastapi import FastAPI
from openenv.core.env_server import create_app

from env import AppLensOpenEnv
from models import Action, Observation

# Create the FASTAPI app using openenv_core
app = create_app(AppLensOpenEnv, Action, Observation)

def main():
    import os
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("server.app:app", host="0.0.0.0", port=port, reload=False)

if __name__ == "__main__":
    main()
