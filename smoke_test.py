from server.environment import TransplantEnv, TransplantGrader, TASKS
from models import ActionType, TransplantAction, TransportMode

env = TransplantEnv("task_easy_clear_match")
obs = env.reset(seed=42)
print(obs.available_donors) 