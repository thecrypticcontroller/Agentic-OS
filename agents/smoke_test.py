from crewai import Agent

agent = Agent(
    role="System Validator",
    goal="Confirm that the Agent OS environment is operational.",
    backstory="You validate local AI infrastructure.",
    verbose=True,
)

print("CrewAI agent created successfully:", agent.role)
