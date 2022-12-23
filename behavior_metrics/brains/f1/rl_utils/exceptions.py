class CreationError(Exception):
    ...

class NoValidEnvironmentType(CreationError):
    def __init__(self, environment_type):
        self.traning_type = environment_type
        self.message = f"[MESSAGE] No valid training type ({environment_type}) in your settings.py file"
        super().__init__(self.message)
