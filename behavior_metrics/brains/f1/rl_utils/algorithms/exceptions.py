class CreationError(Exception):
    ...


class NoValidAlgorithmType(CreationError):
    def __init__(self, algorithm):
        self.algorithm_type = algorithm
        self.message = f"[ERROR] No valid training type ({algorithm}) in your config.yml file or is missing."
        super().__init__(self.message)
