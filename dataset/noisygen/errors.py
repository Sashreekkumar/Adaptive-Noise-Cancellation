"""Exception types shared across modules."""


class ConfigError(Exception):
    pass


class SampleRejected(Exception):
    """An attempt that cannot yield a valid sample (logged, then retried with a new seed)."""
