class SchedulerError(Exception):
    """Base class for errors translated by the HTTP layer."""


class QueueFullError(SchedulerError):
    pass


class RequestDeadlineError(SchedulerError):
    pass


class SchedulerUnavailableError(SchedulerError):
    pass


class BatchExecutionError(SchedulerError):
    pass


class AdapterContractError(SchedulerError):
    pass


class FatalWorkerError(Exception):
    """The adapter cannot continue serving; restart the process."""
