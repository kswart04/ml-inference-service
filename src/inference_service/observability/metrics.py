from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram


class ServiceMetrics:
    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        labels = ("model_id", "model_version", "policy")
        self.requests = Counter(
            "inference_requests_total",
            "Requests by admission decision",
            (*labels, "decision"),
            registry=self.registry,
        )
        self.outcomes = Counter(
            "inference_terminal_outcomes_total",
            "Accepted request terminal states",
            (*labels, "outcome"),
            registry=self.registry,
        )
        self.pending = Gauge(
            "inference_pending_requests", "Live pending requests", labels, registry=self.registry
        )
        self.active_batches = Gauge(
            "inference_active_batches",
            "Submitted or running batches",
            labels,
            registry=self.registry,
        )
        self.batch_size = Histogram(
            "inference_batch_size",
            "Actual dispatched batch size",
            labels,
            buckets=(1, 2, 4, 8, 16, 32),
            registry=self.registry,
        )
        self.request_duration = Histogram(
            "inference_request_duration_seconds",
            "Handler entry to terminal outcome",
            (*labels, "outcome"),
            registry=self.registry,
        )
        self.queue_wait = Histogram(
            "inference_queue_wait_seconds",
            "Admission to worker start",
            labels,
            registry=self.registry,
        )
        self.execution = Histogram(
            "inference_forward_duration_seconds",
            "Adapter batch execution duration",
            labels,
            registry=self.registry,
        )
        self.preprocessing = Histogram(
            "inference_preprocessing_duration_seconds",
            "Adapter-reported preprocessing duration",
            labels,
            registry=self.registry,
        )
        self.postprocessing = Histogram(
            "inference_postprocessing_duration_seconds",
            "Adapter-reported postprocessing duration",
            labels,
            registry=self.registry,
        )
        self.failures = Counter(
            "inference_execution_failures_total",
            "Batch execution failures",
            (*labels, "category"),
            registry=self.registry,
        )
