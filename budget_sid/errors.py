"""Project exceptions. Exit code 2 is reserved for the scientific stop-gate."""


class ConfigRejected(ValueError):
    """A config uses a model or trainer default this project refuses to run."""


class TestLeakError(RuntimeError):
    """Test users were used to train the predictor or to pick a threshold."""

    __test__ = False


class RoutingProtocolError(RuntimeError):
    """Search was started twice instead of continuing a shared first step."""


class HistoryLeakError(ValueError):
    """The target interaction appears inside the history used to predict it."""
