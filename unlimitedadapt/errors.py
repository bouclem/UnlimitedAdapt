"""Framework exceptions, with messages meant to be actionable."""


class UnlimitedAdaptError(Exception):
    """Base class for everything this framework raises deliberately."""


class UnsupportedModelError(UnlimitedAdaptError):
    """The model's architecture has no registered spec.

    Raised when asking for a family that does not exist, or when a model
    does not match the spec it claims to be.
    """


class UnsupportedVersionError(UnlimitedAdaptError):
    """The model's version falls outside the spec's supported range."""


class AdapterAttachmentError(UnlimitedAdaptError):
    """An adapter cannot be attached: wrong dimensions, unknown point,
    or name conflict."""


class AdapterNotFoundError(UnlimitedAdaptError):
    """An operation names an adapter that is not attached."""
