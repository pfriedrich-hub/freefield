import pathlib

__version__ = "1.1.0"

DIR = pathlib.Path(__file__).parent.resolve()


def __getattr__(name):
    if name == "Sensor":
        try:
            from freefield.motion_sensor import Sensor
        except ModuleNotFoundError as exc:
            if exc.name == "mbientlab":
                raise ModuleNotFoundError(
                    "freefield.Sensor requires the optional dependency 'mbientlab'. "
                    "Install 'mbientlab' to use motion sensor functionality."
                ) from exc
            raise

        globals()["Sensor"] = Sensor
        return Sensor

    if name == "Processors":
        from freefield.processors import Processors

        globals()["Processors"] = Processors
        return Processors

    if name == "Cameras":
        from freefield.cameras import Cameras

        globals()["Cameras"] = Cameras
        return Cameras

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


from freefield.freefield import *  # noqa: F401,F403
