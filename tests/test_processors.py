"""Unit tests for :class:`freefield.processors.Processors`.

These tests exercise the Python-side processor logic only. All TDT COM objects
are replaced by ``MagicMock`` instances, so no Windows installation, TDT
software, RCX hardware, or physical processor is required.
"""

import importlib
import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, call
from freefield import setups

import numpy as np
import pytest


processors_module = importlib.import_module("freefield.processors")
freefield_module = importlib.import_module("freefield.freefield")
Processors = processors_module.Processors


# ---------------------------------------------------------------------------
# Helpers and fixtures
# ---------------------------------------------------------------------------


def make_processor_mock(name="processor"):
    """Return a processor mock with successful default return values."""
    processor = MagicMock(name=name)

    processor.ConnectRP2.return_value = 1
    processor.ConnectRX8.return_value = 1
    processor.ConnectRX6.return_value = 1
    processor.ConnectRM1.return_value = 1

    processor.ClearCOF.return_value = 1
    processor.LoadCOF.return_value = 1
    processor.Run.return_value = 1

    processor.SetTagVal.return_value = 1
    processor.GetTagVal.return_value = 1
    processor.ReadTagV.return_value = [1.0, 2.0, 3.0]

    processor.SoftTrg.return_value = 1
    processor.Halt.return_value = 1

    processor._oleobj_.InvokeTypes.return_value = 1

    return processor


@pytest.fixture
def processors():
    return Processors()


@pytest.fixture
def populated_processors():
    instance = Processors()
    instance.processors = {
        "RP2": make_processor_mock("RP2"),
        "RX81": make_processor_mock("RX81"),
        "RX82": make_processor_mock("RX82"),
    }
    return instance


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_init_sets_empty_default_state(processors):
    assert processors.processors == {}
    assert processors.mode is None
    assert processors._zbus is None


# ---------------------------------------------------------------------------
# _initialize_proc
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("model", "connect_method"),
    [
        ("RP2", "ConnectRP2"),
        ("RX8", "ConnectRX8"),
        ("RX6", "ConnectRX6"),
        ("RM1", "ConnectRM1"),
    ],
)
def test_initialize_proc_uses_correct_connect_method(
    monkeypatch,
    model,
    connect_method,
):
    processor = make_processor_mock()
    com_factory = MagicMock(return_value=processor)

    monkeypatch.setattr(processors_module, "win32com", None)
    monkeypatch.setattr(processors_module, "_COM", com_factory)

    result = Processors._initialize_proc(
        model=model,
        circuit="test_circuit.rcx",
        connection="GB",
        index=2,
    )

    com_factory.assert_called_once_with()
    getattr(processor, connect_method).assert_called_once_with("GB", 2)

    for other_method in (
        "ConnectRP2",
        "ConnectRX8",
        "ConnectRX6",
        "ConnectRM1",
    ):
        if other_method != connect_method:
            getattr(processor, other_method).assert_not_called()

    processor.ClearCOF.assert_called_once_with()
    processor.LoadCOF.assert_called_once_with("test_circuit.rcx")
    processor.Run.assert_called_once_with()

    assert result is processor


@pytest.mark.parametrize(
    "model",
    ["rp2", "rx8", "Rx6", "rm1"],
)
def test_initialize_proc_accepts_case_insensitive_models(
    monkeypatch,
    model,
):
    processor = make_processor_mock()

    monkeypatch.setattr(processors_module, "win32com", None)
    monkeypatch.setattr(
        processors_module,
        "_COM",
        MagicMock(return_value=processor),
    )

    result = Processors._initialize_proc(
        model,
        "test.rcx",
        "USB",
        1,
    )

    assert result is processor


@pytest.mark.parametrize(
    ("model", "connect_method"),
    [
        ("RP2", "ConnectRP2"),
        ("RX8", "ConnectRX8"),
        ("RX6", "ConnectRX6"),
        ("RM1", "ConnectRM1"),
    ],
)
def test_initialize_proc_returns_none_when_connection_fails(
    monkeypatch,
    caplog,
    model,
    connect_method,
):
    processor = make_processor_mock()
    getattr(processor, connect_method).return_value = 0

    monkeypatch.setattr(processors_module, "win32com", None)
    monkeypatch.setattr(
        processors_module,
        "_COM",
        MagicMock(return_value=processor),
    )

    with caplog.at_level(logging.WARNING):
        result = Processors._initialize_proc(
            model,
            "test.rcx",
            "GB",
            1,
        )

    assert result is None
    assert f"Unable to connect to {model} processor!" in caplog.text

    processor.ClearCOF.assert_not_called()
    processor.LoadCOF.assert_not_called()
    processor.Run.assert_not_called()


def test_initialize_proc_returns_none_for_unknown_model(
    monkeypatch,
    caplog,
):
    processor = make_processor_mock()

    monkeypatch.setattr(processors_module, "win32com", None)
    monkeypatch.setattr(
        processors_module,
        "_COM",
        MagicMock(return_value=processor),
    )

    with caplog.at_level(logging.WARNING):
        result = Processors._initialize_proc(
            "UNKNOWN",
            "test.rcx",
            "GB",
            1,
        )

    assert result is None
    assert "Unable to connect to UNKNOWN processor!" in caplog.text

    processor.ConnectRP2.assert_not_called()
    processor.ConnectRX8.assert_not_called()
    processor.ConnectRX6.assert_not_called()
    processor.ConnectRM1.assert_not_called()


@pytest.mark.parametrize(
    ("failed_method", "warning_text"),
    [
        (
            "ClearCOF",
            "clearing control object file failed",
        ),
        (
            "LoadCOF",
            "could not load test.rcx.",
        ),
        (
            "Run",
            "Failed to run RX8 processor",
        ),
    ],
)
def test_initialize_proc_logs_stage_failures_but_returns_processor(
    monkeypatch,
    caplog,
    failed_method,
    warning_text,
):
    processor = make_processor_mock()
    getattr(processor, failed_method).return_value = 0

    monkeypatch.setattr(processors_module, "win32com", None)
    monkeypatch.setattr(
        processors_module,
        "_COM",
        MagicMock(return_value=processor),
    )

    with caplog.at_level(logging.WARNING):
        result = Processors._initialize_proc(
            "RX8",
            "test.rcx",
            "GB",
            1,
        )

    assert result is processor
    assert warning_text in caplog.text

    processor.ClearCOF.assert_called_once_with()
    processor.LoadCOF.assert_called_once_with("test.rcx")
    processor.Run.assert_called_once_with()


def test_initialize_proc_calls_steps_in_expected_order(monkeypatch):
    processor = make_processor_mock()

    monkeypatch.setattr(processors_module, "win32com", None)
    monkeypatch.setattr(
        processors_module,
        "_COM",
        MagicMock(return_value=processor),
    )

    Processors._initialize_proc(
        "RX8",
        "test.rcx",
        "GB",
        3,
    )

    assert processor.method_calls == [
        call.ConnectRX8("GB", 3),
        call.ClearCOF(),
        call.LoadCOF("test.rcx"),
        call.Run(),
    ]


def test_initialize_proc_uses_win32com_dispatch_when_available(
    monkeypatch,
):
    processor = make_processor_mock()

    client = MagicMock()
    client.Dispatch.return_value = processor
    fake_win32com = SimpleNamespace(client=client)

    fallback_factory = MagicMock()

    monkeypatch.setattr(
        processors_module,
        "win32com",
        fake_win32com,
    )
    monkeypatch.setattr(
        processors_module,
        "_COM",
        fallback_factory,
    )

    result = Processors._initialize_proc(
        "RP2",
        "test.rcx",
        "USB",
        1,
    )

    client.Dispatch.assert_called_once_with("RPco.X")
    fallback_factory.assert_not_called()
    processor.ConnectRP2.assert_called_once_with("USB", 1)

    assert result is processor


def test_initialize_proc_converts_com_error_to_value_error(
    monkeypatch,
):
    class FakeComError(Exception):
        pass

    client = MagicMock()
    client.Dispatch.side_effect = FakeComError("dispatch failed")
    client.pythoncom = SimpleNamespace(com_error=FakeComError)

    fake_win32com = SimpleNamespace(client=client)

    monkeypatch.setattr(
        processors_module,
        "win32com",
        fake_win32com,
    )

    with pytest.raises(ValueError, match="dispatch failed"):
        Processors._initialize_proc(
            "RX8",
            "test.rcx",
            "GB",
            1,
        )


# ---------------------------------------------------------------------------
# _initialize_zbus
# ---------------------------------------------------------------------------


def test_initialize_zbus_uses_fallback_com_without_win32com(
    monkeypatch,
):
    zbus = MagicMock(name="zbus")
    zbus.ConnectZBUS.return_value = 1

    com_factory = MagicMock(return_value=zbus)

    monkeypatch.setattr(processors_module, "win32com", None)
    monkeypatch.setattr(processors_module, "_COM", com_factory)

    result = Processors._initialize_zbus("USB")

    com_factory.assert_called_once_with()
    zbus.ConnectZBUS.assert_called_once_with("USB")

    assert result is zbus


def test_initialize_zbus_uses_win32com_dispatch_when_available(
    monkeypatch,
):
    fallback_zbus = MagicMock(name="fallback_zbus")
    fallback_factory = MagicMock(return_value=fallback_zbus)

    zbus = MagicMock(name="zbus")
    zbus.ConnectZBUS.return_value = 1

    client = MagicMock()
    client.Dispatch.return_value = zbus

    fake_win32com = SimpleNamespace(client=client)

    monkeypatch.setattr(
        processors_module,
        "win32com",
        fake_win32com,
    )
    monkeypatch.setattr(
        processors_module,
        "_COM",
        fallback_factory,
    )

    result = Processors._initialize_zbus("GB")

    fallback_factory.assert_called_once_with()
    client.Dispatch.assert_called_once_with("ZBUS.x")
    zbus.ConnectZBUS.assert_called_once_with("GB")

    assert result is zbus


def test_initialize_zbus_logs_failed_connection(
    monkeypatch,
    caplog,
):
    zbus = MagicMock(name="zbus")
    zbus.ConnectZBUS.return_value = 0

    monkeypatch.setattr(processors_module, "win32com", None)
    monkeypatch.setattr(
        processors_module,
        "_COM",
        MagicMock(return_value=zbus),
    )

    with caplog.at_level(logging.WARNING):
        result = Processors._initialize_zbus("GB")

    assert result is zbus
    assert "Failed to connect to ZBUS." in caplog.text


def test_initialize_zbus_keeps_fallback_after_dispatch_error(
    monkeypatch,
    caplog,
):
    class FakeComError(Exception):
        pass

    fallback_zbus = MagicMock(name="fallback_zbus")
    fallback_zbus.ConnectZBUS.return_value = 1

    fallback_factory = MagicMock(return_value=fallback_zbus)

    client = MagicMock()
    client.Dispatch.side_effect = FakeComError(
        "zbus dispatch failed"
    )
    client.pythoncom = SimpleNamespace(com_error=FakeComError)

    fake_win32com = SimpleNamespace(client=client)

    monkeypatch.setattr(
        processors_module,
        "win32com",
        fake_win32com,
    )
    monkeypatch.setattr(
        processors_module,
        "_COM",
        fallback_factory,
    )

    with caplog.at_level(logging.WARNING):
        result = Processors._initialize_zbus("GB")

    assert result is fallback_zbus
    assert "zbus dispatch failed" in caplog.text
    fallback_zbus.ConnectZBUS.assert_called_once_with("GB")


# ---------------------------------------------------------------------------
# initialize
# ---------------------------------------------------------------------------


def test_initialize_accepts_single_device_list(
    monkeypatch,
    tmp_path,
    processors,
):
    circuit = tmp_path / "single.rcx"
    circuit.touch()

    processor = make_processor_mock()
    initialize_proc = MagicMock(return_value=processor)

    monkeypatch.setattr(
        Processors,
        "_initialize_proc",
        initialize_proc,
    )

    processors.initialize(
        ["RX81", "RX8", str(circuit)]
    )

    initialize_proc.assert_called_once_with(
        "RX8",
        str(circuit),
        "GB",
        1,
    )

    assert processors.processors == {
        "RX81": processor,
    }
    assert processors.mode == "custom"


def test_initialize_assigns_separate_indexes_per_model(
    monkeypatch,
    tmp_path,
    processors,
):
    rx8_circuit = tmp_path / "rx8.rcx"
    rp2_circuit = tmp_path / "rp2.rcx"

    rx8_circuit.touch()
    rp2_circuit.touch()

    rx81 = make_processor_mock("RX81")
    rp2 = make_processor_mock("RP2")
    rx82 = make_processor_mock("RX82")

    initialize_proc = MagicMock(
        side_effect=[rx81, rp2, rx82]
    )

    monkeypatch.setattr(
        Processors,
        "_initialize_proc",
        initialize_proc,
    )

    devices = [
        ["RX81", "RX8", str(rx8_circuit)],
        ["RP2", "RP2", str(rp2_circuit)],
        ["RX82", "RX8", str(rx8_circuit)],
    ]

    processors.initialize(
        devices,
        connection="USB",
    )

    assert initialize_proc.call_args_list == [
        call(
            "RX8",
            str(rx8_circuit),
            "USB",
            1,
        ),
        call(
            "RP2",
            str(rp2_circuit),
            "USB",
            1,
        ),
        call(
            "RX8",
            str(rx8_circuit),
            "USB",
            2,
        ),
    ]

    assert processors.processors == {
        "RX81": rx81,
        "RP2": rp2,
        "RX82": rx82,
    }


def test_initialize_rejects_duplicate_device_names(processors):
    devices = [
        ["RX81", "RX8", "first.rcx"],
        ["RX81", "RX8", "second.rcx"],
    ]

    with pytest.raises(KeyError, match="unique name"):
        processors.initialize(devices)


def test_initialize_rejects_missing_circuit_file(
    processors,
    tmp_path,
):
    missing = tmp_path / "missing.rcx"

    with pytest.raises(
        FileNotFoundError,
        match="missing.rcx does not exist",
    ):
        processors.initialize(
            ["RX81", "RX8", str(missing)]
        )


def test_initialize_resolves_circuit_from_package_rcx_directory(
    monkeypatch,
    tmp_path,
    processors,
):
    rcx_dir = tmp_path / "data" / "rcx"
    rcx_dir.mkdir(parents=True)

    circuit = rcx_dir / "packaged.rcx"
    circuit.touch()

    processor = make_processor_mock()
    initialize_proc = MagicMock(return_value=processor)

    monkeypatch.setattr(
        processors_module,
        "DIR",
        tmp_path,
    )
    monkeypatch.setattr(
        Processors,
        "_initialize_proc",
        initialize_proc,
    )

    device = [
        ["RX81", "RX8", "packaged.rcx"],
    ]

    processors.initialize(device)

    resolved_path = str(circuit)

    assert device[0][2] == resolved_path

    initialize_proc.assert_called_once_with(
        "RX8",
        resolved_path,
        "GB",
        1,
    )


def test_initialize_initializes_zbus_when_requested(
    monkeypatch,
    tmp_path,
    processors,
):
    circuit = tmp_path / "test.rcx"
    circuit.touch()

    processor = make_processor_mock()
    zbus = MagicMock(name="zbus")

    initialize_proc = MagicMock(return_value=processor)
    initialize_zbus = MagicMock(return_value=zbus)

    monkeypatch.setattr(
        Processors,
        "_initialize_proc",
        initialize_proc,
    )
    monkeypatch.setattr(
        Processors,
        "_initialize_zbus",
        initialize_zbus,
    )

    processors.initialize(
        ["RX81", "RX8", str(circuit)],
        zbus=True,
        connection="USB",
    )

    initialize_zbus.assert_called_once_with("USB")
    assert processors._zbus is zbus


def test_initialize_does_not_overwrite_existing_mode(
    monkeypatch,
    tmp_path,
    processors,
):
    circuit = tmp_path / "test.rcx"
    circuit.touch()

    processors.mode = "play_rec"

    monkeypatch.setattr(
        Processors,
        "_initialize_proc",
        MagicMock(
            return_value=make_processor_mock()
        ),
    )

    processors.initialize(
        ["RX81", "RX8", str(circuit)]
    )

    assert processors.mode == "play_rec"


# ---------------------------------------------------------------------------
# initialize_default
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("mode", "expected_circuits"),
    [
        (
            "play_rec",
            [
                "rec_buf.rcx",
                "play_buf.rcx",
                "play_buf.rcx",
            ]
        ),
        (
            "play_birec",
            [
                "bi_rec_buf.rcx",
                "play_buf.rcx",
                "play_buf.rcx",
            ]
        ),
        (
            "bi_play_rec",
            [
                "bi_play_rec_buf.rcx",
                "bits.rcx",
                "bits.rcx",
            ]
        ),
        (
            "loctest_freefield",
            [
                "button.rcx",
                "play_buf.rcx",
                "play_buf.rcx",
            ]
        ),
        (
            "loctest_headphones",
            [
                "bi_play_buf.rcx",
                "bits.rcx",
                "bits.rcx",
            ]
        ),
        (
            "cam_calibration",
            [
                "button.rcx",
                "bits.rcx",
                "bits.rcx",
            ]
        ),
    ],
)
def test_initialize_default_builds_expected_configuration_for_dome(
    processors,
    mode,
    expected_circuits,
    monkeypatch
):
    monkeypatch.setattr(processors_module.freefield, "SETUP", setups.DOME)

    processors.initialize = MagicMock()

    processors.initialize_default(mode=mode)

    processors.initialize.assert_called_once()

    proc_list, zbus, connection = (
        processors.initialize.call_args.args
    )

    assert [entry[0] for entry in proc_list] == [
        "RP2",
        "RX81",
        "RX82",
    ]
    assert [entry[1] for entry in proc_list] == [
        "RP2",
        "RX8",
        "RX8",
    ]
    assert [
        Path(entry[2]).name
        for entry in proc_list
    ] == expected_circuits

    assert zbus is True
    assert connection == "GB"
    assert processors.mode == mode

@pytest.mark.parametrize(
    ("mode", "expected_circuits"),
    [
        (
            "play_rec",
            [
                "rec_buf.rcx",
                "play_buf.rcx",
                "play_buf.rcx",
            ]
        ),
        (
            "play_birec",
            [
                "bi_rec_buf.rcx",
                "play_buf.rcx",
                "play_buf.rcx",
            ]
        ),
        (
            "bi_play_rec",
            [
                "bi_play_rec_buf.rcx",
                "bits.rcx",
                "bits.rcx",
            ]
        ),
        (
            "loctest_freefield",
            [
                "button.rcx",
                "play_buf.rcx",
                "play_buf.rcx",
            ]
        ),
        (
            "loctest_headphones",
            [
                "bi_play_buf.rcx",
                "bits.rcx",
                "bits.rcx",
            ]
        ),
        (
            "cam_calibration",
            [
                "button.rcx",
                "bits.rcx",
                "bits.rcx",
            ]
        ),
    ],
)
def test_initialize_default_builds_expected_configuration_for_arc(
        processors,
        mode,
        expected_circuits,
        monkeypatch
):
    monkeypatch.setattr(processors_module.freefield, "SETUP", setups.ARC)

    processors.initialize = MagicMock()

    processors.initialize_default(mode=mode)

    processors.initialize.assert_called_once()

    proc_list, zbus, connection = (
        processors.initialize.call_args.args
    )

    assert [entry[0] for entry in proc_list] == [
        "RP2",
        "RX81",
        "RX82",
    ]
    assert [entry[1] for entry in proc_list] == [
        "RP2",
        "RX8",
        "RX8",
    ]
    assert [
               Path(entry[2]).name
               for entry in proc_list
           ] == expected_circuits

    assert zbus is True
    assert connection == "GB"
    assert processors.mode == mode

@pytest.mark.parametrize(
    ("mode", "expected_circuits"),
    [
        (
            "play_rec",
            [
                "rec_buf.rcx",
                "play_buf.rcx",
            ]
        ),
        (
            "play_birec",
            [
                "bi_rec_buf.rcx",
                "play_buf.rcx",
            ]
        ),
        (
            "bi_play_rec",
            [
                "bi_play_rec_buf.rcx",
                "bits.rcx",
            ]
        ),
        (
            "loctest_freefield",
            [
                "button.rcx",
                "play_buf.rcx",
            ]
        ),
        (
            "loctest_headphones",
            [
                "bi_play_buf.rcx",
                "bits.rcx",
            ]
        ),
        (
            "cam_calibration",
            [
                "button.rcx",
                "bits.rcx",
            ]
        ),
    ],
)
def test_initialize_default_builds_expected_configuration_for_distance_array(
    monkeypatch,
    processors,
    mode,
    expected_circuits,
):

    monkeypatch.setattr(processors_module.freefield, "SETUP", setups.DISTANCE_ARRAY)
    processors.initialize = MagicMock()

    processors.initialize_default(mode=mode)

    processors.initialize.assert_called_once()

    proc_list, zbus, connection = (
        processors.initialize.call_args.args
    )

    assert [entry[0] for entry in proc_list] == [
        "RP2",
        "RX81",
    ]
    assert [entry[1] for entry in proc_list] == [
        "RP2",
        "RX8",
    ]
    assert [
               Path(entry[2]).name
               for entry in proc_list
           ] == expected_circuits

    assert zbus is False
    assert connection == "USB"
    assert processors.mode == mode


def test_initialize_default_is_case_insensitive(processors, monkeypatch):
    processors.initialize = MagicMock()
    monkeypatch.setattr(processors_module.freefield, "SETUP", setups.DOME)

    processors.initialize_default(mode="PLAY_REC")

    processors.initialize.assert_called_once()
    assert processors.mode == "play_rec"


def test_initialize_default_rejects_unknown_mode(processors, monkeypatch):
    monkeypatch.setattr(processors_module.freefield, "SETUP", setups.DOME)
    with pytest.raises(
        ValueError
    ):
        processors.initialize_default(mode="unknown")


# ---------------------------------------------------------------------------
# write
# ---------------------------------------------------------------------------


def test_write_scalar_to_single_processor(
    populated_processors,
):
    result = populated_processors.write(
        "level",
        42,
        "RX81",
    )

    populated_processors.processors[
        "RX81"
    ].SetTagVal.assert_called_once_with(
        "level",
        42,
    )

    assert result == 1


@pytest.mark.parametrize(
    "value",
    [
        np.int32(7),
        np.int64(9),
    ],
)
def test_write_converts_numpy_integer_to_builtin_int(
    populated_processors,
    value,
):
    populated_processors.write(
        "value",
        value,
        "RP2",
    )

    written_value = (
        populated_processors.processors[
            "RP2"
        ].SetTagVal.call_args.args[1]
    )

    assert written_value == int(value)
    assert type(written_value) is int


def test_write_all_selects_every_processor(
    populated_processors,
):
    result = populated_processors.write(
        "level",
        1.5,
        "all",
    )

    for processor in (
        populated_processors.processors.values()
    ):
        processor.SetTagVal.assert_called_once_with(
            "level",
            1.5,
        )

    assert result == 1


def test_write_rx8s_selects_only_rx8_processors(
    populated_processors,
):
    result = populated_processors.write(
        "level",
        1.5,
        "RX8s",
    )

    populated_processors.processors[
        "RP2"
    ].SetTagVal.assert_not_called()

    populated_processors.processors[
        "RX81"
    ].SetTagVal.assert_called_once_with(
        "level",
        1.5,
    )

    populated_processors.processors[
        "RX82"
    ].SetTagVal.assert_called_once_with(
        "level",
        1.5,
    )

    assert result == 1


def test_write_accepts_explicit_processor_list(
    populated_processors,
):
    populated_processors.write(
        "level",
        2,
        ["RP2", "RX82"],
    )

    populated_processors.processors[
        "RP2"
    ].SetTagVal.assert_called_once_with(
        "level",
        2,
    )

    populated_processors.processors[
        "RX81"
    ].SetTagVal.assert_not_called()

    populated_processors.processors[
        "RX82"
    ].SetTagVal.assert_called_once_with(
        "level",
        2,
    )


@pytest.mark.parametrize(
    "requested_processors",
    [
        "UNKNOWN",
        ["RX81", "UNKNOWN"],
    ],
)
def test_write_rejects_unknown_processors(
    populated_processors,
    requested_processors,
):
    with pytest.raises(
        ValueError,
        match="Can not find",
    ):
        populated_processors.write(
            "level",
            1,
            requested_processors,
        )


def test_write_vector_uses_invoke_types(
    populated_processors,
):
    value = [0.1, 0.2, 0.3]

    result = populated_processors.write(
        "buffer",
        value,
        "RX81",
    )

    invoke = (
        populated_processors.processors[
            "RX81"
        ]._oleobj_.InvokeTypes
    )

    invoke.assert_called_once()

    args = invoke.call_args.args

    assert args[:7] == (
        15,
        0x0,
        1,
        (3, 0),
        (
            (8, 0),
            (3, 0),
            (0x2005, 0),
        ),
        "buffer",
        0,
    )

    np.testing.assert_array_equal(
        args[7],
        np.asarray(value),
    )

    assert result == 1


def test_write_flattens_multidimensional_array(
    populated_processors,
):
    value = np.array(
        [
            [1, 2],
            [3, 4],
        ]
    )

    populated_processors.write(
        "buffer",
        value,
        "RX81",
    )

    written_value = (
        populated_processors.processors[
            "RX81"
        ]
        ._oleobj_
        .InvokeTypes
        .call_args
        .args[7]
    )

    np.testing.assert_array_equal(
        written_value,
        np.array([1, 2, 3, 4]),
    )


def test_write_logs_warning_when_processor_returns_zero(
    populated_processors,
    caplog,
):
    populated_processors.processors[
        "RX81"
    ].SetTagVal.return_value = 0

    with caplog.at_level(logging.WARNING):
        result = populated_processors.write(
            "level",
            2,
            "RX81",
        )

    assert result == 0
    assert (
        "Unable to set tag level on RX81"
        in caplog.text
    )


def test_write_returns_last_processor_flag(
    populated_processors,
):
    populated_processors.processors[
        "RX81"
    ].SetTagVal.return_value = 0

    populated_processors.processors[
        "RX82"
    ].SetTagVal.return_value = 1

    result = populated_processors.write(
        "level",
        2,
        ["RX81", "RX82"],
    )

    assert result == 1


def test_write_empty_processor_list_returns_zero(
    populated_processors,
):
    result = populated_processors.write(
        "level",
        1,
        [],
    )

    assert result == 0


# ---------------------------------------------------------------------------
# read
# ---------------------------------------------------------------------------


def test_read_scalar_uses_get_tag_val(
    populated_processors,
):
    populated_processors.processors[
        "RP2"
    ].GetTagVal.return_value = 12.5

    result = populated_processors.read(
        "level",
        "RP2",
    )

    populated_processors.processors[
        "RP2"
    ].GetTagVal.assert_called_once_with(
        "level"
    )

    populated_processors.processors[
        "RP2"
    ].ReadTagV.assert_not_called()

    assert result == 12.5


def test_read_vector_uses_read_tag_v_and_returns_numpy_array(
    populated_processors,
):
    populated_processors.processors[
        "RP2"
    ].ReadTagV.return_value = [1, 2, 3]

    result = populated_processors.read(
        "buffer",
        "RP2",
        n_samples=3,
    )

    populated_processors.processors[
        "RP2"
    ].ReadTagV.assert_called_once_with(
        "buffer",
        0,
        3,
    )

    populated_processors.processors[
        "RP2"
    ].GetTagVal.assert_not_called()

    assert isinstance(result, np.ndarray)

    np.testing.assert_array_equal(
        result,
        np.array([1, 2, 3]),
    )


# ---------------------------------------------------------------------------
# halt
# ---------------------------------------------------------------------------


def test_halt_calls_halt_on_all_compatible_processors(
    processors,
):
    first = MagicMock(spec=["Halt"])
    second = MagicMock(spec=["Halt"])

    processors.processors = {
        "first": first,
        "second": second,
    }

    processors.halt()

    first.Halt.assert_called_once_with()
    second.Halt.assert_called_once_with()


def test_halt_skips_objects_without_halt_method(processors):
    with_halt = MagicMock(spec=["Halt"])
    without_halt = object()

    processors.processors = {
        "with_halt": with_halt,
        "without_halt": without_halt,
    }

    processors.halt()

    with_halt.Halt.assert_called_once_with()


# ---------------------------------------------------------------------------
# trigger
# ---------------------------------------------------------------------------


def test_trigger_sends_soft_trigger_to_selected_processors(
    populated_processors,
):
    populated_processors.trigger(
        kind=3,
        proc=["RP2", "RX82"],
    )

    populated_processors.processors[
        "RP2"
    ].SoftTrg.assert_called_once_with(3)

    populated_processors.processors[
        "RX81"
    ].SoftTrg.assert_not_called()

    populated_processors.processors[
        "RX82"
    ].SoftTrg.assert_called_once_with(3)


def test_trigger_requires_processors_for_soft_trigger(
    processors,
):
    with pytest.raises(
        ValueError,
        match="Proc needs to be specified",
    ):
        processors.trigger(
            kind=1,
            proc=None,
        )


@pytest.mark.parametrize(
    "kind",
    [0, 11, -1],
)
def test_trigger_rejects_soft_trigger_outside_valid_range(
    processors,
    kind,
):
    processors.processors = {
        "RX81": make_processor_mock("RX81"),
    }

    with pytest.raises(
        ValueError,
        match="between 1 and 10",
    ):
        processors.trigger(
            kind=kind,
            proc=["RX81"],
        )


def test_string_trigger_without_zbus_falls_back_to_soft_trigger_one(
    populated_processors,
    caplog,
):
    with caplog.at_level(logging.WARNING):
        populated_processors.trigger(
            kind="zBusA"
        )

    assert (
        "zBus trigger not available"
        in caplog.text
    )

    for processor in (
        populated_processors.processors.values()
    ):
        processor.SoftTrg.assert_called_once_with(1)


def test_trigger_zbusa_uses_zbus(
    populated_processors,
):
    zbus = MagicMock(name="zbus")
    populated_processors._zbus = zbus

    populated_processors.trigger(
        kind="zBusA"
    )

    zbus.zBusTrigA.assert_called_once_with(
        0,
        0,
        20,
    )
    zbus.zBusTrigB.assert_not_called()


def test_trigger_zbusb_uses_zbus(
    populated_processors,
):
    zbus = MagicMock(name="zbus")
    populated_processors._zbus = zbus

    populated_processors.trigger(
        kind="ZBUSB"
    )

    zbus.zBusTrigB.assert_called_once_with(
        0,
        0,
        20,
    )
    zbus.zBusTrigA.assert_not_called()


def test_trigger_rejects_unknown_string_when_zbus_exists(
    populated_processors,
):
    populated_processors._zbus = MagicMock(
        name="zbus"
    )

    with pytest.raises(
        ValueError,
        match="Unknown trigger type",
    ):
        populated_processors.trigger(
            kind="invalid"
        )