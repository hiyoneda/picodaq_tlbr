"""GUI smoke test with the simulator (offscreen)."""
import os
import time

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _wait(app, cond, timeout):
    t = time.time()
    while time.time() - t < timeout:
        app.processEvents()
        if cond():
            return True
        time.sleep(0.05)
    return False


def test_gui_tune_then_run(tmp_path, config):
    import pyqtgraph as pg
    import yaml

    from picodaq.gui import MainWindow

    config["sim_realtime"] = True
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(config))
    app = pg.mkQApp()
    win = MainWindow(path, simulate=True)
    win.show()

    win.start()  # tune, not saved
    assert _wait(app, lambda: win.hists.events > 50, 20)
    win.thr_spin.setValue(100)
    win.apply_settings()
    assert _wait(app, lambda: win.threshold_now == 100, 10)
    win.stop()
    assert _wait(app, lambda: not win.running, 15)
    assert not (tmp_path / "data").exists()

    win.mode_box.setCurrentIndex(1)  # run
    assert win.file_label.text().startswith("Next file:") and "tlbr_00000.h5" in win.file_label.text()
    win.base_edit.setText("bad/name")
    assert "⚠" in win.file_label.text()
    win.start()
    assert not win.running  # refused
    win.base_edit.setText("Cs137_run1")
    win.dir_edit.setText(str(tmp_path / "data2"))
    assert "Cs137_run1_00000.h5" in win.file_label.text()
    win.start()
    assert not win.base_edit.isEnabled() and not win.dir_edit.isEnabled()
    assert not win.thr_spin.isEnabled() and not win.apply_btn.isEnabled()
    assert _wait(app, lambda: win.total > 50, 20)
    assert win.file_label.text().startswith("Saving: ") and "Cs137_run1_00000.h5" in win.file_label.text()
    win.stop()
    assert _wait(app, lambda: not win.running, 15)
    assert list((tmp_path / "data2").glob("Cs137_run1_0*.h5"))
    assert "Cs137_run1_00001.h5" in win.file_label.text()  # next one, never overwrite
    win.close()
