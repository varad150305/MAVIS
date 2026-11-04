"""
settings_gui.py — gesture and command configuration window (tkinter, no deps).

Gesture list on the left, editor of the selected gesture on the right:
  * finger checkboxes for pattern gestures,
  * hold/toggle mode with delay and repeat,
  * action and its parameters.

All changes are saved to config/gestures.json on button press.
The tray application watches the file and reloads it automatically.

Launch: python settings_gui.py  (or from the tray menu: "Settings")

Recording a new gesture opens a camera window. If recognition is running
in the tray at the same time — hit Pause there first to free the camera.
"""

from __future__ import annotations

import queue
import threading
import tkinter as tk
from tkinter import messagebox, simpledialog, ttk

from main import load_config, save_config

FINGERS = ["thumb", "index", "middle", "ring", "pinky"]
ACTIONS = ["hotkey", "press", "scroll", "command", "click", "mouse_mode", "none"]

# hints for the parameters field per action
PARAM_HINTS = {
    "hotkey": "keys separated by spaces: ctrl shift esc",
    "press": "keys separated by spaces: volumeup",
    "scroll": "number: -300 (negative = down)",
    "command": "command: notepad",
    "click": "button: left / right / middle",
    "mouse_mode": "no parameters needed",
    "none": "no parameters needed",
}


class SettingsWindow:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.cfg = load_config()
        self.current: str | None = None
        self.msg_queue: queue.Queue = queue.Queue()

        root.title("Gesture Control — Settings")
        root.geometry("720x460")
        root.minsize(640, 420)

        # ---------------- left panel: gesture list ----------------
        left = ttk.Frame(root, padding=8)
        left.pack(side="left", fill="y")

        ttk.Label(left, text="Gestures").pack(anchor="w")
        self.listbox = tk.Listbox(left, width=24, exportselection=False)
        self.listbox.pack(fill="y", expand=True, pady=4)
        self.listbox.bind("<<ListboxSelect>>", self._on_select)

        btns = ttk.Frame(left)
        btns.pack(fill="x")
        ttk.Button(btns, text="+ pattern", command=self._add_pattern).pack(
            side="left", expand=True, fill="x")
        ttk.Button(btns, text="+ record", command=self._record).pack(
            side="left", expand=True, fill="x")
        ttk.Button(left, text="Delete gesture", command=self._delete).pack(
            fill="x", pady=(4, 0))

        # ---------------- right panel: editor ----------------
        right = ttk.Frame(root, padding=8)
        right.pack(side="left", fill="both", expand=True)

        self.title_var = tk.StringVar(value="— select a gesture —")
        ttk.Label(right, textvariable=self.title_var,
                  font=("", 11, "bold")).pack(anchor="w")

        # fingers (pattern gestures only)
        self.fingers_frame = ttk.LabelFrame(right, text="Extended fingers", padding=6)
        self.finger_vars = [tk.IntVar() for _ in FINGERS]
        for var, name in zip(self.finger_vars, FINGERS):
            ttk.Checkbutton(self.fingers_frame, text=name, variable=var).pack(side="left")

        # trigger mode
        mode_frame = ttk.LabelFrame(right, text="Trigger mode", padding=6)
        mode_frame.pack(fill="x", pady=6)
        self.mode_frame = mode_frame
        self.mode_var = tk.StringVar(value="hold")
        ttk.Radiobutton(mode_frame, text="hold — hold the gesture, fires once",
                        variable=self.mode_var, value="hold").pack(anchor="w")
        ttk.Radiobutton(mode_frame, text="toggle — gesture turns on, repeat gesture turns off",
                        variable=self.mode_var, value="toggle").pack(anchor="w")

        row = ttk.Frame(mode_frame)
        row.pack(fill="x", pady=(6, 0))
        ttk.Label(row, text="Hold delay, s:").pack(side="left")
        self.hold_var = tk.StringVar()
        ttk.Entry(row, textvariable=self.hold_var, width=6).pack(side="left", padx=(4, 16))
        ttk.Label(row, text="Repeat every, s:").pack(side="left")
        self.repeat_var = tk.StringVar()
        ttk.Entry(row, textvariable=self.repeat_var, width=6).pack(side="left", padx=4)
        ttk.Label(mode_frame, foreground="gray",
                  text="Empty fields = default delay, no repeat").pack(anchor="w")

        # action
        act_frame = ttk.LabelFrame(right, text="Action", padding=6)
        act_frame.pack(fill="x", pady=6)
        self.action_var = tk.StringVar(value="none")
        combo = ttk.Combobox(act_frame, textvariable=self.action_var,
                             values=ACTIONS, state="readonly", width=14)
        combo.pack(anchor="w")
        combo.bind("<<ComboboxSelected>>", lambda _e: self._update_hint())

        self.params_var = tk.StringVar()
        ttk.Entry(act_frame, textvariable=self.params_var).pack(fill="x", pady=(6, 0))
        self.hint_var = tk.StringVar()
        ttk.Label(act_frame, textvariable=self.hint_var, foreground="gray").pack(anchor="w")

        ttk.Button(right, text="Apply & save",
                   command=self._apply).pack(anchor="e", pady=8)

        self._refresh_list()
        self.root.after(200, self._poll_queue)

    # ---------------- list ----------------

    def _refresh_list(self, select: str | None = None) -> None:
        self.listbox.delete(0, tk.END)
        for name, g in self.cfg["gestures"].items():
            mark = "P" if g["type"] == "pattern" else "R"
            self.listbox.insert(tk.END, f"[{mark}] {name}")
        if select:
            names = list(self.cfg["gestures"])
            if select in names:
                i = names.index(select)
                self.listbox.selection_set(i)
                self.listbox.see(i)
                self._load_gesture(select)

    def _on_select(self, _event) -> None:
        sel = self.listbox.curselection()
        if sel:
            name = list(self.cfg["gestures"])[sel[0]]
            self._load_gesture(name)

    # ---------------- editor ----------------

    def _load_gesture(self, name: str) -> None:
        self.current = name
        g = self.cfg["gestures"][name]
        self.title_var.set(f"{name}  ({g['type']})")

        if g["type"] == "pattern":
            self.fingers_frame.pack(fill="x", pady=6, before=self.mode_frame)
            for var, v in zip(self.finger_vars, g.get("pattern", [0] * 5)):
                var.set(int(v))
        else:
            self.fingers_frame.pack_forget()

        self.mode_var.set(g.get("mode", "hold"))
        self.hold_var.set(str(g["hold_sec"]) if "hold_sec" in g else "")
        self.repeat_var.set(str(g["repeat_sec"]) if "repeat_sec" in g else "")

        do = g.get("do", {"action": "none"})
        self.action_var.set(do.get("action", "none"))
        if "keys" in do:
            self.params_var.set(" ".join(do["keys"]))
        elif "cmd" in do:
            self.params_var.set(do["cmd"])
        elif "amount" in do:
            self.params_var.set(str(do["amount"]))
        elif "button" in do:
            self.params_var.set(do["button"])
        else:
            self.params_var.set("")
        self._update_hint()

    def _update_hint(self) -> None:
        self.hint_var.set(PARAM_HINTS.get(self.action_var.get(), ""))

    def _apply(self) -> None:
        if not self.current:
            return
        g = self.cfg["gestures"][self.current]

        if g["type"] == "pattern":
            g["pattern"] = [v.get() for v in self.finger_vars]

        g["mode"] = self.mode_var.get()
        try:
            g.pop("hold_sec", None)
            g.pop("repeat_sec", None)
            if self.hold_var.get().strip():
                g["hold_sec"] = float(self.hold_var.get().replace(",", "."))
            if self.repeat_var.get().strip():
                g["repeat_sec"] = float(self.repeat_var.get().replace(",", "."))
        except ValueError:
            messagebox.showerror("Error", "Delay and repeat must be numbers, e.g. 0.6")
            return

        kind = self.action_var.get()
        params = self.params_var.get().strip()
        do: dict = {"action": kind}
        if kind in ("hotkey", "press"):
            if not params:
                messagebox.showerror("Error", "Specify keys separated by spaces")
                return
            do["keys"] = params.split()
        elif kind == "scroll":
            try:
                do["amount"] = int(params or "-300")
            except ValueError:
                messagebox.showerror("Error", "The scroll parameter must be an integer")
                return
        elif kind == "command":
            do["cmd"] = params
        elif kind == "click":
            do["button"] = params or "left"
        g["do"] = do

        save_config(self.cfg)
        self._refresh_list(select=self.current)

    # ---------------- add / delete / record ----------------

    def _ask_name(self) -> str | None:
        name = simpledialog.askstring("New gesture", "Gesture name (latin letters):",
                                      parent=self.root)
        if not name:
            return None
        name = name.strip().replace(" ", "_")
        if name in self.cfg["gestures"]:
            messagebox.showerror("Error", f"Gesture '{name}' already exists")
            return None
        return name

    def _add_pattern(self) -> None:
        name = self._ask_name()
        if not name:
            return
        self.cfg["gestures"][name] = {
            "type": "pattern", "pattern": [0, 0, 0, 0, 0],
            "mode": "hold", "do": {"action": "none"},
        }
        save_config(self.cfg)
        self._refresh_list(select=name)

    def _delete(self) -> None:
        if not self.current:
            return
        if messagebox.askyesno("Delete", f"Delete gesture '{self.current}'?"):
            self.cfg["gestures"].pop(self.current, None)
            self.current = None
            self.title_var.set("— select a gesture —")
            save_config(self.cfg)
            self._refresh_list()

    def _record(self) -> None:
        name = self._ask_name()
        if not name:
            return
        messagebox.showinfo(
            "Record gesture",
            "A camera window will open.\n\nHold the gesture and press SPACE to capture "
            "a sample (12 needed, slightly vary the angle).\nQ — cancel.\n\n"
            "If recognition is running in the tray — hit Pause there first!",
        )
        # The cv2 loop runs in a separate thread so the tkinter window
        # doesn't freeze. The thread never touches widgets: results go
        # into a queue that the main thread drains in _poll_queue
        # (tkinter is not thread-safe).
        threading.Thread(target=self._record_worker, args=(name,), daemon=True).start()

    def _record_worker(self, name: str, samples_needed: int = 12) -> None:
        import cv2

        from gesture_engine import landmarks_to_features
        from hand_tracker import HandTracker

        cap = cv2.VideoCapture(self.cfg["settings"].get("camera_index", 0))
        if not cap.isOpened():
            self.msg_queue.put(("error", "Camera unavailable. Hit Pause in the tray."))
            return
        tracker = HandTracker()
        samples: list[list[float]] = []
        try:
            while len(samples) < samples_needed:
                ok, frame = cap.read()
                if not ok:
                    break
                if self.cfg["settings"].get("mirror", True):
                    frame = cv2.flip(frame, 1)
                hands = tracker.process(frame)
                tracker.draw(frame)
                cv2.putText(frame, f"'{name}': {len(samples)}/{samples_needed}  [SPACE=snap  Q=quit]",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.imshow("Record gesture", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    self.msg_queue.put(("info", "Recording cancelled."))
                    return
                if key == ord(" ") and hands:
                    samples.append(landmarks_to_features(hands[0]).tolist())
        finally:
            cap.release()
            tracker.close()
            cv2.destroyAllWindows()

        if len(samples) < samples_needed:
            self.msg_queue.put(("error", "Not enough samples, gesture not saved."))
            return
        self.cfg["gestures"][name] = {"type": "recorded", "samples": samples,
                                      "mode": "hold", "do": {"action": "none"}}
        save_config(self.cfg)
        self.msg_queue.put(("recorded", name))

    def _poll_queue(self) -> None:
        try:
            while True:
                kind, payload = self.msg_queue.get_nowait()
                if kind == "recorded":
                    self._refresh_list(select=payload)
                    messagebox.showinfo("Done",
                                        f"Gesture '{payload}' recorded.\nNow assign it an action.")
                elif kind == "error":
                    messagebox.showerror("Error", payload)
                else:
                    messagebox.showinfo("Info", payload)
        except queue.Empty:
            pass
        self.root.after(200, self._poll_queue)


def main() -> None:
    root = tk.Tk()
    try:
        ttk.Style().theme_use("vista")
    except tk.TclError:
        pass
    SettingsWindow(root)
    root.mainloop()


if __name__ == "__main__":
    main()
