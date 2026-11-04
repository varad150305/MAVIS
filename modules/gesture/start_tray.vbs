' Launch the tray application without a console window (Windows).
' Double-click this file — the icon appears in the tray.
Set fso = CreateObject("Scripting.FileSystemObject")
Set sh = CreateObject("WScript.Shell")
sh.CurrentDirectory = fso.GetParentFolderName(WScript.ScriptFullName)
sh.Run """.venv\Scripts\pythonw.exe"" tray_app.py", 0, False
