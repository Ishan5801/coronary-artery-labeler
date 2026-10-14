Option Explicit

Dim shell, fso, root, command
Set shell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")
root = fso.GetParentFolderName(WScript.ScriptFullName)
shell.CurrentDirectory = root

' Start the local API and static web server without showing a terminal window.
command = "cmd /c cd /d """ & root & """ && (py -3 server.py || python server.py) >> server.log 2>&1"
shell.Run command, 0, False

' Give the server a moment to bind before opening the app.
WScript.Sleep 1500
shell.Run "http://127.0.0.1:8765", 1, False
