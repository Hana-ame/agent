# UI 驱动工具：真实鼠标 / 键盘 / 截屏（Windows + Edge）
# 用法：. C:\Users\lumin\Downloads\agent\scripts\ui.ps1
# 只依赖 .NET / user32，不安装任何东西。键盘走 keybd_event（OS 级注入，不阻塞）。

Add-Type -AssemblyName System.Windows.Forms, System.Drawing

if (-not ("UiNative" -as [type])) {
    Add-Type @"
using System;
using System.Runtime.InteropServices;
public class UiNative {
    [DllImport("user32.dll")] public static extern bool SetCursorPos(int X, int Y);
    [DllImport("user32.dll")] public static extern void mouse_event(uint f, uint dx, uint dy, uint d, IntPtr e);
    [DllImport("user32.dll")] public static extern void keybd_event(byte vk, byte scan, uint flags, IntPtr extra);
    [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr h);
    [DllImport("user32.dll")] public static extern bool ShowWindow(IntPtr h, int c);
    [DllImport("user32.dll")] public static extern bool GetWindowRect(IntPtr h, out RECT r);
    [DllImport("user32.dll")] public static extern IntPtr GetForegroundWindow();
    [StructLayout(LayoutKind.Sequential)] public struct RECT { public int Left, Top, Right, Bottom; }
}
"@
}

$UiVk = @{ ENTER = 0x0D; TAB = 0x09; ESC = 0x1B; CTRL = 0x11; SHIFT = 0x10; ALT = 0x12
           DOWN = 0x28; UP = 0x26; LEFT = 0x25; RIGHT = 0x27; HOME = 0x24; END = 0x23
           PGUP = 0x21; PGDN = 0x22; F5 = 0x74; F6 = 0x75; BACK = 0x08; DEL = 0x2E }

function Ui-Screen {
    param([string]$Path = "$env:TEMP\ui-shot.png")
    $b = [System.Windows.Forms.SystemInformation]::VirtualScreen
    $bmp = New-Object System.Drawing.Bitmap $b.Width, $b.Height
    $g = [System.Drawing.Graphics]::FromImage($bmp)
    $g.CopyFromScreen($b.Left, $b.Top, 0, 0, $bmp.Size)
    $bmp.Save($Path, [System.Drawing.Imaging.ImageFormat]::Png)
    $g.Dispose(); $bmp.Dispose()
    return $Path
}

function Ui-MouseMove { param([int]$X, [int]$Y) [UiNative]::SetCursorPos($X, $Y) | Out-Null; Start-Sleep -Milliseconds 120 }

function Ui-Click {
    param([int]$X = -1, [int]$Y = -1, [switch]$Double)
    if ($X -ge 0) { Ui-MouseMove -X $X -Y $Y }
    [UiNative]::mouse_event(0x0002, 0, 0, 0, [IntPtr]::Zero)
    Start-Sleep -Milliseconds 40
    [UiNative]::mouse_event(0x0004, 0, 0, 0, [IntPtr]::Zero)
    if ($Double) {
        Start-Sleep -Milliseconds 70
        [UiNative]::mouse_event(0x0002, 0, 0, 0, [IntPtr]::Zero)
        Start-Sleep -Milliseconds 40
        [UiNative]::mouse_event(0x0004, 0, 0, 0, [IntPtr]::Zero)
    }
    Start-Sleep -Milliseconds 150
}

function Ui-Press { param([string]$Key = "ENTER", [int]$Times = 1)
    $vk = $UiVk[$Key]; if ($null -eq $vk) { throw "未知按键 $Key" }
    for ($i = 0; $i -lt $Times; $i++) {
        [UiNative]::keybd_event([byte]$vk, 0, 0, [IntPtr]::Zero)
        Start-Sleep -Milliseconds 30
        [UiNative]::keybd_event([byte]$vk, 0, 2, [IntPtr]::Zero)
        Start-Sleep -Milliseconds 60
    }
}

function Ui-Hotkey { param([string]$Combo = "CTRL+L")
    $parts = $Combo.Split('+')
    $mods = @(); $key = $parts[-1]
    foreach ($m in $parts[0..($parts.Count - 2)]) {
        if ($m -and $UiVk.ContainsKey($m.ToUpper())) { $mods += $UiVk[$m.ToUpper()] }
    }
    foreach ($m in $mods) { [UiNative]::keybd_event([byte]$m, 0, 0, [IntPtr]::Zero); Start-Sleep -Milliseconds 20 }
    # 单字符按键用大写 ASCII
    $vk = $UiVk[$key]; if ($null -eq $vk) { $vk = [int][char]$key.ToUpper() }
    [UiNative]::keybd_event([byte]$vk, 0, 0, [IntPtr]::Zero); Start-Sleep -Milliseconds 30
    [UiNative]::keybd_event([byte]$vk, 0, 2, [IntPtr]::Zero); Start-Sleep -Milliseconds 30
    for ($i = $mods.Count - 1; $i -ge 0; $i--) { [UiNative]::keybd_event([byte]$mods[$i], 0, 2, [IntPtr]::Zero); Start-Sleep -Milliseconds 20 }
    Start-Sleep -Milliseconds 60
}

function Ui-TypeText { param([string]$Text, [int]$DelayMs = 8)
    foreach ($ch in $Text.ToCharArray()) {
        $code = [int][char]$ch
        if ($code -gt 255) { continue }   # 非 ASCII 走 Ui-PasteText
        [UiNative]::keybd_event(0, [byte]$code, 4, [IntPtr]::Zero)          # KEYEVENTF_UNICODE down
        [UiNative]::keybd_event(0, [byte]$code, 6, [IntPtr]::Zero)          # unicode up
        Start-Sleep -Milliseconds $DelayMs
    }
}

function Ui-PasteText { param([string]$Text)
    Set-Clipboard -Value $Text
    Start-Sleep -Milliseconds 120
    Ui-Hotkey "CTRL+V"
    Start-Sleep -Milliseconds 150
}

function Ui-EdgeWindows {
    Get-Process msedge -ErrorAction SilentlyContinue | Where-Object { $_.MainWindowHandle -ne 0 } |
        Select-Object Id, MainWindowHandle, MainWindowTitle
}

function Ui-ActivateEdge {
    param([string]$TitleLike = "")
    $procs = Get-Process msedge -ErrorAction SilentlyContinue | Where-Object { $_.MainWindowHandle -ne 0 }
    if (-not $procs) { return $null }
    if ($TitleLike) {
        $hit = $procs | Where-Object { $_.MainWindowTitle -like "*$TitleLike*" } | Select-Object -First 1
        if ($hit) { $procs = @($hit) }
    }
    $p = $procs | Select-Object -First 1
    [UiNative]::ShowWindow($p.MainWindowHandle, 9) | Out-Null
    [UiNative]::SetForegroundWindow($p.MainWindowHandle) | Out-Null
    Start-Sleep -Milliseconds 400
    return $p
}

function Ui-Maximize {
    param([string]$TitleLike = "")
    $p = Ui-ActivateEdge -TitleLike $TitleLike
    if ($p) { [UiNative]::ShowWindow($p.MainWindowHandle, 3) | Out-Null; Start-Sleep -Milliseconds 500 }
    return $p
}

function Ui-WindowRect {
    param([string]$TitleLike = "")
    $procs = Get-Process msedge -ErrorAction SilentlyContinue | Where-Object { $_.MainWindowHandle -ne 0 }
    if ($TitleLike) { $procs = $procs | Where-Object { $_.MainWindowTitle -like "*$TitleLike*" } }
    $p = $procs | Select-Object -First 1
    if (-not $p) { return $null }
    $r = New-Object UiNative+RECT
    [UiNative]::GetWindowRect($p.MainWindowHandle, [ref]$r) | Out-Null
    return [pscustomobject]@{ Left = $r.Left; Top = $r.Top; Width = $r.Right - $r.Left; Height = $r.Bottom - $r.Top }
}

function Ui-CopyPage {
    param([string]$TitleLike = "", [switch]$NoClick)
    if (-not $NoClick) {
        $r = Ui-WindowRect -TitleLike $TitleLike
        if ($r) { Ui-Click -X ([int]($r.Left + $r.Width / 2)) -Y ([int]($r.Top + $r.Height / 2)) }
    }
    Set-Clipboard -Value " "
    Start-Sleep -Milliseconds 100
    Ui-Hotkey "CTRL+A"
    Ui-Hotkey "CTRL+C"
    Start-Sleep -Milliseconds 600
    return (Get-Clipboard -Raw)
}
