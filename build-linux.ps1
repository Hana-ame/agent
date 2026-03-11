param(
    [string]$Path = ".",
    [string]$Output = "",
    [string]$GOOS = "linux",
    [string]$GOARCH = "amd64",
    [switch]$CgoEnabled = $false,
    [string]$Ldflags = ""
)

# 检查Go是否安装
$goCmd = Get-Command go -ErrorAction SilentlyContinue
if (-not $goCmd) {
    Write-Error "Go is not installed or not in PATH."
    exit 1
}

# 确定输出文件名
if (-not $Output) {
    # 如果Path是目录，取目录名；如果是文件，取文件名（无扩展名）
    if (Test-Path $Path -PathType Container) {
        $Output = Split-Path $Path -Leaf
    } else {
        $Output = [System.IO.Path]::GetFileNameWithoutExtension($Path)
    }
}
# 在Windows上，如果输出没有扩展名，没问题，因为Linux二进制无扩展。
# 但如果有.exe扩展名，最好去掉，因为Linux不需要。
if ($Output.EndsWith(".exe")) {
    $Output = $Output -replace "\.exe$", ""
}

# 设置环境变量
$env:GOOS = $GOOS
$env:GOARCH = $GOARCH
if (-not $CgoEnabled) {
    $env:CGO_ENABLED = 0
} else {
    $env:CGO_ENABLED = 1
}

Write-Host "Cross-compiling for $GOOS/$GOARCH..."
Write-Host "Building from: $Path"
Write-Host "Output: $Output"

# 构建命令
$buildArgs = @("build")
if ($Ldflags) {
    $buildArgs += "-ldflags"
    $buildArgs += "$Ldflags"
}
$buildArgs += "-o"
$buildArgs += "$Output"
$buildArgs += "$Path"

# 执行
& $goCmd.Source @buildArgs

if ($LASTEXITCODE -ne 0) {
    Write-Error "Build failed."
    exit $LASTEXITCODE
} else {
    Write-Host "Build succeeded. Output: $Output"
    # 可以显示文件信息
    Get-Item $Output
}

scp golang-server root@cloudcone.moonchan.xyz:~