param(
    [Parameter(Mandatory = $true)]
    [string]$ImagePath,

    [Parameter(Mandatory = $false)]
    [string]$OutputPath
)

$ErrorActionPreference = "Stop"

function Get-AsyncResult {
    param(
        [Parameter(Mandatory = $true)]
        $AsyncOperation,

        [Parameter(Mandatory = $true)]
        [Type]$ResultType
    )

    $method = [System.WindowsRuntimeSystemExtensions].GetMethods() |
        Where-Object { $_.Name -eq 'AsTask' -and $_.IsGenericMethod -and $_.GetParameters().Count -eq 1 } |
        Select-Object -First 1

    if ($null -eq $method) {
        throw "Could not locate the Windows Runtime task bridge."
    }

    $genericMethod = $method.MakeGenericMethod($ResultType)
    $task = $genericMethod.Invoke($null, @($AsyncOperation))
    $task.Wait()
    return $task.Result
}

Add-Type -AssemblyName System.Runtime.WindowsRuntime

$null = [Windows.Storage.StorageFile, Windows.Storage, ContentType = WindowsRuntime]
$null = [Windows.Storage.Streams.IRandomAccessStream, Windows.Storage.Streams, ContentType = WindowsRuntime]
$null = [Windows.Graphics.Imaging.BitmapDecoder, Windows.Graphics.Imaging, ContentType = WindowsRuntime]
$null = [Windows.Graphics.Imaging.SoftwareBitmap, Windows.Graphics.Imaging, ContentType = WindowsRuntime]
$null = [Windows.Media.Ocr.OcrEngine, Windows.Media.Ocr, ContentType = WindowsRuntime]
$null = [Windows.Media.Ocr.OcrResult, Windows.Media.Ocr, ContentType = WindowsRuntime]
$null = [Windows.Globalization.Language, Windows.Globalization, ContentType = WindowsRuntime]

$resolvedPath = (Resolve-Path -LiteralPath $ImagePath).Path
$file = Get-AsyncResult ([Windows.Storage.StorageFile]::GetFileFromPathAsync($resolvedPath)) ([Windows.Storage.StorageFile])
$stream = Get-AsyncResult ($file.OpenAsync([Windows.Storage.FileAccessMode]::Read)) ([Windows.Storage.Streams.IRandomAccessStream])
$decoder = Get-AsyncResult ([Windows.Graphics.Imaging.BitmapDecoder]::CreateAsync($stream)) ([Windows.Graphics.Imaging.BitmapDecoder])
$bitmap = Get-AsyncResult ($decoder.GetSoftwareBitmapAsync()) ([Windows.Graphics.Imaging.SoftwareBitmap])

$engine = [Windows.Media.Ocr.OcrEngine]::TryCreateFromUserProfileLanguages()
if ($null -eq $engine) {
    $engine = [Windows.Media.Ocr.OcrEngine]::TryCreateFromLanguage([Windows.Globalization.Language]::new("en-US"))
}
if ($null -eq $engine) {
    throw "Windows OCR engine could not be initialized."
}

$ocrResult = Get-AsyncResult ($engine.RecognizeAsync($bitmap)) ([Windows.Media.Ocr.OcrResult])
$lineCount = if ($ocrResult.Lines) { @($ocrResult.Lines).Count } else { 0 }
$wordCount = if ([string]::IsNullOrWhiteSpace($ocrResult.Text)) {
    0
} else {
    ($ocrResult.Text -split '\s+' | Where-Object { $_ }).Count
}

$payload = [PSCustomObject]@{
    text = [string]$ocrResult.Text
    line_count = [int]$lineCount
    word_count = [int]$wordCount
    width = [int]$decoder.PixelWidth
    height = [int]$decoder.PixelHeight
    language = if ($engine.RecognizerLanguage) { [string]$engine.RecognizerLanguage.LanguageTag } else { "" }
} | ConvertTo-Json -Compress

if (-not [string]::IsNullOrWhiteSpace($OutputPath)) {
    $directory = Split-Path -Parent $OutputPath
    if (-not [string]::IsNullOrWhiteSpace($directory)) {
        New-Item -ItemType Directory -Path $directory -Force | Out-Null
    }
    [System.IO.File]::WriteAllText($OutputPath, $payload, [System.Text.Encoding]::UTF8)
}

Write-Output $payload
