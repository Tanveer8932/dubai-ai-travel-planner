<#
.SYNOPSIS
    Windows task runner (Makefile equivalent).
.EXAMPLE
    .\scripts\tasks.ps1 setup
    .\scripts\tasks.ps1 test
#>
param(
    [Parameter(Position = 0)]
    [ValidateSet('setup', 'test', 'lint', 'run', 'eval', 'help')]
    [string]$Task = 'help'
)

$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$Py = Join-Path $Root '.venv\Scripts\python.exe'

function Invoke-Checked([string]$Exe, [string[]]$Arguments) {
    & $Exe @Arguments
    if ($LASTEXITCODE -ne 0) { throw "'$Exe $($Arguments -join ' ')' failed with exit code $LASTEXITCODE" }
}

function Assert-Venv {
    if (-not (Test-Path $Py)) { throw "No .venv found. Run: .\scripts\tasks.ps1 setup" }
}

switch ($Task) {
    'setup' {
        if (-not (Test-Path $Py)) {
            Write-Host 'Creating .venv with Python 3.12 ...'
            Invoke-Checked 'py' @('-3.12', '-m', 'venv', '.venv')
        } else {
            Write-Host 'Reusing existing .venv'
        }
        Invoke-Checked $Py @('-m', 'pip', 'install', '--upgrade', 'pip')
        Invoke-Checked $Py @('-m', 'pip', 'install', '-e', '.[dev,langgraph,crewai,ui]')
        if (-not (Test-Path '.env')) {
            Copy-Item '.env.example' '.env'
            Write-Host 'Created .env from .env.example (fill in only what you need).'
        }
    }
    'test' { Assert-Venv; Invoke-Checked $Py @('-m', 'pytest', '-m', 'not llm') }
    'lint' {
        Assert-Venv
        Invoke-Checked $Py @('-m', 'ruff', 'check', '.')
        Invoke-Checked $Py @('-m', 'mypy', 'src')
    }
    'run' {
        Assert-Venv
        if (-not (Test-Path 'app\streamlit_app.py')) { throw 'Streamlit app arrives in Phase 5.' }
        Invoke-Checked $Py @('-m', 'streamlit', 'run', 'app\streamlit_app.py')
    }
    'eval' {
        Assert-Venv
        if (-not (Test-Path 'scripts\run_eval.py')) { throw 'Evaluation runner arrives in Phase 7.' }
        Invoke-Checked $Py @('scripts\run_eval.py', '--provider', 'mock')
    }
    default {
        Write-Host 'Usage: .\scripts\tasks.ps1 <setup|test|lint|run|eval>'
    }
}
