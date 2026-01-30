\
param(
  [Parameter(Mandatory=$true)]
  [string]$RepoUrl,
  [string]$Branch = "main"
)

$ErrorActionPreference = "Stop"

if (!(Test-Path ".git")) {
  git init | Out-Null
}

git add .
$diff = git diff --cached --name-only
if ($diff) {
  git commit -m "Add RAP-Lite trainable baseline" | Out-Null
} else {
  Write-Host "No changes to commit."
}

git branch -M $Branch | Out-Null

$origin = git remote get-url origin 2>$null
if ($LASTEXITCODE -eq 0 -and $origin) {
  git remote set-url origin $RepoUrl | Out-Null
} else {
  git remote add origin $RepoUrl | Out-Null
}

git push -u origin $Branch
Write-Host "✅ Pushed to $RepoUrl ($Branch)"
