$ErrorActionPreference = "Stop"

$Repo = "D:\experiment"
$Wheelhouse = Join-Path $Repo "wheelhouse_py312_linux"
$Log = Join-Path $Repo "wheelhouse_py312_linux.log"
$Remote = "busanbusi@192.168.43.5"
$RemoteWheelhouse = "/home/busanbusi/wheelhouse_py312_linux"
$RemotePython = "/home/busanbusi/.virtualenvs/experiment/bin/python"
$RemoteProject = "/home/busanbusi/experiment"

New-Item -ItemType Directory -Force -Path $Wheelhouse | Out-Null
Set-Location $Repo

"==== wheel download start $(Get-Date -Format o) ====" | Tee-Object -FilePath $Log

python -m pip download `
  --dest $Wheelhouse `
  --only-binary=:all: `
  --platform manylinux_2_28_x86_64 `
  --platform manylinux2014_x86_64 `
  --python-version 312 `
  --implementation cp `
  --abi cp312 `
  -r requirements.txt `
  tushare pyarrow 2>&1 | Tee-Object -FilePath $Log -Append

"==== upload wheels $(Get-Date -Format o) ====" | Tee-Object -FilePath $Log -Append
ssh -o BatchMode=yes -o ConnectTimeout=12 $Remote "mkdir -p $RemoteWheelhouse" 2>&1 | Tee-Object -FilePath $Log -Append
scp "$Wheelhouse\*" "${Remote}:$RemoteWheelhouse/" 2>&1 | Tee-Object -FilePath $Log -Append

"==== remote offline install $(Get-Date -Format o) ====" | Tee-Object -FilePath $Log -Append
ssh -o BatchMode=yes -o ConnectTimeout=12 $Remote "cd $RemoteProject && $RemotePython -m pip install --no-index --find-links $RemoteWheelhouse -r requirements.txt tushare pyarrow" 2>&1 | Tee-Object -FilePath $Log -Append

"==== remote import check $(Get-Date -Format o) ====" | Tee-Object -FilePath $Log -Append
ssh -o BatchMode=yes -o ConnectTimeout=12 $Remote "$RemotePython - <<'PY'
import importlib
mods = 'polars pandas numpy sklearn yaml matplotlib lightgbm tushare pyarrow'.split()
for mod in mods:
    m = importlib.import_module(mod)
    print(mod, getattr(m, '__version__', 'OK'))
PY" 2>&1 | Tee-Object -FilePath $Log -Append

"==== done $(Get-Date -Format o) ====" | Tee-Object -FilePath $Log -Append
